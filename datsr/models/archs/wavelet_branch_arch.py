# datsr/models/archs/wavelet_branch_arch.py
"""
小波频域分支: DWT 分解 + 高频子带对齐迁移.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg


class DWTForward(nn.Module):
    """Haar 小波正变换 (固定滤波器, 无可学习参数)"""

    def __init__(self):
        super(DWTForward, self).__init__()
        # Haar 低通/高通滤波器
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        # (4, 1, 2, 2)
        filts = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filts', filts)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            ll: (B, C, H/2, W/2)
            highfreq: (B, C*3, H/2, W/2)  — LH, HL, HH 沿通道拼接
        """
        B, C, H, W = x.shape
        # 对每个通道独立做卷积: 用 groups=C
        # 扩展滤波器到 (4*C, 1, 2, 2), groups=C
        filts = self.filts.repeat(C, 1, 1, 1)  # (4*C, 1, 2, 2)
        y = F.conv2d(x, filts, stride=2, groups=C)  # (B, 4*C, H/2, W/2)
        # 重排: (B, C, 4, H/2, W/2)
        y = y.reshape(B, C, 4, H // 2, W // 2)
        ll = y[:, :, 0, :, :]  # (B, C, H/2, W/2)
        lh = y[:, :, 1, :, :]  # (B, C, H/2, W/2)
        hl = y[:, :, 2, :, :]  # (B, C, H/2, W/2)
        hh = y[:, :, 3, :, :]  # (B, C, H/2, W/2)
        highfreq = torch.cat([lh, hl, hh], dim=1)  # (B, C*3, H/2, W/2)
        return ll, highfreq


class WaveletFrequencyBranch(nn.Module):
    """小波频域分支: DWT 分解 + 高频迁移"""

    def __init__(self, out_channels=64):
        super(WaveletFrequencyBranch, self).__init__()
        self.dwt = DWTForward()
        self.out_channels = out_channels
        # 高频融合: 9ch (3通道 × 3子带) → out_channels
        self.highfreq_fusion = nn.Sequential(
            nn.Conv2d(9, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )

    def dwt_forward(self, img):
        """
        对图像做 DWT 分解
        Args:
            img: (B, 3, H, W)
        Returns:
            ll: (B, 3, H/2, W/2)
            highfreq: (B, 9, H/2, W/2)  — LH/HL/HH 拼接
        """
        ll, highfreq = self.dwt(img)
        return ll, highfreq

    def warp_highfreq(self, highfreq_r, flow):
        """
        用光流 warp Ref 高频子带, 然后融合为 F_wav
        Args:
            highfreq_r: (B, 9, H, W) — Ref 的 LH/HL/HH 拼接
            flow: (B, H, W, 2) — 光流 (像素偏移量)
        Returns:
            F_wav: (B, out_channels, H, W) — 对齐后的高频特征
        """
        # flow_warp: 用 grid_sample 实现亚像素级 warp
        assert highfreq_r.size()[2:] == flow.size()[1:3], \
            f"Spatial size mismatch: highfreq_r {highfreq_r.shape} vs flow {flow.shape}"

        _, _, h, w = highfreq_r.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=highfreq_r.dtype, device=highfreq_r.device),
            torch.arange(0, w, dtype=highfreq_r.dtype, device=highfreq_r.device))
        grid = torch.stack((grid_x, grid_y), dim=2).float()  # (H, W, 2)
        grid = grid.unsqueeze(0)  # (1, H, W, 2)

        vgrid = grid + flow  # (B, H, W, 2)
        # 归一化到 [-1, 1]
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)  # (B, H, W, 2)

        warped_highfreq = F.grid_sample(
            highfreq_r, vgrid_scaled,
            mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 9, H, W)

        # 融合为 F_wav
        F_wav = self.highfreq_fusion(warped_highfreq)  # (B, out_channels, H, W)
        return F_wav

    def forward(self, img_in_up, img_ref):
        """完整前向 (一般不直接调用, 而是分步调用 dwt_forward + warp_highfreq)"""
        ll_y, _ = self.dwt_forward(img_in_up)
        ll_r, highfreq_r = self.dwt_forward(img_ref)
        return ll_y, ll_r, highfreq_r


class WaveletVGGFeatureExtractor(nn.Module):
    """VGG feature extractor with Haar WavePool for WTRN-style LL matching.

    The LL path replaces VGG max-pooling before conv2/conv3, while the
    discarded LH/HL/HH feature sub-bands are kept for reference texture
    transfer.
    """

    def __init__(self):
        super(WaveletVGGFeatureExtractor, self).__init__()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1'
        ]
        features = getattr(vgg, 'vgg16')(pretrained=True).features[:11]

        self.slice1 = nn.Sequential(OrderedDict(
            (name, layer) for name, layer in zip(vgg16_layers[:4], features[:4])
        ))
        self.slice2 = nn.Sequential(OrderedDict(
            (name, layer) for name, layer in zip(vgg16_layers[5:9], features[5:9])
        ))
        self.slice3 = nn.Sequential(OrderedDict(
            [('conv3_1', features[10])]
        ))
        self.pool1 = DWTForward()
        self.pool2 = DWTForward()
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        x = self.slice1(x)
        ll1, hf1 = self.pool1(x)
        x = self.slice2(ll1)
        ll2, hf2 = self.pool2(x)
        dense = self.slice3(ll2)
        return dense, {'pool1': hf1, 'pool2': hf2}


class WaveletVGGFrequencyBranch(nn.Module):
    """Feature-domain low-frequency matching and high-frequency transfer.

    This branch follows the WTRN idea more closely than RGB-domain DWT:
    matching features are extracted from the VGG LL path, and reference
    LH/HL/HH feature sub-bands are warped with the resulting correspondence.
    The transferred feature-domain high frequencies are projected to the
    existing DATSR medium-scale `F_wav` interface.
    """

    def __init__(self, out_channels=64):
        super(WaveletVGGFrequencyBranch, self).__init__()
        self.out_channels = out_channels
        self.extractor = WaveletVGGFeatureExtractor()
        self.pool1_proj = nn.Sequential(
            nn.Conv2d(64 * 3, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.pool2_proj = nn.Sequential(
            nn.Conv2d(128 * 3, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.hf_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    @staticmethod
    def _flow_warp(x, flow):
        assert x.size()[-2:] == flow.size()[1:3], (
            f"Spatial size mismatch: x {x.shape} vs flow {flow.shape}")
        _, _, h, w = x.size()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, dtype=x.dtype, device=x.device),
            torch.arange(0, w, dtype=x.dtype, device=x.device))
        grid = torch.stack((grid_x, grid_y), dim=2).float().unsqueeze(0)
        vgrid = grid + flow
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        return F.grid_sample(
            x, vgrid_scaled, mode='bilinear',
            padding_mode='zeros', align_corners=True)

    @staticmethod
    def _resize_flow_to(flow, size):
        target_h, target_w = size
        b, h, w, _ = flow.shape
        if (h, w) == (target_h, target_w):
            return flow
        flow_chw = flow.permute(0, 3, 1, 2)
        flow_chw = F.interpolate(
            flow_chw, size=(target_h, target_w),
            mode='bilinear', align_corners=True)
        flow_chw[:, 0] *= target_w / max(w, 1)
        flow_chw[:, 1] *= target_h / max(h, 1)
        return flow_chw.permute(0, 2, 3, 1)

    def extract_matching_features(self, img_in_up, img_ref):
        dense_in, _ = self.extractor(img_in_up)
        dense_ref, ref_hf = self.extractor(img_ref)
        return {
            'dense_features1': dense_in,
            'dense_features2': dense_ref,
            'ref_hf_pool1': ref_hf['pool1'],
            'ref_hf_pool2': ref_hf['pool2'],
        }

    def transfer_ref_hf(self, wavelet_features, pre_flow):
        hf_pool1 = wavelet_features['ref_hf_pool1']
        hf_pool2 = wavelet_features['ref_hf_pool2']

        flow_pool1 = self._resize_flow_to(
            pre_flow['relu2_1'], hf_pool1.shape[-2:])
        flow_pool2 = self._resize_flow_to(
            pre_flow['relu3_1'], hf_pool2.shape[-2:])

        hf_pool1 = self._flow_warp(hf_pool1, flow_pool1)
        hf_pool2 = self._flow_warp(hf_pool2, flow_pool2)

        feat1 = self.pool1_proj(hf_pool1)
        feat2 = self.pool2_proj(hf_pool2)
        feat2 = F.interpolate(
            feat2, size=feat1.shape[-2:], mode='bilinear',
            align_corners=False)
        return self.hf_fusion(torch.cat([feat1, feat2], dim=1))


def upsample_offsets(pre_offset, pre_flow, pre_similarity, scale=2):
    """将 LL 子带上计算的 offset/flow/similarity 上采样到原始分辨率.

    由于匹配在 80x80 的 LL 子带上进行 (VGG conv3_1 输出 20x20),
    而主网络在 160x160 上工作 (期望 VGG conv3_1 对应 40x40),
    需要将所有空间维度 ×scale, 同时 flow/offset 的值也 ×scale.

    Args:
        pre_offset: dict, keys=['relu1_1','relu2_1','relu3_1'],
                    values shape: (B, 9, H, W, 2)
        pre_flow: dict, keys=['relu1_1','relu2_1','relu3_1'],
                  values shape: (B, H, W, 2)
        pre_similarity: dict, keys=['relu1_1','relu2_1','relu3_1'],
                        values shape: (B, 1, H+2, W+2) 或类似
        scale: int, 上采样倍数. Default: 2.

    Returns:
        new_offset, new_flow, new_similarity: 上采样后的 dict
    """
    new_offset = {}
    new_flow = {}
    new_similarity = {}

    for key in pre_flow:
        # flow: (B, H, W, 2) → (B, H*scale, W*scale, 2)
        flow = pre_flow[key]  # (B, h, w, 2)
        B, h, w, _ = flow.shape
        flow_permuted = flow.permute(0, 3, 1, 2)  # (B, 2, h, w)
        flow_up = F.interpolate(
            flow_permuted, scale_factor=scale,
            mode='bilinear', align_corners=True)  # (B, 2, h*s, w*s)
        flow_up = flow_up.permute(0, 2, 3, 1) * scale  # 坐标值 ×scale
        new_flow[key] = flow_up

    for key in pre_offset:
        # offset: (B, 9, H, W, 2)
        offset = pre_offset[key]
        if offset.dim() == 5:
            offset = offset.unsqueeze(1)
        B, K, N, h, w, two = offset.shape
        # reshape to (B*K, 2, h, w) for interpolation
        offset_reshaped = offset.permute(0, 1, 2, 5, 3, 4).reshape(
            B * K * N, two, h, w)
        offset_up = F.interpolate(
            offset_reshaped, scale_factor=scale,
            mode='bilinear', align_corners=True)  # (B*9, 2, h*s, w*s)
        offset_up = offset_up * scale  # 坐标值 ×scale
        _, _, h_new, w_new = offset_up.shape
        offset_up = offset_up.reshape(
            B, K, N, two, h_new, w_new).permute(0, 1, 2, 4, 5, 3)
        new_offset[key] = offset_up

    for key in pre_similarity:
        # similarity: (B, 1, H, W) 或 (B, K, 1, H, W) — 需要检查实际形状
        sim = pre_similarity[key]
        if sim.dim() == 4:
            # (B, 1, h, w)
            sim_up = F.interpolate(
                sim, scale_factor=scale,
                mode='bilinear', align_corners=True)
        elif sim.dim() == 5:
            # (B, K, 1, h, w)
            B, K, c, h, w = sim.shape
            sim_reshaped = sim.reshape(B * K, c, h, w)
            sim_up = F.interpolate(
                sim_reshaped, scale_factor=scale,
                mode='bilinear', align_corners=True)
            _, _, h_new, w_new = sim_up.shape
            sim_up = sim_up.reshape(B, K, c, h_new, w_new)
        else:
            sim_up = sim  # fallback
        new_similarity[key] = sim_up

    return new_offset, new_flow, new_similarity
