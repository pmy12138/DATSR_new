# datsr/models/archs/wavelet_branch_arch.py
"""
小波频域分支: DWT 分解 + 高频子带对齐迁移.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
