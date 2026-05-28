import torch
import torch.nn as nn
import torch.nn.functional as F

from datsr.models.archs.swin_unetv3_ref_restoration_arch import (
    ContentExtractor, DynamicAggregationRestoration)
from datsr.models.archs.wavelet_utils_arch import (
    DWT_2D, FrequencyBranch, WaveletDenoiseModule)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros',
              align_corners=True):
    """Warp a tensor with pixel-space optical flow."""
    assert x.size()[-2:] == flow.size()[1:3], (
        f'Size mismatch: x={x.shape}, flow={flow.shape}')
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, dtype=x.dtype, device=x.device),
        torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)


class ReferenceFrequencyBranch(nn.Module):
    """Wavelet branch that transfers aligned high-frequency ref sub-bands."""

    def __init__(self, in_channels=3, ngf=64):
        super(ReferenceFrequencyBranch, self).__init__()
        self.dwt = DWT_2D(wave='haar')

        self.input_ll = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.input_hf = nn.Sequential(
            nn.Conv2d(in_channels * 3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.ref_hf = nn.Sequential(
            nn.Conv2d(in_channels * 3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 3, ngf, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.upsample = nn.ConvTranspose2d(
            ngf, ngf, kernel_size=4, stride=2, padding=1)

    @staticmethod
    def _split_bands(coeffs, channels):
        return torch.split(coeffs, channels, dim=1)

    @staticmethod
    def _resize_flow_to(flow, size):
        target_h, target_w = size
        b, h, w, _ = flow.shape
        if h == target_h and w == target_w:
            return flow
        flow_chw = flow.permute(0, 3, 1, 2)
        flow_chw = F.interpolate(
            flow_chw, size=(target_h, target_w),
            mode='bilinear', align_corners=True)
        flow_chw[:, 0] *= target_w / max(w, 1)
        flow_chw[:, 1] *= target_h / max(h, 1)
        return flow_chw.permute(0, 2, 3, 1)

    def forward(self, x, img_ref=None, pre_flow=None):
        coeffs = self.dwt(x)
        c = x.shape[1]
        ll, lh, hl, hh = self._split_bands(coeffs, c)
        input_hf = torch.cat([lh, hl, hh], dim=1)

        ll_feat = self.input_ll(ll)
        input_hf_feat = self.input_hf(input_hf)

        if img_ref is not None and pre_flow is not None:
            ref = F.interpolate(
                img_ref, size=x.shape[-2:], mode='bicubic',
                align_corners=False)
            ref_coeffs = self.dwt(ref)
            _, ref_lh, ref_hl, ref_hh = self._split_bands(ref_coeffs, c)
            ref_hf = torch.cat([ref_lh, ref_hl, ref_hh], dim=1)
            flow = self._resize_flow_to(pre_flow, ref_hf.shape[-2:])
            ref_hf = flow_warp(ref_hf, flow)
            ref_hf_feat = self.ref_hf(ref_hf)
        else:
            ref_hf_feat = torch.zeros_like(input_hf_feat)

        fused = self.fusion(torch.cat([ll_feat, input_hf_feat, ref_hf_feat], 1))
        return self.upsample(fused)


class SimilarityGate(nn.Module):
    """Predicts a per-pixel gate for spatial-vs-frequency fusion."""

    def __init__(self, ngf):
        super(SimilarityGate, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ngf * 2 + 1, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _similarity_map(pre_similarity, size):
        if not pre_similarity:
            return None
        sim = pre_similarity.get('relu1_1', None)
        if sim is None:
            sim = next(iter(pre_similarity.values()))
        if sim.dim() == 5:
            sim = sim.mean(dim=1)
        if sim.dim() == 3:
            sim = sim.unsqueeze(1)
        sim = F.interpolate(
            sim, size=size, mode='bilinear', align_corners=False)
        return sim

    def forward(self, spatial_feat, freq_feat, pre_similarity):
        sim = self._similarity_map(pre_similarity, spatial_feat.shape[-2:])
        if sim is None:
            sim = spatial_feat.new_ones(
                spatial_feat.size(0), 1, spatial_feat.size(2),
                spatial_feat.size(3))
        return self.body(torch.cat([spatial_feat, freq_feat, sim], 1))


class ZeroInitResidualFusion(nn.Module):
    """Residual correction branch initialized as an exact no-op."""

    def __init__(self, in_channels, ngf=128, out_channels=3):
        super(ZeroInitResidualFusion, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, out_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return self.body(x)


class RefHFConfidenceAttention(nn.Module):
    """Reference high-frequency attention weighted by match confidence."""

    def __init__(self, in_channels=3, ngf=128):
        super(RefHFConfidenceAttention, self).__init__()
        self.dwt = DWT_2D(wave='haar')
        self.input_hf = nn.Sequential(
            nn.Conv2d(in_channels * 3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.ref_hf = nn.Sequential(
            nn.Conv2d(in_channels * 3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(ngf * 2 + 1, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ngf * 2 + 1, ngf, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.upsample = nn.ConvTranspose2d(
            ngf, ngf, kernel_size=4, stride=2, padding=1)

    @staticmethod
    def _split_hf(coeffs, channels):
        _, lh, hl, hh = torch.split(coeffs, channels, dim=1)
        return torch.cat([lh, hl, hh], dim=1)

    @staticmethod
    def _resize_flow_to(flow, size):
        target_h, target_w = size
        b, h, w, _ = flow.shape
        if h == target_h and w == target_w:
            return flow
        flow_chw = flow.permute(0, 3, 1, 2)
        flow_chw = F.interpolate(
            flow_chw, size=(target_h, target_w),
            mode='bilinear', align_corners=True)
        flow_chw[:, 0] *= target_w / max(w, 1)
        flow_chw[:, 1] *= target_h / max(h, 1)
        return flow_chw.permute(0, 2, 3, 1)

    @staticmethod
    def _similarity_map(pre_similarity, size, ref_tensor):
        if not pre_similarity:
            return ref_tensor.new_ones(
                ref_tensor.size(0), 1, size[0], size[1])
        sim = pre_similarity.get('relu2_1', None)
        if sim is None:
            sim = pre_similarity.get('relu1_1', None)
        if sim is None:
            sim = next(iter(pre_similarity.values()))
        if sim.dim() == 5:
            sim = sim.mean(dim=1)
        if sim.dim() == 3:
            sim = sim.unsqueeze(1)
        return F.interpolate(
            sim, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def _select_flow(pre_flow):
        if not pre_flow:
            return None
        return pre_flow.get('relu2_1', pre_flow.get('relu1_1', None))

    def forward(self, base, img_ref, pre_offset_flow_sim):
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]
        flow = self._select_flow(pre_flow)

        coeffs_in = self.dwt(base)
        c = base.shape[1]
        input_hf = self._split_hf(coeffs_in, c)

        ref = F.interpolate(
            img_ref, size=base.shape[-2:], mode='bicubic',
            align_corners=False)
        coeffs_ref = self.dwt(ref)
        ref_hf = self._split_hf(coeffs_ref, c)
        if flow is not None:
            flow = self._resize_flow_to(flow, ref_hf.shape[-2:])
            ref_hf = flow_warp(ref_hf, flow)

        input_feat = self.input_hf(input_hf)
        ref_feat = self.ref_hf(ref_hf)
        sim = self._similarity_map(
            pre_similarity, input_feat.shape[-2:], input_feat)
        conf = self.confidence(torch.cat([input_feat, ref_feat, sim], dim=1))
        ref_feat = ref_feat * conf
        feat = self.proj(torch.cat([input_feat, ref_feat, sim], dim=1))
        feat = self.upsample(feat)
        conf = F.interpolate(
            conf, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        return feat, conf


class ParallelDualBranchRestoration(nn.Module):
    """Spatial DATSR branch and reference-aware wavelet branch in parallel."""

    def __init__(self, ngf=64, n_blocks=4, groups=4, embed_dim=64,
                 depths=(2, 2), num_heads=(2, 2), window_size=8,
                 use_checkpoint=False, use_ref_frequency=True,
                 use_similarity_gate=True):
        super(ParallelDualBranchRestoration, self).__init__()
        self.use_ref_frequency = use_ref_frequency
        self.use_similarity_gate = use_similarity_gate

        self.spatial_branch = DynamicAggregationRestoration(
            ngf=ngf, n_blocks=n_blocks, groups=groups,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, use_checkpoint=use_checkpoint)
        self.freq_branch = ReferenceFrequencyBranch(in_channels=3, ngf=ngf)

        self.rgb_to_feat = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.gate = SimilarityGate(ngf)
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf // 2, 3, 3, 1, 1),
        )

    def forward(self, x, pre_offset_flow_sim, img_ref_feat, img_ref=None):
        base = F.interpolate(x, None, 4, 'bilinear', False)
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]

        spatial_out = self.spatial_branch(
            base, x, pre_offset_flow_sim, img_ref_feat)
        spatial_feat = self.rgb_to_feat(spatial_out)

        freq_flow = pre_flow.get('relu2_1', pre_flow.get('relu1_1', None))
        freq_ref = img_ref if self.use_ref_frequency else None
        freq_flow = freq_flow if self.use_ref_frequency else None
        freq_feat = self.freq_branch(
            base, img_ref=freq_ref, pre_flow=freq_flow)
        freq_feat = F.interpolate(
            freq_feat, size=spatial_out.shape[-2:],
            mode='bilinear', align_corners=False)

        if self.use_similarity_gate:
            gate = self.gate(spatial_feat, freq_feat, pre_similarity)
        else:
            gate = spatial_feat.new_full(
                (spatial_feat.size(0), 1, spatial_feat.size(2),
                 spatial_feat.size(3)), 0.5)
        gated_spatial = gate * spatial_feat
        gated_freq = (1.0 - gate) * freq_feat
        fused = self.fusion(torch.cat([gated_spatial, gated_freq], dim=1))
        output = self.output_conv(fused)

        return output + base


class OldParallelDualBranchRestoration(nn.Module):
    """Original parallel DATSR + input-wavelet branch for old checkpoints."""

    def __init__(self, ngf=128, n_blocks=8, groups=8, embed_dim=128,
                 depths=(4, 4), num_heads=(4, 4), window_size=8,
                 use_checkpoint=False, use_ref_hf_confidence=False,
                 use_zero_init_residual_fusion=False):
        super(OldParallelDualBranchRestoration, self).__init__()
        self.use_ref_hf_confidence = use_ref_hf_confidence
        self.use_zero_init_residual_fusion = use_zero_init_residual_fusion

        self.spatial_branch = DynamicAggregationRestoration(
            ngf=ngf, n_blocks=n_blocks, groups=groups,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, use_checkpoint=use_checkpoint)
        self.freq_branch = FrequencyBranch(in_channels=3, ngf=ngf)
        self.ref_hf_attn = None
        if self.use_ref_hf_confidence:
            self.ref_hf_attn = RefHFConfidenceAttention(
                in_channels=3, ngf=ngf)
            self.ref_hf_scale = nn.Parameter(torch.zeros(1))

        self.rgb_to_feat = nn.Sequential(
            nn.Conv2d(3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf // 2, 3, 3, 1, 1),
        )
        self.residual_fusion = None
        if self.use_zero_init_residual_fusion:
            self.residual_fusion = ZeroInitResidualFusion(
                in_channels=ngf * 2, ngf=ngf, out_channels=3)

    def forward(self, x, pre_offset_flow_sim, img_ref_feat, img_ref=None):
        base = F.interpolate(x, None, 4, 'bilinear', False)

        spatial_out = self.spatial_branch(
            base, x, pre_offset_flow_sim, img_ref_feat)
        spatial_feat = self.rgb_to_feat(spatial_out)

        freq_feat = self.freq_branch(base)
        if self.ref_hf_attn is not None and img_ref is not None:
            ref_hf_feat, ref_hf_conf = self.ref_hf_attn(
                base, img_ref, pre_offset_flow_sim)
            freq_feat = freq_feat + self.ref_hf_scale * ref_hf_feat * ref_hf_conf
        freq_feat = F.interpolate(
            freq_feat, size=spatial_out.shape[-2:],
            mode='bilinear', align_corners=False)

        fused = self.fusion(torch.cat([spatial_feat, freq_feat], dim=1))
        residual = self.output_conv(fused)

        if self.residual_fusion is not None:
            residual = residual + self.residual_fusion(
                torch.cat([spatial_feat, freq_feat], dim=1))

        return residual + base


class OldWaveletParallelRestorationNet(nn.Module):
    """Checkpoint-compatible wrapper for the 250k old parallel model."""

    def __init__(self, ngf=128, n_blocks=8, groups=8, embed_dim=128,
                 depths=(4, 4), num_heads=(4, 4), window_size=8,
                 use_checkpoint=False, use_wdm=True,
                 use_denoised_matching=False,
                 use_ref_hf_confidence=False,
                 use_zero_init_residual_fusion=False,
                 **kwargs):
        super(OldWaveletParallelRestorationNet, self).__init__()
        self.use_wdm = use_wdm
        self.use_denoised_matching = use_denoised_matching
        if self.use_wdm:
            self.wdm = WaveletDenoiseModule(in_channels=3, hidden=ngf)

        # Kept for strict checkpoint compatibility with the 250k model.
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)
        self.parallel_branch = OldParallelDualBranchRestoration(
            ngf=ngf, n_blocks=n_blocks, groups=groups,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, use_checkpoint=use_checkpoint,
            use_ref_hf_confidence=use_ref_hf_confidence,
            use_zero_init_residual_fusion=use_zero_init_residual_fusion)

    def denoise(self, x):
        if not self.use_wdm:
            return x, None
        return self.wdm(x)

    def forward(self, x, pre_offset_flow_sim, img_ref_feat, img_ref=None,
                x_denoised=None, mask_dict=None):
        if x_denoised is None:
            x_denoised, mask_dict = self.denoise(x)
        output = self.parallel_branch(
            x_denoised, pre_offset_flow_sim, img_ref_feat, img_ref=img_ref)
        return output, mask_dict, x_denoised


class WaveletParallelRestorationNet(nn.Module):
    """Full network with WDM denoising and parallel restoration branches."""

    def __init__(self, ngf=64, n_blocks=4, groups=4, embed_dim=64,
                 depths=(2, 2), num_heads=(2, 2), window_size=8,
                 use_checkpoint=False, use_wdm=True,
                 use_ref_frequency=True, use_similarity_gate=True,
                 use_denoised_matching=True, **kwargs):
        super(WaveletParallelRestorationNet, self).__init__()
        self.use_wdm = use_wdm
        self.use_denoised_matching = use_denoised_matching
        if self.use_wdm:
            self.wdm = WaveletDenoiseModule(in_channels=3, hidden=ngf)

        self.parallel_branch = ParallelDualBranchRestoration(
            ngf=ngf, n_blocks=n_blocks, groups=groups,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, use_checkpoint=use_checkpoint,
            use_ref_frequency=use_ref_frequency,
            use_similarity_gate=use_similarity_gate)

    def denoise(self, x):
        if not self.use_wdm:
            return x, None
        return self.wdm(x)

    def forward(self, x, pre_offset_flow_sim, img_ref_feat, img_ref=None,
                x_denoised=None, mask_dict=None):
        if x_denoised is None:
            x_denoised, mask_dict = self.denoise(x)
        output = self.parallel_branch(
            x_denoised, pre_offset_flow_sim, img_ref_feat, img_ref=img_ref)
        return output, mask_dict, x_denoised
