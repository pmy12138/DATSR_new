# datsr/models/archs/denoise_swin_unetv3_ref_restoration_arch.py

import datsr.models.archs.arch_util as arch_util
import torch
import torch.nn as nn
import torch.nn.functional as F
from datsr.models.archs.dcn_v2 import DCN_sep_pre_multi_offset_flow_similarity as DynAgg
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 复用原始架构中的所有组件
from datsr.models.archs.swin_unetv3_ref_restoration_arch import (
    Mlp, WindowAttention, SwinTransformerBlock, PatchEmbed, PatchUnEmbed,
    BasicLayer, RSTB, ContentExtractor, SwinBlock, DynamicAggregationRestoration
)


class DilatedDenoiseModule(nn.Module):
    """多尺度空洞卷积降噪模块，支持辅助通道输入。"""

    def __init__(self, in_nc=6, nf=64, n_blocks=4):  # ← in_nc=6
        super().__init__()
        self.in_nc = in_nc
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)  # 6→64
        self.lrelu = nn.LeakyReLU(0.1, True)
        self.blocks = nn.ModuleList([DilatedResBlock(nf) for _ in range(n_blocks)])
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)  # 64→3 (输出始终是RGB)

    def forward(self, x):
        """
        Args:
            x: (B, 6, H, W) — 前3ch是noisy RGB，后3ch是albedo RGB
        Returns:
            (B, 3, H, W) — 降噪后的RGB
        """
        noisy_rgb = x[:, :3, :, :]  # 取前3ch用于残差连接
        feat = self.lrelu(self.conv_first(x))  # 6ch → 64ch
        for block in self.blocks:
            feat = block(feat)
        out = self.conv_last(feat)  # 64ch → 3ch
        return out + noisy_rgb  # 残差连接：只加 noisy RGB 部分

class DilatedResBlock(nn.Module):
    """单个残差空洞卷积块，包含 4 个并行的不同 dilation rate 的卷积分支。

    Args:
        nf (int): 特征通道数
    """

    def __init__(self, nf=64):
        super(DilatedResBlock, self).__init__()

        # 4 个并行空洞卷积分支，dilation = 1, 2, 4, 8
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, padding=1, dilation=1)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, padding=2, dilation=2)
        self.conv_d4 = nn.Conv2d(nf, nf, 3, 1, padding=4, dilation=4)
        self.conv_d8 = nn.Conv2d(nf, nf, 3, 1, padding=8, dilation=8)

        # 1x1 卷积融合 4 个分支
        self.fusion = nn.Conv2d(nf * 4, nf, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        arch_util.default_init_weights(
            [self.conv_d1, self.conv_d2, self.conv_d4, self.conv_d8, self.fusion], 0.1)

    def forward(self, x):
        d1 = self.lrelu(self.conv_d1(x))
        d2 = self.lrelu(self.conv_d2(x))
        d4 = self.lrelu(self.conv_d4(x))
        d8 = self.lrelu(self.conv_d8(x))
        fused = self.lrelu(self.fusion(torch.cat([d1, d2, d4, d8], dim=1)))
        return x + fused  # 残差连接


class DenoiseSwinUnetv3RestorationNet(nn.Module):
    def __init__(self, ngf=64, n_blocks=16, groups=8, embed_dim=64,
                 depths=(8, 8), num_heads=(8, 8), window_size=8,
                 use_checkpoint=False, denoise_nf=64, denoise_blocks=4,
                 denoise_in_nc=6):  # ← 新增参数
        super().__init__()
        self.denoise_module = DilatedDenoiseModule(
            in_nc=denoise_in_nc, nf=denoise_nf, n_blocks=denoise_blocks)
        self.content_extractor = ContentExtractor(
            in_nc=3, out_nc=3, nf=ngf, n_blocks=n_blocks)  # 仍然是3ch
        self.dyn_agg_restore = DynamicAggregationRestoration(
            ngf=ngf, n_blocks=n_blocks, groups=groups,
            embed_dim=ngf, depths=depths, num_heads=num_heads,
            window_size=window_size, use_checkpoint=use_checkpoint)
        # ... 初始化权重 ...

    def forward(self, x, pre_offset_flow_sim, img_ref_feat):
        # x: (B, 6, H_lq, W_lq) — 6ch输入
        x_denoised = self.denoise_module(x)  # 6ch → 3ch
        base = F.interpolate(x_denoised, None, 4, 'bilinear', False)
        content_feat = self.content_extractor(x_denoised)  # 3ch输入
        upscale_restore = self.dyn_agg_restore(
            base, content_feat, pre_offset_flow_sim, img_ref_feat)
        return upscale_restore + base