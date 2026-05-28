import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import pywt  
  
  
class DWT_Function(torch.autograd.Function):  
    @staticmethod  
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):  
        x = x.contiguous()  
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)  
        ctx.shape = x.shape  
  
        dim = x.shape[1]  
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)  
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)  
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)  
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)  
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  
        return x  
  
    @staticmethod  
    def backward(ctx, dx):  
        if ctx.needs_input_grad[0]:  
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors  
            B, C, H, W = ctx.shape  
            dx = dx.view(B, 4, -1, H//2, W//2)  
            dx = dx.transpose(1, 2).reshape(B, -1, H//2, W//2)  
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)  
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)  
        return dx, None, None, None, None  
  
  
class IDWT_Function(torch.autograd.Function):  
    @staticmethod  
    def forward(ctx, x, filters):  
        ctx.save_for_backward(filters)  
        ctx.shape = x.shape  
  
        B, _, H, W = x.shape  
        x = x.view(B, 4, -1, H, W).transpose(1, 2)  
        C = x.shape[1]  
        x = x.reshape(B, -1, H, W)  
        filters = filters.repeat(C, 1, 1, 1)  
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)  
        return x  
  
    @staticmethod  
    def backward(ctx, dx):  
        if ctx.needs_input_grad[0]:  
            filters = ctx.saved_tensors[0]  
            B, C, H, W = ctx.shape  
            C = C // 4  
            dx = dx.contiguous()  
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)  
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)  
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)  
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)  
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C)  
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  
        return dx, None  
  
  
class IDWT_2D(nn.Module):  
    def __init__(self, wave='haar'):  
        super(IDWT_2D, self).__init__()  
        w = pywt.Wavelet(wave)  
        rec_hi = torch.Tensor(w.rec_hi)  
        rec_lo = torch.Tensor(w.rec_lo)  
  
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)  
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)  
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)  
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)  
  
        w_ll = w_ll.unsqueeze(0).unsqueeze(1)  
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)  
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)  
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)  
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)  
        self.register_buffer('filters', filters)  
        self.filters = self.filters.to(dtype=torch.float32)  
  
    def forward(self, x):  
        return IDWT_Function.apply(x, self.filters)  
  
  
class DWT_2D(nn.Module):  
    def __init__(self, wave='haar'):  
        super(DWT_2D, self).__init__()  
        w = pywt.Wavelet(wave)  
        dec_hi = torch.Tensor(w.dec_hi[::-1])  
        dec_lo = torch.Tensor(w.dec_lo[::-1])  
  
        w_ll = dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)  
        w_lh = dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)  
        w_hl = dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)  
        w_hh = dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)  
  
        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))  
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))  
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))  
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))  
  
        self.w_ll = self.w_ll.to(dtype=torch.float32)  
        self.w_lh = self.w_lh.to(dtype=torch.float32)  
        self.w_hl = self.w_hl.to(dtype=torch.float32)  
        self.w_hh = self.w_hh.to(dtype=torch.float32)  
  
    def forward(self, x):  
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)  
  
  
class SubbandDenoiser(nn.Module):  
    """Learnable mask-based denoiser for wavelet high-freq sub-bands (方案1)."""  
      
    def __init__(self, channels, hidden=None):  
        super(SubbandDenoiser, self).__init__()  
        hidden = hidden or channels * 2  
        self.mask_net = nn.Sequential(  
            nn.Conv2d(channels, hidden, 3, 1, 1),  
            nn.LeakyReLU(0.1, inplace=True),  
            nn.Conv2d(hidden, hidden, 3, 1, 1),  
            nn.LeakyReLU(0.1, inplace=True),  
            nn.Conv2d(hidden, channels, 3, 1, 1),  
            nn.Sigmoid()  # 输出[0,1]的mask  
        )  
      
    def forward(self, x):  
        mask = self.mask_net(x)  
        return x * mask, mask  # 返回mask用于稀疏性损失  
  
  
class WaveletDenoiseModule(nn.Module):  
    """Wavelet domain denoising module with mask mechanism."""  
      
    def __init__(self, in_channels=3, hidden=64):  
        super(WaveletDenoiseModule, self).__init__()  
        self.dwt = DWT_2D(wave='haar')  
        self.idwt = IDWT_2D(wave='haar')  
          
        # LL子带轻量细化  
        self.ll_refine = nn.Sequential(  
            nn.Conv2d(in_channels, hidden, 3, 1, 1),  
            nn.LeakyReLU(0.1, inplace=True),  
            nn.Conv2d(hidden, in_channels, 3, 1, 1),  
        )  
          
        # 高频子带mask去噪  
        self.lh_denoiser = SubbandDenoiser(in_channels, hidden)  
        self.hl_denoiser = SubbandDenoiser(in_channels, hidden)  
        self.hh_denoiser = SubbandDenoiser(in_channels, hidden)  
      
    def forward(self, x):  
        # DWT分解  
        coeffs = self.dwt(x)  # (B, 4*C, H/2, W/2)  
        C = x.shape[1]  
        ll, lh, hl, hh = torch.split(coeffs, C, dim=1)  
          
        # LL轻量细化  
        ll = ll + self.ll_refine(ll)  
          
        # 高频mask去噪  
        lh, lh_mask = self.lh_denoiser(lh)  
        hl, hl_mask = self.hl_denoiser(hl)  
        hh, hh_mask = self.hh_denoiser(hh)  
          
        # IDWT重建  
        denoised_coeffs = torch.cat([ll, lh, hl, hh], dim=1)  
        denoised = self.idwt(denoised_coeffs)  
          
        # 返回去噪结果和mask字典  
        mask_dict = {'lh': lh_mask, 'hl': hl_mask, 'hh': hh_mask}  
        return denoised, mask_dict


class FrequencyBranch(nn.Module):
    """Pure wavelet domain CNN branch (方案4) - 输出特征图"""

    def __init__(self, in_channels=3, ngf=128):
        super(FrequencyBranch, self).__init__()
        self.dwt = DWT_2D(wave='haar')

        # 处理4个子带
        self.conv_ll = nn.Sequential(
            nn.Conv2d(in_channels, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.conv_hf = nn.Sequential(
            nn.Conv2d(in_channels * 3, ngf, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
        )
        self.fusion = nn.Conv2d(ngf * 2, ngf, 1, 1, 0)

        # 上采样层：H/2×W/2 -> H×W
        self.upsample = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """
        Args:
            x: Input image (B, 3, H, W)
        Returns:
            fused: Feature map (B, ngf, H, W)
        """
        coeffs = self.dwt(x)
        C = x.shape[1]
        ll, lh, hl, hh = torch.split(coeffs, C, dim=1)

        ll_feat = self.conv_ll(ll)
        hf_feat = self.conv_hf(torch.cat([lh, hl, hh], dim=1))

        fused = self.fusion(torch.cat([ll_feat, hf_feat], dim=1))

        # 上采样到原始尺寸
        fused = self.upsample(fused)

        return fused  # (B, ngf, H, W)