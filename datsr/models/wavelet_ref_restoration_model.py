"""
Wavelet-enhanced Reference-based Restoration Model.

Combines:
- Scheme 1: WaveletDenoiseModule (WDM) for input-stage noise separation
- Scheme 2: WaveAttentionBlocks in the U-Net body
- FFTLoss + WaveletLoss for frequency/wavelet-domain supervision
"""

import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
import torch.nn.functional as F

import datsr.models.networks as networks
import datsr.utils.metrics as metrics
from datsr.utils import ProgressBar, tensor2img

from .ref_restoration_model import RefRestorationModel

loss_module = importlib.import_module('datsr.models.losses')
logger = logging.getLogger('base')


class WaveletRefRestorationModel(RefRestorationModel):
    """
    Extends RefRestorationModel with:
    - WDM-based denoising: the net_g (WaveletSwinUnetv3RestorationNet) has an
      internal WDM that denoises the noisy LR before SR.  The denoised LR is
      upsampled and used as match_img_in for correspondence matching (instead
      of the dataset-provided noisy version).
    - Wavelet / FFT losses on the SR output.
    - Denoising loss on the WDM output vs clean LR.
    """

    def __init__(self, opt):
        super().__init__(opt)

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']

        # wavelet loss on SR output
        if train_opt.get('wavelet_opt', None):
            cri_wavelet_cls = getattr(loss_module, 'WaveletLoss')
            self.cri_wavelet = cri_wavelet_cls(
                **train_opt['wavelet_opt']).to(self.device)
        else:
            logger.info('No wavelet loss.')
            self.cri_wavelet = None

        # FFT loss on SR output
        if train_opt.get('fft_opt', None):
            cri_fft_cls = getattr(loss_module, 'FFTLoss')
            self.cri_fft = cri_fft_cls(
                **train_opt['fft_opt']).to(self.device)
        else:
            logger.info('No FFT loss.')
            self.cri_fft = None

        # denoising loss weight
        self.denoise_weight = train_opt.get('denoise_weight', 1.0)

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)          # noisy LR
        self.img_in_lq_clean = data['img_in_lq_clean'].to(self.device)  # clean LR
        self.img_ref = data['img_ref'].to(self.device)
        self.gt = data['img_in'].to(self.device)
        # img_in_up from dataset is from clean LR; we will recompute from
        # denoised LR during optimize_parameters, but keep it as fallback.
        self.match_img_in = data['img_in_up'].to(self.device)
        if 'img_in_ori' in data:
            self.gt_ori = data['img_in_ori'].to(self.device)

    def optimize_parameters(self, step):
        # --- Forward pass through net_g (which includes WDM) ---
        # The net_g returns (sr_output, denoised_lq).
        # We first run WDM to get denoised LR, then use it for matching.

        # Step 1: Run WDM inside net_g to get denoised LR
        denoised_lq = self.net_g.module.wdm(self.img_in_lq) \
            if hasattr(self.net_g, 'module') else self.net_g.wdm(self.img_in_lq)

        # Step 2: Create match image from denoised LR (4x upscale)
        match_img_in_denoised = F.interpolate(
            denoised_lq, scale_factor=4, mode='bicubic', align_corners=False)

        # Step 3: VGG feature extraction using denoised match image
        self.features = self.net_extractor(match_img_in_denoised, self.img_ref)
        self.pre_offset, self.img_ref_feat = self.net_map(
            self.features, self.img_ref)

        # Step 4: Full forward through net_g (WDM runs again internally,
        # but shares the same parameters so results are consistent)
        self.output, self.denoised_lq = self.net_g(
            self.img_in_lq, self.pre_offset, self.img_ref_feat)

        # --- Losses ---
        if step <= self.net_g_pretrain_steps:
            self.optimizer_g.zero_grad()
            l_total = 0

            # pixel loss on SR
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            self.log_dict = OrderedDict()
            self.log_dict['l_pix'] = l_pix.item()

            # denoising loss
            l_denoise = F.l1_loss(self.denoised_lq, self.img_in_lq_clean)
            l_total += self.denoise_weight * l_denoise
            self.log_dict['l_denoise'] = l_denoise.item()

            # wavelet loss
            if self.cri_wavelet:
                l_wavelet = self.cri_wavelet(self.output, self.gt)
                l_total += l_wavelet
                self.log_dict['l_wavelet'] = l_wavelet.item()

            # FFT loss
            if self.cri_fft:
                l_fft = self.cri_fft(self.output, self.gt)
                l_total += l_fft
                self.log_dict['l_fft'] = l_fft.item()

            l_total.backward()
            self.optimizer_g.step()

        else:
            if self.net_d:
                # train net_d
                self.optimizer_d.zero_grad()
                for p in self.net_d.parameters():
                    p.requires_grad = True
                real_d_pred = self.net_d(self.gt)
                l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
                self.log_dict['l_d_real'] = l_d_real.item()
                self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
                fake_d_pred = self.net_d(self.output.detach())
                l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
                self.log_dict['l_d_fake'] = l_d_fake.item()
                self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
                l_d_total = l_d_real + l_d_fake
                if self.cri_grad_penalty:
                    l_grad_penalty = self.cri_grad_penalty(
                        self.net_d, self.gt, self.output)
                    self.log_dict['l_grad_penalty'] = l_grad_penalty.item()
                    l_d_total += l_grad_penalty
                l_d_total.backward()
                self.optimizer_d.step()

            # train net_g
            self.optimizer_g.zero_grad()
            if self.net_d:
                for p in self.net_d.parameters():
                    p.requires_grad = False

            l_g_total = 0
            if ((step - self.net_g_pretrain_steps) % self.net_d_steps == 0
                    and (step - self.net_g_pretrain_steps)
                    > self.net_d_init_steps):

                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()

                # denoising loss
                l_denoise = F.l1_loss(self.denoised_lq, self.img_in_lq_clean)
                l_g_total += self.denoise_weight * l_denoise
                self.log_dict['l_denoise'] = l_denoise.item()

                # wavelet loss
                if self.cri_wavelet:
                    l_wavelet = self.cri_wavelet(self.output, self.gt)
                    l_g_total += l_wavelet
                    self.log_dict['l_wavelet'] = l_wavelet.item()

                # FFT loss
                if self.cri_fft:
                    l_fft = self.cri_fft(self.output, self.gt)
                    l_g_total += l_fft
                    self.log_dict['l_fft'] = l_fft.item()

                if self.cri_perceptual:
                    l_g_percep, _ = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    self.log_dict['l_g_percep'] = l_g_percep.item()
                if self.cri_style:
                    _, l_g_style = self.cri_style(self.output, self.gt)
                    l_g_total += l_g_style
                    self.log_dict['l_g_style'] = l_g_style.item()
                if self.cri_texture:
                    l_g_texture = self.cri_texture(self.output, self.maps,
                                                   self.weights)
                    l_g_total += l_g_texture
                    self.log_dict['l_g_texture'] = l_g_texture.item()

                if self.net_d:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    l_g_total += l_g_gan
                    self.log_dict['l_g_gan'] = l_g_gan.item()

                l_g_total.backward()
                self.optimizer_g.step()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            # Use WDM denoised LR for matching
            net = self.net_g.module if hasattr(self.net_g, 'module') \
                else self.net_g
            denoised_lq = net.wdm(self.img_in_lq)
            match_img_in = F.interpolate(
                denoised_lq, scale_factor=4, mode='bicubic',
                align_corners=False)

            self.features = self.net_extractor(match_img_in, self.img_ref)
            self.pre_offset, self.img_ref_feat = self.net_map(
                self.features, self.img_ref)
            self.output, _ = self.net_g(
                self.img_in_lq, self.pre_offset, self.img_ref_feat)

        self.net_g.train()