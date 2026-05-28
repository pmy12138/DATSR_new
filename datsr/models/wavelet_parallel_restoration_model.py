import importlib
import logging
from collections import OrderedDict

import torch
import torch.nn.functional as F

from datsr.models.ref_restoration_model import RefRestorationModel

loss_module = importlib.import_module('datsr.models.losses')
logger = logging.getLogger('base')


class WaveletParallelRestorationModel(RefRestorationModel):
    """Reference SR model with WDM, denoised matching and wavelet fusion."""

    def __init__(self, opt):
        super(WaveletParallelRestorationModel, self).__init__(opt)
        net_g = self._net_g_module()
        self.use_denoised_matching = getattr(
            net_g, 'use_denoised_matching', True)

        # RefRestorationModel appends optimizer_g after setup_schedulers()
        # because of the inheritance order. Rebuild here so net_g scheduling
        # and the new WDM/fusion parameters are tracked correctly.
        if self.is_train:
            self._rebuild_generator_optimizer()
            self.schedulers = []
            self.setup_schedulers()
            self.log_dict = OrderedDict()

    def init_training_settings(self):
        super(WaveletParallelRestorationModel, self).init_training_settings()
        train_opt = self.opt['train']

        if train_opt.get('mask_sparse_opt', None):
            cri_mask_sparse_cls = getattr(loss_module, 'MaskSparseLoss')
            self.cri_mask_sparse = cri_mask_sparse_cls(
                loss_weight=train_opt['mask_sparse_opt'].get(
                    'loss_weight', 0.01)).to(self.device)
        else:
            self.cri_mask_sparse = None

        self.denoise_weight = train_opt.get('denoise_weight', 0.5)
        self.cri_denoise = torch.nn.L1Loss().to(self.device)

        if train_opt.get('wavelet_opt', None):
            cri_wavelet_cls = getattr(loss_module, 'WaveletLoss')
            self.cri_wavelet = cri_wavelet_cls(
                **train_opt['wavelet_opt']).to(self.device)
        else:
            self.cri_wavelet = None

        if train_opt.get('fft_opt', None):
            cri_fft_cls = getattr(loss_module, 'FFTLoss')
            self.cri_fft = cri_fft_cls(**train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

    def _net_g_module(self):
        return self.net_g.module if hasattr(self.net_g, 'module') else self.net_g

    def _rebuild_generator_optimizer(self):
        if hasattr(self, 'optimizer_g'):
            self.optimizers = [
                opt for opt in self.optimizers if opt is not self.optimizer_g
            ]

        train_opt = self.opt['train']
        weight_decay_g = train_opt.get('weight_decay_g', 0)
        optim_params_g = []
        optim_params_offset = []
        optim_params_relu2_offset = []
        optim_params_relu3_offset = []

        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                logger.warning(f'Params {name} will not be optimized.')
                continue
            if 'offset' in name:
                if 'small' in name:
                    optim_params_relu3_offset.append(param)
                elif 'medium' in name:
                    optim_params_relu2_offset.append(param)
                else:
                    optim_params_offset.append(param)
            else:
                optim_params_g.append(param)

        self.optimizer_g = torch.optim.Adam(
            [{
                'params': optim_params_g
            }, {
                'params': optim_params_offset,
                'lr': train_opt['lr_offset']
            }, {
                'params': optim_params_relu3_offset,
                'lr': train_opt['lr_relu3_offset']
            }, {
                'params': optim_params_relu2_offset,
                'lr': train_opt['lr_relu2_offset']
            }],
            lr=train_opt['lr_g'],
            weight_decay=weight_decay_g,
            betas=train_opt['beta_g'])

        if hasattr(self, 'optimizer_d') and self.optimizer_d in self.optimizers:
            self.optimizers = [self.optimizer_g, self.optimizer_d]
        else:
            self.optimizers = [self.optimizer_g]

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref = data['img_ref'].to(self.device)
        self.gt = data['img_in'].to(self.device)
        self.match_img_in = data['img_in_up'].to(self.device)
        if 'img_in_lq_clean' in data:
            self.img_in_lq_clean = data['img_in_lq_clean'].to(self.device)
        else:
            self.img_in_lq_clean = None
        if 'img_in_ori' in data:
            self.gt_ori = data['img_in_ori'].to(self.device)

    def _prepare_correspondence(self):
        net_g = self._net_g_module()
        x_denoised, mask_dict = net_g.denoise(self.img_in_lq)
        if self.use_denoised_matching:
            scale = int(self.opt.get('scale', 4))
            match_img = F.interpolate(
                x_denoised, scale_factor=scale, mode='bicubic',
                align_corners=False)
        else:
            match_img = self.match_img_in
        self.features = self.net_extractor(match_img, self.img_ref)
        self.pre_offset_flow_sim, self.img_ref_feat = self.net_map(
            self.features, self.img_ref)
        return x_denoised, mask_dict

    def _forward_generator(self):
        x_denoised, mask_dict = self._prepare_correspondence()
        self.output, mask_dict, x_denoised = self.net_g(
            self.img_in_lq,
            self.pre_offset_flow_sim,
            self.img_ref_feat,
            self.img_ref,
            x_denoised=x_denoised,
            mask_dict=mask_dict)
        return mask_dict, x_denoised

    def _accumulate_generator_losses(self, mask_dict, x_denoised,
                                     pixel_key='l_pix'):
        l_total = 0

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            self.log_dict[pixel_key] = l_pix.item()

        if self.img_in_lq_clean is not None:
            clean_lq = self.img_in_lq_clean
            if clean_lq.shape[-2:] != x_denoised.shape[-2:]:
                clean_lq = F.interpolate(
                    clean_lq, size=x_denoised.shape[-2:],
                    mode='bicubic', align_corners=False)
            l_denoise = self.cri_denoise(x_denoised, clean_lq)
            l_total += self.denoise_weight * l_denoise
            self.log_dict['l_denoise'] = l_denoise.item()

        if self.cri_mask_sparse and mask_dict:
            l_mask_sparse = self.cri_mask_sparse(mask_dict)
            l_total += l_mask_sparse
            self.log_dict['l_mask_sparse'] = l_mask_sparse.item()

        if self.cri_wavelet:
            l_wavelet = self.cri_wavelet(self.output, self.gt)
            l_total += l_wavelet
            self.log_dict['l_wavelet'] = l_wavelet.item()

        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            self.log_dict['l_fft'] = l_fft.item()

        if self.cri_perceptual:
            l_percep, _ = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                self.log_dict['l_g_percep'] = l_percep.item()

        if self.cri_style:
            _, l_style = self.cri_style(self.output, self.gt)
            if l_style is not None:
                l_total += l_style
                self.log_dict['l_g_style'] = l_style.item()

        if self.cri_texture and hasattr(self, 'maps') and hasattr(self, 'weights'):
            l_texture = self.cri_texture(self.output, self.maps, self.weights)
            l_total += l_texture
            self.log_dict['l_g_texture'] = l_texture.item()

        return l_total

    def optimize_parameters(self, step):
        self.log_dict = OrderedDict()
        mask_dict, x_denoised = self._forward_generator()

        if step <= self.net_g_pretrain_steps:
            self.optimizer_g.zero_grad()
            l_total = self._accumulate_generator_losses(
                mask_dict, x_denoised, pixel_key='l_pix')
            l_total.backward()
            self.optimizer_g.step()
            return

        if self.net_d:
            self.optimizer_d.zero_grad()
            for p in self.net_d.parameters():
                p.requires_grad = True

            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            l_d_total = l_d_real + l_d_fake
            self.log_dict['l_d_real'] = l_d_real.item()
            self.log_dict['l_d_fake'] = l_d_fake.item()
            self.log_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            self.log_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

            if self.cri_grad_penalty:
                l_grad_penalty = self.cri_grad_penalty(
                    self.net_d, self.gt, self.output)
                l_d_total += l_grad_penalty
                self.log_dict['l_grad_penalty'] = l_grad_penalty.item()

            l_d_total.backward()
            self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        if self.net_d:
            for p in self.net_d.parameters():
                p.requires_grad = False

        l_g_total = 0
        if ((step - self.net_g_pretrain_steps) % self.net_d_steps == 0
                and (step - self.net_g_pretrain_steps)
                > self.net_d_init_steps):
            l_g_total = self._accumulate_generator_losses(
                mask_dict, x_denoised, pixel_key='l_g_pix')

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
            self._forward_generator()
        self.net_g.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
