# datsr/models/ref_restoration_model.py  

import importlib
import logging
import os.path as osp
from collections import OrderedDict

import mmcv
import torch

import datsr.models.networks as networks
import datsr.utils.metrics as metrics
from datsr.utils import ProgressBar, tensor2img, img2tensor

from .sr_model import SRModel
from datsr.models.archs.wavelet_branch_arch import WaveletFrequencyBranch, upsample_offsets
import pdb

loss_module = importlib.import_module('datsr.models.losses')
logger = logging.getLogger('base')
psnr_list = []


class RefRestorationModel(SRModel):

    def __init__(self, opt):
        super(RefRestorationModel, self).__init__(opt)

        # net_map does not have any trainable parameters.  
        self.net_map = networks.define_net_map(opt)
        self.net_map = self.model_to_device(self.net_map)

        # define network for feature extraction  
        self.net_extractor = networks.define_net_extractor(opt)
        self.net_extractor = self.model_to_device(self.net_extractor)
        self.print_network(self.net_extractor)

        # ===== 新增: 小波频域分支 =====  
        self.net_wavelet = WaveletFrequencyBranch(out_channels=64).to(self.device)
        net_g_opt = self.opt.get('network_g', {})
        legacy_wav_concat = net_g_opt.get('legacy_wav_concat', False)
        self.use_wavelet_ll_matching = net_g_opt.get(
            'use_wavelet_ll_matching', False)
        self.use_ref_hf_residual = net_g_opt.get(
            'use_ref_hf_residual', False) or legacy_wav_concat

        # load pretrained feature extractor  
        load_path = self.opt['path'].get('pretrain_model_feature_extractor',
                                         None)
        if load_path is not None:
            self.load_network(self.net_extractor, load_path,
                              self.opt['path']['strict_load'])

            # load pretrained models
        load_path = self.opt['path'].get('pretrain_model_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path']['strict_load'])
        load_path = self.opt['path'].get('pretrain_model_wavelet', None)
        if load_path is not None:
            self.load_network(self.net_wavelet, load_path,
                              self.opt['path']['strict_load'])
        if self.is_train:
            self.net_g.train()

            # optimizers  
            train_opt = self.opt['train']
            weight_decay_g = train_opt.get('weight_decay_g', 0)
            optim_params_g = []
            optim_params_offset = []
            optim_params_relu2_offset = []
            optim_params_relu3_offset = []
            if train_opt.get('lr_relu3_offset', None):
                optim_params_relu3_offset = []
            for name, v in self.net_g.named_parameters():
                if v.requires_grad:
                    if 'offset' in name:
                        if 'small' in name:
                            logger.info(name)
                            optim_params_relu3_offset.append(v)
                        elif 'medium' in name:
                            logger.info(name)
                            optim_params_relu2_offset.append(v)
                        else:
                            optim_params_offset.append(v)
                    else:
                        optim_params_g.append(v)

                        # 将小波分支的可学习参数加入优化器
            if self.use_wavelet_ll_matching or self.use_ref_hf_residual:
                optim_params_wavelet = list(self.net_wavelet.parameters())
            else:
                optim_params_wavelet = []

            self.optimizer_g = torch.optim.Adam(
                [{
                    'params': optim_params_g + optim_params_wavelet
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
            self.schedulers = []
            self.setup_schedulers()
            self.log_dict = OrderedDict()

    def init_training_settings(self):
        train_opt = self.opt['train']

        if self.opt.get('network_d', None):
            self.net_d = networks.define_net_d(self.opt)
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            load_path = self.opt['path'].get('pretrain_model_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path,
                                  self.opt['path']['strict_load'])
        else:
            logger.info('No discriminator.')
            self.net_d = None

        if self.net_d:
            self.net_d.train()

            # define losses
        if train_opt['pixel_weight'] > 0:
            cri_pix_cls = getattr(loss_module, train_opt['pixel_criterion'])
            self.cri_pix = cri_pix_cls(
                loss_weight=train_opt['pixel_weight'],
                reduction='mean').to(self.device)
        else:
            logger.info('Remove pixel loss.')
            self.cri_pix = None

        if train_opt.get('perceptual_opt', None):
            cri_perceptual_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            logger.info('Remove perceptual loss.')
            self.cri_perceptual = None

        if train_opt.get('style_opt', None):
            cri_style_cls = getattr(loss_module, 'PerceptualLoss')
            self.cri_style = cri_style_cls(**train_opt['style_opt']).to(
                self.device)
        else:
            logger.info('Remove style loss.')
            self.cri_style = None

        if train_opt.get('texture_opt', None):
            cri_texture_cls = getattr(loss_module, 'TextureLoss')
            self.cri_texture = cri_texture_cls(**train_opt['texture_opt']).to(
                self.device)
        else:
            logger.info('Remove texture loss.')
            self.cri_texture = None

        if train_opt.get('gan_type', None):
            cri_gan_cls = getattr(loss_module, 'GANLoss')
            self.cri_gan = cri_gan_cls(
                train_opt['gan_type'],
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=train_opt['gan_weight']).to(self.device)

            if train_opt['grad_penalty_weight'] > 0:
                cri_grad_penalty_cls = getattr(loss_module,
                                               'GradientPenaltyLoss')
                self.cri_grad_penalty = cri_grad_penalty_cls(
                    loss_weight=train_opt['grad_penalty_weight']).to(
                    self.device)
            else:
                logger.info('Remove gradient penalty.')
                self.cri_grad_penalty = None
        else:
            logger.info('Remove GAN loss.')
            self.cri_gan = None

        self.net_g_pretrain_steps = train_opt['net_g_pretrain_steps']
        self.net_d_steps = train_opt['net_d_steps'] if train_opt[
            'net_d_steps'] else 1
        self.net_d_init_steps = train_opt['net_d_init_steps'] if train_opt[
            'net_d_init_steps'] else 0

        # optimizers  
        if self.net_d:
            weight_decay_d = train_opt.get('weight_decay_d', 0)
            self.optimizer_d = torch.optim.Adam(
                self.net_d.parameters(),
                lr=train_opt['lr_d'],
                weight_decay=weight_decay_d,
                betas=train_opt['beta_d'])
            self.optimizers.append(self.optimizer_d)

        self.setup_schedulers()
        self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.img_in_lq = data['img_in_lq'].to(self.device)
        self.img_ref = data['img_ref'].to(self.device)
        self.gt = data['img_in'].to(self.device)  # gt  
        self.match_img_in = data['img_in_up'].to(self.device)
        if 'img_in_ori' in data:
            self.gt_ori = data['img_in_ori'].to(self.device)

    def _legacy_wavelet_forward(self):
        """小波频域分支的前向传播  

        Returns:  
            pre_offset_flow_sim: [pre_offset, pre_flow, pre_similarity] (上采样到原始分辨率)  
            img_ref_feat: VGG 多尺度特征 (从原始 img_ref 提取, 160x160 分辨率)  
            F_wav: (B, 64, 80, 80) 对齐后的高频特征  
        """
        # Step 1: DWT 分解  
        # match_img_in: (B, 3, 160, 160) — 含噪 LR bicubic 上采样  
        # img_ref: (B, 3, 160, 160) — 干净 Ref  
        ll_y, ll_r, highfreq_r = self.net_wavelet(self.match_img_in, self.img_ref)
        # ll_y: (B, 3, 80, 80), ll_r: (B, 3, 80, 80)  
        # highfreq_r: {'LH': (B,3,80,80), 'HL': (B,3,80,80), 'HH': (B,3,80,80)}  

        # Step 2: 用 LL 子带做特征提取 (噪声鲁棒匹配)  
        features = self.net_extractor(ll_y, ll_r)
        # dense_features1: (B, 256, 20, 20)  
        # dense_features2: (B, 256, 20, 20)  

        # Step 3: 对应关系生成  
        # 匹配在 LL 特征上进行, VGG ref features 从原始 img_ref (160x160) 提取 (选择 A)  
        pre_offset_flow_sim, img_ref_feat = self.net_map(
            features, self.img_ref)
        # pre_flow['relu3_1']: (B, 20, 20, 2)  
        # pre_flow['relu2_1']: (B, 40, 40, 2)  
        # pre_flow['relu1_1']: (B, 80, 80, 2)  
        # img_ref_feat: VGG(img_ref) → relu1_1:(B,64,160,160), relu2_1:(B,128,80,80), relu3_1:(B,256,40,40)  

        pre_offset = pre_offset_flow_sim[0]
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]

        # Step 4: 用 relu1_1 flow (80x80) warp Ref 高频子带  
        # pre_flow['relu1_1'] 与 highfreq_r 分辨率一致 (都是 80x80)  
        F_wav = self.net_wavelet.warp_highfreq(highfreq_r, pre_flow['relu1_1'])
        # F_wav: (B, 64, 80, 80)  

        # Step 5: Offset/Flow/Similarity 上采样 ×2 (从 LL 分辨率恢复到原始分辨率)  
        # LL 匹配基准: relu3_1 @ 20x20 → 原始需要 40x40  
        # relu2_1 @ 40x40 → 原始需要 80x80  
        # relu1_1 @ 80x80 → 原始需要 160x160  
        pre_offset_up, pre_flow_up, pre_similarity_up = upsample_offsets(
            pre_offset, pre_flow, pre_similarity, scale=2)
        # pre_flow_up['relu3_1']: (B, 40, 40, 2)  
        # pre_flow_up['relu2_1']: (B, 80, 80, 2)  
        # pre_flow_up['relu1_1']: (B, 160, 160, 2)  

        pre_offset_flow_sim_up = [pre_offset_up, pre_flow_up, pre_similarity_up]

        return pre_offset_flow_sim_up, img_ref_feat, F_wav

    def _wavelet_forward(self):
        """Build correspondence and optional aligned reference HF features."""
        highfreq_r = None
        needs_upsample = False

        if self.use_wavelet_ll_matching:
            ll_y, ll_r, highfreq_r = self.net_wavelet(
                self.match_img_in, self.img_ref)
            features = self.net_extractor(ll_y, ll_r)
            pre_offset_flow_sim, img_ref_feat = self.net_map(
                features, self.img_ref)
            needs_upsample = True
            hf_flow_key = 'relu1_1'
        else:
            features = self.net_extractor(self.match_img_in, self.img_ref)
            pre_offset_flow_sim, img_ref_feat = self.net_map(
                features, self.img_ref)
            hf_flow_key = 'relu2_1'

        pre_offset = pre_offset_flow_sim[0]
        pre_flow = pre_offset_flow_sim[1]
        pre_similarity = pre_offset_flow_sim[2]

        F_wav = None
        if self.use_ref_hf_residual:
            if highfreq_r is None:
                _, highfreq_r = self.net_wavelet.dwt_forward(self.img_ref)
            F_wav = self.net_wavelet.warp_highfreq(
                highfreq_r, pre_flow[hf_flow_key])

        if needs_upsample:
            pre_offset, pre_flow, pre_similarity = upsample_offsets(
                pre_offset, pre_flow, pre_similarity, scale=2)

        return [pre_offset, pre_flow, pre_similarity], img_ref_feat, F_wav

    def optimize_parameters(self, step):

        # ===== 小波频域 + 空域串行 =====  
        pre_offset_flow_sim, img_ref_feat, F_wav = self._wavelet_forward()

        # 空域分支: DATSR 主网络  
        self.output = self.net_g(self.img_in_lq, pre_offset_flow_sim,
                                 img_ref_feat, F_wav)

        if step <= self.net_g_pretrain_steps:
            # pretrain the net_g with pixel Loss  
            self.optimizer_g.zero_grad()
            l_pix = self.cri_pix(self.output, self.gt)
            l_pix.backward()
            self.optimizer_g.step()

            # set log  
            self.log_dict['l_pix'] = l_pix.item()
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
            if (step - self.net_g_pretrain_steps) % self.net_d_steps == 0 and (
                    step - self.net_g_pretrain_steps) > self.net_d_init_steps:
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    self.log_dict['l_g_pix'] = l_g_pix.item()
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
        net_g_module = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
        collect_hf_stats = self.opt.get('val', {}).get(
            'collect_hf_stats', False) or self.opt.get(
            'collect_hf_stats', False)
        if hasattr(net_g_module, 'dyn_agg_restore'):
            net_g_module.dyn_agg_restore.collect_hf_stats = collect_hf_stats
        with torch.no_grad():
            pre_offset_flow_sim, img_ref_feat, F_wav = self._wavelet_forward()
            self.output = self.net_g(self.img_in_lq, pre_offset_flow_sim,
                                     img_ref_feat, F_wav)
        self.net_g.train()

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['img_in_lq'] = self.img_in_lq.detach().cpu()
        out_dict['rlt'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        # 保存小波分支  
        self.save_network(self.net_wavelet, 'net_wavelet', current_iter)
        if self.net_d:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img):
        net_g_module = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
        collect_hf_stats = self.opt.get('val', {}).get(
            'collect_hf_stats', False) or self.opt.get(
            'collect_hf_stats', False)
        if collect_hf_stats and hasattr(net_g_module, 'reset_hf_stats'):
            net_g_module.reset_hf_stats()

        pbar = ProgressBar(len(dataloader))
        avg_psnr = 0.
        avg_psnr_y = 0.
        avg_ssim_y = 0.
        avg_lpips = 0.
        dataset_name = dataloader.dataset.opt['name']
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img, gt_img = tensor2img([visuals['rlt'], visuals['gt']])

            if 'multi' in dataset_name:
                _, h, w, _ = self.gt_ori.shape
                sr_img = sr_img[:h, :w, :]
                gt_img = gt_img[:h, :w, :]

            if 'padding' in val_data.keys():
                padding = val_data['padding']
                original_size = val_data['original_size']
                if padding:
                    sr_img = sr_img[:original_size[0], :original_size[1]]

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f"{img_name}_{self.opt['name']}.png")
                    if self.opt['suffix']:
                        save_img_path = save_img_path.replace(
                            '.png', f'_{self.opt["suffix"]}.png')
                mmcv.imwrite(sr_img, save_img_path)

            del self.img_in_lq
            del self.output
            del self.gt
            torch.cuda.empty_cache()

            # --- 尺寸对齐 (处理测试图尺寸不整除 scale 的情况) ---
            min_h = min(sr_img.shape[0], gt_img.shape[0])
            min_w = min(sr_img.shape[1], gt_img.shape[1])
            sr_img = sr_img[:min_h, :min_w, :]
            gt_img = gt_img[:min_h, :min_w, :]
            # --- end ---
            psnr = metrics.psnr(
                sr_img, gt_img, crop_border=self.opt['crop_border'])
            psnr_list.append(psnr)
            avg_psnr += psnr
            sr_img_y = metrics.bgr2ycbcr(sr_img / 255., only_y=True)
            gt_img_y = metrics.bgr2ycbcr(gt_img / 255., only_y=True)
            psnr_y = metrics.psnr(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_psnr_y += psnr_y
            ssim_y = metrics.ssim(
                sr_img_y * 255,
                gt_img_y * 255,
                crop_border=self.opt['crop_border'])
            avg_ssim_y += ssim_y

            if not self.is_train:
                logger.info(f'# img {img_name} # PSNR: {psnr:.4e} '
                            f'# PSNR_Y: {psnr_y:.4e} # SSIM_Y: {ssim_y:.4e}.')

            pbar.update(f'Test {img_name}')

        avg_psnr = avg_psnr / (idx + 1)
        avg_psnr_y = avg_psnr_y / (idx + 1)
        avg_ssim_y = avg_ssim_y / (idx + 1)

        logger.info(f'# Validation {dataset_name} # PSNR: {avg_psnr:.4e} '
                    f'# PSNR_Y: {avg_psnr_y:.4e} # SSIM_Y: {avg_ssim_y:.4e}.')

        if collect_hf_stats and hasattr(net_g_module, 'get_hf_stats'):
            hf_stats = net_g_module.get_hf_stats(reset=True)
            if hf_stats:
                hf_stats_msg = ', '.join(
                    f'{key}: {value:.4e}'
                    for key, value in sorted(hf_stats.items()))
                logger.info(f'# HF contribution stats # {hf_stats_msg}.')
            else:
                logger.info('# HF contribution stats # no HF residual was collected.')

        if tb_logger:
            tb_logger.add_scalar('psnr', avg_psnr, current_iter)
            tb_logger.add_scalar('psnr_y', avg_psnr_y, current_iter)
            tb_logger.add_scalar('ssim_y', avg_ssim_y, current_iter)
