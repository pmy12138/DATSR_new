import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from datsr.models.archs.arch_util import tensor_shift
from datsr.models.archs.ref_map_util import (feature_match_index,
                                              geometry_guided_feature_match_index,
                                              probabilistic_geometry_feature_match_index)
from datsr.models.archs.vgg_arch import VGGFeatureExtractor

logger = logging.getLogger('base')


class FlowSimCorrespondenceGenerationArch(nn.Module):

    def __init__(self,
                 patch_size=3,
                 stride=1,
                 vgg_layer_list=['relu3_1', 'relu2_1', 'relu1_1'],
                 vgg_type='vgg19',
                 use_freq_matching=False,
                 use_matching_geo_prior=False,
                 matching_geo_mode='depth',
                 normalize_similarity=False,
                 geometry_prior_strength=1.0,
                 geometry_prior_floor=0.25,
                 geometry_prior_grad_scale=4.0,
                 geometry_prior_blur_kernel=5,
                 geometry_search_radius=4,
                 geometry_position_weight=0.0,
                 geometry_position_sigma=2.0,
                 geometry_soft_prior_strength=1.0,
                 geometry_soft_prior_sigma=0.1,
                 geometry_confidence_mode='valid_mask',
                 geometry_depth_edge_scale=0.5,
                 geometry_depth_edge_floor=0.1,
                 geometry_auxiliary_logit_scale=10.0,
                 geometry_auxiliary_max_queries=512,
                 use_geometry_adaptive_dcn=False,
                 gar_dcn_base_radius=10.0,
                 gar_dcn_radius_min=6.0,
                 gar_dcn_radius_max=14.0,
                 gar_dcn_extent_min=0.6,
                 gar_dcn_extent_max=1.4,
                 gar_dcn_mask_floor=0.25):
        super(FlowSimCorrespondenceGenerationArch, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.use_freq_matching = use_freq_matching
        self.use_matching_geo_prior = use_matching_geo_prior
        self.matching_geo_mode = matching_geo_mode
        self.normalize_similarity = normalize_similarity
        self.geometry_prior_strength = float(geometry_prior_strength)
        self.geometry_prior_floor = float(geometry_prior_floor)
        self.geometry_prior_grad_scale = float(geometry_prior_grad_scale)
        self.geometry_prior_blur_kernel = int(geometry_prior_blur_kernel)
        self.geometry_search_radius = int(geometry_search_radius)
        self.geometry_position_weight = float(geometry_position_weight)
        self.geometry_position_sigma = float(geometry_position_sigma)
        self.geometry_soft_prior_strength = float(
            geometry_soft_prior_strength)
        self.geometry_soft_prior_sigma = float(geometry_soft_prior_sigma)
        self.geometry_confidence_mode = geometry_confidence_mode
        self.geometry_depth_edge_scale = float(geometry_depth_edge_scale)
        self.geometry_depth_edge_floor = float(geometry_depth_edge_floor)
        self.geometry_auxiliary_logit_scale = float(
            geometry_auxiliary_logit_scale)
        self.geometry_auxiliary_max_queries = int(
            geometry_auxiliary_max_queries)
        self.use_geometry_adaptive_dcn = bool(use_geometry_adaptive_dcn)
        self.gar_dcn_base_radius = float(gar_dcn_base_radius)
        self.gar_dcn_radius_min = float(gar_dcn_radius_min)
        self.gar_dcn_radius_max = float(gar_dcn_radius_max)
        self.gar_dcn_extent_min = float(gar_dcn_extent_min)
        self.gar_dcn_extent_max = float(gar_dcn_extent_max)
        self.gar_dcn_mask_floor = float(gar_dcn_mask_floor)
        self.matching_aux_losses = {}

        if (self.use_matching_geo_prior
                and self.matching_geo_mode not in [
                    'depth', 'projective_window', 'projective_soft_prior']):
            raise ValueError(
                'matching_geo_mode must be depth, projective_window, or '
                'projective_soft_prior.')
        if not 0.0 <= self.geometry_prior_strength <= 1.0:
            raise ValueError('geometry_prior_strength must be in [0, 1].')
        if not 0.0 <= self.geometry_prior_floor <= 1.0:
            raise ValueError('geometry_prior_floor must be in [0, 1].')
        if self.geometry_prior_blur_kernel < 1 or (
                self.geometry_prior_blur_kernel % 2 == 0):
            raise ValueError(
                'geometry_prior_blur_kernel must be a positive odd integer.')
        if self.geometry_search_radius < 0:
            raise ValueError('geometry_search_radius must be non-negative.')
        if self.geometry_position_weight < 0:
            raise ValueError('geometry_position_weight must be non-negative.')
        if self.geometry_position_sigma <= 0:
            raise ValueError('geometry_position_sigma must be positive.')
        if self.geometry_soft_prior_strength < 0:
            raise ValueError(
                'geometry_soft_prior_strength must be non-negative.')
        if self.geometry_soft_prior_sigma <= 0:
            raise ValueError('geometry_soft_prior_sigma must be positive.')
        if self.geometry_confidence_mode not in ['valid_mask', 'depth_edge']:
            raise ValueError(
                'geometry_confidence_mode must be valid_mask or depth_edge.')
        if self.geometry_depth_edge_scale < 0:
            raise ValueError('geometry_depth_edge_scale must be non-negative.')
        if not 0 <= self.geometry_depth_edge_floor <= 1:
            raise ValueError('geometry_depth_edge_floor must be in [0, 1].')
        if self.geometry_auxiliary_logit_scale <= 0:
            raise ValueError(
                'geometry_auxiliary_logit_scale must be positive.')
        if self.geometry_auxiliary_max_queries < 1:
            raise ValueError(
                'geometry_auxiliary_max_queries must be at least 1.')
        if self.gar_dcn_base_radius <= 0:
            raise ValueError('gar_dcn_base_radius must be positive.')
        if not 0 < self.gar_dcn_radius_min <= self.gar_dcn_radius_max:
            raise ValueError(
                'GAR-DCN radius bounds must be positive and ordered.')
        if not 0 < self.gar_dcn_extent_min <= self.gar_dcn_extent_max:
            raise ValueError(
                'GAR-DCN extent bounds must be positive and ordered.')
        if not 0 <= self.gar_dcn_mask_floor <= 1:
            raise ValueError('gar_dcn_mask_floor must be in [0, 1].')

        self.vgg_layer_list = vgg_layer_list
        self.vgg = VGGFeatureExtractor(
            layer_name_list=vgg_layer_list, vgg_type=vgg_type)
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1., 0., 1.],
                          [-2., 0., 2.],
                          [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1., -2., -1.],
                          [0., 0., 0.],
                          [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3))

    def index_to_flow(self, max_idx):
        device = max_idx.device
        h, w = max_idx.size()
        flow_w = max_idx % w
        flow_h = max_idx // w

        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h).to(device),
            torch.arange(0, w).to(device))
        grid = torch.stack((grid_x, grid_y), 2).unsqueeze(0).float().to(device)
        grid.requires_grad = False
        flow = torch.stack((flow_w, flow_h),
                           dim=2).unsqueeze(0).float().to(device)
        flow = flow - grid
        flow = torch.nn.functional.pad(flow, (0, 0, 0, 2, 0, 2))

        return flow

    def frequency_domain_matching(self, feat_in, feat_ref):
        """频域相关性计算（方案2）- 使用FFT代替DWT以支持多通道特征"""
        feat_in = feat_in.unsqueeze(0)
        feat_ref = feat_ref.unsqueeze(0)

        feat_in_fft = torch.fft.fft2(feat_in, dim=(-2, -1))
        feat_ref_fft = torch.fft.fft2(feat_ref, dim=(-2, -1))

        mag_in = torch.abs(feat_in_fft)
        mag_ref = torch.abs(feat_ref_fft)

        mag_in_flat = mag_in.reshape(mag_in.shape[0], mag_in.shape[1], -1)
        mag_ref_flat = mag_ref.reshape(mag_ref.shape[0], mag_ref.shape[1], -1)
        corr = F.cosine_similarity(mag_in_flat, mag_ref_flat, dim=1).unsqueeze(1)

        corr = corr.squeeze(0)
        return corr

    @staticmethod
    def _normalize_tensor_map(x):
        flat = x.reshape(x.size(0), -1)
        x_min = flat.min(dim=1)[0].reshape(-1, 1, 1, 1)
        x_max = flat.max(dim=1)[0].reshape(-1, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def _build_matching_geo_prior(self, depth_guidance, target_size):
        if depth_guidance is None:
            return None
        if depth_guidance.dim() == 3:
            depth_guidance = depth_guidance.unsqueeze(1)
        if depth_guidance.shape[1] != 1:
            raise ValueError(
                'Depth guidance for matching must have exactly one channel.')
        if depth_guidance.shape[-2:] != target_size:
            depth_guidance = F.interpolate(
                depth_guidance,
                size=target_size,
                mode='bilinear',
                align_corners=False)
        depth_guidance = self._normalize_tensor_map(depth_guidance.float())
        grad_x = F.conv2d(depth_guidance, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth_guidance, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-12)
        grad_mag = grad_mag / (
            grad_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        confidence = torch.exp(-self.geometry_prior_grad_scale * grad_mag)
        if self.geometry_prior_blur_kernel > 1:
            padding = self.geometry_prior_blur_kernel // 2
            confidence = F.avg_pool2d(
                confidence,
                kernel_size=self.geometry_prior_blur_kernel,
                stride=1,
                padding=padding)
        return confidence.clamp_(0.0, 1.0)

    def _build_depth_edge_confidence(self, depth_guidance, target_size):
        """Down-weight projective priors at noisy depth discontinuities."""
        if depth_guidance is None:
            raise ValueError(
                'depth_edge confidence requires target metric depth.')
        if depth_guidance.dim() == 3:
            depth_guidance = depth_guidance.unsqueeze(1)
        if depth_guidance.shape[1] != 1:
            raise ValueError('Metric depth must have exactly one channel.')
        depth = depth_guidance.float()
        if depth.shape[-2:] != target_size:
            depth = F.interpolate(
                depth, size=target_size, mode='nearest')
        valid = torch.isfinite(depth) & (depth > 0)
        safe_depth = torch.where(valid, depth, torch.ones_like(depth))
        log_depth = torch.log(safe_depth.clamp_min(1e-6))
        grad_x = F.conv2d(log_depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(log_depth, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square())
        valid_float = valid.float()
        grad_scale = (
            (grad_mag * valid_float).sum(dim=(2, 3), keepdim=True)
            / valid_float.sum(dim=(2, 3), keepdim=True).clamp_min(1.0))
        normalized_grad = grad_mag / (grad_scale + 1e-6)
        confidence = torch.exp(
            -self.geometry_depth_edge_scale * normalized_grad)
        confidence = (self.geometry_depth_edge_floor
                      + (1.0 - self.geometry_depth_edge_floor) * confidence)
        return confidence.clamp(0.0, 1.0) * valid_float

    def _apply_similarity_guidance(self, sim_map, depth_guidance):
        sim_out = torch.sigmoid(sim_map) if self.normalize_similarity else sim_map
        if (not self.use_matching_geo_prior
                or self.matching_geo_mode in [
                    'projective_window', 'projective_soft_prior']):
            return sim_out
        if depth_guidance is None:
            raise ValueError(
                'use_matching_geo_prior=True requires depth guidance input.')
        geo_prior = self._build_matching_geo_prior(
            depth_guidance, sim_out.shape[-2:])
        geo_weight = self.geometry_prior_floor + (
            1.0 - self.geometry_prior_floor) * geo_prior
        guided = sim_out * geo_weight
        return ((1.0 - self.geometry_prior_strength) * sim_out +
                self.geometry_prior_strength * guided)

    def _build_geometry_adaptive_dcn_guidance(
            self, projected_ref_coords, geometry_valid_mask, depth_guidance,
            target_sizes):
        if projected_ref_coords is None or geometry_valid_mask is None:
            raise ValueError(
                'Geometry-adaptive DCN requires projected coordinates and a '
                'validity mask.')

        coords = projected_ref_coords.float()
        valid = geometry_valid_mask.float()
        if coords.dim() == 3:
            coords = coords.unsqueeze(0)
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        if coords.shape[1] != 2 or valid.shape[1] != 1:
            raise ValueError(
                'GAR-DCN coordinates/mask must have 2 and 1 channels.')

        coords = torch.nan_to_num(coords, nan=0.0, posinf=1.0, neginf=0.0)
        radius_maps = {}
        confidence_maps = {}
        for level, target_size in target_sizes.items():
            target_h, target_w = target_size
            coords_level = F.interpolate(
                coords, size=target_size, mode='nearest')
            valid_level = F.interpolate(
                valid, size=target_size, mode='nearest').clamp(0.0, 1.0)

            coords_pixels = torch.cat([
                coords_level[:, 0:1] * max(target_w - 1, 1),
                coords_level[:, 1:2] * max(target_h - 1, 1),
            ], dim=1)
            horizontal_delta = (
                coords_pixels[:, :, :, 1:] - coords_pixels[:, :, :, :-1])
            vertical_delta = (
                coords_pixels[:, :, 1:, :] - coords_pixels[:, :, :-1, :])
            horizontal_span = torch.sqrt(
                horizontal_delta.square().sum(dim=1, keepdim=True) + 1e-12)
            vertical_span = torch.sqrt(
                vertical_delta.square().sum(dim=1, keepdim=True) + 1e-12)
            horizontal_valid = (
                valid_level[:, :, :, 1:] * valid_level[:, :, :, :-1])
            vertical_valid = (
                valid_level[:, :, 1:, :] * valid_level[:, :, :-1, :])

            span_sum = (
                F.pad(horizontal_span * horizontal_valid, (1, 0, 0, 0))
                + F.pad(horizontal_span * horizontal_valid, (0, 1, 0, 0))
                + F.pad(vertical_span * vertical_valid, (0, 0, 1, 0))
                + F.pad(vertical_span * vertical_valid, (0, 0, 0, 1)))
            support_count = (
                F.pad(horizontal_valid, (1, 0, 0, 0))
                + F.pad(horizontal_valid, (0, 1, 0, 0))
                + F.pad(vertical_valid, (0, 0, 1, 0))
                + F.pad(vertical_valid, (0, 0, 0, 1)))
            local_span = span_sum / support_count.clamp_min(1.0)
            local_span = local_span.clamp(
                self.gar_dcn_extent_min, self.gar_dcn_extent_max)

            if depth_guidance is None:
                depth_confidence = torch.ones_like(valid_level)
            else:
                depth_confidence = self._build_depth_edge_confidence(
                    depth_guidance, target_size)
            support_confidence = (support_count / 4.0).clamp(0.0, 1.0)
            geometry_reliability = depth_confidence * support_confidence
            has_local_support = support_count >= 2.0

            candidate_radius = (
                self.gar_dcn_base_radius * local_span).clamp(
                    self.gar_dcn_radius_min, self.gar_dcn_radius_max)
            radius = self.gar_dcn_base_radius + geometry_reliability * (
                candidate_radius - self.gar_dcn_base_radius)
            use_adaptive_radius = (valid_level > 0.5) & has_local_support
            radius = torch.where(
                use_adaptive_radius, radius,
                torch.full_like(radius, self.gar_dcn_base_radius))

            confidence = self.gar_dcn_mask_floor + (
                1.0 - self.gar_dcn_mask_floor) * geometry_reliability
            confidence = torch.where(
                valid_level > 0.5, confidence, torch.ones_like(confidence))
            radius_maps[level] = radius
            confidence_maps[level] = confidence.clamp(0.0, 1.0)

        return radius_maps, confidence_maps

    def forward(self, dense_features, img_ref_hr, depth_guidance=None,
                projected_ref_coords=None, geometry_valid_mask=None,
                compute_matching_auxiliary=False):
        batch_offset_relu3 = []
        batch_offset_relu2 = []
        batch_offset_relu1 = []
        flows_relu3 = []
        flows_relu2 = []
        flows_relu1 = []
        similarity_relu3 = []
        similarity_relu2 = []
        similarity_relu1 = []
        auxiliary_matching = []
        auxiliary_projection = []
        auxiliary_valid_ratio = []
        auxiliary_confidence_mean = []

        for ind in range(img_ref_hr.size(0)):
            feat_in = dense_features['dense_features1'][ind]
            feat_ref = dense_features['dense_features2'][ind]
            c, h, w = feat_in.size()
            feat_in = F.normalize(feat_in.reshape(c, -1), dim=0).view(c, h, w)
            feat_ref = F.normalize(
                feat_ref.reshape(c, -1), dim=0).view(c, h, w)

            if (self.use_matching_geo_prior
                    and self.matching_geo_mode == 'projective_window'):
                if projected_ref_coords is None or geometry_valid_mask is None:
                    raise ValueError(
                        'projective_window matching requires geometry tensors.')
                _max_idx, _max_val = geometry_guided_feature_match_index(
                    feat_in,
                    feat_ref,
                    projected_ref_coords[ind],
                    geometry_valid_mask[ind],
                    patch_size=self.patch_size,
                    input_stride=self.stride,
                    ref_stride=self.stride,
                    is_norm=True,
                    norm_input=True,
                    search_radius=self.geometry_search_radius,
                    position_weight=self.geometry_position_weight,
                    position_sigma=self.geometry_position_sigma)
            elif (self.use_matching_geo_prior
                  and self.matching_geo_mode == 'projective_soft_prior'):
                if projected_ref_coords is None or geometry_valid_mask is None:
                    raise ValueError(
                        'projective_soft_prior matching requires geometry '
                        'tensors.')
                geometry_confidence = None
                if self.geometry_confidence_mode == 'depth_edge':
                    if depth_guidance is None:
                        raise ValueError(
                            'depth_edge confidence requires depth guidance.')
                    geometry_confidence = self._build_depth_edge_confidence(
                        depth_guidance[ind:ind + 1],
                        projected_ref_coords[ind].shape[-2:])
                _max_idx, _max_val, auxiliary = \
                    probabilistic_geometry_feature_match_index(
                        feat_in,
                        feat_ref,
                        projected_ref_coords[ind],
                        geometry_valid_mask[ind],
                        geometry_confidence=(
                            geometry_confidence[0]
                            if geometry_confidence is not None else None),
                        patch_size=self.patch_size,
                        input_stride=self.stride,
                        ref_stride=self.stride,
                        is_norm=True,
                        norm_input=True,
                        prior_strength=self.geometry_soft_prior_strength,
                        prior_sigma=self.geometry_soft_prior_sigma,
                        compute_auxiliary=compute_matching_auxiliary,
                        auxiliary_logit_scale=(
                            self.geometry_auxiliary_logit_scale),
                        auxiliary_max_queries=(
                            self.geometry_auxiliary_max_queries))
                auxiliary_matching.append(auxiliary['matching'])
                auxiliary_projection.append(auxiliary['projection'])
                auxiliary_valid_ratio.append(auxiliary['valid_ratio'])
                auxiliary_confidence_mean.append(
                    auxiliary['confidence_mean'])
            else:
                _max_idx, _max_val = feature_match_index(
                    feat_in,
                    feat_ref,
                    patch_size=self.patch_size,
                    input_stride=self.stride,
                    ref_stride=self.stride,
                    is_norm=True,
                    norm_input=True)

            sim_relu3 = F.pad(_max_val, (1, 1, 1, 1)).unsqueeze(0).unsqueeze(0)

            # 添加频域匹配（方案2）
            if self.use_freq_matching:
                freq_corr = self.frequency_domain_matching(feat_in, feat_ref)
                freq_corr = F.interpolate(freq_corr.unsqueeze(0).unsqueeze(0),
                                          size=sim_relu3.shape[-2:],
                                          mode='bilinear', align_corners=False)
                sim_relu3 = sim_relu3 * freq_corr

            depth_slice = None
            if (self.matching_geo_mode == 'depth'
                    and depth_guidance is not None):
                depth_slice = depth_guidance[ind:ind + 1]
            sim_relu3 = self._apply_similarity_guidance(
                sim_relu3, depth_slice)

            similarity_relu3.append(sim_relu3)

            offset_relu3 = self.index_to_flow(_max_idx)
            flows_relu3.append(offset_relu3)

            shifted_offset_relu3 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu3, (i, j))
                    shifted_offset_relu3.append(flow_shift)
            shifted_offset_relu3 = torch.cat(shifted_offset_relu3, dim=0)
            batch_offset_relu3.append(shifted_offset_relu3)

            # similarity for relu2_1 - 保持与relu3_1相同的尺寸
            sim_relu2 = F.interpolate(sim_relu3, scale_factor=2, mode='bilinear', align_corners=False)
            similarity_relu2.append(sim_relu2)

            offset_relu2 = torch.repeat_interleave(offset_relu3, 2, 1)
            offset_relu2 = torch.repeat_interleave(offset_relu2, 2, 2)
            offset_relu2 *= 2
            flows_relu2.append(offset_relu2)

            shifted_offset_relu2 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu2, (i * 2, j * 2))
                    shifted_offset_relu2.append(flow_shift)
            shifted_offset_relu2 = torch.cat(shifted_offset_relu2, dim=0)
            batch_offset_relu2.append(shifted_offset_relu2)

            # similarity for relu1_1 - 保持与relu3_1相同的尺寸
            sim_relu1 = F.interpolate(sim_relu3, scale_factor=4, mode='bilinear', align_corners=False)
            similarity_relu1.append(sim_relu1)

            offset_relu1 = torch.repeat_interleave(offset_relu3, 4, 1)
            offset_relu1 = torch.repeat_interleave(offset_relu1, 4, 2)
            offset_relu1 *= 4
            flows_relu1.append(offset_relu1)

            shifted_offset_relu1 = []
            for i in range(0, 3):
                for j in range(0, 3):
                    flow_shift = tensor_shift(offset_relu1, (i * 4, j * 4))
                    shifted_offset_relu1.append(flow_shift)
            shifted_offset_relu1 = torch.cat(shifted_offset_relu1, dim=0)
            batch_offset_relu1.append(shifted_offset_relu1)

        batch_offset_relu3 = torch.stack(batch_offset_relu3, dim=0).unsqueeze(1)
        batch_offset_relu2 = torch.stack(batch_offset_relu2, dim=0).unsqueeze(1)
        batch_offset_relu1 = torch.stack(batch_offset_relu1, dim=0).unsqueeze(1)

        pre_flow = {}
        pre_flow['relu3_1'] = torch.cat(flows_relu3, dim=0)
        pre_flow['relu2_1'] = torch.cat(flows_relu2, dim=0)
        pre_flow['relu1_1'] = torch.cat(flows_relu1, dim=0)

        pre_offset = {}
        pre_offset['relu1_1'] = batch_offset_relu1
        pre_offset['relu2_1'] = batch_offset_relu2
        pre_offset['relu3_1'] = batch_offset_relu3

        pre_similarity = {}
        pre_similarity['relu3_1'] = torch.stack(similarity_relu3, dim=0)
        pre_similarity['relu2_1'] = torch.stack(similarity_relu2, dim=0)
        pre_similarity['relu1_1'] = torch.stack(similarity_relu1, dim=0)

        geometry_radius = None
        geometry_confidence = None
        if self.use_geometry_adaptive_dcn:
            target_sizes = {
                level: similarity.shape[-2:]
                for level, similarity in pre_similarity.items()
            }
            geometry_radius, geometry_confidence = \
                self._build_geometry_adaptive_dcn_guidance(
                    projected_ref_coords, geometry_valid_mask,
                    depth_guidance, target_sizes)

        img_ref_feat = self.vgg(img_ref_hr)

        self.matching_aux_losses = {}
        if auxiliary_matching:
            self.matching_aux_losses = {
                'matching': torch.stack(auxiliary_matching).mean(),
                'projection': torch.stack(auxiliary_projection).mean(),
                'valid_ratio': torch.stack(auxiliary_valid_ratio).mean(),
                'confidence_mean': torch.stack(
                    auxiliary_confidence_mean).mean(),
            }

        correspondence = [pre_offset, pre_flow, pre_similarity]
        if geometry_radius is not None:
            correspondence.extend([geometry_radius, geometry_confidence])
        return correspondence, img_ref_feat

    def get_matching_aux_losses(self):
        return self.matching_aux_losses
