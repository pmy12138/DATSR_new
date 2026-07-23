import csv
import os

import cv2
import mmcv
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from datsr.data.transforms import augment, paired_random_crop, totensor
from datsr.data.geometry_ref_crop import (compute_projected_ref_box,
                                          crop_projected_ref,
                                          load_camera_manifest,
                                          project_target_depth_map,
                                          project_target_patch,
                                          read_metric_depth)
from datsr.utils import FileClient


def _resolve_dataroot(path):
    if path is None:
        return None
    resolved = os.path.normpath(path)
    if os.path.exists(resolved):
        return resolved
    unix_like = path.replace('\\', '/')
    prefix = 'datasets/'
    if unix_like.startswith(prefix):
        suffix = unix_like[len(prefix):]
        alt = os.path.join(os.sep, 'root', 'datasets', *suffix.split('/'))
        if os.path.exists(alt):
            return alt
    return resolved


def _read_guidance_depth(path):
    lower = path.lower()
    if lower.endswith(('.npy', '.npz', '.pfm', '.exr')):
        depth = read_metric_depth(path)
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
    if depth.ndim == 2:
        depth = depth[..., None]
    return depth


class MCRefSRDataset(data.Dataset):
    """Monte Carlo noisy RefSR dataset.

    Expected folder structure for each split:
        input/  noisy LR images, e.g. 320x180
        gt/     clean HR targets, e.g. 1280x720
        ref/    clean HR reference images from a different view
        albedo/ optional HR albedo maps
        normal/ optional HR surface-normal maps

    Returned fields follow RefRestorationModel:
        img_in_lq: noisy LR input
        img_in_up: bicubic upsampled noisy LR for correspondence matching
        img_ref: clean HR reference, or albedo when ref_source is albedo
        img_in: clean HR target
    """

    def __init__(self, opt):
        super(MCRefSRDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()

        self.in_folder = _resolve_dataroot(opt['dataroot_in'])
        self.gt_folder = _resolve_dataroot(opt['dataroot_gt'])
        self.ref_folder = _resolve_dataroot(opt.get('dataroot_ref', None))
        self.albedo_folder = _resolve_dataroot(opt.get('dataroot_albedo', None))
        self.normal_folder = _resolve_dataroot(opt.get('dataroot_normal', None))
        self.ref_source = opt.get('ref_source', 'ref')
        self.ref_mode = opt.get('ref_mode', 'normal')
        self.ref_shuffle_offset = opt.get('ref_shuffle_offset', None)
        if self.ref_shuffle_offset is not None:
            self.ref_shuffle_offset = int(self.ref_shuffle_offset)
        self.use_albedo = opt.get('use_albedo', False)
        self.use_normal = opt.get('use_normal', False)
        self.use_matching_geo_prior = opt.get('use_matching_geo_prior', False)
        self.matching_geo_mode = opt.get('matching_geo_mode', 'depth')
        self.use_projective_matching = (
            self.use_matching_geo_prior
            and self.matching_geo_mode in [
                'projective_window', 'projective_soft_prior'])
        self.albedo_mode = opt.get('albedo_mode', 'normal')
        self.albedo_shuffle_offset = int(opt.get('albedo_shuffle_offset', 1))
        self.normal_mode = opt.get('normal_mode', 'normal')
        self.normal_shuffle_offset = int(opt.get('normal_shuffle_offset', 1))
        self.test_filenames = opt.get('test_filenames', None)
        self.sample_enlarge_ratio = int(opt.get('sample_enlarge_ratio', 1))
        self.sample_enlarge_ratio = max(self.sample_enlarge_ratio, 1)
        self.lq_noise_aug = opt.get('lq_noise_aug', False)
        self.lq_noise_aug_sigma_max = float(
            opt.get('lq_noise_aug_sigma_max', 0))
        self.use_geometry_ref_crop = opt.get('use_geometry_ref_crop', False)
        self.depth_folder = _resolve_dataroot(
            opt.get('dataroot_depth_metric', None))
        self.guidance_depth_folder = _resolve_dataroot(
            opt.get('dataroot_depth', opt.get('dataroot_depth_metric', None)))
        self.camera_manifest_path = _resolve_dataroot(
            opt.get('camera_manifest', None))
        self.geometry_min_overlap = float(
            opt.get('geometry_min_overlap', 0.60))
        self.geometry_max_attempts = int(
            opt.get('geometry_max_attempts', 20))
        self.geometry_fallback_grid_stride = int(
            opt.get('geometry_fallback_grid_stride', 4))
        self.geometry_sample_stride = int(
            opt.get('geometry_sample_stride', 4))
        self.geometry_ref_margin = float(
            opt.get('geometry_ref_margin', 1.2))
        self.geometry_ref_min_scale = float(
            opt.get('geometry_ref_min_scale', 0.5))
        self.geometry_ref_max_scale = float(
            opt.get('geometry_ref_max_scale', 4.0))
        self.geometry_projection_trim_percentile = float(
            opt.get('geometry_projection_trim_percentile', 1.0))
        self.geometry_min_crop_coverage = float(
            opt.get('geometry_min_crop_coverage', 0.90))
        self.geometry_depth_type = opt.get(
            'geometry_depth_type', 'ray_distance')
        self.geometry_use_ref_depth_consistency = opt.get(
            'geometry_use_ref_depth_consistency', False)
        self.geometry_depth_consistency_rel_tol = float(
            opt.get('geometry_depth_consistency_rel_tol', 0.05))
        self.geometry_depth_consistency_abs_tol = float(
            opt.get('geometry_depth_consistency_abs_tol', 0.10))
        self.geometry_fail_policy = opt.get('geometry_fail_policy', 'error')
        self.geometry_strict_pairs = opt.get('geometry_strict_pairs', True)
        self.camera_manifest = None
        if self.use_geometry_ref_crop or self.use_projective_matching:
            if self.ref_source != 'ref' or self.ref_mode != 'normal':
                raise ValueError(
                    'Camera projection requires ref_source=ref and ref_mode=normal.')
            if self.depth_folder is None or self.camera_manifest_path is None:
                raise ValueError(
                    'dataroot_depth_metric and camera_manifest are required '
                    'for geometry crop/projective-window matching.')
            self.camera_manifest = load_camera_manifest(
                self.camera_manifest_path)
        if self.use_geometry_ref_crop:
            if opt.get('phase') != 'train':
                raise ValueError(
                    'use_geometry_ref_crop currently applies to training only.')
            if self.geometry_fail_policy not in ['error', 'best']:
                raise ValueError(
                    'geometry_fail_policy must be error or best.')
            if not 0 <= self.geometry_min_overlap <= 1:
                raise ValueError('geometry_min_overlap must be in [0, 1].')
            if self.geometry_max_attempts < 1:
                raise ValueError('geometry_max_attempts must be at least 1.')
            if self.geometry_fallback_grid_stride < 1:
                raise ValueError(
                    'geometry_fallback_grid_stride must be at least 1.')
            if self.geometry_sample_stride < 1:
                raise ValueError('geometry_sample_stride must be at least 1.')
            if (self.geometry_ref_margin <= 0
                    or self.geometry_ref_min_scale <= 0
                    or self.geometry_ref_max_scale
                    < self.geometry_ref_min_scale):
                raise ValueError('Invalid geometry Ref crop margin/scale bounds.')
            if not 0 <= self.geometry_projection_trim_percentile < 50:
                raise ValueError(
                    'geometry_projection_trim_percentile must be in [0, 50).')
            if not 0 <= self.geometry_min_crop_coverage <= 1:
                raise ValueError('geometry_min_crop_coverage must be in [0, 1].')
        if self.ref_source not in ['ref', 'albedo']:
            raise ValueError(
                f'Unsupported ref_source: {self.ref_source}. '
                'Supported choices are ref and albedo.')
        if self.ref_source == 'ref' and self.ref_folder is None:
            raise ValueError('dataroot_ref is required when ref_source=ref.')
        if self.ref_source == 'albedo' and self.albedo_folder is None:
            raise ValueError(
                'dataroot_albedo is required when ref_source=albedo.')
        if self.ref_mode not in ['normal', 'zero', 'shuffled']:
            raise ValueError(
                f'Unsupported ref_mode: {self.ref_mode}. '
                'Supported choices are normal, zero and shuffled.')
        if self.ref_source != 'ref' and self.ref_mode != 'normal':
            raise ValueError(
                'Ref ablation modes are only supported when ref_source=ref.')
        if self.use_albedo and self.albedo_folder is None:
            raise ValueError('dataroot_albedo is required when use_albedo=True.')
        if self.use_normal and self.normal_folder is None:
            raise ValueError('dataroot_normal is required when use_normal=True.')
        if (self.use_normal and (opt.get('use_flip', False)
                                 or opt.get('use_rot', False))):
            raise ValueError(
                'Disable use_flip/use_rot when use_normal=True. Spatially '
                'flipping or rotating a normal map also requires transforming '
                'its vector components, which the generic image augment does '
                'not perform.')
        if self.use_matching_geo_prior:
            if self.matching_geo_mode not in [
                    'depth', 'projective_window', 'projective_soft_prior']:
                raise ValueError(
                    'matching_geo_mode must be depth, projective_window, or '
                    'projective_soft_prior.')
            if (self.matching_geo_mode == 'depth'
                    and self.guidance_depth_folder is None):
                raise ValueError(
                    'dataroot_depth or dataroot_depth_metric is required when '
                    'use_matching_geo_prior=True.')
            if (self.use_projective_matching and opt.get('phase') == 'train'
                    and not self.use_geometry_ref_crop):
                raise ValueError(
                    'Training with projective matching currently requires '
                    'use_geometry_ref_crop=True so target/Ref crop transforms '
                    'remain known.')
            if (self.use_projective_matching
                    and (opt.get('use_flip', False)
                         or opt.get('use_rot', False))):
                raise ValueError(
                    'Disable use_flip/use_rot for projective matching; '
                    'camera-projected coordinates must stay in the original '
                    'image orientation.')
            if (self.geometry_use_ref_depth_consistency
                    and not self.use_projective_matching):
                raise ValueError(
                    'Ref-depth consistency is only supported by '
                    'projective matching modes.')
            if (self.geometry_depth_consistency_rel_tol < 0
                    or self.geometry_depth_consistency_abs_tol < 0):
                raise ValueError(
                    'Geometry depth-consistency tolerances must be non-negative.')
        if self.albedo_mode not in ['normal', 'zero', 'shuffled']:
            raise ValueError(
                f'Unsupported albedo_mode: {self.albedo_mode}. '
                'Supported choices are normal, zero and shuffled.')
        if self.normal_mode not in ['normal', 'zero', 'shuffled']:
            raise ValueError(
                f'Unsupported normal_mode: {self.normal_mode}. '
                'Supported choices are normal, zero and shuffled.')
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.dataset_root = None
        self.dataset_split = None
        self.manifest_rows = self._load_manifest_rows()

        self.paths = []
        if self.manifest_rows is not None:
            names = [row['name'] for row in self.manifest_rows]
        else:
            names = sorted([
                f for f in os.listdir(self.in_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))
            ])
        missing_geometry = []
        missing_metric_depth = []
        missing_camera_manifest = []
        missing_matching_geo = []
        missing_normal = []
        for name in names:
            in_path = os.path.join(self.in_folder, name)
            gt_path = os.path.join(self.gt_folder, name)
            if self.ref_source == 'albedo':
                ref_path = os.path.join(self.albedo_folder, name)
            else:
                ref_path = os.path.join(self.ref_folder, name)
            albedo_path = os.path.join(self.albedo_folder, name) if self.use_albedo else None
            normal_path = os.path.join(self.normal_folder, name) if self.use_normal else None
            metric_depth_path = self._find_metric_depth(name)
            ref_metric_depth_path = self._find_ref_metric_depth(name)
            guidance_depth_path = self._find_guidance_depth(name)
            has_albedo = (not self.use_albedo or os.path.exists(albedo_path))
            has_normal = (not self.use_normal or os.path.exists(normal_path))
            if not has_normal:
                missing_normal.append(name)
            has_metric_depth = metric_depth_path is not None
            needs_camera = self.use_geometry_ref_crop or self.use_projective_matching
            has_camera_manifest = (
                not needs_camera or name in self.camera_manifest)
            has_geometry = (not needs_camera or
                            (has_metric_depth and has_camera_manifest))
            if self.use_projective_matching:
                has_matching_geo = (has_metric_depth and has_camera_manifest
                                    and (not self.geometry_use_ref_depth_consistency
                                         or ref_metric_depth_path is not None))
            else:
                has_matching_geo = (not self.use_matching_geo_prior or
                                    guidance_depth_path is not None)
            if needs_camera and not has_geometry:
                missing_geometry.append(name)
                if not has_metric_depth:
                    missing_metric_depth.append(name)
                if not has_camera_manifest:
                    missing_camera_manifest.append(name)
            if self.use_matching_geo_prior and not has_matching_geo:
                missing_matching_geo.append(name)
            if (os.path.exists(gt_path) and os.path.exists(ref_path)
                    and has_albedo and has_normal and has_geometry
                    and has_matching_geo):
                self.paths.append({
                    'in_path': in_path,
                    'gt_path': gt_path,
                    'ref_path': ref_path,
                    'albedo_path': albedo_path,
                    'normal_path': normal_path,
                    'depth_path': metric_depth_path,
                    'ref_depth_path': ref_metric_depth_path,
                    'guidance_depth_path': guidance_depth_path,
                    'name': name,
                })

        if ((self.use_geometry_ref_crop or self.use_projective_matching)
                and self.geometry_strict_pairs
                and missing_geometry):
            preview = ', '.join(missing_geometry[:5])
            metric_preview = ', '.join(missing_metric_depth[:5])
            camera_preview = ', '.join(missing_camera_manifest[:5])
            camera_keys = list(self.camera_manifest.keys())[:5] \
                if self.camera_manifest is not None else []
            raise ValueError(
                f'{len(missing_geometry)} input images are missing metric '
                f'depth or camera manifest entries (examples: {preview}). '
                f'Missing metric depth: {len(missing_metric_depth)} '
                f'(examples: {metric_preview}). Missing camera manifest: '
                f'{len(missing_camera_manifest)} '
                f'(examples: {camera_preview}). Resolved depth_metric root: '
                f'{self.depth_folder}. Resolved camera manifest: '
                f'{self.camera_manifest_path}. First camera manifest keys: '
                f'{camera_keys}. '
                'Set geometry_strict_pairs=false only for an intentional '
                'partial-dataset experiment.')
        if self.use_matching_geo_prior and missing_matching_geo:
            preview = ', '.join(missing_matching_geo[:5])
            raise ValueError(
                f'{len(missing_matching_geo)} input images are missing depth '
                f'guidance maps required by use_matching_geo_prior '
                f'(examples: {preview}).')
        if self.use_normal and missing_normal:
            preview = ', '.join(missing_normal[:5])
            raise ValueError(
                f'{len(missing_normal)} input images are missing paired normal '
                f'maps (examples: {preview}). Resolved normal root: '
                f'{self.normal_folder}.')

        if not self.paths:
            if self.use_geometry_ref_crop:
                raise ValueError(
                    'No geometry-ready MC RefSR pairs were found. Ensure '
                    'dataroot_depth_metric contains float depth files with '
                    'the same stems as the input images and camera_manifest '
                    'contains every image name.')
            raise ValueError(
                f'No paired MC RefSR images found in {self.in_folder}, '
                f'{self.gt_folder}, {self.ref_folder}.')
        if self.test_filenames is not None:
            if self.opt.get('phase') == 'train':
                raise ValueError('test_filenames is only supported outside training.')
            requested_names = set(self.test_filenames)
            self.paths = [
                item for item in self.paths
                if os.path.basename(item['in_path']) in requested_names
            ]
            if not self.paths:
                raise ValueError(
                    'None of test_filenames were found in the paired MC RefSR dataset.')
        if (self.use_albedo and self.albedo_mode == 'shuffled'
                and len(self.paths) < 2):
            raise ValueError(
                'albedo_mode=shuffled requires at least two paired samples.')
        if (self.use_normal and self.normal_mode == 'shuffled'
                and len(self.paths) < 2):
            raise ValueError(
                'normal_mode=shuffled requires at least two paired samples.')
        if self.ref_mode == 'shuffled' and len(self.paths) < 2:
            raise ValueError(
                'ref_mode=shuffled requires at least two paired samples.')

    def _load_manifest_rows(self):
        in_path = os.path.normpath(self.in_folder)
        in_parent = os.path.dirname(in_path)
        split_name = os.path.basename(in_parent)
        dataset_root = os.path.dirname(in_parent)
        manifest_path = os.path.join(dataset_root, 'manifest.csv')
        if (not os.path.exists(manifest_path)
                or os.path.basename(in_path).lower() != 'input'
                or split_name not in ['train', 'test', 'val']):
            return None
        with open(manifest_path, 'r', encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        split_rows = [row for row in rows if row.get('split') == split_name]
        if not split_rows:
            return None
        self.dataset_root = dataset_root
        self.dataset_split = split_name
        return split_rows

    def _find_metric_depth(self, name):
        if not (self.use_geometry_ref_crop or self.use_projective_matching):
            return None
        stem = os.path.splitext(name)[0]
        for extension in ['.pfm', '.npy', '.npz', '.exr']:
            path = os.path.join(self.depth_folder, stem + extension)
            if os.path.exists(path):
                return path
        return None

    def _find_guidance_depth(self, name):
        if (not self.use_matching_geo_prior or self.use_projective_matching):
            return None
        stem = os.path.splitext(name)[0]
        for extension in ['.pfm', '.npy', '.npz', '.exr',
                          '.png', '.jpg', '.jpeg', '.bmp', '.tif']:
            path = os.path.join(self.guidance_depth_folder, stem + extension)
            if os.path.exists(path):
                return path
        return None

    def _find_ref_metric_depth(self, name):
        if not self.geometry_use_ref_depth_consistency:
            return None
        camera_pair = self.camera_manifest.get(name)
        if camera_pair is None:
            return None
        scene = camera_pair.get('scene')
        ref_idx = camera_pair.get('ref_idx')
        if scene is None or ref_idx is None:
            return None
        stem = f'{scene}_scene_{int(ref_idx):04d}'
        for extension in ['.pfm', '.npy', '.npz', '.exr']:
            path = os.path.join(self.depth_folder, stem + extension)
            if os.path.exists(path):
                return path
        return None

    def _geometry_random_crop(self, img_gt, img_lq, img_ref, img_albedo,
                              img_normal, img_depth,
                              path_info, gt_size, scale):
        if gt_size % scale != 0:
            raise ValueError(
                f'gt_size={gt_size} must be divisible by scale={scale}.')
        depth = read_metric_depth(path_info['depth_path'])
        if depth.shape != img_gt.shape[:2]:
            depth = cv2.resize(
                depth, (img_gt.shape[1], img_gt.shape[0]),
                interpolation=cv2.INTER_NEAREST)
        camera_pair = self.camera_manifest[path_info['name']]
        original_ref_size = img_ref.shape[:2]
        lq_size = gt_size // scale
        max_top = img_lq.shape[0] - lq_size
        max_left = img_lq.shape[1] - lq_size
        if max_top < 0 or max_left < 0:
            raise ValueError(
                f'GT size {gt_size} is larger than {path_info["gt_path"]}.')

        best = None
        selected = None

        def evaluate_candidate(top_lq, left_lq):
            top_gt = top_lq * scale
            left_gt = left_lq * scale
            projection = project_target_patch(
                depth, camera_pair['target'], camera_pair['ref'],
                (top_gt, left_gt, gt_size, gt_size), img_ref.shape[:2],
                sample_stride=self.geometry_sample_stride,
                depth_type=self.geometry_depth_type,
                trim_percentile=self.geometry_projection_trim_percentile)
            if projection is None or 'center' not in projection:
                return None
            ref_box, crop_coverage = compute_projected_ref_box(
                img_ref.shape[:2], projection, gt_size,
                margin=self.geometry_ref_margin,
                min_scale=self.geometry_ref_min_scale,
                max_scale=self.geometry_ref_max_scale)
            return (projection, crop_coverage, ref_box, top_lq,
                    left_lq, top_gt, left_gt)

        def consider_candidate(candidate):
            nonlocal best
            if candidate is None:
                return False
            projection, crop_coverage = candidate[:2]
            candidate_score = projection['overlap'] * crop_coverage
            if best is None or candidate_score > best[0]['overlap'] * best[1]:
                best = candidate
            return (projection['overlap'] >= self.geometry_min_overlap
                    and crop_coverage >= self.geometry_min_crop_coverage)

        for _ in range(self.geometry_max_attempts):
            top_lq = np.random.randint(0, max_top + 1)
            left_lq = np.random.randint(0, max_left + 1)
            candidate = evaluate_candidate(top_lq, left_lq)
            if consider_candidate(candidate):
                selected = candidate
                break

        # Random rejection sampling can miss a valid region by chance. Fall
        # back to a shuffled lattice so a single unlucky sample cannot abort a
        # long training run. The lattice is only evaluated after all random
        # attempts fail, so normal training keeps the original randomness and
        # cost profile.
        if selected is None:
            stride = self.geometry_fallback_grid_stride
            top_positions = list(range(0, max_top + 1, stride))
            left_positions = list(range(0, max_left + 1, stride))
            if top_positions[-1] != max_top:
                top_positions.append(max_top)
            if left_positions[-1] != max_left:
                left_positions.append(max_left)
            fallback_positions = [
                (top_lq, left_lq)
                for top_lq in top_positions
                for left_lq in left_positions
            ]
            np.random.shuffle(fallback_positions)
            for top_lq, left_lq in fallback_positions:
                candidate = evaluate_candidate(top_lq, left_lq)
                if consider_candidate(candidate):
                    selected = candidate
                    break

        if selected is None:
            if best is None or self.geometry_fail_policy == 'error':
                best_overlap = 0.0 if best is None else best[0]['overlap']
                best_coverage = 0.0 if best is None else best[1]
                raise RuntimeError(
                    f'No geometry crop reached overlap '
                    f'{self.geometry_min_overlap:.3f} and crop coverage '
                    f'{self.geometry_min_crop_coverage:.3f} for '
                    f'{path_info["name"]}; best overlap={best_overlap:.3f}, '
                    f'best coverage={best_coverage:.3f}.')
            selected = best

        (projection, crop_coverage, expected_ref_box, top_lq, left_lq,
         top_gt, left_gt) = selected
        img_gt = img_gt[top_gt:top_gt + gt_size,
                        left_gt:left_gt + gt_size]
        img_lq = img_lq[top_lq:top_lq + lq_size,
                        left_lq:left_lq + lq_size]
        if img_albedo is not None:
            img_albedo = img_albedo[top_gt:top_gt + gt_size,
                                    left_gt:left_gt + gt_size]
        if img_normal is not None:
            img_normal = img_normal[top_gt:top_gt + gt_size,
                                    left_gt:left_gt + gt_size]
        if img_depth is not None:
            img_depth = img_depth[top_gt:top_gt + gt_size,
                                  left_gt:left_gt + gt_size]
        img_ref, ref_box = crop_projected_ref(
            img_ref, projection, gt_size,
            margin=self.geometry_ref_margin,
            min_scale=self.geometry_ref_min_scale,
            max_scale=self.geometry_ref_max_scale)
        if ref_box != expected_ref_box:
            raise RuntimeError('Geometry Ref crop box changed after selection.')
        geometry_info = {
            'geometry_overlap': projection['overlap'],
            'geometry_fov_overlap': projection['fov_overlap'],
            'geometry_valid_depth_ratio': projection['valid_depth_ratio'],
            'geometry_crop_coverage': crop_coverage,
            'target_crop_box': (top_gt, left_gt, gt_size, gt_size),
            'ref_crop_box': ref_box,
        }
        geo_ref_coords, geo_valid_mask = None, None
        if self.use_projective_matching:
            ref_depth = None
            if self.geometry_use_ref_depth_consistency:
                ref_depth = read_metric_depth(path_info['ref_depth_path'])
                if ref_depth.shape != original_ref_size:
                    ref_depth = cv2.resize(
                        ref_depth,
                        (original_ref_size[1], original_ref_size[0]),
                        interpolation=cv2.INTER_NEAREST)
            geo_ref_coords, geo_valid_mask = project_target_depth_map(
                depth,
                camera_pair['target'],
                camera_pair['ref'],
                geometry_info['target_crop_box'],
                original_ref_size,
                ref_box=geometry_info['ref_crop_box'],
                output_size=img_gt.shape[:2],
                ref_output_size=img_ref.shape[:2],
                depth_type=self.geometry_depth_type,
                ref_depth=ref_depth,
                depth_consistency_rel_tol=(
                    self.geometry_depth_consistency_rel_tol),
                depth_consistency_abs_tol=(
                    self.geometry_depth_consistency_abs_tol))
        return (img_gt, img_lq, img_ref, img_albedo, img_normal, img_depth,
                geometry_info, geo_ref_coords, geo_valid_mask)

    def _read_img(self, path, key):
        img_bytes = self.file_client.get(path, key)
        return mmcv.imfrombytes(img_bytes).astype(np.float32) / 255.

    @staticmethod
    def _resize_to(img, size):
        target_h, target_w = size
        if img.shape[:2] == (target_h, target_w):
            return img
        pil_img = Image.fromarray(
            cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR).astype(
            np.float32) / 255.

    @staticmethod
    def _resize_depth_to(depth, size):
        target_h, target_w = size
        if depth.shape[:2] == (target_h, target_w):
            return depth
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = cv2.resize(
            depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return depth[..., None]

    @staticmethod
    def _bicubic_up(img_lq, size):
        target_h, target_w = size
        pil_img = Image.fromarray(
            cv2.cvtColor((img_lq * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((target_w, target_h), Image.BICUBIC)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR).astype(
            np.float32) / 255.

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 4)
        base_index = index % len(self.paths)
        path_info = self.paths[base_index]
        img_lq = self._read_img(path_info['in_path'], 'in')
        img_gt = self._read_img(path_info['gt_path'], 'gt')
        if self.ref_mode == 'zero':
            img_ref = np.zeros_like(img_gt)
        else:
            ref_path = path_info['ref_path']
            if self.ref_mode == 'shuffled':
                offset = self.ref_shuffle_offset
                if offset is None:
                    offset = max(1, len(self.paths) // 2)
                offset %= len(self.paths)
                offset = offset if offset != 0 else 1
                ref_path = self.paths[
                    (base_index + offset) % len(self.paths)]['ref_path']
            img_ref = self._read_img(ref_path, 'ref')
        if not self.use_albedo:
            img_albedo = None
        elif self.albedo_mode == 'zero':
            img_albedo = np.zeros_like(img_gt)
        else:
            albedo_path = path_info['albedo_path']
            if self.albedo_mode == 'shuffled':
                offset = self.albedo_shuffle_offset % len(self.paths)
                offset = offset if offset != 0 else 1
                albedo_path = self.paths[
                    (base_index + offset) % len(self.paths)]['albedo_path']
            img_albedo = self._read_img(albedo_path, 'albedo')
        if not self.use_normal:
            img_normal = None
        elif self.normal_mode == 'zero':
            img_normal = np.zeros_like(img_gt)
        else:
            normal_path = path_info['normal_path']
            if self.normal_mode == 'shuffled':
                offset = self.normal_shuffle_offset % len(self.paths)
                offset = offset if offset != 0 else 1
                normal_path = self.paths[
                    (base_index + offset) % len(self.paths)]['normal_path']
            img_normal = self._read_img(normal_path, 'normal')
        if self.use_projective_matching:
            img_depth = read_metric_depth(
                path_info['depth_path']).astype(np.float32)[..., None]
        elif self.use_matching_geo_prior:
            img_depth = _read_guidance_depth(
                path_info['guidance_depth_path']).astype(np.float32)
        else:
            img_depth = None

        target_lq_size = (img_gt.shape[0] // scale, img_gt.shape[1] // scale)
        img_lq = self._resize_to(img_lq, target_lq_size)
        img_ref = self._resize_to(img_ref, img_gt.shape[:2])
        if img_albedo is not None:
            img_albedo = self._resize_to(img_albedo, img_gt.shape[:2])
        if img_normal is not None:
            img_normal = self._resize_to(img_normal, img_gt.shape[:2])
        if img_depth is not None:
            img_depth = self._resize_depth_to(img_depth, img_gt.shape[:2])
        full_ref_size = img_ref.shape[:2]
        full_projective_depth = (img_depth[..., 0].copy()
                                 if self.use_projective_matching else None)
        full_ref_projective_depth = None
        if (self.use_projective_matching
                and self.geometry_use_ref_depth_consistency):
            full_ref_projective_depth = read_metric_depth(
                path_info['ref_depth_path'])
            if full_ref_projective_depth.shape != full_ref_size:
                full_ref_projective_depth = cv2.resize(
                    full_ref_projective_depth,
                    (full_ref_size[1], full_ref_size[0]),
                    interpolation=cv2.INTER_NEAREST)

        geo_ref_coords = None
        geo_valid_mask = None

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            geometry_info = None
            if self.use_geometry_ref_crop:
                (img_gt, img_lq, img_ref, img_albedo, img_normal, img_depth,
                 geometry_info, geo_ref_coords, geo_valid_mask) = \
                    self._geometry_random_crop(
                        img_gt, img_lq, img_ref, img_albedo, img_normal,
                        img_depth,
                        path_info, gt_size, scale)
            else:
                gt_list = [img_gt, img_ref]
                if img_albedo is not None:
                    gt_list.append(img_albedo)
                if img_normal is not None:
                    gt_list.append(img_normal)
                if img_depth is not None:
                    gt_list.append(img_depth)
                img_gts, img_lq = paired_random_crop(
                    gt_list, img_lq, gt_size, scale,
                    path_info['gt_path'])
                img_gt, img_ref = img_gts[:2]
                next_index = 2
                if img_albedo is not None:
                    img_albedo = img_gts[next_index]
                    next_index += 1
                if img_normal is not None:
                    img_normal = img_gts[next_index]
                    next_index += 1
                if img_depth is not None:
                    img_depth = img_gts[next_index]

            if self.opt.get('use_flip', False) or self.opt.get('use_rot', False):
                aug_list = [img_gt, img_ref, img_lq]
                if img_albedo is not None:
                    aug_list.append(img_albedo)
                if img_depth is not None:
                    aug_list.append(img_depth)
                aug_list = augment(
                    aug_list,
                    self.opt.get('use_flip', False),
                    self.opt.get('use_rot', False))
                img_gt, img_ref, img_lq = aug_list[:3]
                next_index = 3
                if img_albedo is not None:
                    img_albedo = aug_list[next_index]
                    next_index += 1
                if img_depth is not None:
                    img_depth = aug_list[next_index]
            if self.lq_noise_aug and self.lq_noise_aug_sigma_max > 0:
                sigma = np.random.uniform(0, self.lq_noise_aug_sigma_max)
                noise = np.random.normal(
                    0, sigma, img_lq.shape).astype(np.float32)
                img_lq = np.clip(img_lq + noise, 0, 1)
            original_size = (gt_size, gt_size)
            padding = False
        else:
            h_lq, w_lq = img_lq.shape[:2]
            window_size = self.opt.get('window_size', 8)
            h_lq = h_lq - h_lq % window_size
            w_lq = w_lq - w_lq % window_size
            h_gt = h_lq * scale
            w_gt = w_lq * scale
            img_lq = img_lq[:h_lq, :w_lq, :]
            img_gt = img_gt[:h_gt, :w_gt, :]
            img_ref = img_ref[:h_gt, :w_gt, :]
            if img_albedo is not None:
                img_albedo = img_albedo[:h_gt, :w_gt, :]
            if img_normal is not None:
                img_normal = img_normal[:h_gt, :w_gt, :]
            if img_depth is not None:
                img_depth = img_depth[:h_gt, :w_gt, :]
            if self.use_projective_matching:
                camera_pair = self.camera_manifest[path_info['name']]
                geo_ref_coords, geo_valid_mask = project_target_depth_map(
                    full_projective_depth,
                    camera_pair['target'],
                    camera_pair['ref'],
                    (0, 0, h_gt, w_gt),
                    full_ref_size,
                    ref_box=(0, 0, h_gt, w_gt),
                    output_size=(h_gt, w_gt),
                    ref_output_size=(h_gt, w_gt),
                    depth_type=self.geometry_depth_type,
                    ref_depth=full_ref_projective_depth,
                    depth_consistency_rel_tol=(
                        self.geometry_depth_consistency_rel_tol),
                    depth_consistency_abs_tol=(
                        self.geometry_depth_consistency_abs_tol))
            original_size = (h_gt, w_gt)
            padding = False

        img_in_up = self._bicubic_up(img_lq, img_gt.shape[:2])

        tensor_list = [img_gt, img_lq, img_in_up, img_ref]
        if img_albedo is not None:
            tensor_list.append(img_albedo)
        if img_normal is not None:
            tensor_list.append(img_normal)
        if img_depth is not None:
            tensor_list.append(img_depth)
        tensor_list = totensor(
            tensor_list,
            bgr2rgb=True,
            float32=True)
        img_gt, img_lq, img_in_up, img_ref = tensor_list[:4]
        next_index = 4
        if img_albedo is not None:
            img_albedo = tensor_list[next_index]
            next_index += 1
        if img_normal is not None:
            img_normal = tensor_list[next_index]
            next_index += 1
        if img_depth is not None:
            img_depth = tensor_list[next_index]

        return_dict = {
            'img_in_lq': img_lq,
            'img_in_up': img_in_up,
            'img_ref': img_ref,
            'img_in': img_gt,
        }
        if img_albedo is not None:
            return_dict['img_albedo'] = img_albedo
        if img_normal is not None:
            return_dict['img_normal'] = img_normal
        if img_depth is not None:
            return_dict['img_depth'] = img_depth
        if geo_ref_coords is not None:
            return_dict['geo_ref_coords'] = torch.from_numpy(
                np.ascontiguousarray(geo_ref_coords.transpose(2, 0, 1))).float()
            return_dict['geo_valid_mask'] = torch.from_numpy(
                np.ascontiguousarray(geo_valid_mask.transpose(2, 0, 1))).float()
        if self.opt['phase'] == 'train' and geometry_info is not None:
            return_dict.update(geometry_info)

        if self.opt['phase'] != 'train':
            return_dict['lq_path'] = path_info['in_path']
            return_dict['padding'] = padding
            return_dict['original_size'] = original_size

        return return_dict

    def __len__(self):
        if self.opt['phase'] == 'train':
            return len(self.paths) * self.sample_enlarge_ratio
        return len(self.paths)
