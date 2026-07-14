import json
import math

import cv2
import numpy as np


def read_metric_depth(path):
    """Read a metric single-channel depth map without image normalization."""
    lower = path.lower()
    if lower.endswith('.npy'):
        depth = np.load(path)
    elif lower.endswith('.npz'):
        archive = np.load(path)
        key = 'depth' if 'depth' in archive else archive.files[0]
        depth = archive[key]
    elif lower.endswith('.pfm'):
        with open(path, 'rb') as file:
            def read_data_line():
                line = file.readline()
                while line and (not line.strip() or line.lstrip().startswith(b'#')):
                    line = file.readline()
                if not line:
                    raise ValueError(f'Unexpected end of PFM header in {path}.')
                return line.decode('ascii').strip()

            header = read_data_line()
            if header not in ('Pf', 'PF'):
                raise ValueError(f'Invalid PFM header in {path}: {header}')
            width, height = map(int, read_data_line().split())
            scale = float(read_data_line())
            if width <= 0 or height <= 0 or scale == 0:
                raise ValueError(f'Invalid PFM dimensions or scale in {path}.')
            endian = '<' if scale < 0 else '>'
            channels = 1 if header == 'Pf' else 3
            count = width * height * channels
            values = np.fromfile(file, dtype=endian + 'f4', count=count)
            if values.size != count:
                raise ValueError(f'Unexpected PFM data size in {path}.')
            depth = values.astype(np.float32).reshape(height, width, channels)
            # The sign encodes endianness; the magnitude is a value scale.
            depth *= abs(scale)
            depth = np.flipud(depth)
            if channels == 1:
                depth = depth[..., 0]
    else:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(path)
        if depth.dtype in (np.uint8, np.uint16):
            raise ValueError(
                f'{path} is integer depth. Geometry crop requires metric '
                'float depth in PFM, NPY, NPZ or float EXR format.')
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def load_camera_manifest(path):
    with open(path, 'r', encoding='utf-8-sig') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError('Camera manifest must be a JSON object keyed by image name.')
    return data


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError('Camera direction has zero length.')
    return vector / norm


def camera_basis(camera):
    position = np.asarray(camera['position'], dtype=np.float32)
    forward = _normalize(np.asarray(camera['look_at']) - position)
    up_hint = _normalize(camera.get('up', [0, 1, 0]))
    cross = np.cross(forward, up_hint)
    if np.linalg.norm(cross) < 1e-8:
        raise ValueError('Camera forward direction is parallel to its up vector.')
    right = _normalize(cross)
    up = _normalize(np.cross(right, forward))
    return position, right, up, forward


def _fov_tangents(camera, width, height):
    tan_half = math.tan(math.radians(float(camera['fov'])) * 0.5)
    axis = camera.get('fov_axis', 'horizontal')
    aspect = width / float(height)
    if axis == 'vertical':
        return tan_half * aspect, tan_half
    if axis != 'horizontal':
        raise ValueError(f'Unsupported fov_axis: {axis}')
    return tan_half, tan_half / aspect


def project_target_patch(depth, target_camera, ref_camera, target_box,
                         ref_size, sample_stride=4, depth_type='ray_distance',
                         trim_percentile=1.0):
    """Project sampled target-patch pixels into the reference view.

    ``overlap`` is the fraction of all sampled target pixels that have valid
    depth and project inside the Ref field of view. It is a coarse geometric
    availability score, not an occlusion-aware visibility score.
    """
    top, left, patch_h, patch_w = target_box
    target_h, target_w = depth.shape
    ref_h, ref_w = ref_size
    if sample_stride < 1:
        raise ValueError('sample_stride must be at least 1.')
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError('Target patch dimensions must be positive.')
    if (top < 0 or left < 0 or top + patch_h > target_h
            or left + patch_w > target_w):
        raise ValueError('Target patch lies outside the depth map.')
    if ref_h <= 0 or ref_w <= 0:
        raise ValueError('Reference image dimensions must be positive.')
    if not 0 <= trim_percentile < 50:
        raise ValueError('trim_percentile must be in [0, 50).')
    ys = np.arange(top, top + patch_h, sample_stride, dtype=np.int32)
    xs = np.arange(left, left + patch_w, sample_stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    z = depth[yy, xx]
    valid_depth = np.isfinite(z) & (z > 0)
    if not np.any(valid_depth):
        return None

    yy = yy[valid_depth].astype(np.float32)
    xx = xx[valid_depth].astype(np.float32)
    z = z[valid_depth].astype(np.float32)

    target_pos, target_right, target_up, target_forward = camera_basis(
        target_camera)
    tan_x, tan_y = _fov_tangents(target_camera, target_w, target_h)
    nx = ((xx + 0.5) / target_w) * 2.0 - 1.0
    ny = 1.0 - ((yy + 0.5) / target_h) * 2.0
    rays = (target_forward[None] + nx[:, None] * tan_x * target_right[None]
            + ny[:, None] * tan_y * target_up[None])
    if depth_type == 'ray_distance':
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        world = target_pos[None] + rays * z[:, None]
    elif depth_type == 'camera_z':
        world = target_pos[None] + rays * z[:, None]
    else:
        raise ValueError(f'Unsupported geometry_depth_type: {depth_type}')

    ref_pos, ref_right, ref_up, ref_forward = camera_basis(ref_camera)
    relative = world - ref_pos[None]
    rz = relative @ ref_forward
    rx = relative @ ref_right
    ry = relative @ ref_up
    tan_ref_x, tan_ref_y = _fov_tangents(ref_camera, ref_w, ref_h)
    in_front = rz > 1e-6
    u = np.full_like(rz, np.nan)
    v = np.full_like(rz, np.nan)
    u[in_front] = ((rx[in_front] / (rz[in_front] * tan_ref_x) + 1) *
                   0.5 * ref_w - 0.5)
    v[in_front] = ((1 - ry[in_front] / (rz[in_front] * tan_ref_y)) *
                   0.5 * ref_h - 0.5)
    # A tiny tolerance keeps identity-camera border pixels from being rejected
    # solely by floating-point round-off (e.g. u == width - 1 + 1e-6).
    border_eps = 1e-4
    inside = (in_front & (u >= -border_eps) &
              (u <= ref_w - 1 + border_eps) &
              (v >= -border_eps) & (v <= ref_h - 1 + border_eps))
    total_count = int(valid_depth.size)
    valid_count = int(np.count_nonzero(valid_depth))
    inside_count = int(np.count_nonzero(inside))
    overlap = float(inside_count) / float(total_count)
    fov_overlap = float(inside_count) / float(valid_count)
    summary = {
        'overlap': overlap,
        'fov_overlap': fov_overlap,
        'valid_depth_ratio': float(valid_depth.mean()),
        'sample_count': total_count,
        'valid_depth_count': valid_count,
        'inside_count': inside_count,
    }
    if inside_count < 4:
        return summary

    projected = np.stack([
        np.clip(u[inside], 0, ref_w - 1),
        np.clip(v[inside], 0, ref_h - 1),
    ], axis=1)
    low = np.percentile(projected, trim_percentile, axis=0)
    high = np.percentile(projected, 100 - trim_percentile, axis=0)
    summary.update({
        'center': (low + high) * 0.5,
        'extent': np.maximum(high - low, 1.0),
        'projected_low': low,
        'projected_high': high,
    })
    return summary


def project_target_depth_map(depth, target_camera, ref_camera, target_box,
                             ref_size, ref_box=None, output_size=None,
                             ref_output_size=None,
                             depth_type='ray_distance', ref_depth=None,
                             depth_consistency_rel_tol=0.05,
                             depth_consistency_abs_tol=0.10):
    """Project every current target pixel to normalized Ref coordinates.

    The camera calculation is performed in the uncropped target/Ref image
    coordinate systems. ``target_box`` and ``ref_box`` then map those
    projections into the tensors that are actually sent to the matcher. This
    makes the same function usable for geometry-cropped training patches and
    uncropped validation/test images.

    Returns:
        coords (ndarray): float32 array with shape (H, W, 2). The last
            dimension stores normalized Ref x/y coordinates in [0, 1].
        valid (ndarray): float32 visibility/FOV mask with shape (H, W, 1).
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError('Metric depth must be a two-dimensional array.')
    target_h, target_w = depth.shape
    ref_h, ref_w = ref_size
    top, left, box_h, box_w = target_box
    if output_size is None:
        output_h, output_w = box_h, box_w
    else:
        output_h, output_w = output_size
    if ref_box is None:
        ref_top, ref_left, ref_box_h, ref_box_w = 0, 0, ref_h, ref_w
    else:
        ref_top, ref_left, ref_box_h, ref_box_w = ref_box
    if ref_output_size is None:
        ref_output_h, ref_output_w = ref_box_h, ref_box_w
    else:
        ref_output_h, ref_output_w = ref_output_size
    if min(output_h, output_w, ref_output_h, ref_output_w) <= 0:
        raise ValueError('Projection input/output dimensions must be positive.')

    # Map current tensor pixel centres back to the full target image.
    ys = (top + (np.arange(output_h, dtype=np.float32) + 0.5)
          * box_h / float(output_h) - 0.5)
    xs = (left + (np.arange(output_w, dtype=np.float32) + 0.5)
          * box_w / float(output_w) - 0.5)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    depth_y = np.clip(np.rint(yy).astype(np.int32), 0, target_h - 1)
    depth_x = np.clip(np.rint(xx).astype(np.int32), 0, target_w - 1)
    z = depth[depth_y, depth_x]
    valid = np.isfinite(z) & (z > 0)

    target_pos, target_right, target_up, target_forward = camera_basis(
        target_camera)
    tan_x, tan_y = _fov_tangents(target_camera, target_w, target_h)
    nx = ((xx + 0.5) / target_w) * 2.0 - 1.0
    ny = 1.0 - ((yy + 0.5) / target_h) * 2.0
    rays = (target_forward[None, None]
            + nx[..., None] * tan_x * target_right[None, None]
            + ny[..., None] * tan_y * target_up[None, None])
    if depth_type == 'ray_distance':
        rays /= np.maximum(
            np.linalg.norm(rays, axis=2, keepdims=True), 1e-8)
        world = target_pos[None, None] + rays * z[..., None]
    elif depth_type == 'camera_z':
        world = target_pos[None, None] + rays * z[..., None]
    else:
        raise ValueError(f'Unsupported geometry_depth_type: {depth_type}')

    ref_pos, ref_right, ref_up, ref_forward = camera_basis(ref_camera)
    relative = world - ref_pos[None, None]
    rz = relative @ ref_forward
    rx = relative @ ref_right
    ry = relative @ ref_up
    tan_ref_x, tan_ref_y = _fov_tangents(ref_camera, ref_w, ref_h)
    in_front = rz > 1e-6
    safe_rz = np.where(in_front, rz, 1.0)
    u = ((rx / (safe_rz * tan_ref_x) + 1.0) * 0.5 * ref_w - 0.5)
    v = ((1.0 - ry / (safe_rz * tan_ref_y)) * 0.5 * ref_h - 0.5)

    border_eps = 1e-4
    valid &= in_front
    valid &= (u >= ref_left - border_eps)
    valid &= (u <= ref_left + ref_box_w - 1 + border_eps)
    valid &= (v >= ref_top - border_eps)
    valid &= (v <= ref_top + ref_box_h - 1 + border_eps)

    if ref_depth is not None:
        ref_depth = np.asarray(ref_depth, dtype=np.float32)
        if ref_depth.ndim == 3:
            ref_depth = ref_depth[..., 0]
        if ref_depth.shape != (ref_h, ref_w):
            ref_depth = cv2.resize(
                ref_depth, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
        sample_x = np.clip(
            np.rint(np.nan_to_num(u, nan=0.0)).astype(np.int32),
            0, ref_w - 1)
        sample_y = np.clip(
            np.rint(np.nan_to_num(v, nan=0.0)).astype(np.int32),
            0, ref_h - 1)
        ref_z = ref_depth[sample_y, sample_x]
        if depth_type == 'ray_distance':
            projected_z = np.linalg.norm(relative, axis=2)
        else:
            projected_z = rz
        valid_ref_depth = np.isfinite(ref_z) & (ref_z > 0)
        tolerance = (depth_consistency_abs_tol
                     + depth_consistency_rel_tol * np.abs(ref_z))
        valid &= valid_ref_depth
        valid &= np.abs(projected_z - ref_z) <= tolerance

    # Match OpenCV/PIL resize pixel-centre convention when a Ref ROI is
    # resized to the network input size.
    u_current = ((u - ref_left + 0.5) * ref_output_w
                 / float(ref_box_w) - 0.5)
    v_current = ((v - ref_top + 0.5) * ref_output_h
                 / float(ref_box_h) - 0.5)
    valid &= (u_current >= -border_eps)
    valid &= (u_current <= ref_output_w - 1 + border_eps)
    valid &= (v_current >= -border_eps)
    valid &= (v_current <= ref_output_h - 1 + border_eps)

    denom_x = max(ref_output_w - 1, 1)
    denom_y = max(ref_output_h - 1, 1)
    coords = np.stack([
        np.clip(u_current / denom_x, 0.0, 1.0),
        np.clip(v_current / denom_y, 0.0, 1.0),
    ], axis=2).astype(np.float32)
    coords[~valid] = 0.0
    return coords, valid[..., None].astype(np.float32)


def compute_projected_ref_box(ref_size, projection, output_size, margin=1.2,
                              min_scale=0.5, max_scale=3.0):
    """Compute a square Ref crop and its projected-region coverage.

    Coverage is measured against the robust projected bounding box. It catches
    cases where ``max_scale`` or an image boundary makes the final crop discard
    a large part of the geometrically corresponding region.
    """
    height, width = ref_size
    if output_size <= 0:
        raise ValueError('output_size must be positive.')
    if margin <= 0 or min_scale <= 0 or max_scale < min_scale:
        raise ValueError('Invalid geometry crop margin or scale bounds.')
    if 'center' not in projection or 'extent' not in projection:
        raise ValueError('Projection has no crop center/extent.')
    center_x, center_y = np.asarray(projection['center'], dtype=np.float32)
    extent = float(np.max(np.asarray(projection['extent'], dtype=np.float32)))
    if not np.isfinite([center_x, center_y, extent]).all():
        raise ValueError('Projection contains non-finite crop values.')
    extent *= margin
    crop_size = int(round(np.clip(
        extent, output_size * min_scale, output_size * max_scale)))
    crop_size = max(2, min(crop_size, height, width))
    left = int(round(center_x - crop_size * 0.5))
    top = int(round(center_y - crop_size * 0.5))
    left = int(np.clip(left, 0, width - crop_size))
    top = int(np.clip(top, 0, height - crop_size))

    low = np.asarray(projection.get('projected_low'), dtype=np.float32)
    high = np.asarray(projection.get('projected_high'), dtype=np.float32)
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError('Projection has no robust projected bounds.')
    intersection_low = np.maximum(low, [left, top])
    intersection_high = np.minimum(high, [left + crop_size, top + crop_size])
    intersection_extent = np.maximum(intersection_high - intersection_low, 0)
    projected_area = float(np.prod(np.maximum(high - low, 1e-6)))
    crop_coverage = float(np.prod(intersection_extent) / projected_area)
    crop_coverage = float(np.clip(crop_coverage, 0, 1))
    return (top, left, crop_size, crop_size), crop_coverage


def crop_projected_ref(img_ref, projection, output_size, margin=1.2,
                       min_scale=0.5, max_scale=3.0):
    """Crop a square projected region and resize it to output_size."""
    height, width = img_ref.shape[:2]
    ref_box, _ = compute_projected_ref_box(
        (height, width), projection, output_size, margin, min_scale, max_scale)
    top, left, crop_size, _ = ref_box
    crop = img_ref[top:top + crop_size, left:left + crop_size]
    crop = cv2.resize(crop, (output_size, output_size),
                      interpolation=cv2.INTER_CUBIC)
    return crop, ref_box
