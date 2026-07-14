import torch
import torch.nn.functional as F
import pdb


def sample_patches(inputs, patch_size=3, stride=1):
    """Extract sliding local patches from an input feature tensor.
    The sampled pathes are row-major.

    Args:
        inputs (Tensor): the input feature maps, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.

    Returns:
        patches (Tensor): extracted patches, shape: (c, patch_size,
            patch_size, n_patches).
    """

    c, h, w = inputs.shape
    patches = inputs.unfold(1, patch_size, stride)\
                    .unfold(2, patch_size, stride)\
                    .reshape(c, -1, patch_size, patch_size)\
                    .permute(0, 2, 3, 1)
    return patches


def feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False):
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)   # [64, 3, 3, 1444]
    #
    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape  # [64, 40, 40]
    batch_size = int(1024.**2 * 512 / (h * w))  # 335544
    n_patches = patches_ref.shape[-1]   # 1444

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)  # [64, 3, 3, 1444]
        corr = F.conv2d(feat_input.unsqueeze(0),
                        batch.permute(3, 0, 1, 2),   # [1444, 64, 3, 3]
                        stride=input_stride)         # [1, 1444, 38, 38]

        max_val_tmp, max_idx_tmp = corr.squeeze(0).max(dim=0) # [38, 38], [38, 38]

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp       # [38, 38], [38, 38]
        else:
            indices = max_val_tmp > max_val
            max_val[indices] = max_val_tmp[indices]
            max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val


def geometry_guided_feature_match_index(
        feat_input,
        feat_ref,
        projected_ref_coords,
        geometry_valid_mask,
        patch_size=3,
        input_stride=1,
        ref_stride=1,
        is_norm=True,
        norm_input=False,
        search_radius=4,
        position_weight=0.0,
        position_sigma=2.0):
    """Match Q patches only near their camera/depth-predicted Ref position.

    ``projected_ref_coords`` contains normalized x/y coordinates in the Ref
    image. For geometrically valid query locations, Ref candidates outside a
    square window are assigned a matching logit of negative infinity. Invalid
    projections deliberately fall back to global appearance matching.
    """
    if projected_ref_coords is None or geometry_valid_mask is None:
        raise ValueError(
            'Projective-window matching requires Ref coordinates and a valid mask.')
    if search_radius < 0:
        raise ValueError('search_radius must be non-negative.')
    if position_weight < 0:
        raise ValueError('position_weight must be non-negative.')
    if position_sigma <= 0:
        raise ValueError('position_sigma must be positive.')

    _, input_h, input_w = feat_input.shape
    _, ref_h, ref_w = feat_ref.shape
    query_h = int((input_h - patch_size) / input_stride + 1)
    query_w = int((input_w - patch_size) / input_stride + 1)
    ref_grid_h = int((ref_h - patch_size) / ref_stride + 1)
    ref_grid_w = int((ref_w - patch_size) / ref_stride + 1)

    coords = projected_ref_coords
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
    mask = geometry_valid_mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    if coords.shape[1] != 2 or mask.shape[1] != 1:
        raise ValueError(
            'Projected Ref coordinates/mask must have 2 and 1 channels.')
    coords = F.interpolate(coords.float(), size=(query_h, query_w),
                           mode='nearest').squeeze(0)
    mask = F.interpolate(mask.float(), size=(query_h, query_w),
                         mode='nearest').squeeze(0).squeeze(0) > 0.5

    # Convert normalized Ref image coordinates to indices of Ref patch
    # centres. Clamping ensures that every valid projection has candidates.
    patch_center = (patch_size - 1) * 0.5
    pred_x_feat = coords[0] * max(ref_w - 1, 1)
    pred_y_feat = coords[1] * max(ref_h - 1, 1)
    pred_x = ((pred_x_feat - patch_center) / ref_stride).clamp(
        0, max(ref_grid_w - 1, 0))
    pred_y = ((pred_y_feat - patch_center) / ref_stride).clamp(
        0, max(ref_grid_h - 1, 0))
    # Unfold Q/K once. Unlike global convolution, the valid path below gathers
    # only (2r+1)^2 Ref patches for each query, so geometry reduces the actual
    # QK computation rather than merely masking an already-global score map.
    query_patches = F.unfold(
        feat_input.unsqueeze(0), kernel_size=patch_size,
        stride=input_stride).squeeze(0)
    ref_patches = F.unfold(
        feat_ref.unsqueeze(0), kernel_size=patch_size,
        stride=ref_stride).squeeze(0)
    if norm_input:
        query_patches = F.normalize(query_patches, p=2, dim=0, eps=1e-5)
    if is_norm:
        ref_patches = F.normalize(ref_patches, p=2, dim=0, eps=1e-5)

    n_queries = query_patches.shape[1]
    n_ref_patches = ref_patches.shape[1]
    descriptor_dim = query_patches.shape[0]
    pred_x = pred_x.reshape(-1)
    pred_y = pred_y.reshape(-1)
    mask = mask.reshape(-1)
    max_idx = torch.empty(n_queries, dtype=torch.long, device=feat_input.device)
    max_val = torch.empty(n_queries, dtype=feat_input.dtype,
                          device=feat_input.device)

    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-search_radius, search_radius + 1,
                     device=feat_input.device),
        torch.arange(-search_radius, search_radius + 1,
                     device=feat_input.device))
    offsets_x = offsets_x.reshape(-1)
    offsets_y = offsets_y.reshape(-1)
    candidate_count = offsets_x.numel()

    valid_queries = torch.nonzero(mask, as_tuple=False).squeeze(1)
    # Bound the temporary selected-patch tensor to roughly 64 MiB for fp32.
    local_chunk = max(1, int(16 * 1024**2 /
                             max(descriptor_dim * candidate_count, 1)))
    for start in range(0, valid_queries.numel(), local_chunk):
        query_indices = valid_queries[start:start + local_chunk]
        centre_x = pred_x[query_indices].round().long()
        centre_y = pred_y[query_indices].round().long()
        candidate_x = centre_x[:, None] + offsets_x[None]
        candidate_y = centre_y[:, None] + offsets_y[None]
        in_bounds = ((candidate_x >= 0) & (candidate_x < ref_grid_w)
                     & (candidate_y >= 0) & (candidate_y < ref_grid_h))
        candidate_x_safe = candidate_x.clamp(0, ref_grid_w - 1)
        candidate_y_safe = candidate_y.clamp(0, ref_grid_h - 1)
        candidate_indices = (candidate_y_safe * ref_grid_w
                             + candidate_x_safe)
        selected_ref = ref_patches[:, candidate_indices.reshape(-1)].view(
            descriptor_dim, query_indices.numel(), candidate_count)
        appearance = (query_patches[:, query_indices, None]
                      * selected_ref).sum(dim=0)
        logits = appearance.masked_fill(
            ~in_bounds, torch.finfo(appearance.dtype).min)
        if position_weight > 0:
            delta_x = candidate_x.float() - pred_x[query_indices, None]
            delta_y = candidate_y.float() - pred_y[query_indices, None]
            logits = logits - position_weight * (
                delta_x.square() + delta_y.square()) / (
                    2.0 * position_sigma * position_sigma)
        _, local_choice = logits.max(dim=1)
        max_idx[query_indices] = candidate_indices.gather(
            1, local_choice[:, None]).squeeze(1)
        max_val[query_indices] = appearance.gather(
            1, local_choice[:, None]).squeeze(1)

    # Depth holes, points behind the Ref camera, and projections outside the
    # Ref field of view have no reliable geometric centre. Only those queries
    # fall back to the original global appearance match.
    invalid_queries = torch.nonzero(~mask, as_tuple=False).squeeze(1)
    global_chunk = max(1, int(16 * 1024**2 / max(n_ref_patches, 1)))
    for start in range(0, invalid_queries.numel(), global_chunk):
        query_indices = invalid_queries[start:start + global_chunk]
        appearance = query_patches[:, query_indices].transpose(0, 1) @ ref_patches
        values, indices = appearance.max(dim=1)
        max_idx[query_indices] = indices
        max_val[query_indices] = values

    return max_idx.view(query_h, query_w), max_val.view(query_h, query_w)


def topk_feature_match_index(feat_input,
                             feat_ref,
                             patch_size=3,
                             input_stride=1,
                             ref_stride=1,
                             is_norm=True,
                             norm_input=False,
                             K=2):
    """Patch matching between input and reference features.

    Args:
        feat_input (Tensor): the feature of input, shape: (c, h, w).
        feat_ref (Tensor): the feature of reference, shape: (c, h, w).
        patch_size (int): the spatial size of sampled patches. Default: 3.
        stride (int): the stride of sampling. Default: 1.
        is_norm (bool): determine to normalize the ref feature or not.
            Default:True.

    Returns:
        max_idx (Tensor): The indices of the most similar patches.
        max_val (Tensor): The correlation values of the most similar patches.
    """

    # patch decomposition, shape: (c, patch_size, patch_size, n_patches)
    patches_ref = sample_patches(feat_ref, patch_size, ref_stride)   # [64, 3, 3, 1444]
    #
    # normalize reference feature for each patch in both channel and
    # spatial dimensions.

    # batch-wise matching because of memory limitation
    _, h, w = feat_input.shape  # [64, 40, 40]
    batch_size = int(1024.**2 * 512 / (h * w))  # 335544
    n_patches = patches_ref.shape[-1]   # 1444

    max_idx, max_val = None, None
    for idx in range(0, n_patches, batch_size):
        batch = patches_ref[..., idx:idx + batch_size]
        if is_norm:
            batch = batch / (batch.norm(p=2, dim=(0, 1, 2)) + 1e-5)  # [64, 3, 3, 1444]
        corr = F.conv2d(feat_input.unsqueeze(0),
                        batch.permute(3, 0, 1, 2),   # [1444, 64, 3, 3]
                        stride=input_stride)         # [1, 1444, 38, 38]

        max_val_tmp, max_idx_tmp = torch.topk(corr.squeeze(0), K, 0) # [K, 38, 38], [K, 38, 38]

        if max_idx is None:
            max_idx, max_val = max_idx_tmp, max_val_tmp       # [K, 38, 38], [K, 38, 38]
        else:
            for k in range(K):
                indices = max_val_tmp[k] > max_val[k]
                max_val[indices] = max_val_tmp[indices]
                max_idx[indices] = max_idx_tmp[indices] + idx

    if norm_input:
        patches_input = sample_patches(feat_input, patch_size, input_stride)
        norm = patches_input.norm(p=2, dim=(0, 1, 2)) + 1e-5
        norm = norm.view(
            int((h - patch_size) / input_stride + 1),
            int((w - patch_size) / input_stride + 1))
        max_val = max_val / norm

    return max_idx, max_val

if __name__ == '__main__':
    H, W = 160, 160
    h, w = 40, 40
    feat_input = torch.rand(64, h, w)
    feat_ref = torch.rand(64, h, w)

    feature_match_index(feat_input,
                        feat_ref,
                        patch_size=3,
                        input_stride=1,
                        ref_stride=1,
                        is_norm=True,
                        norm_input=False)
