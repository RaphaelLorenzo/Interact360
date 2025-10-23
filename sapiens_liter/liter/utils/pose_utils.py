# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import cv2
import torch
import torch.nn.functional as F


def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # print("N : ", N)
    # print("K : ", K)
    # print("H : ", H)
    # print("W : ", W)
    # print("heatmaps.shape : ", heatmaps.shape) # (Nkpts, 256, 192) = (Nkpts, 1024/4, 768/4) = (Nkpts, orig_h/4, orig_w/4)

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge')
    
    # print("heatmaps_pad.shape : ", heatmaps_pad.shape) # (Nkpts, 258, 194)
    
    heatmaps_pad = heatmaps_pad.flatten()

    
    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints

# import time

def udp_decode(heatmaps, input_size, heatmap_size, blur_kernel_size=11) -> np.ndarray:
    """UDP decoding for keypoint location refinement.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        blur_kernel_size (int): Gaussian kernel size (K) for modulation, which
            should match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Refined keypoint locations.
    """
    # tic = time.time()
    keypoints, scores = get_heatmap_maximum(heatmaps)
    # tac = time.time()
    # print(f"get_heatmap_maximum time: {(tac - tic)*1000:.0f} ms")
    # unsqueeze the instance dimension for single-instance results
    keypoints = keypoints[None]
    scores = scores[None]
    # tic = time.time()
    keypoints = refine_keypoints_dark_udp(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size)
    # tac = time.time()
    # print(f"refine_keypoints_dark_udp time: {(tac - tic)*1000:.0f} ms")
    
    W, H = heatmap_size
    keypoints = (keypoints / [W - 1, H - 1]) * input_size
    return keypoints, scores


def get_udp_warp_matrix(
    center: np.ndarray,
    scale: np.ndarray,
    rot: float,
    output_size,
) -> np.ndarray:
    """Calculate the affine transformation matrix under the unbiased
    constraint. See `UDP (CVPR 2020)`_ for details.

    Note:

        - The bbox number: N

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (tuple): Size ([w, h]) of the output image

    Returns:
        np.ndarray: A 2x3 transformation matrix

    .. _`UDP (CVPR 2020)`: https://arxiv.org/abs/1911.07524
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    input_size = center * 2
    rot_rad = np.deg2rad(rot)
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    scale_x = (output_size[0] - 1) / scale[0]
    scale_y = (output_size[1] - 1) / scale[1]
    warp_mat[0, 0] = math.cos(rot_rad) * scale_x
    warp_mat[0, 1] = -math.sin(rot_rad) * scale_x
    warp_mat[0, 2] = scale_x * (-0.5 * input_size[0] * math.cos(rot_rad) +
                                0.5 * input_size[1] * math.sin(rot_rad) +
                                0.5 * scale[0])
    warp_mat[1, 0] = math.sin(rot_rad) * scale_y
    warp_mat[1, 1] = math.cos(rot_rad) * scale_y
    warp_mat[1, 2] = scale_y * (-0.5 * input_size[0] * math.sin(rot_rad) -
                                0.5 * input_size[1] * math.cos(rot_rad) +
                                0.5 * scale[1])
    return warp_mat


def top_down_affine_transform(img, bbox, padding=1.25):
    # TODO simplify and comment this function
    """
    Args:
        img (np.ndarray): Image to be transformed.
        bbox (np.ndarray): Bounding box to be transformed.
        padding (int): Padding size.

    Returns:
        np.ndarray: Transformed image.
        np.ndarray: Transformed bounding box.
    """
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    h, w = img.shape[:2]
    warp_size = (int(w), int(h))
    aspect_ratio = 768 / 1024

    # reshape bbox to fixed aspect ratio
    box_w, box_h = np.hsplit(scale, [1])
    scale = np.where(box_w > box_h * aspect_ratio,
                            np.hstack([box_w, box_w / aspect_ratio]),
                            np.hstack([box_h * aspect_ratio, box_h]))

    rot = 0.

    warp_mat = get_udp_warp_matrix(
            center, scale, rot, output_size=(w, h))

    img = cv2.warpAffine(
            img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, [center], [scale]


def nms(dets: np.ndarray, thr: float):
    """Greedily select boxes with high confidence and overlap <= thr.

    Args:
        dets (np.ndarray): [[x1, y1, x2, y2, score]].
        thr (float): Retain overlap < thr.

    Returns:
        list: Indexes to keep.
    """
    if len(dets) == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thr)[0]
        order = order[inds + 1]

    return keep


# ===============================
# PyTorch versions for faster GPU processing
# ===============================

def gaussian_blur_torch(heatmaps: torch.Tensor, kernel: int = 11) -> torch.Tensor:
    """Modulate heatmap distribution with Gaussian using PyTorch.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (torch.Tensor[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        torch.Tensor ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1
    
    # Create Gaussian kernel
    sigma = kernel / 6.0  # Common rule of thumb: kernel_size = 6*sigma + 1
    x = torch.arange(kernel, dtype=heatmaps.dtype, device=heatmaps.device) - kernel // 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
    # Create 2D Gaussian kernel
    gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
    gaussian_2d = gaussian_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    K, H, W = heatmaps.shape
    original_max = heatmaps.view(K, -1).max(dim=1)[0]  # [K]
    
    # Apply Gaussian blur to each heatmap
    heatmaps_blurred = F.conv2d(
        heatmaps.unsqueeze(1),  # [NKpts, 1, H, W], NKpts as batch dimension, 1 as channel dimension
        gaussian_2d, # [1, 1, K, K] = [out_channels, groups, kernel_height, kernel_width]
        padding=kernel // 2,
    ).squeeze(1) # [NKpts, H, W]
    
    # # print("heatmaps_blurred.shape : ", heatmaps_blurred.shape)
    
    # Normalize to preserve original maximum
    current_max = heatmaps_blurred.view(K, -1).max(dim=1)[0]  # [K]
    scale_factor = original_max / (current_max + 1e-8)
    heatmaps_blurred = heatmaps_blurred * scale_factor.view(K, 1, 1)
    
    # # print("heatmaps_blurred.shape : ", heatmaps_blurred.shape)
    
    return heatmaps_blurred


def get_heatmap_maximum_torch(heatmaps: torch.Tensor):
    """Get maximum response location and value from heatmaps using PyTorch.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (torch.Tensor): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (torch.Tensor): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (torch.Tensor): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps, torch.Tensor), 'heatmaps should be torch.Tensor'
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f'Invalid shape {heatmaps.shape}'

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.view(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.view(B * K, -1)

    vals, indices = torch.max(heatmaps_flatten, dim=1)
    y_locs = indices // W
    x_locs = indices % W
    locs = torch.stack((x_locs, y_locs), dim=-1).float()
    
    # Set locations with zero or negative values to -1
    mask = vals <= 0.
    locs[mask] = -1

    if B:
        locs = locs.view(B, K, 2)
        vals = vals.view(B, K)

    return locs, vals


def refine_keypoints_dark_udp_torch(keypoints: torch.Tensor, heatmaps: torch.Tensor,
                                    blur_kernel_size: int) -> torch.Tensor:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP using PyTorch. See `UDP`_ for details. The operation is in-place.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (torch.Tensor): The keypoint coordinates in shape (N, K, D)
        heatmaps (torch.Tensor): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        torch.Tensor: Refined keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    K_hm, H, W = heatmaps.shape
    
    assert K == K_hm, f"Keypoint number mismatch: {K} vs {K_hm}"

    # Modulate heatmaps
    heatmaps = gaussian_blur_torch(heatmaps, blur_kernel_size)
    heatmaps = torch.clamp(heatmaps, min=1e-3, max=50.0)
    heatmaps = torch.log(heatmaps)

    # Pad heatmaps
    heatmaps_pad = F.pad(heatmaps, (1, 1, 1, 1), mode='replicate')  # [K, H+2, W+2]
    heatmaps_pad = heatmaps_pad.flatten()  # Flatten to 1D for indexing

    device = keypoints.device
    
    for n in range(N):
        # Calculate indices for each keypoint
        x_coords = keypoints[n, :, 0] + 1  # [K]
        y_coords = keypoints[n, :, 1] + 1  # [K]
        
        # Base index in flattened padded heatmap
        base_index = x_coords + y_coords * (W + 2)  # [K]
        
        # Add offset for each keypoint channel
        k_offset = torch.arange(K, device=device) * (W + 2) * (H + 2)
        index = (base_index + k_offset).long()  # [K]
        
        # Extract values at different positions
        i_ = heatmaps_pad[index]  # [K]
        ix1 = heatmaps_pad[index + 1]  # [K]
        iy1 = heatmaps_pad[index + W + 2]  # [K]
        ix1y1 = heatmaps_pad[index + W + 3]  # [K]
        ix1_y1_ = heatmaps_pad[index - W - 3]  # [K]
        ix1_ = heatmaps_pad[index - 1]  # [K]
        iy1_ = heatmaps_pad[index - 2 - W]  # [K]

        # Calculate first derivatives
        dx = 0.5 * (ix1 - ix1_)  # [K]
        dy = 0.5 * (iy1 - iy1_)  # [K]
        derivative = torch.stack([dx, dy], dim=1)  # [K, 2]

        # Calculate second derivatives (Hessian)
        dxx = ix1 - 2 * i_ + ix1_  # [K]
        dyy = iy1 - 2 * i_ + iy1_  # [K]
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)  # [K]
        
        # Build Hessian matrix for each keypoint
        hessian = torch.zeros(K, 2, 2, device=device, dtype=heatmaps.dtype)
        hessian[:, 0, 0] = dxx
        hessian[:, 0, 1] = dxy
        hessian[:, 1, 0] = dxy
        hessian[:, 1, 1] = dyy
        
        # Add small epsilon to diagonal for numerical stability
        eps_eye = torch.finfo(heatmaps.dtype).eps * torch.eye(2, device=device)
        hessian = hessian + eps_eye.unsqueeze(0)
        
        # Invert Hessian and apply correction
        try:
            hessian_inv = torch.inverse(hessian)  # [K, 2, 2]
            correction = torch.bmm(hessian_inv, derivative.unsqueeze(-1)).squeeze(-1)  # [K, 2]
            keypoints[n] -= correction
        except RuntimeError:
            # Handle singular matrices by skipping correction for those keypoints
            for k in range(K):
                try:
                    hess_k_inv = torch.inverse(hessian[k])
                    corr_k = torch.mv(hess_k_inv, derivative[k])
                    keypoints[n, k] -= corr_k
                except RuntimeError:
                    # Skip correction for this keypoint if Hessian is singular
                    pass

    return keypoints


def udp_decode_torch(heatmaps: torch.Tensor, input_size, heatmap_size, 
                     blur_kernel_size: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    """UDP decoding for keypoint location refinement using PyTorch.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (torch.Tensor[K, H, W]): model predicted heatmaps.
        input_size (tuple): Size of the input image (W, H).
        heatmap_size (tuple): Size of the heatmap (W, H).
        blur_kernel_size (int): Gaussian kernel size (K) for modulation, which
            should match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        tuple:
        - keypoints (torch.Tensor): Refined keypoint locations in input image coordinates
        - scores (torch.Tensor): Confidence scores for each keypoint
    """
    # Get initial keypoint locations
    keypoints, scores = get_heatmap_maximum_torch(heatmaps)
    
    # Add instance dimension for single-instance results
    keypoints = keypoints.unsqueeze(0)  # [1, K, 2]
    scores = scores.unsqueeze(0)  # [1, K]
    
    # Refine keypoints using UDP
    keypoints = refine_keypoints_dark_udp_torch(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size)
    
    # Convert from heatmap coordinates to input image coordinates
    W, H = heatmap_size
    scale_x = input_size[0] / (W - 1)
    scale_y = input_size[1] / (H - 1)
    
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y
    
    return keypoints, scores
