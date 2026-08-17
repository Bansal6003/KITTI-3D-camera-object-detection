"""
Points -> pillars, and box labels -> CenterPoint-style dense targets
(heatmap + offset + z + dim + rotation), for the single "Car" class.

Whole-dataset precompute: only 443 frames and the scene geometry never
changes across epochs, so we pillarize + build targets once in __init__
and cache in RAM instead of redoing it every __getitem__ call.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

import kitti_utils as ku
import pp_config as cfg


def points_to_pillars(points):
    """points: Nx4 (x,y,z,r) in velodyne frame -> (pillars, coords, n_pillars)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    in_range = (
        (x >= cfg.X_MIN) & (x < cfg.X_MAX) &
        (y >= cfg.Y_MIN) & (y < cfg.Y_MAX) &
        (z >= cfg.Z_MIN) & (z < cfg.Z_MAX)
    )
    pts = points[in_range]

    pillars = np.zeros((cfg.MAX_PILLARS, cfg.MAX_POINTS_PER_PILLAR, cfg.PILLAR_FEAT_DIM), dtype=np.float32)
    coords = np.zeros((cfg.MAX_PILLARS, 2), dtype=np.int64)
    num_points = np.zeros((cfg.MAX_PILLARS,), dtype=np.int64)
    if pts.shape[0] == 0:
        return pillars, coords, num_points, 0

    ix = np.clip(((pts[:, 0] - cfg.X_MIN) / cfg.VOXEL_X).astype(np.int64), 0, cfg.GRID_W - 1)
    iy = np.clip(((pts[:, 1] - cfg.Y_MIN) / cfg.VOXEL_Y).astype(np.int64), 0, cfg.GRID_H - 1)
    pillar_id = iy * cfg.GRID_W + ix

    order = np.argsort(pillar_id, kind="stable")
    pts_sorted, ix_sorted, iy_sorted = pts[order], ix[order], iy[order]
    uniq_ids, start_idx, counts = np.unique(pillar_id[order], return_index=True, return_counts=True)

    n_pillars = min(len(uniq_ids), cfg.MAX_PILLARS)
    for k in range(n_pillars):
        s, c = start_idx[k], min(counts[k], cfg.MAX_POINTS_PER_PILLAR)
        chunk = pts_sorted[s:s + c]  # (c, 4): x,y,z,r
        cx, cy = ix_sorted[s], iy_sorted[s]
        coords[k] = (cy, cx)
        num_points[k] = c

        mean_xyz = chunk[:, :3].mean(axis=0)
        pillar_center_x = cfg.X_MIN + (cx + 0.5) * cfg.VOXEL_X
        pillar_center_y = cfg.Y_MIN + (cy + 0.5) * cfg.VOXEL_Y

        feat = np.zeros((c, cfg.PILLAR_FEAT_DIM), dtype=np.float32)
        feat[:, 0:4] = chunk
        feat[:, 4:7] = chunk[:, :3] - mean_xyz
        feat[:, 7] = chunk[:, 0] - pillar_center_x
        feat[:, 8] = chunk[:, 1] - pillar_center_y
        pillars[k, :c] = feat

    return pillars, coords, num_points, n_pillars


# --- CenterNet-style gaussian heatmap target (standard formulas) -----------

def _gaussian_radius(height, width, min_overlap=0.7):
    a1, b1 = 1, (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + np.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2

    a2, b2 = 4, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + np.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2

    a3, b3 = 4 * min_overlap, -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + np.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return max(0, int(min(r1, r2, r3)))


def _draw_gaussian(heatmap, center_xy, radius):
    diameter = 2 * radius + 1
    m = (diameter - 1.0) / 2.0
    yy, xx = np.ogrid[-m:m + 1, -m:m + 1]
    sigma = diameter / 6.0
    gaussian = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma + 1e-9))

    x, y = center_xy
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    masked_hm = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_hm, masked_g, out=masked_hm)


def build_targets(boxes):
    """boxes: list of (x,y,z,l,w,h,yaw) in velodyne frame -> dict of target tensors."""
    heatmap = np.zeros((1, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)
    offset = np.zeros((2, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)
    z_t = np.zeros((1, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)
    dim_t = np.zeros((3, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)
    rot_t = np.zeros((2, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)
    mask = np.zeros((1, cfg.FEAT_H, cfg.FEAT_W), dtype=np.float32)

    for x, y, z, l, w, h, yaw in boxes:
        fx = (x - cfg.X_MIN) / cfg.VOXEL_X / cfg.BACKBONE_DOWNSAMPLE
        fy = (y - cfg.Y_MIN) / cfg.VOXEL_Y / cfg.BACKBONE_DOWNSAMPLE
        if not (0 <= fx < cfg.FEAT_W and 0 <= fy < cfg.FEAT_H):
            continue
        ifx, ify = int(fx), int(fy)

        l_cells = l / cfg.VOXEL_X / cfg.BACKBONE_DOWNSAMPLE
        w_cells = w / cfg.VOXEL_Y / cfg.BACKBONE_DOWNSAMPLE
        radius = max(1, _gaussian_radius(w_cells, l_cells))
        _draw_gaussian(heatmap[0], (ifx, ify), radius)

        mask[0, ify, ifx] = 1.0
        offset[:, ify, ifx] = [fx - ifx, fy - ify]
        z_t[0, ify, ifx] = z
        dim_t[:, ify, ifx] = [
            np.log(l / cfg.CAR_ANCHOR_L), np.log(w / cfg.CAR_ANCHOR_W), np.log(h / cfg.CAR_ANCHOR_H)
        ]
        rot_t[:, ify, ifx] = [np.sin(yaw), np.cos(yaw)]

    return {
        "heatmap": heatmap, "offset": offset, "z": z_t,
        "dim": dim_t, "rot": rot_t, "mask": mask,
    }


def load_boxes(frame_idx):
    path = fr"{cfg.LABELS_DIR}\{frame_idx:06d}.txt"
    boxes = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                vals = [float(v) for v in line.split()]
                if vals:
                    boxes.append(tuple(vals))
    return boxes


class KittiPillarDataset(Dataset):
    def __init__(self, frame_indices, verbose=True):
        self.samples = []
        for i, frame_idx in enumerate(frame_indices):
            velo_path = fr"{cfg.VELO_DIR}\{frame_idx:010d}.bin"
            points = ku.load_velodyne_points(velo_path)
            pillars, coords, num_points, n_pillars = points_to_pillars(points)
            boxes = load_boxes(frame_idx)
            targets = build_targets(boxes)
            self.samples.append({
                "frame_idx": frame_idx,
                "pillars": torch.from_numpy(pillars),
                "coords": torch.from_numpy(coords),
                "num_points": torch.from_numpy(num_points),
                "n_pillars": n_pillars,
                "n_boxes": len(boxes),
                **{k: torch.from_numpy(v) for k, v in targets.items()},
            })
            if verbose and (i + 1) % 100 == 0:
                print(f"  precomputed {i + 1}/{len(frame_indices)} frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
