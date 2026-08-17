"""Turn raw head outputs (heatmap/offset/z/dim/rot) into (x,y,z,l,w,h,yaw,score) boxes.

Peak-picking via 3x3 max-pool ("is this pixel a local max?") stands in for
NMS — the standard CenterNet trick that avoids needing rotated-box NMS.
"""
import torch
import torch.nn.functional as F

import pp_config as cfg


def decode_heatmap(pred, score_thresh=0.3, batch_idx=0):
    heatmap = torch.sigmoid(pred["heatmap"][batch_idx, 0])  # (H, W)
    hmax = F.max_pool2d(heatmap[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
    keep = (hmax == heatmap) & (heatmap > score_thresh)
    ys, xs = torch.where(keep)

    boxes = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        score = heatmap[y, x].item()
        off = pred["offset"][batch_idx, :, y, x]
        fx = x + off[0].item()
        fy = y + off[1].item()
        real_x = cfg.X_MIN + fx * cfg.VOXEL_X * cfg.BACKBONE_DOWNSAMPLE
        real_y = cfg.Y_MIN + fy * cfg.VOXEL_Y * cfg.BACKBONE_DOWNSAMPLE

        z = pred["z"][batch_idx, 0, y, x].item()
        dim = pred["dim"][batch_idx, :, y, x]
        l = float(torch.exp(dim[0])) * cfg.CAR_ANCHOR_L
        w = float(torch.exp(dim[1])) * cfg.CAR_ANCHOR_W
        h = float(torch.exp(dim[2])) * cfg.CAR_ANCHOR_H

        rot = pred["rot"][batch_idx, :, y, x]
        yaw = float(torch.atan2(rot[0], rot[1]))

        boxes.append((real_x, real_y, z, l, w, h, yaw, score))
    return boxes
