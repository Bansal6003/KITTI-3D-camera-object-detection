"""Shared box-drawing helpers for BEV plots and camera-image overlays.

Split out from 03_visualize_3d_predictions.py so 04_run_3d_tracking.py (and
anything else) can import them — a module whose filename starts with a
digit can't itself be imported with a normal `import` statement.
"""
import numpy as np
import cv2

import kitti_utils as ku


def box3d_corners_from_center(x, y, z, l, w, h, yaw):
    """8x3 box corners in velodyne frame from a box-CENTER (x,y,z) — the
    convention used by our labels/predictions (unlike raw tracklets, which
    use ground-contact tz)."""
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_c = np.array([-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2])
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    corners = R @ np.vstack([x_c, y_c, z_c])
    corners[0] += x
    corners[1] += y
    corners[2] += z
    return corners.T


def draw_boxes_on_image(img, boxes, calib, color, labels=None):
    """boxes: list of (x,y,z,l,w,h,yaw). labels: optional parallel list of strings."""
    out = img.copy()
    for i, (x, y, z, l, w, h, yaw) in enumerate(boxes):
        corners = box3d_corners_from_center(x, y, z, l, w, h, yaw)
        pixels, depth = ku.velo_to_cam2_image(corners, calib)
        if np.any(depth <= 0):
            continue
        pixels = pixels.astype(int)
        for a, b in ku.BOX_EDGES:
            cv2.line(out, tuple(pixels[a]), tuple(pixels[b]), color, 2)
        if labels is not None:
            cv2.putText(out, str(labels[i]), tuple(pixels[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2, cv2.LINE_AA)
    return out


def plot_box_bev(ax, x, y, l, w, yaw, color, label=None):
    corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2], [l/2, w/2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    c = corners @ R.T
    c[:, 0] += x
    c[:, 1] += y
    ax.plot(-c[:, 1], c[:, 0], color=color, linewidth=1.5)
    if label:
        ax.text(-y, x, str(label), color=color, fontsize=8)
