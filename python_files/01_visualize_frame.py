"""
Step 1: prove calib + tracklet parsing is correct before building any model.

For one frame, this:
  1. projects the LiDAR point cloud onto the image_02 camera (colored by depth)
  2. projects each tracklet's 3D box onto the same image
  3. draws a bird's-eye-view (BEV) plot of the points + box footprints

Run:
    python 01_visualize_frame.py --frame 20
"""
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

import kitti_utils as ku

DATA_ROOT = r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice"
CALIB_DIR = fr"{DATA_ROOT}\2011_09_26_calib\2011_09_26"
SYNC_DIR = fr"{DATA_ROOT}\2011_09_26_drive_0009_sync\2011_09_26\2011_09_26_drive_0009_sync"
TRACKLET_XML = fr"{DATA_ROOT}\2011_09_26_drive_0009_tracklets\2011_09_26\2011_09_26_drive_0009_sync\tracklet_labels.xml"


def draw_points_on_image(img, points_velo, calib, max_depth=60.0):
    pixels, depth = ku.velo_to_cam2_image(points_velo[:, :3], calib)

    h, w = img.shape[:2]
    in_front = depth > 0
    in_bounds = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )
    keep = in_front & in_bounds

    pixels, depth = pixels[keep], depth[keep]
    depth_norm = np.clip(depth / max_depth, 0, 1)
    colors = (plt.cm.turbo(1 - depth_norm)[:, :3] * 255).astype(np.uint8)

    out = img.copy()
    for (x, y), c in zip(pixels.astype(int), colors):
        cv2.circle(out, (x, y), 1, tuple(int(v) for v in c[::-1]), -1)
    return out


def draw_boxes_on_image(img, tracklets, frame_idx, calib):
    out = img.copy()
    for tr, pose in ku.tracklets_for_frame(tracklets, frame_idx):
        corners_velo = ku.box3d_corners_velo(tr, pose)
        pixels, depth = ku.velo_to_cam2_image(corners_velo, calib)
        if np.any(depth <= 0):
            continue  # box behind camera, skip

        pixels = pixels.astype(int)
        for i, j in ku.BOX_EDGES:
            cv2.line(out, tuple(pixels[i]), tuple(pixels[j]), (0, 255, 0), 2)

        label_pos = tuple(pixels[0])
        cv2.putText(out, tr.obj_type, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def plot_bev(points_velo, tracklets, frame_idx, save_path, x_range=(0, 60), y_range=(-30, 30)):
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.scatter(-points_velo[:, 1], points_velo[:, 0], s=0.5, c="gray")

    for tr, pose in ku.tracklets_for_frame(tracklets, frame_idx):
        corners = ku.box3d_corners_velo(tr, pose)
        bottom = corners[:4]  # footprint
        xs = -np.append(bottom[:, 1], bottom[0, 1])
        ys = np.append(bottom[:, 0], bottom[0, 0])
        ax.plot(xs, ys, c="red", linewidth=1.5)
        ax.text(-pose["ty"], pose["tx"], tr.obj_type, color="red", fontsize=8)

    ax.set_xlim(-y_range[1], -y_range[0])
    ax.set_ylim(x_range)
    ax.set_xlabel("left (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"BEV — frame {frame_idx}")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=20)
    args = parser.parse_args()
    frame_idx = args.frame

    calib = ku.load_calib(CALIB_DIR)
    tracklets = ku.parse_tracklets(TRACKLET_XML)

    img_path = fr"{SYNC_DIR}\image_02\data\{frame_idx:010d}.png"
    velo_path = fr"{SYNC_DIR}\velodyne_points\data\{frame_idx:010d}.bin"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    points = ku.load_velodyne_points(velo_path)

    print(f"Frame {frame_idx}: image {img.shape}, {points.shape[0]} lidar points, "
          f"{sum(1 for _ in ku.tracklets_for_frame(tracklets, frame_idx))} tracklets")

    img_with_points = draw_points_on_image(img, points, calib)
    img_with_boxes = draw_boxes_on_image(img_with_points, tracklets, frame_idx, calib)

    out_img_path = f"frame_{frame_idx:04d}_projection.png"
    cv2.imwrite(out_img_path, img_with_boxes)
    print(f"Saved camera projection -> {out_img_path}")

    out_bev_path = f"frame_{frame_idx:04d}_bev.png"
    plot_bev(points, tracklets, frame_idx, out_bev_path)
    print(f"Saved BEV plot -> {out_bev_path}")


if __name__ == "__main__":
    main()
