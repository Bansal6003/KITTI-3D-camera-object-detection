"""
Step 3: load the best checkpoint (lowest val loss, not the final overfit
epoch) and inspect predictions on held-out val frames — BEV plot (same style
as training-time monitoring) and predicted boxes projected onto the camera
image via the same velo->cam0->rect->P2 chain from Step 1.

Run:
    python 03_visualize_3d_predictions.py --frame 215
"""
import argparse
import torch
import matplotlib.pyplot as plt

import kitti_utils as ku
import pp_config as cfg
from dataset_3d import load_boxes, points_to_pillars
from pointpillars_model import PointPillarsCenterNet
from decode_3d import decode_heatmap
from viz_utils import draw_boxes_on_image, plot_box_bev
import cv2

CKPT_PATH = r"D:\Behavioral_genetics\Robotics\kitti_perception\pointpillars_centernet_best.pth"
CALIB_DIR = r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice\2011_09_26_calib\2011_09_26"
IMG_DIR = (
    r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice"
    r"\2011_09_26_drive_0009_sync\2011_09_26\2011_09_26_drive_0009_sync\image_02\data"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=215)
    parser.add_argument("--score_thresh", type=float, default=0.3)
    args = parser.parse_args()
    frame_idx = args.frame

    model = PointPillarsCenterNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    velo_path = fr"{cfg.VELO_DIR}\{frame_idx:010d}.bin"
    points = ku.load_velodyne_points(velo_path)
    gt_boxes = load_boxes(frame_idx)

    pillars, coords, num_points, n_pillars = points_to_pillars(points)
    batch = {
        "pillars": torch.from_numpy(pillars).unsqueeze(0).to(device),
        "coords": torch.from_numpy(coords).unsqueeze(0).to(device),
        "num_points": torch.from_numpy(num_points).unsqueeze(0).to(device),
        "n_pillars": torch.tensor([n_pillars]),
    }
    with torch.no_grad():
        pred = model(batch["pillars"], batch["coords"], batch["num_points"], batch["n_pillars"])
    pred_boxes = decode_heatmap(pred, score_thresh=args.score_thresh)

    print(f"Frame {frame_idx}: {len(gt_boxes)} GT boxes, {len(pred_boxes)} predicted boxes "
          f"(score > {args.score_thresh})")

    # --- BEV plot ---
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.scatter(-points[:, 1], points[:, 0], s=0.5, c="gray")
    for x, y, z, l, w, h, yaw in gt_boxes:
        plot_box_bev(ax, x, y, l, w, yaw, color="red")
    for x, y, z, l, w, h, yaw, score in pred_boxes:
        plot_box_bev(ax, x, y, l, w, yaw, color="lime", label=f"{score:.2f}")
    ax.set_xlim(cfg.Y_MAX, cfg.Y_MIN)
    ax.set_ylim(cfg.X_MIN, cfg.X_MAX)
    ax.set_aspect("equal")
    ax.set_title(f"Frame {frame_idx} — BEV (red=GT, green=pred, best checkpoint)")
    bev_path = f"final_frame_{frame_idx:04d}_bev.png"
    fig.savefig(bev_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {bev_path}")

    # --- Camera projection ---
    calib = ku.load_calib(CALIB_DIR)
    img_path = fr"{IMG_DIR}\{frame_idx:010d}.png"
    img = cv2.imread(img_path)
    gt_xyzlwh = [b[:7] for b in gt_boxes]
    pred_xyzlwh = [b[:7] for b in pred_boxes]
    pred_labels = [f"{b[7]:.2f}" for b in pred_boxes]
    img = draw_boxes_on_image(img, gt_xyzlwh, calib, color=(0, 0, 255))                       # red = GT
    img = draw_boxes_on_image(img, pred_xyzlwh, calib, color=(0, 255, 0), labels=pred_labels)  # green = pred
    cam_path = f"final_frame_{frame_idx:04d}_camera.png"
    cv2.imwrite(cam_path, img)
    print(f"Saved -> {cam_path}")


if __name__ == "__main__":
    main()
