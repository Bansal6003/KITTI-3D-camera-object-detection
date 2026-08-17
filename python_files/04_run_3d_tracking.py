"""
Step 4: run the trained detector frame-by-frame over the whole drive (in
temporal order — unlike training, tracking needs a real sequence, not a
random train/val shuffle) and feed its detections into the 3D tracker.
Renders tracked boxes + persistent IDs onto the camera images as an mp4.

Run:
    python 04_run_3d_tracking.py
"""
import os
import colorsys
import numpy as np
import cv2
import torch

import kitti_utils as ku
import pp_config as cfg
from dataset_3d import points_to_pillars
from pointpillars_model import PointPillarsCenterNet
from decode_3d import decode_heatmap
from tracker import Tracker3D
from viz_utils import draw_boxes_on_image

CKPT_PATH = r"D:\Behavioral_genetics\Robotics\kitti_perception\pointpillars_centernet_best.pth"
CALIB_DIR = r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice\2011_09_26_calib\2011_09_26"
# IMG_DIR = (
#     r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice"
#     r"\2011_09_26_drive_0009_sync\2011_09_26\2011_09_26_drive_0009_sync\image_02\data"
# )

IMG_DIR = (
    r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice\sample_video\frames"
)

OUT_VIDEO = r"D:\Behavioral_genetics\Robotics\kitti_perception\sample_tracking_output.mp4"
SCORE_THRESH = 0.35
FPS = 10  # KITTI velodyne is ~10Hz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def id_color(track_id):
    hue = (track_id * 0.618033988749895) % 1.0  # golden-ratio spacing for visually distinct colors
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for cv2


def main():
    model = PointPillarsCenterNet().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {CKPT_PATH}")

    calib = ku.load_calib(CALIB_DIR)
    frame_indices = sorted(int(f[:-4]) for f in os.listdir(cfg.VELO_DIR) if f.endswith(".bin"))
    print(f"Running over {len(frame_indices)} frames ({frame_indices[0]}-{frame_indices[-1]})")

    tracker = Tracker3D(max_age=3, min_hits=2, gate_dist=3.0, dt=1.0 / FPS)

    sample_img = cv2.imread(fr"{IMG_DIR}\{frame_indices[0]:010d}.png")
    h, w = sample_img.shape[:2]
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    all_ids_seen = set()
    with torch.no_grad():
        for frame_idx in frame_indices:
            velo_path = fr"{cfg.VELO_DIR}\{frame_idx:010d}.bin"
            points = ku.load_velodyne_points(velo_path)
            pillars, coords, num_points, n_pillars = points_to_pillars(points)

            batch_pillars = torch.from_numpy(pillars).unsqueeze(0).to(device)
            batch_coords = torch.from_numpy(coords).unsqueeze(0).to(device)
            batch_num_points = torch.from_numpy(num_points).unsqueeze(0).to(device)
            batch_n_pillars = torch.tensor([n_pillars])

            pred = model(batch_pillars, batch_coords, batch_num_points, batch_n_pillars)
            detections = [b[:8] for b in decode_heatmap(pred, score_thresh=SCORE_THRESH)]

            tracked = tracker.step(detections)
            all_ids_seen.update(t[8] for t in tracked)

            img_path = fr"{IMG_DIR}\{frame_idx:010d}.png"
            img = cv2.imread(img_path)
            boxes_xyzlwh = [t[:7] for t in tracked]
            labels = [f"ID{t[8]}" for t in tracked]
            for box, track_id in zip(boxes_xyzlwh, [t[8] for t in tracked]):
                img = draw_boxes_on_image(img, [box], calib, color=id_color(track_id),
                                           labels=[f"ID{track_id}"])
            writer.write(img)

    writer.release()
    print(f"\nSaved tracking video -> {OUT_VIDEO}")
    print(f"Unique track IDs used across sequence: {len(all_ids_seen)}")

    all_hits = [hits for _, hits in tracker.dead_track_log] + [t.hits for t in tracker.tracks]
    all_hits = np.array(all_hits)
    print(f"Track lifespans (hits) — n={len(all_hits)}, "
          f"mean={all_hits.mean():.1f}, median={np.median(all_hits):.0f}, max={all_hits.max()}")
    for lo, hi in [(1, 2), (3, 5), (6, 10), (11, 1000)]:
        n = ((all_hits >= lo) & (all_hits <= hi)).sum()
        print(f"  hits {lo}-{hi if hi < 1000 else '+'}: {n} tracks")


if __name__ == "__main__":
    main()
