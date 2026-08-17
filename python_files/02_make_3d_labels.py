"""
Step 2a: convert tracklets -> per-frame 3D label files for the "Car" class.

Each output file labels_3d/{frame:06d}.txt has one line per Car:
    x y z l w h yaw
in the velodyne frame, where (x,y,z) is the BOX CENTER (not ground contact —
tz from the tracklet is the bottom, so we add h/2 here once, up front, so
every downstream consumer uses box-center convention consistently).
"""
import os
import kitti_utils as ku

DATA_ROOT = r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice"
TRACKLET_XML = fr"{DATA_ROOT}\2011_09_26_drive_0009_tracklets\2011_09_26\2011_09_26_drive_0009_sync\tracklet_labels.xml"
VELO_DIR = fr"{DATA_ROOT}\2011_09_26_drive_0009_sync\2011_09_26\2011_09_26_drive_0009_sync\velodyne_points\data"
OUT_DIR = r"D:\Behavioral_genetics\Robotics\kitti_perception\labels_3d"
TARGET_CLASS = "Car"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tracklets = ku.parse_tracklets(TRACKLET_XML)

    # Frame numbering isn't contiguous — this drive drops frames 177-180
    # (443 files spanning indices 0-446), so build the valid set from the
    # actual filenames rather than assuming range(num_files).
    valid_frames = sorted(int(f[:-4]) for f in os.listdir(VELO_DIR) if f.endswith(".bin"))
    valid_set = set(valid_frames)

    per_frame = {i: [] for i in valid_frames}
    skipped_missing = 0
    for tr in tracklets:
        if tr.obj_type != TARGET_CLASS:
            continue
        for pose in tr.poses:
            if pose["frame"] not in valid_set:
                skipped_missing += 1
                continue
            cz = pose["tz"] + tr.h / 2.0  # bottom -> box center
            per_frame[pose["frame"]].append(
                (pose["tx"], pose["ty"], cz, tr.l, tr.w, tr.h, pose["rz"])
            )

    total_boxes = 0
    for frame_idx, boxes in per_frame.items():
        out_path = fr"{OUT_DIR}\{frame_idx:06d}.txt"
        with open(out_path, "w") as f:
            for x, y, z, l, w, h, yaw in boxes:
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {l:.4f} {w:.4f} {h:.4f} {yaw:.4f}\n")
        total_boxes += len(boxes)

    print(f"Wrote {len(valid_frames)} label files to {OUT_DIR} (frame range {valid_frames[0]}-{valid_frames[-1]})")
    print(f"Total Car boxes: {total_boxes} across {sum(1 for b in per_frame.values() if b)} non-empty frames")
    if skipped_missing:
        print(f"Skipped {skipped_missing} tracklet poses referencing frames with no velodyne data (e.g. 177-180)")


if __name__ == "__main__":
    main()
