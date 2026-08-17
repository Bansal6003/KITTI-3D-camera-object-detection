"""
calib parser

KITTI raw-data utilities: calib parsing, tracklet XML parsing, and the
velodyne -> camera -> image projection chain.

Folder layout this expects (matches your dataset_3Dperception_practice folder):
  2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt
  2011_09_26_calib/2011_09_26/calib_velo_to_cam.txt
  2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/image_02/data/*.png
  2011_09_26_drive_0009_sync/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/*.bin
  2011_09_26_drive_0009_tracklets/2011_09_26/2011_09_26_drive_0009_sync/tracklet_labels.xml
"""
import numpy as np
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _read_calib_file(path):
    """Parse a KITTI 'key: v1 v2 v3 ...' calib file into {key: np.array}."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                continue  # e.g. calib_time string field
    return data


def load_calib(calib_dir):
    """
    Returns a dict with:
      P2        : 3x4 projection matrix, rectified cam0 -> image_02 pixels
      R_rect_00 : 4x4 homogeneous rectifying rotation for cam0
      Tr_velo_cam : 4x4 homogeneous transform, velodyne -> unrectified cam0
    """
    cam = _read_calib_file(f"{calib_dir}/calib_cam_to_cam.txt")
    velo = _read_calib_file(f"{calib_dir}/calib_velo_to_cam.txt")

    P2 = cam["P_rect_02"].reshape(3, 4)

    R_rect_00 = np.eye(4)
    R_rect_00[:3, :3] = cam["R_rect_00"].reshape(3, 3)

    Tr_velo_cam = np.eye(4)
    Tr_velo_cam[:3, :3] = velo["R"].reshape(3, 3)
    Tr_velo_cam[:3, 3] = velo["T"]

    return {"P2": P2, "R_rect_00": R_rect_00, "Tr_velo_cam": Tr_velo_cam}


def velo_to_cam2_image(points_velo_xyz, calib):
    """
    Project Nx3 points in velodyne frame to Nx2 pixel coords in image_02,
    following the standard chain: velo -> cam0 -> rectified cam0 -> P2 pixels.
    Returns (pixels Nx2, depth N) — depth is distance along cam0's z-axis,
    used to keep only points in front of the camera.
    """
    n = points_velo_xyz.shape[0]
    pts_h = np.hstack([points_velo_xyz, np.ones((n, 1))])  # Nx4 homogeneous

    pts_cam0 = (calib["Tr_velo_cam"] @ pts_h.T).T          # velo -> cam0
    pts_rect = (calib["R_rect_00"] @ pts_cam0.T).T         # rectify

    depth = pts_rect[:, 2]

    pts_img_h = (calib["P2"] @ pts_rect.T).T                # rectified -> pixels
    pixels = pts_img_h[:, :2] / pts_img_h[:, 2:3]
    return pixels, depth


# ---------------------------------------------------------------------------
# Velodyne point cloud
# ---------------------------------------------------------------------------

def load_velodyne_points(bin_path):
    """Returns Nx4 array: x, y, z, reflectance (velodyne frame, meters)."""
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)


# ---------------------------------------------------------------------------
# Tracklets (3D object labels, one XML per drive, poses per frame)
# ---------------------------------------------------------------------------

class Tracklet:
    def __init__(self, obj_type, h, w, l, first_frame):
        self.obj_type = obj_type
        self.h, self.w, self.l = h, w, l
        self.first_frame = first_frame
        self.poses = []  # list of dict(frame, tx, ty, tz, rz, occlusion, truncation)


def parse_tracklets(xml_path):
    """Parse tracklet_labels.xml -> list[Tracklet]."""
    root = ET.parse(xml_path).getroot()
    tracklets_elem = root.find("tracklets")

    tracklets = []
    for item in tracklets_elem.findall("item"):
        obj_type = item.find("objectType").text
        h = float(item.find("h").text)
        w = float(item.find("w").text)
        l = float(item.find("l").text)
        first_frame = int(item.find("first_frame").text)

        tr = Tracklet(obj_type, h, w, l, first_frame)

        poses_elem = item.find("poses")
        for frame_offset, pose in enumerate(poses_elem.findall("item")):
            tr.poses.append({
                "frame": first_frame + frame_offset,
                "tx": float(pose.find("tx").text),
                "ty": float(pose.find("ty").text),
                "tz": float(pose.find("tz").text),
                "rz": float(pose.find("rz").text),
                "occlusion": float(pose.find("occlusion").text),
                "truncation": float(pose.find("truncation").text),
            })
        tracklets.append(tr)
    return tracklets


def tracklets_for_frame(tracklets, frame_idx):
    """Yield (tracklet, pose) for every tracklet present at frame_idx."""
    for tr in tracklets:
        for pose in tr.poses:
            if pose["frame"] == frame_idx:
                yield tr, pose
                break


def box3d_corners_velo(tr, pose):
    """
    8x3 box corners in velodyne frame, following the KITTI devkit convention:
    (tx,ty,tz) is the ground-contact center (bottom of box), z_corners run 0->h.
    Corner order: 0-3 = bottom face, 4-7 = top face (same x,y order).
    """
    l, w, h = tr.l, tr.w, tr.h
    x_corners = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2])
    y_corners = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2])
    z_corners = np.array([   0,    0,    0,    0,    h,    h,    h,    h])

    rz = pose["rz"]
    R = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz),  np.cos(rz), 0],
        [0,           0,          1],
    ])
    corners = R @ np.vstack([x_corners, y_corners, z_corners])
    corners[0, :] += pose["tx"]
    corners[1, :] += pose["ty"]
    corners[2, :] += pose["tz"]
    return corners.T  # 8x3


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]
