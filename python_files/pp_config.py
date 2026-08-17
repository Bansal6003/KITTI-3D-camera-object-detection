"""Shared config for the pillar encoder, model, dataset, and training script."""

# LiDAR point cloud range we detect within (velodyne frame): x=forward, y=left, z=up
X_MIN, X_MAX = 0.0, 70.4
Y_MIN, Y_MAX = -40.0, 40.0
Z_MIN, Z_MAX = -3.0, 1.0

# Pillar (voxel with a single z-bin) footprint in meters
VOXEL_X, VOXEL_Y = 0.32, 0.32

GRID_W = round((X_MAX - X_MIN) / VOXEL_X)  # 220 columns (x / forward)
GRID_H = round((Y_MAX - Y_MIN) / VOXEL_Y)  # 250 rows    (y / left)

MAX_POINTS_PER_PILLAR = 32
MAX_PILLARS = 8000

PILLAR_FEAT_DIM = 9   # x,y,z,r, xc,yc,zc (offset from pillar mean), xp,yp (offset from pillar center)
PILLAR_OUT_CHANNELS = 64

BACKBONE_DOWNSAMPLE = 2       # feature map is GRID/2 in each dim
FEAT_H = GRID_H // BACKBONE_DOWNSAMPLE
FEAT_W = GRID_W // BACKBONE_DOWNSAMPLE

# Fixed-size "Car" anchor used only to make regression targets scale-normalized
CAR_ANCHOR_L = 3.9
CAR_ANCHOR_W = 1.6
CAR_ANCHOR_H = 1.5

LABELS_DIR = r"D:\Behavioral_genetics\Robotics\kitti_perception\labels_3d"
VELO_DIR = (
    r"D:\Behavioral_genetics\Robotics\dataset_3Dperception_practice"
    r"\2011_09_26_drive_0009_sync\2011_09_26\2011_09_26_drive_0009_sync\velodyne_points\data"
)
