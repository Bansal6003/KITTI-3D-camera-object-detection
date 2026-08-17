"""
AB3DMOT-style 3D multi-object tracker: constant-velocity Kalman filter per
track (position only) + Hungarian assignment on BEV center distance +
track birth/death management.

Why position-only KF: filtering yaw needs angle-wraparound handling (mean of
359 deg and 1 deg is 0 deg, not 180 deg) which is easy to get subtly wrong.
Position (x,y,z) has no such issue, and it's the part that actually benefits
from smoothing/prediction — so yaw and dimensions are just carried over from
the latest matched detection instead.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class Track:
    _next_id = 1

    def __init__(self, det, dt):
        x, y, z, l, w, h, yaw, score = det
        self.id = Track._next_id
        Track._next_id += 1

        self.dt = dt
        self.state = np.array([x, y, z, 0.0, 0.0, 0.0])  # x,y,z,vx,vy,vz
        self.P = np.eye(6) * 1.0

        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt
        self.Q = np.eye(6) * 0.5   # process noise (how much we trust the motion model)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1
        self.R = np.eye(3) * 0.5   # measurement noise (how much we trust a detection)

        self.dims = np.array([l, w, h])
        self.yaw = yaw
        self.score = score
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.state[:3]

    def update(self, det):
        x, y, z, l, w, h, yaw, score = det
        z_meas = np.array([x, y, z])
        residual = z_meas - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ residual
        self.P = (np.eye(6) - K @ self.H) @ self.P

        self.dims = np.array([l, w, h])
        self.yaw = yaw
        self.score = score
        self.hits += 1
        self.time_since_update = 0

    def as_box(self):
        x, y, z = self.state[:3]
        l, w, h = self.dims
        return (x, y, z, l, w, h, self.yaw, self.score, self.id)


class Tracker3D:
    def __init__(self, max_age=3, min_hits=1, gate_dist=3.0, dt=0.1):
        """
        max_age: frames a track can go unmatched before it's deleted (survives brief occlusion/missed detection)
        min_hits: matches needed before a track is reported (suppresses one-off false-positive detections)
        gate_dist: max BEV distance (m) for a detection to be considered a match to a track
        dt: seconds between frames (KITTI velodyne is ~10Hz)
        """
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_dist = gate_dist
        self.dt = dt
        self.dead_track_log = []  # (id, hits) for every track dropped, for diagnostics

    def step(self, detections):
        """detections: list of (x,y,z,l,w,h,yaw,score) for the current frame.
        Returns list of (x,y,z,l,w,h,yaw,score,track_id) for currently-alive confirmed tracks."""
        preds = [t.predict() for t in self.tracks]

        matched_dets = set()
        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, p in enumerate(preds):
                for j, d in enumerate(detections):
                    cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.gate_dist:
                    self.tracks[r].update(detections[c])
                    matched_dets.add(c)

        for j, d in enumerate(detections):
            if j not in matched_dets:
                self.tracks.append(Track(d, dt=self.dt))

        dying = [t for t in self.tracks if t.time_since_update > self.max_age]
        self.dead_track_log.extend((t.id, t.hits) for t in dying)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t.as_box() for t in self.tracks if t.hits >= self.min_hits]
