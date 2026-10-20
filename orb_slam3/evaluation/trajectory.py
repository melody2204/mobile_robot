"""Write keyframe trajectory in TUM RGB-D format (timestamp tx ty tz qx qy qz qw)."""
from __future__ import annotations
from pathlib import Path
from scipy.spatial.transform import Rotation
from geometry import inverse_pose


def write_tum_trajectory(path, trajectory):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for timestamp, T_cw in trajectory:
            T_wc = inverse_pose(T_cw)
            t = T_wc[:3, 3]
            q = Rotation.from_matrix(T_wc[:3, :3]).as_quat()
            handle.write(f"{timestamp:.9f} " + " ".join(f"{v:.9f}" for v in [*t, *q]) + "\n")
