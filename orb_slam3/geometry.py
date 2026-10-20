"""Camera geometry and SE(3) helpers.

Convention throughout: T_cw maps world coordinates into camera coordinates,
p_c = R_cw @ p_w + t_cw. Camera trajectory files export T_wc.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    @classmethod
    def from_json(cls, path: str) -> "PinholeCamera":
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def project(self, world_points: np.ndarray, T_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        cam = (T_cw[:3, :3] @ p.T).T + T_cw[:3, 3]
        z = cam[:, 2]
        u = np.zeros((len(p), 2), dtype=np.float64)
        valid = z > 1e-8
        u[valid, 0] = self.fx * cam[valid, 0] / z[valid] + self.cx
        u[valid, 1] = self.fy * cam[valid, 1] / z[valid] + self.cy
        valid &= (u[:, 0] >= 0) & (u[:, 0] < self.width) & (u[:, 1] >= 0) & (u[:, 1] < self.height)
        return u, valid


def make_pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inverse_pose(T: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inverse


def triangulate(K: np.ndarray, T1: np.ndarray, T2: np.ndarray,
                uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    P1, P2 = K @ T1[:3], K @ T2[:3]
    homogeneous = cv2.triangulatePoints(P1, P2,
                                         np.asarray(uv1, dtype=np.float64).T,
                                         np.asarray(uv2, dtype=np.float64).T)
    w = homogeneous[3]
    w = np.where(np.abs(w) < 1e-12, np.nan, w)
    return (homogeneous[:3] / w).T


def reprojection_errors(camera: PinholeCamera, xyz: np.ndarray, T: np.ndarray,
                        observed: np.ndarray) -> np.ndarray:
    projected, valid = camera.project(xyz, T)
    errors = np.linalg.norm(projected - observed, axis=1)
    return np.where(valid, errors, np.inf)


def rotation_angle(R: np.ndarray) -> float:
    return float(np.linalg.norm(Rotation.from_matrix(R).as_rotvec()))
