"""Robust 3D-to-2D camera localisation from existing map points."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from geometry import make_pose, reprojection_errors


@dataclass
class PoseEstimate:
    T_cw: np.ndarray
    inlier_indices: np.ndarray  # indices into provided xyz/uv arrays


def estimate_pose(camera, xyz, uv, initial_pose=None, min_inliers=12) -> PoseEstimate | None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    if len(xyz) < 6:
        return None
    kwargs = {}
    if initial_pose is not None:
        rvec, _ = cv2.Rodrigues(initial_pose[:3, :3])
        kwargs = dict(rvec=rvec.copy(), tvec=initial_pose[:3, 3].reshape(3, 1).copy(),
                      useExtrinsicGuess=True)
    try:
        ok, rvec, tvec, indices = cv2.solvePnPRansac(
            xyz, uv, camera.K, None, iterationsCount=120,
            reprojectionError=4.0, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE, **kwargs)
        if not ok or indices is None or len(indices) < min_inliers:
            return None
        ids = indices.reshape(-1)
        if len(ids) >= 6:
            rvec, tvec = cv2.solvePnPRefineLM(xyz[ids], uv[ids], camera.K, None, rvec, tvec)
        T = make_pose(cv2.Rodrigues(rvec)[0], tvec)
        reproj = reprojection_errors(camera, xyz[ids], T, uv[ids])
        ids = ids[reproj < 4.0]
        if len(ids) < min_inliers:
            return None
        return PoseEstimate(T, ids)
    except cv2.error:
        return None
