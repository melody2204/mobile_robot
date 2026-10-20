"""Simple aligned trajectory RMSE. Monocular trajectories require Sim(3) alignment."""
from __future__ import annotations
import numpy as np


def align_trajectory_umeyama(estimated_xyz, reference_xyz, with_scale=True):
    """Find s,R,t that maps estimated XYZ to reference XYZ; synchronized samples required."""
    X = np.asarray(estimated_xyz, dtype=float)
    Y = np.asarray(reference_xyz, dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3 or len(X) < 3:
        raise ValueError("Expected matching Nx3 trajectories, N>=3")
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    x, y = X-mx, Y-my
    U, singular, Vt = np.linalg.svd((y.T @ x)/len(X))
    D = np.eye(3)
    if np.linalg.det(U@Vt) < 0:
        D[-1, -1] = -1
    R = U@D@Vt
    scale = float(np.sum(singular*np.diag(D)) / np.mean(np.sum(x*x, axis=1))) if with_scale else 1.0
    t = my - scale*R@mx
    aligned = (scale*R@X.T).T + t
    rmse = float(np.sqrt(np.mean(np.sum((aligned-Y)**2, axis=1))))
    return {"rmse": rmse, "scale": scale, "R": R, "t": t, "aligned": aligned}
