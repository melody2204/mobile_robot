"""Small sliding-window BA, with sparse residual Jacobian pattern.

Educational solver, not ORB-SLAM3's tightly coupled visual-inertial BA.
We fix the two oldest keyframe poses in the window to remove monocular
gauge freedoms (reference frame and scale). Use local BA conservatively.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from geometry import make_pose


class LocalBundleAdjuster:
    def __init__(self, camera, window_size=5, max_points=250, max_nfev=15):
        self.camera = camera
        self.window_size = window_size
        self.max_points = max_points
        self.max_nfev = max_nfev

    def optimize(self, sparse_map) -> dict:
        kfs = list(sparse_map.keyframes.values())[-self.window_size:]
        if len(kfs) < 3:
            return {"optimized": False, "reason": "need >=3 keyframes"}
        kf_by_id = {kf.id: kf for kf in kfs}
        ids = {kf.id for kf in kfs}
        points = [p for p in sparse_map.points.values() if not p.bad and len(ids.intersection(p.observations)) >= 2]
        points = sorted(points, key=lambda p: -len(p.observations))[:self.max_points]
        if len(points) < 20:
            return {"optimized": False, "reason": "need >=20 shared landmarks"}
        free_kfs = kfs[2:]
        pose_to_index = {kf.id: i for i, kf in enumerate(free_kfs)}
        point_to_index = {p.id: i for i, p in enumerate(points)}
        measurements = [(kf_id, p.id, kf_by_id[kf_id].xy[feat_id])
                        for p in points for kf_id, feat_id in p.observations.items() if kf_id in kf_by_id]
        if len(measurements) < 50:
            return {"optimized": False, "reason": "too few observations"}
        n_pose_vars = 6 * len(free_kfs)
        initial_poses = np.concatenate([np.r_[Rotation.from_matrix(kf.T_cw[:3, :3]).as_rotvec(),
                                               kf.T_cw[:3, 3]] for kf in free_kfs])
        initial_xyz = np.concatenate([p.xyz for p in points])
        x0 = np.r_[initial_poses, initial_xyz]
        fixed_poses = {kf.id: kf.T_cw.copy() for kf in kfs[:2]}
        target = np.asarray([u for _, _, u in measurements])
        cam = self.camera

        def residual(x):
            poses = dict(fixed_poses)
            for kf in free_kfs:
                i = 6 * pose_to_index[kf.id]
                poses[kf.id] = make_pose(Rotation.from_rotvec(x[i:i+3]).as_matrix(), x[i+3:i+6])
            xyz = x[n_pose_vars:].reshape(-1, 3)
            out = np.empty((len(measurements), 2))
            for j, (kf_id, point_id, _) in enumerate(measurements):
                T = poses[kf_id]
                pw = xyz[point_to_index[point_id]]
                pc = T[:3, :3] @ pw + T[:3, 3]
                if pc[2] <= 1e-8:
                    out[j] = 1000.0
                else:
                    out[j] = [cam.fx*pc[0]/pc[2]+cam.cx-target[j, 0],
                              cam.fy*pc[1]/pc[2]+cam.cy-target[j, 1]]
            return out.ravel()

        jac = lil_matrix((2 * len(measurements), len(x0)), dtype=np.int8)
        for j, (kf_id, point_id, _) in enumerate(measurements):
            if kf_id in pose_to_index:
                i = 6 * pose_to_index[kf_id]
                jac[2*j:2*j+2, i:i+6] = 1
            i = n_pose_vars + 3 * point_to_index[point_id]
            jac[2*j:2*j+2, i:i+3] = 1
        r0 = residual(x0)
        result = least_squares(residual, x0, jac_sparsity=jac.tocsr(), loss="huber", f_scale=2.,
                               method="trf", max_nfev=self.max_nfev)
        # Commit only if the numerical objective improves and is finite.
        if not np.isfinite(result.cost) or np.sum(result.fun**2) >= np.sum(r0**2):
            return {"optimized": False, "reason": "solver did not improve residuals"}
        for kf in free_kfs:
            i = 6 * pose_to_index[kf.id]
            kf.T_cw = make_pose(Rotation.from_rotvec(result.x[i:i+3]).as_matrix(), result.x[i+3:i+6])
        xyz = result.x[n_pose_vars:].reshape(-1, 3)
        for p in points:
            p.xyz = xyz[point_to_index[p.id]].copy()
        return {"optimized": True, "observations": len(measurements),
                "rmse_px_before": float(np.sqrt(np.mean(r0**2))),
                "rmse_px_after": float(np.sqrt(np.mean(result.fun**2)))}
