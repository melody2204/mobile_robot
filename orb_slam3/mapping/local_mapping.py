"""Triangulate new landmarks between calibrated keyframes."""
from __future__ import annotations
import numpy as np
from geometry import triangulate, reprojection_errors


class LocalMapper:
    def __init__(self, camera, matcher, max_new_points=450):
        self.camera, self.matcher, self.max_new_points = camera, matcher, max_new_points

    def triangulate_between(self, old_kf, new_kf, sparse_map) -> int:
        matches = self.matcher.match(old_kf.descriptors, new_kf.descriptors)
        if not matches:
            return 0
        matches = [m for m in matches if m.queryIdx not in old_kf.observations
                   and m.trainIdx not in new_kf.observations]
        if not matches:
            return 0
        uv1 = np.array([old_kf.xy[m.queryIdx] for m in matches])
        uv2 = np.array([new_kf.xy[m.trainIdx] for m in matches])
        xyz = triangulate(self.camera.K, old_kf.T_cw, new_kf.T_cw, uv1, uv2)
        e1 = reprojection_errors(self.camera, xyz, old_kf.T_cw, uv1)
        e2 = reprojection_errors(self.camera, xyz, new_kf.T_cw, uv2)
        c1 = -old_kf.T_cw[:3, :3].T @ old_kf.T_cw[:3, 3]
        c2 = -new_kf.T_cw[:3, :3].T @ new_kf.T_cw[:3, 3]
        a, b = xyz - c1, xyz - c2
        parallax = np.degrees(np.arccos(np.clip(np.sum(a*b, axis=1) /
            (np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1) + 1e-12), -1, 1)))
        good = np.isfinite(xyz).all(axis=1) & (e1 < 3) & (e2 < 3) & (parallax > 0.5)
        count = 0
        for j in np.flatnonzero(good):
            m = matches[j]
            sparse_map.new_point(xyz[j], new_kf.descriptors[m.trainIdx],
                                 [(old_kf.id, m.queryIdx), (new_kf.id, m.trainIdx)])
            count += 1
            if count >= self.max_new_points:
                break
        return count
