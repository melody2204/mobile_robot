"""Track current frame against visible map landmarks, estimate 6-DoF pose."""
from __future__ import annotations
import numpy as np
from optimization.pose_optimizer import estimate_pose


class Tracker:
    def __init__(self, camera, matcher, min_inliers=20):
        self.camera, self.matcher, self.min_inliers = camera, matcher, min_inliers

    def track(self, frame, sparse_map, prediction=None) -> int:
        points = sparse_map.visible_points(self.camera, prediction)
        # If inaccurate pose prediction hides the map, allow global matching.
        if len(points) < 35:
            points = sparse_map.visible_points(self.camera)
        if len(points) < self.min_inliers or frame.descriptors is None:
            return 0
        point_desc = np.stack([p.descriptor for p in points])
        matches = self.matcher.match(frame.descriptors, point_desc)
        if len(matches) < self.min_inliers:
            return 0
        xyz = np.stack([points[m.trainIdx].xyz for m in matches])
        uv = np.stack([frame.xy[m.queryIdx] for m in matches])
        result = estimate_pose(self.camera, xyz, uv, prediction, self.min_inliers)
        # Retry without initial guess in case motion model is significantly wrong.
        if result is None and prediction is not None:
            result = estimate_pose(self.camera, xyz, uv, None, self.min_inliers)
        if result is None:
            return 0
        frame.T_cw = result.T_cw
        for i in result.inlier_indices:
            m = matches[int(i)]
            frame.associations[m.queryIdx] = points[m.trainIdx].id
        return len(result.inlier_indices)
