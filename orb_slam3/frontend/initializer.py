"""Calibrated, two-view monocular map initialisation.

Uses essential-matrix RANSAC; highly planar/pure-rotation scenes are rejected
when triangulation does not provide enough parallax. Global scale is arbitrary.
"""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from geometry import make_pose, triangulate, reprojection_errors


@dataclass
class InitializationResult:
    T_1: np.ndarray
    T_2: np.ndarray
    points: np.ndarray
    feature_pairs: list[tuple[int, int]]


class MonocularInitializer:
    def __init__(self, camera, matcher, min_matches=60, min_points=35):
        self.camera, self.matcher = camera, matcher
        self.min_matches, self.min_points = min_matches, min_points

    def initialize(self, first, second) -> InitializationResult | None:
        matches = self.matcher.match(first.descriptors, second.descriptors)
        if len(matches) < self.min_matches:
            return None
        xy_a, xy_b = first.xy, second.xy
        a = np.array([xy_a[m.queryIdx] for m in matches], dtype=np.float64)
        b = np.array([xy_b[m.trainIdx] for m in matches], dtype=np.float64)
        E, inlier_mask = cv2.findEssentialMat(a, b, self.camera.K, cv2.RANSAC, 0.999, 1.5)
        if E is None or inlier_mask is None:
            return None
        try:
            _count, R, t, cheirality = cv2.recoverPose(E, a, b, self.camera.K, mask=inlier_mask)
        except cv2.error:
            return None
        indices = np.flatnonzero(cheirality.reshape(-1) > 0)
        if len(indices) < self.min_points:
            return None
        T1, T2 = np.eye(4), make_pose(R, t)
        xyz = triangulate(self.camera.K, T1, T2, a[indices], b[indices])
        err1 = reprojection_errors(self.camera, xyz, T1, a[indices])
        err2 = reprojection_errors(self.camera, xyz, T2, b[indices])
        C1, C2 = np.zeros(3), -R.T @ t.reshape(3)
        rays1 = xyz - C1
        rays2 = xyz - C2
        cos = np.sum(rays1 * rays2, axis=1) / (np.linalg.norm(rays1, axis=1) * np.linalg.norm(rays2, axis=1) + 1e-12)
        parallax = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        good = (np.isfinite(xyz).all(axis=1) & (err1 < 3.0) & (err2 < 3.0) & (parallax > 0.7))
        good_ids = indices[good]
        if len(good_ids) < self.min_points:
            return None
        return InitializationResult(T1, T2, xyz[good],
                                    [(matches[i].queryIdx, matches[i].trainIdx) for i in good_ids])
