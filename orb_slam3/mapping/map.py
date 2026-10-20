"""Sparse landmark map, with bidirectional observation consistency."""
from __future__ import annotations
from .keyframe import KeyFrame
from .map_point import MapPoint


class SparseMap:
    def __init__(self):
        self.keyframes: dict[int, KeyFrame] = {}
        self.points: dict[int, MapPoint] = {}
        self._next_point_id = 0

    def add_keyframe(self, keyframe: KeyFrame) -> None:
        if keyframe.id in self.keyframes:
            raise ValueError(f"Keyframe {keyframe.id} already exists")
        self.keyframes[keyframe.id] = keyframe
        for feature_id, point_id in list(keyframe.observations.items()):
            if point_id in self.points and not self.points[point_id].bad:
                self.points[point_id].observations[keyframe.id] = feature_id
            else:
                del keyframe.observations[feature_id]

    def new_point(self, xyz, descriptor, measurements: list[tuple[int, int]]) -> MapPoint:
        mp = MapPoint(self._next_point_id, xyz.copy(), descriptor.copy())
        self._next_point_id += 1
        self.points[mp.id] = mp
        for kf_id, feat_id in measurements:
            self.add_observation(kf_id, feat_id, mp.id)
        return mp

    def add_observation(self, kf_id: int, feat_id: int, point_id: int) -> bool:
        kf, mp = self.keyframes[kf_id], self.points[point_id]
        old = kf.observations.get(feat_id)
        if old is not None and old != point_id:
            return False
        old_feature = mp.observations.get(kf_id)
        if old_feature is not None and old_feature != feat_id:
            return False
        kf.observations[feat_id] = point_id
        mp.observations[kf_id] = feat_id
        return True

    def mark_bad_point(self, point_id: int) -> None:
        mp = self.points[point_id]
        mp.bad = True
        for kf_id, feat_id in list(mp.observations.items()):
            if kf_id in self.keyframes:
                self.keyframes[kf_id].observations.pop(feat_id, None)
        mp.observations.clear()

    def visible_points(self, camera, T_prediction=None):
        """Return point IDs and descriptors, with optional predicted frustum filtering."""
        valid_points = [p for p in self.points.values() if not p.bad]
        if T_prediction is not None and valid_points:
            import numpy as np
            xyz = np.array([p.xyz for p in valid_points])
            _, valid = camera.project(xyz, T_prediction)
            valid_points = [p for p, seen in zip(valid_points, valid) if seen]
        return valid_points
