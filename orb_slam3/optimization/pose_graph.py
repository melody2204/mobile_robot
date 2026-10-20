"""Standalone educational SE(3) pose-graph optimiser (not yet integrated).

Each edge measurement T_ji maps coordinates in camera i to camera j.
This supports pose corrections but does not automatically update map landmarks.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from geometry import inverse_pose, make_pose


@dataclass
class PoseGraphEdge:
    source: int
    target: int
    T_ji: np.ndarray
    translation_weight: float = 1.0
    rotation_weight: float = 1.0


class PoseGraph:
    def __init__(self):
        self.poses: dict[int, np.ndarray] = {}
        self.edges: list[PoseGraphEdge] = []

    def add_node(self, index: int, T_cw: np.ndarray) -> None:
        self.poses[index] = T_cw.copy()

    def add_edge(self, source: int, target: int, T_ji: np.ndarray,
                 translation_weight=1.0, rotation_weight=1.0) -> None:
        self.edges.append(PoseGraphEdge(source, target, T_ji.copy(), translation_weight, rotation_weight))

    def optimize(self, max_nfev: int = 40) -> dict[int, np.ndarray]:
        ids = sorted(self.poses)
        if len(ids) < 2 or not self.edges:
            return dict(self.poses)
        anchor = ids[0]
        free = ids[1:]
        params = np.concatenate([np.r_[Rotation.from_matrix(self.poses[i][:3, :3]).as_rotvec(),
                                        self.poses[i][:3, 3]] for i in free])
        def residual(x):
            poses = {anchor: self.poses[anchor]}
            for j, i in enumerate(free):
                p = x[6*j:6*j+6]
                poses[i] = make_pose(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])
            errors = []
            for edge in self.edges:
                pred = poses[edge.target] @ inverse_pose(poses[edge.source])
                delta = pred @ inverse_pose(edge.T_ji)
                errors.extend(edge.rotation_weight * Rotation.from_matrix(delta[:3, :3]).as_rotvec())
                errors.extend(edge.translation_weight * delta[:3, 3])
            return np.asarray(errors)
        result = least_squares(residual, params, loss="huber", max_nfev=max_nfev)
        for j, i in enumerate(free):
            p = result.x[6*j:6*j+6]
            self.poses[i] = make_pose(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])
        return dict(self.poses)
