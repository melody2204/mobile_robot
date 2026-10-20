"""Persistent, selected frame; stores only necessary visual SLAM state."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .frame import Frame


@dataclass
class KeyFrame:
    id: int
    timestamp: float
    T_cw: np.ndarray
    xy: np.ndarray
    descriptors: np.ndarray | None
    observations: dict[int, int] = field(default_factory=dict)  # feature index -> map point id

    @classmethod
    def from_frame(cls, frame: Frame) -> "KeyFrame":
        if frame.T_cw is None:
            raise ValueError("Cannot construct keyframe before pose estimation")
        return cls(frame.id, frame.timestamp, frame.T_cw.copy(), frame.xy.copy(),
                   None if frame.descriptors is None else frame.descriptors.copy(),
                   dict(frame.associations))
