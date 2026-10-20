"""3D landmark and its persistent keyframe measurements."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MapPoint:
    id: int
    xyz: np.ndarray
    descriptor: np.ndarray
    observations: dict[int, int] = field(default_factory=dict)  # keyframe id -> feature index
    bad: bool = False
