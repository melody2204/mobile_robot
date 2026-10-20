"""Image frame with extracted features, estimated pose and landmark associations."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Frame:
    id: int
    timestamp: float
    image: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None
    T_cw: np.ndarray | None = None
    associations: dict[int, int] = field(default_factory=dict)  # feature index -> map point id

    @property
    def xy(self) -> np.ndarray:
        return np.array([k.pt for k in self.keypoints], dtype=np.float64).reshape(-1, 2)

    @classmethod
    def create(cls, frame_id: int, timestamp: float, gray: np.ndarray, extractor) -> "Frame":
        keypoints, descriptors = extractor.extract(gray)
        return cls(frame_id, timestamp, gray, keypoints, descriptors)
