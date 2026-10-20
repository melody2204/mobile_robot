"""Mutual ratio matching for ORB binary descriptors."""
from __future__ import annotations
import cv2
import numpy as np


class FeatureMatcher:
    def __init__(self, ratio: float = 0.8):
        self.ratio = ratio
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> list[cv2.DMatch]:
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return []
        forward = self.matcher.knnMatch(desc_a, desc_b, k=2)
        reverse = self.matcher.knnMatch(desc_b, desc_a, k=2)
        reverse_best = {m.queryIdx: m.trainIdx for pair in reverse
                        if len(pair) == 2 for m, n in [pair] if m.distance < self.ratio * n.distance}
        return [m for pair in forward if len(pair) == 2 for m, n in [pair]
                if m.distance < self.ratio * n.distance and reverse_best.get(m.trainIdx) == m.queryIdx]
