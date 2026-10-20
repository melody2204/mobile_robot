"""Simple ORB descriptor candidate retrieval; NOT DBoW2/ORB-SLAM3 place recognition."""
from __future__ import annotations


class PlaceRecognizer:
    def __init__(self, matcher, min_keyframe_separation=20, min_matches=50):
        self.matcher = matcher
        self.min_keyframe_separation = min_keyframe_separation
        self.min_matches = min_matches

    def candidates(self, current_kf, sparse_map, max_candidates=5):
        scores = []
        for kf in sparse_map.keyframes.values():
            if kf.id == current_kf.id or abs(current_kf.id - kf.id) < self.min_keyframe_separation:
                continue
            n_matches = len(self.matcher.match(current_kf.descriptors, kf.descriptors))
            if n_matches >= self.min_matches:
                scores.append((n_matches, kf.id))
        return sorted(scores, reverse=True)[:max_candidates]
