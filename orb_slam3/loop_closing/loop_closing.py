"""Loop-closure backbone. Detection is implemented; map correction is future work."""
from __future__ import annotations
from .place_recognition import PlaceRecognizer


class LoopCloser:
    def __init__(self, matcher):
        self.recognizer = PlaceRecognizer(matcher)
        self.pending_candidates: list[tuple[int, int, int]] = []

    def inspect(self, keyframe, sparse_map):
        """Find appearance-only candidates, never declare an unverified loop."""
        candidates = self.recognizer.candidates(keyframe, sparse_map)
        for score, other_id in candidates:
            self.pending_candidates.append((keyframe.id, other_id, score))
        return candidates

    def close_loop(self, *args, **kwargs):
        raise NotImplementedError(
            "Loop closure requires geometric verification, Sim(3)/SE(3) estimation, "
            "pose-graph correction, landmark updates, and map consistency checks.")
