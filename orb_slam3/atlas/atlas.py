"""Collection of maps and active-map selection. Map merging is deliberately TODO."""
from __future__ import annotations
from mapping.map import SparseMap


class Atlas:
    def __init__(self):
        self.maps: dict[int, SparseMap] = {0: SparseMap()}
        self.active_id = 0
        self._next_id = 1

    @property
    def active_map(self) -> SparseMap:
        return self.maps[self.active_id]

    def create_map(self) -> int:
        map_id = self._next_id
        self._next_id += 1
        self.maps[map_id] = SparseMap()
        self.active_id = map_id
        return map_id

    def select_map(self, map_id: int):
        if map_id not in self.maps:
            raise KeyError(f"Unknown map {map_id}")
        self.active_id = map_id

    def merge_maps(self, *args, **kwargs):
        raise NotImplementedError("Atlas map alignment, landmark fusion and merging not implemented")
