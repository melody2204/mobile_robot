"""API contract for future tightly coupled visual-inertial optimisation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class InertialState:
    T_wb: np.ndarray
    velocity_w: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray


class VisualInertialEstimator:
    def initialize(self, *args, **kwargs):
        raise NotImplementedError("VIO initialisation (scale, gravity, velocity, bias) not implemented")

    def optimize(self, *args, **kwargs):
        raise NotImplementedError("Joint IMU+visual factor optimisation not implemented")
