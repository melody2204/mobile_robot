"""Small bias-corrected IMU preintegration illustration.

Assumes gyro rad/s, acceleration m/s^2, and constant biases. Does not yet
propagate preintegration covariance or bias Jacobians. NOT production VIO.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class IMUPreintegrator:
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    delta_v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    delta_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    total_dt: float = 0.0

    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> None:
        if dt <= 0:
            raise ValueError("IMU sampling interval must be positive")
        omega = np.asarray(gyro) - self.gyro_bias
        a_body = np.asarray(accel) - self.accel_bias
        a_start = self.delta_R @ a_body
        self.delta_p += self.delta_v*dt + 0.5*a_start*dt**2
        self.delta_v += a_start*dt
        self.delta_R = self.delta_R @ Rotation.from_rotvec(omega*dt).as_matrix()
        self.total_dt += dt

    def reset(self):
        self.delta_R = np.eye(3)
        self.delta_v = np.zeros(3)
        self.delta_p = np.zeros(3)
        self.total_dt = 0.0
