"""Staubli RX-90L preset configuration and helpers."""
from __future__ import annotations

import numpy as np

from ..kinematics import SerialDHRobot
from ..types import DHLink, Frames, Limits, SerialRobotConfig


def rx90l_config() -> SerialRobotConfig:
    dh = (
        DHLink(0.000, -np.pi / 2, 0.350, 0.0, True),
        DHLink(0.450, 0.0, 0.0, -np.pi / 2, True),
        DHLink(0.050, -np.pi / 2, 0.0, np.pi / 2, True),
        DHLink(0.425, np.pi / 2, 0.0, 0.0, True),
        DHLink(0.000, -np.pi / 2, 0.0, 0.0, True),
        DHLink(0.000, 0.0, 0.100, 0.0, True),
    )
    q_limits = np.deg2rad(
        np.array(
            [
                [-160, 160],
                [-137.5, 137.5],
                [-142.5, 142.5],
                [-270, 270],
                [-105, 120],
                [-270, 270],
            ],
            dtype=float,
        )
    )
    limits = Limits(q_min=q_limits[:, 0], q_max=q_limits[:, 1])
    frames = Frames(T_base=np.eye(4), T_tool=np.eye(4))
    return SerialRobotConfig(dh=dh, limits=limits, frames=frames, name="Staubli RX-90L")


def create_robot() -> SerialDHRobot:
    """Instantiate the RX-90L robot from its configuration."""

    return SerialDHRobot(rx90l_config())