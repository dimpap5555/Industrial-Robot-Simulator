"""MotoMini (per your layout) — DH config + helpers."""
from __future__ import annotations
import numpy as np
from ..kinematics import SerialDHRobot
from ..types import DHLink, Frames, Limits, SerialRobotConfig

def motomini_config() -> SerialRobotConfig:
    dh = (
        DHLink(0.000, -np.pi/2, 0.068, 0.0,     True),   # J1 Yaw
        DHLink(0.103,  0.0,     0.000, -np.pi/2, True),  # J2 Pitch
        DHLink(0.165, -np.pi/2, 0.000, +np.pi/2, True),  # J3 Pitch
        DHLink(0.165, +np.pi/2, 0.000, 0.0,     True),   # J4 Yaw
        DHLink(0.000, -np.pi/2, 0.000, 0.0,     True),   # J5 Pitch
        DHLink(0.000,  0.0,     0.040, 0.0,     True),   # J6 / TCP offset
    )

    # Generic limits — adjust to your controller if needed
    q_limits = np.deg2rad(np.array([
        [-170, 170],   # J1
        [ -90,  90],   # J2
        [ -90,  90],   # J3
        [-140, 140],   # J4
        [-120, 120],   # J5
        [-360, 360],   # J6
    ], dtype=float))
    limits = Limits(q_min=q_limits[:,0], q_max=q_limits[:,1])

    frames = Frames(T_base=np.eye(4), T_tool=np.eye(4))  # tool drop is in d6, so identity here
    return SerialRobotConfig(dh=dh, limits=limits, frames=frames, name="MotoMini (from layout)")

def create_robot() -> SerialDHRobot:
    return SerialDHRobot(motomini_config())