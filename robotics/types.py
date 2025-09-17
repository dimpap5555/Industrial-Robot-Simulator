"""Shared dataclasses and protocols for the robotics framework."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DHLink:
    """Standard Denavit–Hartenberg link description."""

    a: float
    alpha: float
    d: float
    theta0: float
    revolute: bool = True


@dataclass(frozen=True)
class Limits:
    """Joint limits for a serial manipulator."""

    q_min: np.ndarray
    q_max: np.ndarray

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.q_min), self.q_max)


@dataclass(frozen=True)
class Frames:
    """Base and tool frames for a robot description."""

    T_base: np.ndarray
    T_tool: np.ndarray


@dataclass(frozen=True)
class SerialRobotConfig:
    """Configuration describing a serial robot in DH form."""

    dh: tuple[DHLink, ...]
    limits: Limits
    frames: Frames
    name: str