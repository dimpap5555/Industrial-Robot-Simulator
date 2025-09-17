"""Kinematics utilities and robot implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np

from .types import DHLink, SerialRobotConfig


class SerialKinematics(Protocol):
    """Protocol for serial manipulators that expose forward kinematics."""

    @property
    def dof(self) -> int:
        """Number of joints of the manipulator."""

    def fk_Ts(self, q: np.ndarray) -> List[np.ndarray]:
        """Return homogeneous transforms for base, joints and tool."""


@dataclass(frozen=True)
class FKOptions:
    with_tool: bool = True
    with_base: bool = True
    return_jacobians: bool = False


@dataclass
class FKResult:
    Ts: List[np.ndarray]
    points: np.ndarray
    J_list: Optional[List[np.ndarray]] = None


def _dh_matrix(link: DHLink, q_i: float) -> np.ndarray:
    """Compute the homogeneous transform for a single DH link."""

    if link.revolute:
        theta = link.theta0 + q_i
        d = link.d
    else:
        theta = link.theta0
        d = link.d + q_i
    ca = np.cos(link.alpha)
    sa = np.sin(link.alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, link.a * ct],
            [st, ct * ca, -ct * sa, link.a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


class SerialDHRobot:
    """Concrete serial manipulator defined by Denavit–Hartenberg parameters."""

    def __init__(self, config: SerialRobotConfig):
        self.config = config
        self._T_base = np.array(config.frames.T_base, dtype=float)
        self._T_tool = np.array(config.frames.T_tool, dtype=float)

    @property
    def dof(self) -> int:
        return len(self.config.dh)

    def fk_Ts(self, q: np.ndarray) -> List[np.ndarray]:
        if q.shape[0] != self.dof:
            raise ValueError(f"Expected q of shape ({self.dof},), got {q.shape}")
        T = self._T_base.copy()
        Ts = [T.copy()]
        for link, qi in zip(self.config.dh, q):
            T = T @ _dh_matrix(link, float(qi))
            Ts.append(T.copy())
        Ts[-1] = Ts[-1] @ self._T_tool
        return Ts


def fk(robot: SerialKinematics, q: np.ndarray, opts: FKOptions | None = None) -> FKResult:
    """Evaluate forward kinematics with configurable output options."""

    if opts is None:
        opts = FKOptions()
    Ts = list(robot.fk_Ts(q))
    if not opts.with_tool and len(Ts) > 1:
        Ts = Ts[:-1]
    if not opts.with_base and len(Ts) > 0:
        Ts = Ts[1:]
    points = np.array([T[:3, 3] for T in Ts], dtype=float)
    if opts.return_jacobians:
        raise NotImplementedError("Analytic Jacobians are not yet implemented")
    return FKResult(Ts=Ts, points=points, J_list=None)


def numerical_jacobian(fun, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Finite difference Jacobian of a vector-valued function."""

    base = fun(q)
    m = base.size
    J = np.zeros((m, q.size), dtype=float)
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        J[:, i] = (fun(q + dq) - base) / eps
    return J
