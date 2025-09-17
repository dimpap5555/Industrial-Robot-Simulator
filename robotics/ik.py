"""Inverse kinematics task framework."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np

from .kinematics import FKOptions, SerialKinematics, fk, numerical_jacobian
from .types import Limits


ErrFun = Callable[[np.ndarray], np.ndarray]
JacFun = Callable[[np.ndarray], np.ndarray]


@dataclass
class IKTask:
    name: str
    error: ErrFun
    weight: float | np.ndarray = 1.0
    jacobian: Optional[JacFun] = None


@dataclass(frozen=True)
class IKLimits:
    q_min: np.ndarray
    q_max: np.ndarray

    @classmethod
    def from_limits(cls, limits: Limits) -> "IKLimits":
        return cls(q_min=limits.q_min, q_max=limits.q_max)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.q_min), self.q_max)


@dataclass(frozen=True)
class IKStop:
    pos_tol: float = 1e-4
    ori_tol: float = 1e-3
    max_iters: int = 150


@dataclass
class IKProblem:
    robot: SerialKinematics
    tasks: Iterable[IKTask]
    q_seed: np.ndarray
    limits: IKLimits
    damp: float = 2e-2
    stop: IKStop = IKStop()


def _task_weight_matrix(weight: float | np.ndarray, dim: int) -> np.ndarray:
    if np.isscalar(weight):
        return float(weight) * np.eye(dim)
    w = np.asarray(weight, dtype=float)
    if w.shape == (dim,):
        return np.diag(w)
    if w.shape == (dim, dim):
        return w
    raise ValueError("Invalid weight shape for task")


def _task_tolerance(task: IKTask, stop: IKStop, err_dim: int) -> float:
    name = task.name.lower()
    if any(key in name for key in ("pos", "xyz", "xy")):
        return stop.pos_tol
    if any(key in name for key in ("ori", "align", "dir")):
        return stop.ori_tol
    return stop.pos_tol if err_dim <= 3 else stop.ori_tol


def solve_ik(problem: IKProblem) -> np.ndarray:
    q = problem.q_seed.astype(float).copy()
    tasks = list(problem.tasks)
    stop = problem.stop

    for _ in range(stop.max_iters):
        errs: List[np.ndarray] = []
        J_blocks: List[np.ndarray] = []
        tol_ok = []
        for task in tasks:
            err = task.error(q)
            if task.jacobian is not None:
                J = task.jacobian(q)
            else:
                J = numerical_jacobian(task.error, q)
            W = _task_weight_matrix(task.weight, err.size)
            errs.append(W @ err)
            J_blocks.append(W @ J)
            tol = _task_tolerance(task, stop, err.size)
            tol_ok.append(np.linalg.norm(err) <= tol)
        if all(tol_ok):
            break
        if not errs:
            return q
        e = np.concatenate(errs)
        J = np.vstack(J_blocks)
        JJt = J @ J.T
        lam = problem.damp
        dq = J.T @ np.linalg.solve(JJt + (lam ** 2) * np.eye(JJt.shape[0]), -e)
        q = q + dq
        q = problem.limits.clamp(q)
    return q


def task_position(robot: SerialKinematics, target_p: np.ndarray, tip_offset: np.ndarray | None = None) -> ErrFun:
    target = np.asarray(target_p, dtype=float)
    offset = np.zeros(3) if tip_offset is None else np.asarray(tip_offset, dtype=float)

    def error(q: np.ndarray) -> np.ndarray:
        res = fk(robot, q, FKOptions(with_base=True, with_tool=True))
        T_tip = res.Ts[-1]
        p = T_tip[:3, 3] + T_tip[:3, :3] @ offset
        return p - target

    return error


def task_tool_z_align(robot: SerialKinematics, target_dir: np.ndarray, tip_offset: np.ndarray | None = None) -> ErrFun:
    direction = np.asarray(target_dir, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    offset = np.zeros(3) if tip_offset is None else np.asarray(tip_offset, dtype=float)

    def error(q: np.ndarray) -> np.ndarray:
        res = fk(robot, q, FKOptions(with_base=True, with_tool=True))
        T_tip = res.Ts[-1]
        z_axis = T_tip[:3, :3] @ np.array([0.0, 0.0, 1.0])
        _ = offset  # reserved for future frame support
        z_norm = z_axis / (np.linalg.norm(z_axis) + 1e-12)
        return np.cross(z_norm, direction)

    return error


def task_xy(robot: SerialKinematics, target_xy: np.ndarray, tip_offset: np.ndarray | None = None) -> ErrFun:
    target = np.asarray(target_xy, dtype=float)
    offset = np.zeros(3) if tip_offset is None else np.asarray(tip_offset, dtype=float)

    def error(q: np.ndarray) -> np.ndarray:
        res = fk(robot, q, FKOptions(with_base=True, with_tool=True))
        T_tip = res.Ts[-1]
        p = T_tip[:3, 3] + T_tip[:3, :3] @ offset
        return p[:2] - target

    return error