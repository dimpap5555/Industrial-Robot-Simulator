"""High-level RX-90L demo utilities built on top of the robotics framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ..ik import IKLimits, IKProblem, IKStop, IKTask, solve_ik, task_position, task_tool_z_align, task_xy
from ..kinematics import FKOptions, fk
from ..types import SerialRobotConfig
from .rx90l import create_robot, rx90l_config


@dataclass(slots=True)
class OCPResult:
    """Container returned by :meth:`Rx90LDemo.ocp_demo`."""

    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    tgrid: np.ndarray
    T: float


class Rx90LDemo:
    """Convenience wrapper exposing FK, IK and OCP demos for the RX-90L."""

    def __init__(self) -> None:
        self.config: SerialRobotConfig = rx90l_config()
        self.robot = create_robot()
        self._ik_limits = IKLimits.from_limits(self.config.limits)

    # ------------------------------------------------------------------
    # Forward kinematics helpers
    # ------------------------------------------------------------------
    def fk_points(self, q: np.ndarray) -> np.ndarray:
        """Return XYZ coordinates for base, joints and TCP."""

        res = fk(self.robot, q, FKOptions(with_base=True, with_tool=True))
        return res.points

    def forward_demo(self, T_final: float = 10.0, fps: int = 30) -> np.ndarray:
        """Generate a smooth forward-kinematics joint trajectory."""

        t = np.linspace(0.0, T_final, int(T_final * fps))
        q_limits = np.vstack([self.config.limits.q_min, self.config.limits.q_max]).T
        amp = 0.4 * (q_limits[:, 1] - q_limits[:, 0]) / 2.0
        center = (q_limits[:, 1] + q_limits[:, 0]) / 2.0
        freqs = np.array([0.2, 0.31, 0.17, 0.27, 0.23, 0.29])
        phases = np.linspace(0.0, np.pi, self.robot.dof)
        q_traj = np.zeros((len(t), self.robot.dof))
        for j in range(self.robot.dof):
            q_traj[:, j] = center[j] + amp[j] * np.sin(2.0 * np.pi * freqs[j] * t + phases[j])
        return q_traj

    # ------------------------------------------------------------------
    # Inverse kinematics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(vec)
        return vec if nrm < 1e-12 else vec / nrm

    def ik_xyz_dir(
        self,
        q_seed: np.ndarray,
        target_xyz: Iterable[float],
        target_dir: Iterable[float],
        *,
        d6: float = 0.0,
        iters: int = 150,
        lam: float = 2e-2,
    ) -> np.ndarray:
        """Solve IK for a desired TCP position and tool Z direction."""

        offset = np.array([0.0, 0.0, d6], dtype=float)
        target_p = np.asarray(target_xyz, dtype=float)
        direction = self._normalise(np.asarray(target_dir, dtype=float))

        tasks = [
            IKTask("pos", task_position(self.robot, target_p, tip_offset=offset)),
            IKTask("tool_align", task_tool_z_align(self.robot, direction, tip_offset=offset), weight=0.7),
        ]
        stop = IKStop(max_iters=iters, pos_tol=1e-4, ori_tol=1e-3)
        problem = IKProblem(
            robot=self.robot,
            tasks=tasks,
            q_seed=np.asarray(q_seed, dtype=float),
            limits=self._ik_limits,
            damp=lam,
            stop=stop,
        )
        return solve_ik(problem)

    def build_xy_circle(self, center_xy=(0.6, 0.0), radius: float = 0.15, samples: int = 240) -> np.ndarray:
        """Planar circular path for IK demo."""

        t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        xs = center_xy[0] + radius * np.cos(t)
        ys = center_xy[1] + radius * np.sin(t)
        return np.stack([xs, ys], axis=1)

    def solve_path_ik(
        self,
        q_seed: np.ndarray,
        xy_path: np.ndarray,
        *,
        target_dir: Iterable[float] = (0.0, 0.0, 1.0),
        d6: float = 0.1,
        iters: int = 120,
        lam: float = 2e-2,
    ) -> np.ndarray:
        """Follow an XY path while keeping the tool z-axis aligned."""

        direction = self._normalise(np.asarray(target_dir, dtype=float))
        offset = np.array([0.0, 0.0, d6], dtype=float)
        stop = IKStop(max_iters=iters, pos_tol=1e-4, ori_tol=1e-3)

        q = np.asarray(q_seed, dtype=float).copy()
        qs = []
        for xy in xy_path:
            target_xy = np.asarray(xy, dtype=float)
            tasks = [
                IKTask("xy", task_xy(self.robot, target_xy, tip_offset=offset)),
                IKTask("tool_align", task_tool_z_align(self.robot, direction, tip_offset=offset), weight=0.7),
            ]
            problem = IKProblem(
                robot=self.robot,
                tasks=tasks,
                q_seed=q,
                limits=self._ik_limits,
                damp=lam,
                stop=stop,
            )
            q = solve_ik(problem)
            qs.append(q.copy())
        return np.array(qs)

    def ik_demo(self, samples: int = 240) -> np.ndarray:
        path = self.build_xy_circle(samples=samples)
        q0 = np.zeros(self.robot.dof)
        return self.solve_path_ik(q0, path, target_dir=(0.0, 0.0, 1.0), d6=0.10, iters=120, lam=2e-2)

    # ------------------------------------------------------------------
    # Trajectory optimisation demo (CasADi multiple shooting)
    # ------------------------------------------------------------------
    @staticmethod
    def rpy_to_dir(rpy_rad: Iterable[float]) -> np.ndarray:
        roll, pitch, yaw = np.asarray(rpy_rad, dtype=float)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        R = Rz @ Ry @ Rx
        return R[:, 2]

    def ocp_demo(self) -> OCPResult:
        """Solve the minimum-time joint-space transfer OCP."""

        try:
            import casadi as ca
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("CasADi is required for the OCP demo") from exc

        links = self.config.dh

        def dh_T(a, alpha, d, theta):
            ca_, sa_ = ca.cos(alpha), ca.sin(alpha)
            ct, st = ca.cos(theta), ca.sin(theta)
            return ca.vertcat(
                ca.horzcat(ct, -st * ca_, st * sa_, a * ct),
                ca.horzcat(st, ct * ca_, -ct * sa_, a * st),
                ca.horzcat(0.0, sa_, ca_, d),
                ca.horzcat(0.0, 0.0, 0.0, 1.0),
            )

        T_base = ca.DM(self.config.frames.T_base)
        T_tool = ca.DM(self.config.frames.T_tool)

        def fk_pose(q):
            T = T_base
            for idx, link in enumerate(links):
                qi = q[idx]
                if link.revolute:
                    theta_i = link.theta0 + qi
                    d_i = link.d
                else:
                    theta_i = link.theta0
                    d_i = link.d + qi
                T = ca.mtimes(T, dh_T(link.a, link.alpha, d_i, theta_i))
            return ca.mtimes(T, T_tool)

        def ee_pos(q):
            return fk_pose(q)[0:3, 3]

        def casadi_forward_dyn():
            q = ca.SX.sym("q", self.robot.dof)
            dq = ca.SX.sym("dq", self.robot.dof)
            tau = ca.SX.sym("tau", self.robot.dof)

            p = ee_pos(q)
            Jp = ca.jacobian(p, q)
            m_eff = 3.0
            Fg = ca.vertcat(0, 0, -9.81 * m_eff)
            g_tau = ca.mtimes(Jp.T, Fg)

            B = ca.DM([2.0, 2.0, 1.5, 0.6, 0.4, 0.2])
            Cc = ca.DM([2.5, 2.0, 1.5, 0.8, 0.6, 0.3])
            visc = ca.diag(B) @ dq
            coul = ca.diag(Cc) @ ca.tanh(50 * dq)

            Minv = ca.diag(1.0 / ca.DM([7.0, 6.0, 3.5, 1.2, 0.8, 0.4]))
            ddq = Minv @ (tau - visc - coul - g_tau)
            return ca.Function("fd", [q, dq, tau], [ddq])

        fd = casadi_forward_dyn()

        def rk4_step(xk, uk, dt):
            def f(x, u):
                nq = self.robot.dof
                q, dq = x[0:nq], x[nq : 2 * nq]
                ddq = fd(q, dq, u)
                return ca.vertcat(dq, ddq)

            k1 = f(xk, uk)
            k2 = f(xk + 0.5 * dt * k1, uk)
            k3 = f(xk + 0.5 * dt * k2, uk)
            k4 = f(xk + dt * k3, uk)
            return xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        nq = self.robot.dof
        nx = 2 * nq
        nu = nq
        N = 60
        T_min, T_max = 2.0, 8.0
        T_sym = ca.SX.sym("T")
        dt = T_sym / N
        X = ca.SX.sym("X", nx, N + 1)
        U = ca.SX.sym("U", nu, N)

        pose_start = np.array([0.6, 0.1, 0.50, 0.0, 0.0, 0.0], dtype=float)
        pose_goal = np.array([0.8, 0.3, 0.50, 0.0, 0.0, 90.0], dtype=float)
        dir_start = self.rpy_to_dir(np.deg2rad(pose_start[3:]))
        dir_goal = self.rpy_to_dir(np.deg2rad(pose_goal[3:]))

        q_start = self.ik_xyz_dir(np.zeros(nq), pose_start[:3], dir_start, d6=0.10, iters=120, lam=2e-2)
        q_goal = self.ik_xyz_dir(q_start, pose_goal[:3], dir_goal, d6=0.10, iters=120, lam=2e-2)
        dq_start = np.zeros(nq)
        dq_goal = np.zeros(nq)

        g_constr, g_l, g_u = [], [], []

        g_constr += [X[0:nq, 0] - ca.DM(q_start)]
        g_l += [ca.DM.zeros(nq)]
        g_u += [ca.DM.zeros(nq)]
        g_constr += [X[nq : 2 * nq, 0] - ca.DM(dq_start)]
        g_l += [ca.DM.zeros(nq)]
        g_u += [ca.DM.zeros(nq)]

        for k in range(N):
            x_next = rk4_step(X[:, k], U[:, k], dt)
            g_constr += [X[:, k + 1] - x_next]
            g_l += [ca.DM.zeros(nx)]
            g_u += [ca.DM.zeros(nx)]

        g_constr += [X[0:nq, -1] - ca.DM(q_goal)]
        g_l += [ca.DM.zeros(nq)]
        g_u += [ca.DM.zeros(nq)]
        g_constr += [X[nq : 2 * nq, -1] - ca.DM(dq_goal)]
        g_l += [ca.DM.zeros(nq)]
        g_u += [ca.DM.zeros(nq)]

        W_mech, W_visc, W_i2r = 0.0, 0.0, 0.0
        W_du = 0.0
        eps = 1e-6
        Bvec = ca.DM([2.0, 2.0, 1.5, 0.6, 0.4, 0.2])

        J = 0
        for k in range(N):
            dqk = X[nq : 2 * nq, k]
            uk = U[:, k]
            pow_elem = uk * dqk
            P_mech = ca.sum1(ca.sqrt(pow_elem * pow_elem + eps))
            P_visc = ca.dot(Bvec, dqk * dqk)
            P_i2r = ca.dot(uk, uk)
            stage = W_mech * P_mech + W_visc * P_visc + W_i2r * P_i2r
            J += stage * dt

        J_du = 0
        for k in range(1, N):
            duk = U[:, k] - U[:, k - 1]
            J_du += ca.dot(duk, duk) / dt
        J += W_du * J_du

        w = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)), T_sym)

        q_min, q_max = self.config.limits.q_min, self.config.limits.q_max
        w_l, w_u = [], []
        for _ in range(N + 1):
            w_l += list(q_min)
            w_u += list(q_max)
            w_l += list(-np.deg2rad([356, 356, 296, 409, 480, 1125]))
            w_u += list(np.deg2rad([356, 356, 296, 409, 480, 1125]))
        for _ in range(N):
            w_l += list(-np.array([70, 70, 50, 30, 20, 12]))
            w_u += list(np.array([70, 70, 50, 30, 20, 12]))
        w_l += [T_min]
        w_u += [T_max]
        w_l = ca.DM(w_l)
        w_u = ca.DM(w_u)

        g = ca.vertcat(*[gc.reshape((-1, 1)) for gc in g_constr])
        g_l = ca.DM.zeros(g.size1(), 1)
        g_u = ca.DM.zeros(g.size1(), 1)

        T0 = 0.5 * (T_min + T_max)
        w0 = []
        for k in range(N + 1):
            s = k / float(N)
            sigma = 10 * s**3 - 15 * s**4 + 6 * s**5
            dsigma = (30 * s**2 - 60 * s**3 + 30 * s**4) / T0
            qk = q_start + sigma * (q_goal - q_start)
            dqk = dsigma * (q_goal - q_start)
            w0 += list(qk)
            w0 += list(dqk)
        for _ in range(N):
            w0 += [0.0] * nu
        w0 += [T0]
        w0 = ca.DM(w0)

        nlp = {"x": w, "f": J, "g": g}
        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": 800,
                "mu_strategy": "adaptive",
                "linear_solver": "mumps",
                "tol": 1e-5,
                "acceptable_tol": 1e-4,
            }
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        sol = solver(x0=w0, lbx=w_l, ubx=w_u, lbg=g_l, ubg=g_u)
        w_opt = sol["x"].full().flatten()

        stats = solver.stats()
        print("status:", stats.get("return_status"))
        print("iter:", stats.get("iter_count"))
        print("obj:", float(sol["f"]))
        print("T_opt:", float(w_opt[-1]))

        idx = 0
        X_opt = np.zeros((nx, N + 1))
        U_opt = np.zeros((nu, N))
        for k in range(N + 1):
            X_opt[:, k] = w_opt[idx : idx + nx]
            idx += nx
        for k in range(N):
            U_opt[:, k] = w_opt[idx : idx + nu]
            idx += nu
        T_opt = float(w_opt[idx])

        q_opt = X_opt[0:nq, :].T
        dq_opt = X_opt[nq : 2 * nq, :].T
        tau_opt = U_opt.T
        tgrid = np.linspace(0.0, T_opt, N + 1)
        return OCPResult(q=q_opt, dq=dq_opt, tau=tau_opt, tgrid=tgrid, T=T_opt)

    def save_ocp_result(self, path: Path | str, result: OCPResult) -> Path:
        """Persist the optimal trajectory to an ``.npz`` archive."""

        dest = Path(path)
        if dest.suffix != ".npz":
            dest = dest.with_suffix(".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, q=result.q, dq=result.dq, tau=result.tau, tgrid=result.tgrid, T=result.T)
        return dest

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------
    def animate(self, q_traj: np.ndarray, fps: int = 30, title: str | None = None):
        pts = self.fk_points(q_traj[0])
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        reach = 1.3
        ax.set_xlim([-reach, reach])
        ax.set_ylim([-reach, reach])
        ax.set_zlim([0.0, reach * 1.2])
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title(title or "RX-90L animation")
        link_lines = []
        joint_scatter = ax.scatter([], [], [], s=20)
        tcp_trail, = ax.plot([], [], [], lw=1, alpha=0.5)
        for _ in range(self.robot.dof):
            line, = ax.plot([], [], [], lw=3)
            link_lines.append(line)
        trail_pts: list[np.ndarray] = []

        def init():
            for line in link_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            tcp_trail.set_data([], [])
            tcp_trail.set_3d_properties([])
            return link_lines + [tcp_trail, joint_scatter]

        def update(frame):
            q = q_traj[frame % len(q_traj)]
            pts = self.fk_points(q)
            xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
            for i, line in enumerate(link_lines):
                line.set_data(xs[i : i + 2], ys[i : i + 2])
                line.set_3d_properties(zs[i : i + 2])
            joint_scatter._offsets3d = (xs[:-1], ys[:-1], zs[:-1])
            trail_pts.append(pts[-1])
            tp = np.array(trail_pts[-300:])
            tcp_trail.set_data(tp[:, 0], tp[:, 1])
            tcp_trail.set_3d_properties(tp[:, 2])
            ax.view_init(elev=25, azim=35 + 0.4 * frame)
            return link_lines + [tcp_trail, joint_scatter]

        anim = FuncAnimation(fig, update, frames=len(q_traj), init_func=init, interval=1000 / fps, blit=False)
        plt.show()
        return anim

    def plot_trajectory(self, tgrid: np.ndarray, q: np.ndarray, dq: np.ndarray | None = None, tau: np.ndarray | None = None):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axs[0].plot(tgrid, np.rad2deg(q))
        axs[0].set_ylabel("q [deg]")
        if dq is not None:
            axs[1].plot(tgrid, np.rad2deg(dq))
            axs[1].set_ylabel("dq [deg/s]")
        if tau is not None:
            axs[2].plot(tgrid[:-1], tau)
            axs[2].set_ylabel("tau [Nm]")
        axs[2].set_xlabel("t [s]")
        for ax in axs:
            ax.grid(True)
        plt.tight_layout()
        plt.show()
        return fig
