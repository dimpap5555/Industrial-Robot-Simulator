"""RX-90L demos executed directly without a command-line parser."""

from __future__ import annotations

import numpy as np

from rx90l import Rx90L


def main() -> None:
    robot = Rx90L()

    # Uncomment the following block to run the forward kinematics demo.
    T_fk = 2.0
    fps = 30
    q_fk = robot.forward_demo(T_final=T_fk, fps=fps)
    t_fk = np.linspace(0.0, T_fk, q_fk.shape[0])
    robot.plot_trajectory(t_fk, q_fk)
    robot.animate(q_fk, title="Forward kinematics demo")

    # Uncomment to run the inverse kinematics demo.
    q_ik = robot.ik_demo(samples=120)
    t_ik = np.linspace(0.0, q_ik.shape[0] / fps, q_ik.shape[0])
    robot.plot_trajectory(t_ik, q_ik)
    robot.animate(q_ik, title="Inverse kinematics demo")

    # Run the optimal control demo directly from code.
    try:
        result = robot.ocp_demo()
    except RuntimeError as exc:
        print(exc)
        print("Install CasADi (pip install casadi) to enable the OCP demo.")
    else:
        robot.plot_trajectory(result.tgrid, result.q, result.dq, result.tau)
        robot.animate(result.q, title="OCP demo")


if __name__ == "__main__":
    main()
