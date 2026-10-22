#!/usr/bin/env python3
"""Runnable demo: gradient-descent path planning in a non-uniform river flow.

This implementation follows the inverse-square source/sink flow expression in
Simple_Path_Planning_Report.pdf. The submitted source script is preserved in
original/Path_Planning_Nonuniform.py for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RiverParameters:
    """Parameters describing the river and its source/sink flow field."""

    width: float = 10.0
    x1: float = -1.0
    y1: float = 1.0
    x2: float = 1.0
    y2: float = -1.0
    source_strength: float = 2.0
    downstream_speed: float = 0.5


def river_flow(
    x: float | np.ndarray,
    y: float | np.ndarray,
    params: RiverParameters,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return (vx, vy) for the source/sink pair plus uniform downstream flow.

    The distance-squared denominator follows Equation (15) in the report.
    A tiny floor prevents division by zero if a point lands exactly on a source
    or sink, where the idealized point-flow model is singular.
    """
    r1_sq = np.maximum((x - params.x1) ** 2 + (y - params.y1) ** 2, 1e-8)
    r2_sq = np.maximum((x - params.x2) ** 2 + (y - params.y2) ** 2, 1e-8)
    factor = params.source_strength / (2.0 * np.pi)
    vx = factor * ((x - params.x1) / r1_sq - (x - params.x2) / r2_sq)
    vy = -params.downstream_speed + factor * (
        (y - params.y1) / r1_sq - (y - params.y2) / r2_sq
    )
    return vx, vy


def simulate_path(
    angles: np.ndarray,
    dt: float,
    swimmer_speed: float,
    target_y: float,
    params: RiverParameters,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Simulate one angle sequence and return cost, endpoint and full path."""
    x = -params.width / 2.0
    y = 0.0
    x_history = [x]
    y_history = [y]

    for theta in angles:
        flow_x, flow_y = river_flow(x, y, params)
        x += dt * (swimmer_speed * np.cos(theta) + flow_x)
        y += dt * (swimmer_speed * np.sin(theta) + flow_y)
        x_history.append(float(x))
        y_history.append(float(y))

    cost = (x - params.width / 2.0) ** 2 + (y - target_y) ** 2
    return (
        float(cost),
        float(x),
        float(y),
        np.asarray(x_history),
        np.asarray(y_history),
    )


def _batch_costs(
    angle_sequences: np.ndarray,
    dt: float,
    swimmer_speed: float,
    target_y: float,
    params: RiverParameters,
) -> np.ndarray:
    """Evaluate many finite-difference perturbations in parallel with NumPy."""
    batch_size, n_steps = angle_sequences.shape
    x = np.full(batch_size, -params.width / 2.0, dtype=float)
    y = np.zeros(batch_size, dtype=float)

    for k in range(n_steps):
        flow_x, flow_y = river_flow(x, y, params)
        theta = angle_sequences[:, k]
        x += dt * (swimmer_speed * np.cos(theta) + flow_x)
        y += dt * (swimmer_speed * np.sin(theta) + flow_y)

    return (x - params.width / 2.0) ** 2 + (y - target_y) ** 2


def optimize_angles(
    *,
    initial_angles: np.ndarray,
    dt: float,
    swimmer_speed: float,
    target_y: float,
    params: RiverParameters,
    learning_rate: float,
    fd_step: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, list[float], bool]:
    """Use central finite differences and gradient descent to optimize angles."""
    angles = initial_angles.astype(float, copy=True)
    n_steps = len(angles)
    cost_history: list[float] = []
    converged = False

    for _iteration in range(max_iterations):
        # Row 2*i perturbs angle i upward; row 2*i+1 perturbs it downward.
        perturbed = np.repeat(angles[np.newaxis, :], 2 * n_steps, axis=0)
        indices = np.arange(n_steps)
        perturbed[2 * indices, indices] += fd_step
        perturbed[2 * indices + 1, indices] -= fd_step
        costs = _batch_costs(perturbed, dt, swimmer_speed, target_y, params)
        gradient = (costs[0::2] - costs[1::2]) / (2.0 * fd_step)

        angles -= learning_rate * gradient
        cost, *_ = simulate_path(angles, dt, swimmer_speed, target_y, params)
        distance_error = float(np.sqrt(cost))
        cost_history.append(cost)

        if distance_error <= tolerance:
            converged = True
            break

    return angles, cost_history, converged


def make_plots(
    angles: np.ndarray,
    cost_history: list[float],
    path_x: np.ndarray,
    path_y: np.ndarray,
    *,
    params: RiverParameters,
    target_y: float,
    swimmer_speed: float,
    save_dir: Path | None,
    show: bool,
) -> None:
    """Plot convergence, the river field and the optimized path."""
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Convergence plot
    fig1, ax1 = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax1.plot(np.arange(1, len(cost_history) + 1), np.sqrt(cost_history), linewidth=2)
    ax1.set_xlabel("Gradient-descent iteration")
    ax1.set_ylabel("Distance to target (m)")
    ax1.set_title("Optimization convergence")
    ax1.grid(True, alpha=0.3)
    if save_dir is not None:
        fig1.savefig(save_dir / "convergence.png", dpi=180)

    # Flow field and swimmer's path
    grid_x, grid_y = np.meshgrid(
        np.linspace(-params.width / 2.0, params.width / 2.0, 120),
        np.linspace(-5.0, 7.0, 120),
    )
    field_x, field_y = river_flow(grid_x, grid_y, params)
    fig2, ax2 = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax2.streamplot(grid_x, grid_y, field_x, field_y, density=1.7, linewidth=0.8, arrowsize=0.8)
    ax2.plot(path_x, path_y, linewidth=2.5, label="Optimized swimmer path")
    ax2.scatter(
        [-params.width / 2.0, params.width / 2.0],
        [0.0, target_y],
        s=65,
        zorder=5,
        label="Start and target",
    )
    ax2.scatter([params.x1, params.x2], [params.y1, params.y2], marker="x", s=70, zorder=6,
                label="Source / sink locations")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Trajectory in a spatially varying river flow")
    ax2.set_xlim(-params.width / 2.0, params.width / 2.0)
    ax2.set_ylim(-5.0, 7.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc="best", fontsize=8)
    if save_dir is not None:
        fig2.savefig(save_dir / "nonuniform-flow-path.png", dpi=180)

    # Relative swimming heading versus ground-velocity direction.
    step_indices = np.arange(0, len(angles), max(1, len(angles) // 15))
    x_sample = path_x[step_indices]
    y_sample = path_y[step_indices]
    theta_sample = angles[step_indices]
    flow_x, flow_y = river_flow(x_sample, y_sample, params)
    total_x = swimmer_speed * np.cos(theta_sample) + flow_x
    total_y = swimmer_speed * np.sin(theta_sample) + flow_y
    total_norm = np.maximum(np.hypot(total_x, total_y), 1e-9)
    total_x, total_y = total_x / total_norm, total_y / total_norm

    fig3, ax3 = plt.subplots(figsize=(8.5, 6.0), constrained_layout=True)
    ax3.streamplot(grid_x, grid_y, field_x, field_y, density=1.7, linewidth=0.8, arrowsize=0.8)
    ax3.plot(path_x, path_y, linewidth=2.0, label="Optimized path")
    ax3.quiver(x_sample, y_sample, np.cos(theta_sample), np.sin(theta_sample),
               angles="xy", scale_units="xy", scale=1.5, width=0.004,
               label="Swimmer heading")
    ax3.quiver(x_sample, y_sample, total_x, total_y, angles="xy", scale_units="xy", scale=1.5,
               width=0.004, label="Ground-velocity direction")
    ax3.scatter([-params.width / 2.0, params.width / 2.0], [0.0, target_y], s=55, zorder=5)
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_title("Swimmer heading and resulting direction of motion")
    ax3.set_xlim(-params.width / 2.0, params.width / 2.0)
    ax3.set_ylim(-5.0, 7.0)
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, alpha=0.2)
    ax3.legend(loc="best", fontsize=8)
    if save_dir is not None:
        fig3.savefig(save_dir / "heading-vs-ground-velocity.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a swimmer's time-dependent heading in a non-uniform river flow."
    )
    parser.add_argument("--width", type=float, default=10.0, help="River width in metres (default: 10).")
    parser.add_argument("--target-y", type=float, default=0.0,
                        help="Target y-coordinate in metres (default: 0, matching the submitted script).")
    parser.add_argument("--swimmer-speed", type=float, default=1.0,
                        help="Constant swimmer speed relative to the water, in m/s (default: 1).")
    parser.add_argument("--downstream-speed", type=float, default=0.5,
                        help="Uniform downstream component V, in m/s (default: 0.5).")
    parser.add_argument("--source-strength", type=float, default=2.0,
                        help="Source/sink strength Q (default: 2).")
    parser.add_argument("--total-time", type=float, default=12.7,
                        help="Fixed simulation horizon in seconds (default: 12.7).")
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of discrete heading controls (default: 100).")
    parser.add_argument("--learning-rate", type=float, default=0.01,
                        help="Gradient-descent learning rate (default: 0.01).")
    parser.add_argument("--initial-angle", type=float, default=0.87,
                        help="Initial heading for all steps, in radians (default: 0.87).")
    parser.add_argument("--fd-step", type=float, default=1e-3,
                        help="Central finite-difference step in radians (default: 0.001).")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Stop when endpoint distance to target is at most this many metres (default: 0.1).")
    parser.add_argument("--max-iterations", type=int, default=400,
                        help="Maximum gradient-descent iterations (default: 400).")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Optional directory in which to save the plots as PNG files.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save requested plots and exit without opening plot windows.")
    args = parser.parse_args()

    if args.width <= 0 or args.total_time <= 0 or args.steps < 1:
        parser.error("--width and --total-time must be positive, and --steps must be at least 1.")
    if args.swimmer_speed <= 0 or args.fd_step <= 0 or args.learning_rate <= 0:
        parser.error("--swimmer-speed, --fd-step, and --learning-rate must be positive.")
    if args.tolerance < 0 or args.max_iterations < 1:
        parser.error("--tolerance must be non-negative and --max-iterations must be at least 1.")
    return args


def main() -> int:
    args = parse_args()
    params = RiverParameters(
        width=args.width,
        source_strength=args.source_strength,
        downstream_speed=args.downstream_speed,
    )
    dt = args.total_time / args.steps
    initial_angles = np.full(args.steps, args.initial_angle, dtype=float)

    angles, costs, converged = optimize_angles(
        initial_angles=initial_angles,
        dt=dt,
        swimmer_speed=args.swimmer_speed,
        target_y=args.target_y,
        params=params,
        learning_rate=args.learning_rate,
        fd_step=args.fd_step,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    cost, final_x, final_y, path_x, path_y = simulate_path(
        angles, dt, args.swimmer_speed, args.target_y, params
    )
    error = float(np.sqrt(cost))

    print("\nNon-uniform river-flow path-planning demo")
    print("-----------------------------------------")
    print(f"Iterations performed : {len(costs)} / {args.max_iterations}")
    print(f"Final position       : ({final_x:.3f}, {final_y:.3f}) m")
    print(f"Target position      : ({params.width / 2.0:.3f}, {args.target_y:.3f}) m")
    print(f"Distance to target   : {error:.3f} m")
    print(f"Status               : {'tolerance reached' if converged else 'maximum iterations reached'}")
    print(f"Mean selected heading: {np.degrees(np.mean(angles)):.2f} degrees")
    if args.save_dir is not None:
        print(f"Plots saved to       : {args.save_dir.resolve()}")

    make_plots(
        angles,
        costs,
        path_x,
        path_y,
        params=params,
        target_y=args.target_y,
        swimmer_speed=args.swimmer_speed,
        save_dir=args.save_dir,
        show=not args.no_show,
    )
    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
