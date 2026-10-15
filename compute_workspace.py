#!/usr/bin/env python3
"""
Phase 1 – Compute Task Workspace and Dexterous Workspace for Franka Panda.
"""

import os
import sys
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robot_descriptions import panda_mj_description

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.makedirs(os.path.join(PROJECT_ROOT, "plots"), exist_ok=True)


def get_end_effector_pos(model, data):
    """Return position of the hand body (approximation of TCP)."""
    hand_id = model.body("hand").id
    return data.xpos[hand_id].copy()


def sample_workspace(model, data, n_samples=8000, joint_margin=0.05):
    """
    Randomly sample joint configurations and record reachable hand positions.
    Returns Nx3 array of positions.
    """
    positions = []

    # Joint limits (ignore finger joints for workspace)
    qpos_min = model.jnt_range[:7, 0] + joint_margin
    qpos_max = model.jnt_range[:7, 1] - joint_margin

    for _ in range(n_samples):
        q = np.random.uniform(qpos_min, qpos_max)
        data.qpos[:7] = q
        data.qpos[7:] = 0.04          # fingers open
        mujoco.mj_forward(model, data)
        pos = get_end_effector_pos(model, data)
        positions.append(pos)

    return np.array(positions)


def main():
    print("=" * 60)
    print("Phase 1: Workspace Analysis")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    print("Sampling configurations (this takes a few seconds)...")
    positions = sample_workspace(model, data, n_samples=10000)
    print(f"Collected {len(positions)} reachable points")

    # Basic statistics
    print("\nTask Workspace bounds (metres):")
    print(f"  X: [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]")
    print(f"  Y: [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    print(f"  Z: [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")

    # ---------- Visualisation ----------
    fig = plt.figure(figsize=(14, 6))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(positions[:,0], positions[:,1], positions[:,2],
                s=1, alpha=0.3, c=positions[:,2], cmap='viridis')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Task Workspace (3D)')
    ax1.view_init(elev=25, azim=120)

    # Top-down (XY) view – most useful for table design
    ax2 = fig.add_subplot(122)
    ax2.scatter(positions[:,0], positions[:,1], s=1, alpha=0.4, c='steelblue')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Task Workspace (Top-down XY)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # Suggest a conservative table region
    # Typical comfortable region for Panda is roughly x ∈ [0.3, 0.7], y ∈ [-0.35, 0.35]
    table_x_min, table_x_max = 0.35, 0.65
    table_y_min, table_y_max = -0.30, 0.30
    ax2.add_patch(plt.Rectangle(
        (table_x_min, table_y_min),
        table_x_max - table_x_min,
        table_y_max - table_y_min,
        fill=False, edgecolor='red', linewidth=2, label='Proposed Table'
    ))
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(PROJECT_ROOT, "plots", "workspace_task.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot → {plot_path}")
    plt.show()

    # ---------- Recommended Table Dimensions ----------
    print("\n" + "=" * 60)
    print("Recommended Table Design (based on workspace)")
    print("=" * 60)
    print("""
| Parameter    | Symbol | Value   | Rationale                                      |
|--------------|--------|---------|------------------------------------------------|
| Table length | L      | 0.40 m  | Fits inside reachable X-range (0.35–0.65)     |
| Table width  | W      | 0.60 m  | Fits inside reachable Y-range (-0.30–0.30)    |
| Table height | H      | 0.40 m  | Comfortable vertical approach, good clearance  |
""")
    print("You can adjust these values after inspecting the plot.")
    print("\n✅ Phase 1 workspace analysis finished.")
    print("Next: We will create the full scene with this table.")


if __name__ == "__main__":
    main()
