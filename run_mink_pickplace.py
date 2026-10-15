#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment_heal import HealPickPlaceEnv
from src.mink_heal import HealMinkController


def set_arm_configuration(env, q):
    for i, value in enumerate(q):
        joint_name = f"joint_{i + 1}"

        joint_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")

        adr = env.model.jnt_qposadr[joint_id]
        env.data.qpos[adr] = value

    mujoco.mj_forward(env.model, env.data)


def get_ee_position(env):
    return env.data.site_xpos[env.ee_site_id].copy()


def get_cube_position(env):
    return env.get_cube_pose()[0]


def print_status(env, label):
    ee = get_ee_position(env)
    cube = get_cube_position(env)

    xy_error = np.linalg.norm(ee[:2] - cube[:2])

    print(
        f"{label:20s} | "
        f"EE z={ee[2]:.4f} | "
        f"Cube z={cube[2]:.4f} | "
        f"XY error={xy_error:.4f}"
    )


def move_to_target(
    env,
    ctrl,
    viewer,
    target,
    quat,
    steps,
    gripper_open,
    label
):
    print(f"\n→ {label}")

    for i in range(steps):
        ctrl.move_to(
            target,
            quat,
            n_steps=1,
            gripper_open=gripper_open
        )

        if i % 25 == 0 or i == steps - 1:
            mujoco.mj_forward(env.model, env.data)

            ee = get_ee_position(env)
            cube = get_cube_position(env)
            error = np.linalg.norm(target - ee)

            print(
                f"Step {i:3d} | "
                f"EE z={ee[2]:.4f} | "
                f"Cube z={cube[2]:.4f} | "
                f"Position error={error:.4f}"
            )

        viewer.sync()
        time.sleep(0.01)

    print_status(env, label)


def main():
    env = HealPickPlaceEnv()

    cube_pos, _ = env.reset(
        cube_xy=(0.615, -0.020),
        cube_yaw=0.0,
        seed=42
    )

    print("Start cube:", cube_pos.round(4))

    initial_q = [0.0, -0.3, 0.5, 0.0, 0.4, 0.0]
    set_arm_configuration(env, initial_q)

    ctrl = HealMinkController(env.model, env.data)

    ctrl.ee_task.position_cost = 100.0
    ctrl.ee_task.orientation_cost = 0.05

    ee_rotation = env.data.site_xmat[
        env.ee_site_id
    ].reshape(3, 3).copy()

    q_xyzw = R.from_matrix(ee_rotation).as_quat()

    down_quat = np.array([
        q_xyzw[3],
        q_xyzw[0],
        q_xyzw[1],
        q_xyzw[2]
    ])

    with mujoco.viewer.launch_passive(
        env.model,
        env.data
    ) as viewer:

        for _ in range(40):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        print_status(env, "Initial")

        # Stage 1: High approach
        target = cube_pos.copy()
        target[2] = env.table_top_z + 0.25

        move_to_target(
            env, ctrl, viewer,
            target, down_quat,
            180, 0.03, "High approach"
        )

        # Stage 2: Mid approach
        target[2] = env.table_top_z + 0.14

        move_to_target(
            env, ctrl, viewer,
            target, down_quat,
            120, 0.03, "Mid approach"
        )

        # Stage 3: Final approach
        cube_pos, _ = env.get_cube_pose()
        target = cube_pos.copy()

        move_to_target(
            env, ctrl, viewer,
            target, down_quat,
            250, 0.03, "Final approach"
        )

        # Stage 4: Close gripper
        print("\n→ Close gripper")

        cube_before_close = get_cube_position(env)

        for i in range(100):
            opening = 0.03 * (1.0 - i / 99.0)

            ctrl.move_to(
                target,
                down_quat,
                n_steps=1,
                gripper_open=opening
            )

            if i % 20 == 0 or i == 99:
                mujoco.mj_forward(env.model, env.data)

                cube = get_cube_position(env)
                ee = get_ee_position(env)

                print(
                    f"Close step {i:3d} | "
                    f"EE z={ee[2]:.4f} | "
                    f"Cube z={cube[2]:.4f} | "
                    f"Cube displacement="
                    f"{np.linalg.norm(cube - cube_before_close):.4f}"
                )

            viewer.sync()
            time.sleep(0.01)

        print_status(env, "After close")

        cube_after_close = get_cube_position(env)

        print(
            "Cube displacement during closing:",
            np.linalg.norm(
                cube_after_close - cube_before_close
            ).round(4)
        )

        # Stage 5: Stabilization
        print("\n→ Grasp stabilization")

        for _ in range(100):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        cube_after_pause = get_cube_position(env)

        print(
            "Cube displacement during pause:",
            np.linalg.norm(
                cube_after_pause - cube_after_close
            ).round(4)
        )

        # Stage 6: Gradual lift
        print("\n→ Lift")

        lift_target = target.copy()
        lift_target[2] = env.table_top_z + 0.25

        cube_before_lift = get_cube_position(env)

        for i in range(200):
            fraction = (i + 1) / 200.0

            intermediate_target = target.copy()
            intermediate_target[2] = (
                target[2]
                + fraction * (
                    lift_target[2] - target[2]
                )
            )

            ctrl.move_to(
                intermediate_target,
                down_quat,
                n_steps=1,
                gripper_open=0.0
            )

            if i % 20 == 0 or i == 199:
                mujoco.mj_forward(env.model, env.data)

                ee = get_ee_position(env)
                cube = get_cube_position(env)

                print(
                    f"Lift step {i:3d} | "
                    f"EE z={ee[2]:.4f} | "
                    f"Cube z={cube[2]:.4f} | "
                    f"Cube rise="
                    f"{cube[2] - cube_before_lift[2]:.4f}"
                )

            viewer.sync()
            time.sleep(0.01)

        final_cube = get_cube_position(env)
        final_ee = get_ee_position(env)

        print("\n" + "=" * 55)
        print("PICK-AND-PLACE RESULTS")
        print("=" * 55)
        print("Initial cube position:", cube_pos.round(4))
        print("Cube before lift:", cube_before_lift.round(4))
        print("Final cube position:", final_cube.round(4))
        print("Final EE position:", final_ee.round(4))
        print(
            "Cube vertical displacement:",
            round(final_cube[2] - cube_before_lift[2], 4)
        )
        print("=" * 55)

        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
