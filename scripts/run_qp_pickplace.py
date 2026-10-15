#!/usr/bin/env python3
"""
Generalised QP-IK Pick-and-Place script
Supports physical grasp trials and kinematic attachment.
"""

import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import mujoco
import mujoco.viewer
from src.environment import PickPlaceEnv
from src.ik_qp import QPIKController

# ===================== CONFIG =====================
TRIAL_NAME = "QP Trial 3 – Kinematic Attachment"
USE_KINEMATIC = True
CLOSE_AMOUNT = 0.020
GRASP_Z_OFFSET = 0.025
BIAS_X = 0.09
LAMBDA_DAMPING = 0.12
# ==================================================

print(f"=== {TRIAL_NAME} ===")
print(f"Kinematic attachment: {USE_KINEMATIC}")
print("Imports OK")


def attach_cube(env):
    env.grasp_offset = np.array([0.0, 0.0, -0.038])
    env.grasped = True
    print("  [Cube attached]")


def update_cube(env):
    if not getattr(env, "grasped", False):
        return
    hand_pos = env.data.xpos[env.hand_body_id].copy()
    new_pos = hand_pos + env.grasp_offset
    qadr = env.cube_joint_id
    env.data.qpos[qadr:qadr+3] = new_pos
    env.data.qvel[qadr:qadr+6] = 0.0


def release_cube(env):
    env.grasped = False
    print("  [Cube released]")


def main():
    env = PickPlaceEnv()
    env.grasped = False
    controller = QPIKController(env.model, env.data, lambda_damping=LAMBDA_DAMPING)

    cube_pos, _ = env.reset(seed=42)
    print(f"Start cube: {cube_pos.round(4)}")

    down_quat = np.array([0.0, 0.7071, 0.0, 0.7071])
    table_z = env.table_top_z
    high_z = table_z + 0.18
    mid_z = table_z + 0.08
    grasp_z = table_z + GRASP_Z_OFFSET

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(30):
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.01)

        env.data.qpos[7:9] = 0.04
        mujoco.mj_forward(env.model, env.data)

        # Approach
        target = cube_pos.copy()
        target[2] = high_z
        print("→ High")
        controller.move_to(target, down_quat, n_steps=120, gripper_open=0.04)
        time.sleep(0.2)

        target[2] = mid_z
        print("→ Mid")
        controller.move_to(target, down_quat, n_steps=80, gripper_open=0.04)
        time.sleep(0.2)

        target[2] = grasp_z
        print("→ Descent")
        controller.move_to(target, down_quat, n_steps=90, gripper_open=0.04)
        time.sleep(0.4)

        # Close
        print(f"→ Closing to {CLOSE_AMOUNT}")
        frozen = env.data.qpos[:7].copy()
        for i in range(100):
            env.data.qpos[:7] = frozen
            env.data.qvel[:7] = 0.0
            open_amt = 0.04 - (0.04 - CLOSE_AMOUNT) * min(1.0, i / 70)
            env.data.qpos[7:9] = open_amt
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.012)
        time.sleep(0.6)

        if USE_KINEMATIC:
            attach_cube(env)
            update_cube(env)
            mujoco.mj_forward(env.model, env.data)
            time.sleep(0.4)

        # Lift
        target[2] = high_z
        print("→ Lift")
        if USE_KINEMATIC:
            for _ in range(80):
                update_cube(env)
                controller.move_to(target, down_quat, n_steps=1, gripper_open=CLOSE_AMOUNT)
                update_cube(env)
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
        else:
            controller.move_to(target, down_quat, n_steps=140, gripper_open=CLOSE_AMOUNT)
        time.sleep(0.3)

        # Above tray
        target = env.tray_pos.copy()
        target[0] += BIAS_X
        target[2] = high_z
        print("→ Above tray")
        if USE_KINEMATIC:
            for _ in range(90):
                update_cube(env)
                controller.move_to(target, down_quat, n_steps=1, gripper_open=CLOSE_AMOUNT)
                update_cube(env)
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
        else:
            controller.move_to(target, down_quat, n_steps=150, gripper_open=CLOSE_AMOUNT)
        time.sleep(0.3)

        # Place
        target[2] = table_z + 0.035
        print("→ Place")
        if USE_KINEMATIC:
            for _ in range(60):
                update_cube(env)
                controller.move_to(target, down_quat, n_steps=1, gripper_open=CLOSE_AMOUNT)
                update_cube(env)
                mujoco.mj_forward(env.model, env.data)
                viewer.sync()
        else:
            controller.move_to(target, down_quat, n_steps=90, gripper_open=CLOSE_AMOUNT)
        time.sleep(0.5)

        # Release
        print("→ Release")
        if USE_KINEMATIC:
            release_cube(env)
        for i in range(80):
            open_amt = CLOSE_AMOUNT + (0.04 - CLOSE_AMOUNT) * (i / 80)
            env.data.qpos[7:9] = open_amt
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(0.015)
        time.sleep(0.4)

        # Retreat
        print("→ Retreat")
        retreat = target.copy()
        retreat[2] = high_z
        for _ in range(100):
            controller.move_to(retreat, down_quat, n_steps=1, gripper_open=0.04)
            viewer.sync()
            time.sleep(0.012)

        final, _ = env.get_cube_pose()
        tray_xy = env.tray_pos[:2]
        offset = final[:2] - tray_xy
        print(f"FINAL CUBE: {final.round(4)}")
        print(f"Offset from tray centre (XY): {offset.round(4)}")
        print(f"{TRIAL_NAME} finished.")

        while viewer.is_running():
            if USE_KINEMATIC and env.grasped:
                update_cube(env)
            mujoco.mj_step(env.model, env.data)
            viewer.sync()


if __name__ == "__main__":
    main()
