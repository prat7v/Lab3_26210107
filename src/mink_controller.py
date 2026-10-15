#!/usr/bin/env python3
"""
Mink Controller with strong position tracking
"""

import numpy as np
import mujoco
import mink
from mink import SE3, SO3


class MinkPickPlaceController:
    def __init__(self, model, data, ee_site_name="ee_site"):
        self.model = model
        self.data = data
        self.configuration = mink.Configuration(model)

        # Strong position, weak orientation, very weak posture
        self.ee_task = mink.FrameTask(
            frame_name=ee_site_name,
            frame_type="site",
            position_cost=5.0,          # much stronger
            orientation_cost=0.1,       # weaker
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=1e-3)   # weaker

        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model)]

        self.solver = "daqp"
        self.dt = 0.002
        self.max_ik_iters = 12          # more iterations

    def update(self):
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def set_gripper(self, open_amount: float):
        self.data.qpos[7] = open_amount
        self.data.qpos[8] = open_amount
        if self.model.nu >= 8:
            self.data.ctrl[7] = open_amount * 255.0

    def open_gripper(self):
        self.set_gripper(0.04)

    def close_gripper(self):
        self.set_gripper(0.0)

    def move_to(self, target_pos, target_quat=None, n_steps=300, gripper_open=None):
        self.update()

        if target_quat is None:
            current = self.configuration.get_transform_frame_to_world("ee_site", "site")
            target_se3 = SE3.from_rotation_and_translation(current.rotation(), np.asarray(target_pos))
        else:
            rot = SO3(wxyz=np.asarray(target_quat))
            target_se3 = SE3.from_rotation_and_translation(rot, np.asarray(target_pos))

        self.ee_task.set_target(target_se3)

        for _ in range(n_steps):
            self.update()

            for _ in range(self.max_ik_iters):
                try:
                    vel = mink.solve_ik(
                        self.configuration,
                        self.tasks,
                        dt=self.dt,
                        solver=self.solver,
                        damping=1e-2,
                        limits=self.limits,
                    )
                    self.configuration.integrate_inplace(vel, self.dt)
                except Exception:
                    break

            self.data.qpos[:7] = self.configuration.q[:7]

            if gripper_open is not None:
                self.set_gripper(gripper_open)

            mujoco.mj_step(self.model, self.data)
