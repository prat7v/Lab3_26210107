#!/usr/bin/env python3
"""
Closed-form Damped Least Squares IK (DLS-IK)
"""

import numpy as np
import mujoco


class DLSIKController:
    def __init__(self, model, data, site_name="ee_site", lambda_damping=0.05):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = model.site(site_name).id
        self.lambda_damping = lambda_damping
        self.dt = 0.002

    def get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        J = np.vstack((jacp[:, :7], jacr[:, :7]))
        return J

    def get_pose_error(self, target_pos, target_quat=None):
        current_pos = self.data.site_xpos[self.site_id].copy()
        pos_err = target_pos - current_pos

        if target_quat is None:
            ori_err = np.zeros(3)
        else:
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.site_id].reshape(9))
            ori_err = np.zeros(3)
            mujoco.mju_subQuat(ori_err, target_quat, current_quat)

        return np.concatenate([pos_err, ori_err])

    def solve(self, target_pos, target_quat=None):
        J = self.get_jacobian()
        err = self.get_pose_error(target_pos, target_quat)
        JTJ = J.T @ J
        damping = self.lambda_damping * np.eye(7)
        q_dot = np.linalg.solve(JTJ + damping, J.T @ err)
        return q_dot

    def move_to(self, target_pos, target_quat=None, n_steps=200, gripper_open=None):
        for _ in range(n_steps):
            q_dot = self.solve(target_pos, target_quat)
            self.data.qpos[:7] += q_dot * self.dt

            for i in range(7):
                self.data.qpos[i] = np.clip(
                    self.data.qpos[i],
                    self.model.jnt_range[i, 0],
                    self.model.jnt_range[i, 1]
                )

            if gripper_open is not None:
                self.data.qpos[7] = gripper_open
                self.data.qpos[8] = gripper_open
                if self.model.nu >= 8:
                    self.data.ctrl[7] = gripper_open * 255.0

            mujoco.mj_step(self.model, self.data)
