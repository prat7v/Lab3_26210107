#!/usr/bin/env python3
"""
Quadratic Programming IK with joint limits.
Uses qpsolvers (OSQP / DAQP).
"""

import numpy as np
import mujoco
import qpsolvers


class QPIKController:
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
        """
        Position error + orientation error.
        Returns 6D error vector [pos_err; ori_err]
        """
        current_pos = self.data.site_xpos[self.site_id].copy()
        pos_err = target_pos - current_pos

        if target_quat is None:
            ori_err = np.zeros(3)
        else:
            # Current orientation as quaternion (scalar-first)
            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(current_quat, self.data.site_xmat[self.site_id].reshape(9))

            # Orientation error as 3D vector (axis-angle style)
            ori_err = np.zeros(3)
            mujoco.mju_subQuat(ori_err, target_quat, current_quat)

        return np.concatenate([pos_err, ori_err])

    def solve(self, target_pos, target_quat=None):
        """
        Solve the QP:
          min  0.5 * dq^T H dq + c^T dq
          s.t. G dq <= h   (joint limits)
        """
        J = self.get_jacobian()
        err = self.get_pose_error(target_pos, target_quat)

        # Hessian and linear term from regularized tracking cost
        H = J.T @ J + self.lambda_damping * np.eye(7)
        c = -J.T @ err

        # Joint limit inequalities for one step
        q = self.data.qpos[:7]
        q_min = self.model.jnt_range[:7, 0]
        q_max = self.model.jnt_range[:7, 1]

        # dq <= q_max - q
        # -dq <= q - q_min
        G = np.vstack([np.eye(7), -np.eye(7)])
        h = np.concatenate([q_max - q, q - q_min])

        # Scale by dt so limits are meaningful for the velocity
        G = G * self.dt
        # (h stays the same; we solve for velocity then integrate)

        try:
            dq = qpsolvers.solve_qp(H, c, G, h, solver="osqp", verbose=False)
            if dq is None:
                dq = np.linalg.solve(H, -c)   # fallback to unconstrained
        except Exception:
            dq = np.linalg.solve(H, -c)

        return dq

    def move_to(self, target_pos, target_quat=None, n_steps=200, gripper_open=None):
        for _ in range(n_steps):
            dq = self.solve(target_pos, target_quat)
            self.data.qpos[:7] += dq * self.dt

            # Hard clip as extra safety
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
