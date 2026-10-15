#!/usr/bin/env python3
"""
MuJoCo Pick-and-Place Environment – Baseline for Trial #1
(normal friction, normal mass)
"""

import numpy as np
import mujoco
from robot_descriptions import panda_mj_description


class PickPlaceEnv:
    def __init__(self):
        self.table_pos = np.array([0.55, 0.0, 0.25])
        self.table_half_size = np.array([0.175, 0.30, 0.25])
        self.table_top_z = 0.50

        self.cube_half_size = 0.02
        self.cube_spawn_margin = 0.04

        self.tray_pos = np.array([0.55, 0.0, self.table_top_z + 0.01])
        self.tray_half_size = np.array([0.08, 0.08, 0.01])

        self._build_scene()

    def _build_scene(self):
        spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)

        # Table
        table_body = spec.worldbody.add_body(name="table", pos=self.table_pos.tolist())
        table_body.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.table_half_size.tolist(),
            rgba=[0.6, 0.45, 0.3, 1],
            friction=[1.2, 0.5, 0.01],
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.001, 0.5, 2]
        )

        # Cube – baseline friction and mass
        cube_body = spec.worldbody.add_body(
            name="cube",
            pos=[0.55, 0.0, self.table_top_z + self.cube_half_size]
        )
        cube_body.add_freejoint(name="cube_joint")
        cube_body.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.cube_half_size] * 3,
            rgba=[0.9, 0.2, 0.2, 1],
            friction=[8.0, 1, 0.01],
            mass=0.25,
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.001, 0.5, 2]
        )

        # Tray
        tray_body = spec.worldbody.add_body(name="tray", pos=self.tray_pos.tolist())
        tray_body.add_geom(
            name="tray_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.tray_half_size.tolist(),
            rgba=[0.2, 0.6, 0.3, 1],
            contype=0,
            conaffinity=0
        )

        # End-effector site
        hand_body = spec.body("hand")
        hand_body.add_site(
            name="ee_site",
            pos=[0.0, 0.0, 0.105],
            size=[0.006, 0.006, 0.006],
            rgba=[0, 1, 0, 0.7]
        )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.cube_body_id = self.model.body("cube").id
        self.cube_joint_id = self.model.joint("cube_joint").qposadr[0]
        self.hand_body_id = self.model.body("hand").id
        self.ee_site_id = self.model.site("ee_site").id

    def reset(self, cube_xy=None, cube_yaw=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.data.qvel[:] = 0.0
        home_q = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.7, 0.8, 0.04, 0.04])
        self.data.qpos[:9] = home_q

        if cube_xy is None:
            x_min = self.table_pos[0] - self.table_half_size[0] + self.cube_spawn_margin
            x_max = self.table_pos[0] + self.table_half_size[0] - self.cube_spawn_margin
            y_min = self.table_pos[1] - self.table_half_size[1] + self.cube_spawn_margin
            y_max = self.table_pos[1] + self.table_half_size[1] - self.cube_spawn_margin
            cube_x = np.random.uniform(x_min, x_max)
            cube_y = np.random.uniform(y_min, y_max)
        else:
            cube_x, cube_y = cube_xy

        cube_z = self.table_top_z + self.cube_half_size + 0.001
        quat = self._yaw_to_quat(cube_yaw)

        self.data.qpos[self.cube_joint_id:self.cube_joint_id+3] = [cube_x, cube_y, cube_z]
        self.data.qpos[self.cube_joint_id+3:self.cube_joint_id+7] = quat

        mujoco.mj_forward(self.model, self.data)
        return self.get_cube_pose()

    def _yaw_to_quat(self, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        return np.array([cy, 0.0, 0.0, sy])

    def get_cube_pose(self):
        pos = self.data.xpos[self.cube_body_id].copy()
        quat = self.data.xquat[self.cube_body_id]
        yaw = np.arctan2(2*(quat[0]*quat[3] + quat[1]*quat[2]),
                         1 - 2*(quat[2]**2 + quat[3]**2))
        return pos, yaw
