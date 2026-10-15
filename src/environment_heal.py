import os
import mujoco
import numpy as np

class HealPickPlaceEnv:
    def __init__(self):
        self.table_pos = np.array([0.50, 0.0, 0.25])
        self.table_half_size = np.array([0.22, 0.28, 0.25])
        self.table_top_z = 0.50
        self.cube_half_size = 0.02
        self.cube_spawn_margin = 0.05
        self.tray_pos = np.array([0.50, 0.0, self.table_top_z + 0.01])
        self.tray_half_size = np.array([0.10, 0.10, 0.01])
        self._build_scene()

    def _build_scene(self):
        heal_xml = "/home/pratyaksh/Dual_Arm_Manipulation-Addverb_and_IITGN_Robotics/Other/Addverb_RL/heal/robot.xml"

        spec = mujoco.MjSpec.from_file(heal_xml)

        # Table
        table_body = spec.worldbody.add_body(name="table", pos=self.table_pos.tolist())
        table_body.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.table_half_size.tolist(),
            rgba=[0.6, 0.45, 0.3, 1],
            friction=[1.5, 0.5, 0.01],
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.001, 0.5, 0.2]
        )

        # Tray
        tray_body = spec.worldbody.add_body(name="tray", pos=self.tray_pos.tolist())
        tray_body.add_geom(
            name="tray_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=self.tray_half_size.tolist(),
            rgba=[0.2, 0.6, 0.9, 1],
            friction=[1.0, 0.4, 0.01],
            solref=[0.02, 1],
            solimp=[0.9, 0.95, 0.001, 0.5, 0.2]
        )

        # Cube
        cube_body = spec.worldbody.add_body(
            name="cube",
            pos=[0.50, 0.0, self.table_top_z + self.cube_half_size]
        )
        cube_body.add_freejoint(name="cube_joint")
        cube_body.mass = 2
        cube_body.ipos = [0, 0, 0]
        cube_body.inertia = [0.0001, 0.0001, 0.0001]
        cube_body.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.cube_half_size] * 3,
            rgba=[0.9, 0.2, 0.2, 1],
            friction=[3.0, 0.8, 0.01],
            solref=[0.01, 1],
            solimp=[0.95, 0.99, 0.001, 0.5, 0.2]
        )

        # Find right_hand
        right_hand = None
        for body in spec.bodies:
            if body.name == "right_hand":
                right_hand = body
                break
        if right_hand is None:
            raise RuntimeError("right_hand body not found")

        # Simple two-finger gripper
        gripper_body = right_hand.add_body(name="gripper", pos=[0, 0, 0.03])
        gripper_body.mass = 0.1
        gripper_body.ipos = [0, 0, 0]
        gripper_body.inertia = [0.001, 0.001, 0.001]
        gripper_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.04, 0.03, 0.05],
            rgba=[0.3, 0.3, 0.3, 1],
            name="gripper_base"
        )

        # Left finger
        left = gripper_body.add_body(name="left_finger", pos=[0, 0.04, 0.06])
        left.mass = 0.02
        left.ipos = [0, 0, 0]
        left.inertia = [1e-5, 1e-5, 1e-5]
        left.add_joint(name="left_finger_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], range=[0, 0.04])
        left.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.04], rgba=[0.7, 0.7, 0.7, 1], friction=[10.0,1.5,0.01])

        # Right finger
        right = gripper_body.add_body(name="right_finger", pos=[0, -0.04, 0.06])
        right.mass = 0.02
        right.ipos = [0, 0, 0]
        right.inertia = [1e-5, 1e-5, 1e-5]
        right.add_joint(name="right_finger_joint", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -1, 0], range=[0, 0.04])
        right.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.04], rgba=[0.7, 0.7, 0.7, 1], friction=[10.0,1.5,0.01])

        # EE site
        gripper_body.add_site(
            name="ee_site",
            pos=[0, 0, 0.10],
            size=[0.008],
            rgba=[0, 1, 0, 1]
        )

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_joint_id = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        ]

        self.gripper_act_id = None
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name and ("finger" in name.lower() or "gripper" in name.lower()):
                self.gripper_act_id = i
                break

    def reset(self, seed=None, cube_xy=None, cube_yaw=0.0):
        if seed is not None:
            np.random.seed(seed)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        if self.model.nq >= 6:
            self.data.qpos[:6] = [0.0, -0.3, 0.0, -1.8, 0.0, 1.5]

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

    def get_ee_pose(self):
        return self.data.site_xpos[self.ee_site_id].copy(), self.data.site_xmat[self.ee_site_id].copy()

    def set_gripper(self, open_amount=0.04):
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and "finger" in name.lower():
                adr = self.model.jnt_qposadr[i]
                self.data.qpos[adr] = open_amount

