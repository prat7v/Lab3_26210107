import numpy as np
import mujoco
import mink

class HealMinkController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.configuration = mink.Configuration(model)
        self.ee_task = mink.FrameTask(
            frame_name="ee_site",
            frame_type="site",
            position_cost=5.0,
            orientation_cost=0.1,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model, cost=1e-2)
        self.tasks = [self.ee_task, self.posture_task]
        self.limits = [mink.ConfigurationLimit(model)]
        self.solver = "daqp"
        self.dt = 0.002
        self.KP = 800.0
        self.KD = 80.0

    def update(self):
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

    def move_to(self, target_pos, target_quat=None, n_steps=100, gripper_open=None):
        self.update()
        if target_quat is None:
            current = self.configuration.get_transform_frame_to_world("ee_site", "site")
            target_se3 = mink.SE3.from_rotation_and_translation(
                current.rotation(), np.asarray(target_pos)
            )
        else:
            rot = mink.SO3(wxyz=np.asarray(target_quat))
            target_se3 = mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos))
        self.ee_task.set_target(target_se3)

        for _ in range(n_steps):
            self.update()
            try:
                vel = mink.solve_ik(
                    self.configuration, self.tasks, dt=self.dt,
                    solver=self.solver, damping=1e-2, limits=self.limits
                )
                self.configuration.integrate_inplace(vel, self.dt)
            except Exception:
                pass

            # PD on first 6 joints
            target_q = self.configuration.q[:6]
            for i in range(6):
                err = target_q[i] - self.data.qpos[i]
                self.data.ctrl[i] = np.clip(
                    self.data.qfrc_bias[i] + self.KP * err - self.KD * self.data.qvel[i],
                    -300, 300
                )
            mujoco.mj_step(self.model, self.data)
