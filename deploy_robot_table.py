#!/usr/bin/env python3
"""
Phase 1 end – Deploy Franka Panda + designed table and visualise.
"""

import os
import sys
import mujoco
import mujoco.viewer
from robot_descriptions import panda_mj_description

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def create_scene_xml():
    """Create a combined scene string with Panda + table."""
    panda_xml_path = panda_mj_description.MJCF_PATH

    # Read the original panda.xml
    with open(panda_xml_path, "r") as f:
        panda_content = f.read()

    # We will build a new scene that includes the panda content + table
    scene = f"""
<mujoco model="pick_place_deployed">
  <compiler angle="radian" meshdir="{os.path.dirname(panda_xml_path)}/assets"/>

  <!-- Copy key sections from panda.xml would be complex; 
       instead we load panda fully and add table via model editing later.
       For now we use a simple include-style approach via Python. -->
</mujoco>
"""
    return panda_xml_path


def main():
    print("=" * 60)
    print("Deploying Franka Panda + Table")
    print("=" * 60)

    # Load pure Panda
    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    data = mujoco.MjData(model)

    # ---------- Add table programmatically ----------
    # Table parameters (your choice)
    table_pos = [0.55, 0.0, 0.25]          # centre of the box
    table_size = [0.175, 0.30, 0.25]       # half-sizes → L=0.35, W=0.60, H=0.50

    # We create a new model specification and add the table
    # (MuJoCo 3.x supports MjSpec for easy editing)
    try:
        spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)

        # Add table body
        table_body = spec.worldbody.add_body(name="table", pos=table_pos)
        table_body.add_geom(
            name="table_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=table_size,
            rgba=[0.6, 0.45, 0.3, 1],
            friction=[1.0, 0.005, 0.0001]
        )

        # Add floor if not present
        # (Panda scene already has floor in some variants; we keep it simple)

        model = spec.compile()
        data = mujoco.MjData(model)
        print("Table successfully added via MjSpec.")
    except Exception as e:
        print(f"MjSpec method failed ({e}). Falling back to pure Panda.")
        model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
        data = mujoco.MjData(model)

    # Set a nice initial configuration (slightly stretched forward)
    data.qpos[:7] = [0, -0.4, 0, -2.0, 0, 1.7, 0.8]
    data.qpos[7:] = 0.04
    mujoco.mj_forward(model, data)

    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if table_geom_id >= 0:
        centre_z = data.geom_xpos[table_geom_id][2]
        half_height = model.geom_size[table_geom_id][2]
        top_surface = centre_z + half_height
        print(f"\n===== TABLE HEIGHT CHECK =====")
        print(f"Table centre Z     : {centre_z:.4f} m")
        print(f"Table half-height  : {half_height:.4f} m")
        print(f"Table TOP surface  : {top_surface:.4f} m")
        print("==============================\n")
    else:
        print("WARNING: table_top geom not found!")

    print(f"Model nq = {model.nq}")
    print(f"Table height (top surface) ≈ 0.50 m")
    print("Launching interactive viewer... (close the window when finished)")

    # Launch the official MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
