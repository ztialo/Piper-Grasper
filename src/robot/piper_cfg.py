import os
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

script_dir = os.path.dirname(os.path.abspath(__file__))
piper_usd_path = os.path.join(
    script_dir, "..", "assets", "piper_description", "urdf","piper_description_v100_camera", "piper_description_v100_camera.usd")

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path = piper_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),      

)