import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

script_dir = os.path.dirname(os.path.abspath(__file__))
piper_usd_path = os.path.join(
    script_dir, "..", "..", "..", "assets", "piper_description", "urdf","piper_description_v100_camera", "piper_description_v100_camera.usd")

# piper_usd_path = os.path.join(
#     script_dir, "..", "..", "..", "assets", "piper_description", "urdf","piper_description_v200", "piper_description_v200.usd")

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=piper_usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "joint5": 0.0,
                "joint6": 0.0,
                "joint7": 0.05,
                "joint8": -0.05,
            },
            # rot=(0.7071067811865475, 0.0, 0.0, -0.7071067811865475),  # 90 deg around x-axis
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=2000.0,
            stiffness=2500.0,
            damping=250.0,
        ),
    },
)

""""
effort_limit_sim=200.0,
stiffness=1400.0,
damping=500.0,
"""
