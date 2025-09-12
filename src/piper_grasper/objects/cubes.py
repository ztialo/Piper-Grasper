
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

RED_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_1",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1, 0.0, 0.0),
            roughness=1.0,),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)

BLUE_CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_2",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 0.0, 1.0),
            roughness=1.0,),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
)