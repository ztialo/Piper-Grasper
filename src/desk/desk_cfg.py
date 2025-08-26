import os

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

mass_props = schemas_cfg.MassPropertiesCfg(density=49.0)
rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
collision_props = schemas_cfg.CollisionPropertiesCfg()

script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(script_dir, "..", "assets", "workDesk_v1", "workDesk_v1.stl")
dest_path = os.path.join(script_dir, "..", "assets", "workDesk_v1")

DESK_CFG = MeshConverterCfg(
    mass_props=mass_props,
    rigid_props=rigid_props,
    collision_props=collision_props,
    asset_path=mesh_path,
    force_usd_conversion=True,
    usd_dir=os.path.dirname(dest_path),
    usd_file_name=os.path.basename(dest_path),
    make_instanceable=True,
    collision_approximation="convexDecomposition",
)