import os

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
import isaaclab.utils.math as math_utils
import math
import torch

mass_props = schemas_cfg.MassPropertiesCfg(density=200.0)
rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
    rigid_body_enabled=False,
    kinematic_enabled=False,
)
collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_path = os.path.join(script_dir, "..", "..", "..", "assets", "workDesk_v1", "workDesk_v1.stl")
dest_path = os.path.join(script_dir, "..", "..", "..", "assets", "workDesk_v1")

# Euler angle representation (0, 0, -90) in degrees
pitch = torch.tensor(0.0)
roll  = torch.tensor(0.0)
yaw   = torch.tensor(-math.pi / 2)

(w, x, y, z) = math_utils.quat_from_euler_xyz(pitch, roll, yaw)
Quaternion_rotation = (float(w), float(x), float(y), float(z))

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
    rotation=Quaternion_rotation,
)