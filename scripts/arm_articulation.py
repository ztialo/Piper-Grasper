import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from src.desk.desk_cfg import DESK_CFG
from isaaclab.sim.converters import MeshConverter
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Desk
    desk_origin = [0.0, 0.0, 0.0]
    prim_utils.create_prim("/World/DeskOrigin", "Xform", translation=desk_origin)

    desk_converter_cfg = DESK_CFG()
    desk_converter = MeshConverter(desk_converter_cfg)

    desk_cfg = sim_utils.UsdFileCfg(usd_path=desk_converter.usd_path)
    desk_cfg.func("/World/DeskOrigin/Desk", desk_cfg)

    # Articulation
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Origin.*/Robot"
    cartpole = Articulation(cfg=cartpole_cfg)

    # return the scene information
    scene_entities = {"cartpole": cartpole}
    return scene_entities, origins