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
import isaacsim.core.utils.rotations as rot_utils

import isaaclab.sim as sim_utils
from piper_grasper import DESK_CFG, PIPER_CFG
import torch
import numpy as np
from isaaclab.sim.converters import MeshConverter
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext



def design_scene() -> tuple[dict, list[list[float]], tuple[float]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Desk
    desk_origin = [0.0, 0.0, 0.65]
    prim_utils.create_prim("/World/DeskOrigin", "Xform", translation=desk_origin)

    desk_converter_cfg = DESK_CFG
    desk_converter = MeshConverter(desk_converter_cfg)

    desk_cfg = sim_utils.UsdFileCfg(usd_path=desk_converter.usd_path)
    desk_cfg.func("/World/DeskOrigin/Desk", desk_cfg)

    visual_material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.533, 0.3))
    visual_material_cfg.func("/World/Looks/DeskMaterial", visual_material_cfg)
    sim_utils.bind_visual_material("/World/DeskOrigin/Desk", "/World/Looks/DeskMaterial")

    # Articulation
    piper_origins = [[0.1, 0.255, 0.795], [-0.5, 0.255, 0.795]]
    piper_orientation_euler = ([0.0, 0.0, -90.0])
    piper_quaternion_np = rot_utils.euler_angles_to_quat(piper_orientation_euler, degrees=True)
    piper_quaternion_tuple = tuple(float(v) for v in piper_quaternion_np)
    prim_utils.create_prim("/World/DeskOrigin/Piper_Origin1", "Xform", translation=piper_origins[0])
    prim_utils.create_prim("/World/DeskOrigin/Piper_Origin2", "Xform", translation=piper_origins[1])
    piper_cfg = PIPER_CFG.copy()
    piper_cfg.prim_path = "/World/DeskOrigin/Piper_Origin.*/Piper_arm"
    piper = Articulation(cfg=piper_cfg)
    scene_entities = {"piper": piper}

    # Cuboid 
    cfg_cuboid = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.1),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.4)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    cfg_cuboid.func("/World/DeskOrigin/Cuboid", cfg_cuboid, translation=(0.0, 0.0, 0.5))
    scene_entities["cuboid"] = prim_utils.get_prim_at_path("/World/DeskOrigin/Cuboid")


    # return the scene information
    return scene_entities, piper_origins, piper_quaternion_tuple


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, orientation: tuple[float]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["piper"]
    cuboid = entities["cuboid"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 5000 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            # coordinate transformation
            root_state[:, :3] += origins
            # orientation update
            root_state[:, 3:7] = torch.tensor(orientation, dtype=root_state.dtype, device=root_state.device)
            # print("set orientation:", orientation)
            # print(root_state)
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1 (not going to need since im not training a model)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action (not applying effort since im not training a model)
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)

        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins, scene_rotation = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, scene_rotation)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()