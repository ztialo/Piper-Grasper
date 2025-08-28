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
import math
from isaaclab.sim.converters import MeshConverter
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.sensors import CameraCfg, Camera
import isaaclab.utils.math as math_utils


def euler_to_quat(euler: list[float], degrees: bool = True) -> tuple[float]:
    quaternion_np = rot_utils.euler_angles_to_quat(euler, degrees=degrees)
    quaternion_tuple = tuple(float(v) for v in quaternion_np)
    return quaternion_tuple


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
    piper_quaternion_tuple = euler_to_quat(piper_orientation_euler, degrees=True)
    prim_utils.create_prim("/World/DeskOrigin/Piper_Origin1", "Xform", translation=piper_origins[0])
    prim_utils.create_prim("/World/DeskOrigin/Piper_Origin2", "Xform", translation=piper_origins[1])
    piper_cfg = PIPER_CFG.copy()
    piper_cfg.prim_path = "/World/DeskOrigin/Piper_Origin.*/Piper_arm"
    piper = Articulation(cfg=piper_cfg)
    scene_entities = {"piper": piper}

    # Cuboid 
    cube_origin = [0.0, 0.0, 0.5]
    prim_utils.create_prim("/World/DeskOrigin/Cube_origin", "Xform", translation=cube_origin)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/DeskOrigin/Cube_origin/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.4)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
    )
    # cube_cfg.prim_path = "/World/DeskOrigin/Cuboid"
    cuboid = RigidObject(cfg=cube_cfg)
    scene_entities["cuboid"] = cuboid

    # Camera sensors
    camera_rotation_euler = ([180, 4.5, -90])
    camera_quaternion_tuple = euler_to_quat(camera_rotation_euler, degrees=True)
    camera_cfg = CameraCfg(
        prim_path="/World/DeskOrigin/Piper_Origin.*/Piper_arm/link6/hand_cam/camera_sensor",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=camera_quaternion_tuple, convention="ros"),
    )
    camera = Camera(cfg=camera_cfg)
    scene_entities["camera"] = camera

    # return the scene information
    return scene_entities, piper_origins, piper_quaternion_tuple


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor, orientation: tuple[float]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["piper"]
    cuboid = entities["cuboid"]
    cuboid_offset = torch.tensor((0.0, 0.0, 0.9), device="cuda:0") 
    camera = entities["camera"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Euler angle representation (13, 0, 0) in degrees
    roll  = torch.tensor(0.293)
    pitch = torch.tensor(0.0)
    yaw   = torch.tensor(0.0)

    (w, x, y, z) = math_utils.quat_from_euler_xyz(pitch, roll, yaw)
    camera_orientations = torch.stack([w, x, y, z], dim=0).unsqueeze(0).repeat(2, 1).to(sim.device)
    print("adding camera rot:", camera_orientations)
    print("pre-cam data:", camera.data.quat_w_world)
    camera.set_world_poses(None, camera_orientations, convention="world")
    

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            cuboid_state = cuboid.data.default_root_state.clone()
            # coordinate transformation
            root_state[:, :3] += origins
            cuboid_state[:, :3] += cuboid_offset
            cuboid.write_root_pose_to_sim(cuboid_state[:, :7])
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
            print("post-cam data:", camera.data.quat_w_world)
        # Apply random action (not applying effort since im not training a model)
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)

        # -- write data to sim
        cuboid.write_data_to_sim()
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        cuboid.update(sim_dt)
        robot.update(sim_dt)
        camera.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(camera)
        # print("Received shape of rgb   image: ", camera.data.output["rgb"].shape)
        # print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)


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