"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Piper arms using the differential IK controller."
)
# parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import os
import numpy as np
import cv2

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import subtract_frame_transforms

from piper_grasper import PIPER_CFG
from piper_grasper import cube_detection as cube_detect
from piper_grasper import gripper_control

script_dir = os.path.dirname(os.path.abspath(__file__))


@configclass
class WorkstationSceneCfg(InteractiveSceneCfg):
    """Configuration for a workstation scene with a table and piper robot arms."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.795)),
    )

    # Lights
    distant_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Desk
    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(
                script_dir, "..", "assets", "workDesk_v1", "workDesk_v1.usd"
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=200.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.533, 0.3)
            ),
        ),
    )
    desk.init_state.pos = (0.255, 0.2, -0.145)
    desk.init_state.rot = (0.7071068, 0, 0, 0.7071068)   # 90 deg around z-axis

    # articulation
    piper_arm = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Piper_arm")

    # cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
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
    cube.init_state.pos = (0.35, 0.19, 0.2)

    # desk camera
    desk_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/desk_sensor",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=400.0,
            horizontal_aperture=24,
            clipping_range=(0.1, 1.0e5),
        ),
    )

    # hand camera
    hand_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Piper_arm/link6/hand_cam/camera_sensor",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(-0.6916548, 0.1470158, 0.1470158, -0.6916548),
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop"""
    # Extract scene entities
    piper_arm = scene["piper_arm"]
    # desk = scene["desk"]
    cube = scene["cube"]
    desk_cam = scene["desk_cam"]
    # hand_cam = scene["hand_cam"]

    # pose the cameras
    desk_cam.set_world_poses(
        torch.as_tensor((0.0, 0.0, 0.72), dtype=torch.float32, device=sim.device).unsqueeze(0),
        torch.as_tensor((0.8435394, 0, 0.5370673, 0),
                        dtype=torch.float32, device=sim.device).unsqueeze(0),
        convention="world"  # (0, 65, 0) in degrees world -> (0, -25, -90) in local frame
    )
    
    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # Markers
    # frame_marker_cfg = FRAME_MARKER_CFG.copy()
    # frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    # ee_marker = VisualizationMarkers(
    #     frame_marker_cfg.replace(prim_path="/Visuals/ee_current")
    # )
    # goal_marker = VisualizationMarkers(
    #     frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
    # )
    drop_off_marker = VisualizationMarkers(
        cfg=VisualizationMarkersCfg(
            prim_path="/Visuals/drop_off_zone",
            markers={
                "cube": sim_utils.CuboidCfg(
                    size=(0.075, 0.075, 0.075),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        emissive_color=(0.0, 0.2, 0.0),
                        roughness=1.0, # so that the surface don't reflect light for camera
                    ),
                )
            }
        ),
    )
     # update marker positions
    pos = np.atleast_2d(np.array([0.31, -0.23, -0.072], dtype=np.float32))  # (1,3)
    orn = np.atleast_2d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # (1,4)
    drop_off_marker.visualize(pos, orn)

    # Specify robot-specifc parameters
    piper_arm_entity_cfg = SceneEntityCfg(
        "piper_arm", joint_names=["joint.*"], body_names=["link8"]
    )

    # Resolving the scene entities
    piper_arm_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    print("Left piper body ids", piper_arm_entity_cfg.body_ids)
    print("Left piper body names", piper_arm_entity_cfg.body_names)
    # piper ee is link 8
    ee_jacobi_idx = (
        piper_arm_entity_cfg.body_ids[0] - 1
    )  # minus 1 because the jacobian does not include the base

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # initialize pick and place task variables
    ready2pick = False
    next_pos_flag = False
    action_flag = False
    set_lift = False
    lowered_flag = False
    pos_tolerance = 0.005
    state = "idle"
    gripper_counter = 0

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 5000 == 0:
            print("[INFO] Resetting scene")
            # reset time
            count = 0
            # reset joint state
            joint_pos = piper_arm.data.default_joint_pos.clone()
            joint_vel = piper_arm.data.default_joint_vel.clone()
            piper_arm.write_joint_state_to_sim(joint_pos, joint_vel)
            piper_arm.reset()
            
            # reset controller
            diff_ik_controller.reset()

            # randomize target cube pos on desk
            # rand_x = torch.rand(scene.num_envs, 1, device=sim.device) * 0.30 + 0.05         # [0.05, 0.35)
            # rand_y = (torch.rand(scene.num_envs, 1, device=sim.device) - 0.5) * 0.6         # [-0.3, 0.3)
            # z = torch.full((scene.num_envs, 1), 0.2, device=sim.device)                # fixed 0.2
            # target_offset = torch.cat([rand_x, rand_y, z], dim=-1)  # (num_envs, 3)
            # cube_state = cube.data.default_root_state.clone()
            # cube_state[:, :3] += target_offset
            # cube.write_root_state_to_sim(cube_state)

            ready2pick = False
            detections = None
            state = "idle"
        
        elif count % 15 == 0:
            camera_color_img = desk_cam.data.output['rgb'][0].clone().cpu().numpy()
            camera_depth_img = desk_cam.data.output['distance_to_image_plane'][0].clone().cpu().numpy()
            annotated, detections = cube_detect.detect_multi_color_cubes(camera_color_img, camera_depth_img)
            # print(detections)
            cv2.imshow("desk camera annotated", annotated)
            # cv2.imshow("desk camera depth", camera_depth_img)
            cv2.waitKey(10)

            if ready2pick is False:
                if detections.get("target") and detections.get("goal"):
                    ready2pick = True
                    print(detections)

                    # Define goals for the arm
                    ee_goals = []
                    target_pose = [
                        detections["target"][0]["x_w"]-0.03, # hard code offset for testing
                        detections["target"][0]["y_w"]+0.03,
                        detections["target"][0]["z_w"]+0.015, # offset so gripper not touching object
                        0.7071, 0.7071,  0.0000, 0.0000  # default orientation
                        ]
                    goal_pose = [
                        detections["goal"][0]["x_w"],
                        detections["goal"][0]["y_w"],
                        detections["goal"][0]["z_w"],
                        0.7071, 0.7071,  0.0000, 0.0000  # default orientation
                    ]
                    ee_goals.extend([target_pose, goal_pose])
                    ee_goals = torch.tensor(ee_goals, device=sim.device)

                    # Track the given command
                    current_goal_idx = 0
                    # Create buffers to store actions
                    ik_commands = torch.zeros(
                        scene.num_envs, diff_ik_controller.action_dim, device=piper_arm.device
                    )  # create for left arm for now
                    ik_commands[:] = ee_goals[current_goal_idx]
                    diff_ik_controller.set_command(ik_commands)
                    next_pos_flag = True

        else:
            # state machine
            if state == "idle":
                gripper_control.open(piper_arm, piper_arm_entity_cfg, sim.device, scene)
                if next_pos_flag is True:
                    next_pos_flag = False
                    state = "to_pose"
                    print("[STATE]: idle -> to_pose")
            elif state == "to_pose":
                # allow arm action to current goal pose
                action_flag = True

                ee_pose_w = piper_arm.data.body_pose_w[
                    :, piper_arm_entity_cfg.body_ids[0]
                ]
                # for now only the position difference
                pos_diff = ee_goals[current_goal_idx, :3] - ee_pose_w[:, :3]
                pos_err = torch.linalg.norm(pos_diff, dim=-1)
                pos_ok = pos_err <= pos_tolerance

                if pos_ok:
                    print(f"    pos_diff: {ee_goals[current_goal_idx, :3]} - {ee_pose_w[:, :3]} = {pos_diff}")
                    pos_ok = False
                    action_flag = False
                    if current_goal_idx == 0:
                        state = "pick"
                        print("[STATE]: to_pose -> pick")
                    if current_goal_idx == 1:
                        state = "place"
                        print("[STATE]: to_pose -> place")
            elif state == "pick":
                # lower gripper first
                if lowered_flag is False:
                    # ''' temporary debug lines '''
                    # lowered_flag = True
                    # ''''''
                    lower_offset = torch.tensor([0.0, 0.0, -0.005], device=sim.device, dtype=torch.float32)  # 3cm
                    lower_pos = ee_pose_w[:, :3] + lower_offset
                    lower_quat = ee_pose_w[:, 3:7]  # keep orientation
                    lower_pose = torch.cat([lower_pos, lower_quat], dim=-1)  # shape (N,7)

                    ik_commands[:] = lower_pose
                    diff_ik_controller.set_command(ik_commands)
                    set_lift = True
                    action_flag = True

                    pos_diff = lower_pose[:, :3] - ee_pose_w[:, :3]
                    pos_err = torch.linalg.norm(pos_diff, dim=-1)
                    pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        pos_ok = False
                        lowered_flag = True
                        set_lift = False
                        print("     lowered gripper")
                else:

                    # close gripper and lift for small amount
                    gripper_control.close(piper_arm, piper_arm_entity_cfg, sim.device, scene)
                    gripper_counter += 1

                    if gripper_counter >= 200:
                        ee_pose_w = piper_arm.data.body_pose_w[:, piper_arm_entity_cfg.body_ids[0]]  # (N,7)
                        if set_lift is False:
                            lift_offset = torch.tensor([0.0, 0.0, 0.00], device=sim.device, dtype=torch.float32)  # 3cm
                            lift_pos = ee_pose_w[:, :3] + lift_offset
                            lift_quat = ee_pose_w[:, 3:7]  # keep orientation
                            lift_pose = torch.cat([lift_pos, lift_quat], dim=-1)  # shape (N,7)

                            ik_commands[:] = lift_pose
                            diff_ik_controller.set_command(ik_commands)
                            set_lift = True
                            action_flag = True

                        pos_diff = lift_pose[:, :3] - ee_pose_w[:, :3]
                        pos_err = torch.linalg.norm(pos_diff, dim=-1)
                        pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        # reset params
                        pos_ok = False
                        set_lift = False
                        lowered_flag = False
                        gripper_counter = 0
                        current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
                        ik_commands[:] = ee_goals[current_goal_idx]
                        diff_ik_controller.set_command(ik_commands)
                        state = "to_pose"
                        print("[STATE]: pick -> to_pose")
            elif state == "place":
                gripper_control.open(piper_arm, piper_arm_entity_cfg, sim.device, scene)
                gripper_counter += 1

                if gripper_counter >= 200:
                    gripper_counter = 0
                    state = "idle"
                    next_pos_flag = False
                    print("[STATE]: place -> idle")

            if action_flag:
                # obtain quantities from simulation
                jacobian = piper_arm.root_physx_view.get_jacobians()[
                    :, ee_jacobi_idx, :, piper_arm_entity_cfg.joint_ids
                ]
                ee_pose_w = piper_arm.data.body_pose_w[
                    :, piper_arm_entity_cfg.body_ids[0]
                ]
                root_pose_w = piper_arm.data.root_pose_w
                joint_pos = piper_arm.data.joint_pos[:, piper_arm_entity_cfg.joint_ids]
                # compute frame in root frame
                ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3],
                    root_pose_w[:, 3:7],
                    ee_pose_w[:, 0:3],
                    ee_pose_w[:, 3:7],
                )
                # compute the joint commands
                joint_pos_des = diff_ik_controller.compute(
                    ee_pos_b, ee_quat_b, jacobian, joint_pos
                )

                piper_arm.set_joint_position_target(
                    joint_pos_des, joint_ids=piper_arm_entity_cfg.joint_ids
                )

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantitiess from simuation
        ee_pose_w = piper_arm.data.body_state_w[
            :, piper_arm_entity_cfg.body_ids[0], 0:7
        ]
        
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(
        #     ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7]
        # )


def main():
    """Main function"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1, 1, 0.6], [0.0, 0.0, 0.0])
    # Desine scene
    scene_cfg = WorkstationSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close the app
    simulation_app.close()
    cv2.destroyAllWindows()

""" executable command
    ./isaaclab.sh -p ../Piper-Grasper/scripts/piper_pick_and_place.py --enable_cameras
"""
