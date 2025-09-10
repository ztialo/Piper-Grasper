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
from piper_grasper import arm_control

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

    # left arm articulation
    left_piper_arm = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Left_piper")
    left_piper_arm.init_state.pos = (0.0, 0.32, 0.0)

    # right arm articulation
    right_piper_arm = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Right_piper")
    right_piper_arm.init_state.pos = (0.0, -0.32, 0.0)

    # cubes
    cube_1 = RigidObjectCfg(
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
    cube_1.init_state.pos = (0.32, 0.25, 0.2)

    cube_2 = RigidObjectCfg(
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
    cube_2.init_state.pos = (0.44, -0.35, 0.2)

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

    # hand cameras
    hand_cam_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Left_piper/link6/hand_cam/left_camera",
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

    hand_cam_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Right_piper/link6/hand_cam/right_camera",
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
    left_piper_arm = scene["left_piper_arm"]
    right_piper_arm = scene["right_piper_arm"]
    # desk = scene["desk"]
    # cube_1 = scene["cube_1"]
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
    left_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    left_ik_controller = DifferentialIKController(
        left_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    right_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    )
    right_ik_controller = DifferentialIKController(
        right_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

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
    pos = np.atleast_2d(np.array([0.31, 0.1, -0.074], dtype=np.float32))  # (1,3)
    orn = np.atleast_2d(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # (1,4)
    drop_off_marker.visualize(pos, orn)

    # Specify robot-specifc parameters
    # piper ee is link 8
    left_piper_arm_entity_cfg = SceneEntityCfg(
        "left_piper_arm", joint_names=["joint[1-6]"], body_names=["link[8]"]
    )
    left_gripper_entity_cfg = SceneEntityCfg(
        "left_piper_arm", joint_names=["joint.*"], body_names=["link[7-8]"]
    )
    right_piper_arm_entity_cfg = SceneEntityCfg(
        "right_piper_arm", joint_names=["joint[1-6]"], body_names=["link[8]"]
    )
    right_gripper_entity_cfg = SceneEntityCfg(
        "right_piper_arm", joint_names=["joint.*"], body_names=["link[7-8]"]
    )

    # Resolving the scene entities
    left_piper_arm_entity_cfg.resolve(scene)
    left_gripper_entity_cfg.resolve(scene)
    right_piper_arm_entity_cfg.resolve(scene)
    right_gripper_entity_cfg.resolve(scene)

    # Obtain the frame index of the end-effector
    print("Left piper body ids", left_piper_arm_entity_cfg.body_ids)
    print("Left piper body names", left_piper_arm_entity_cfg.body_names)
    
    left_ee_jacobi_idx = (
        left_piper_arm_entity_cfg.body_ids[0] - 1
    )  # minus 1 because the jacobian does not include the base
    right_ee_jacobi_idx = (
        right_piper_arm_entity_cfg.body_ids[0] - 1
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # initialize pick and place task variables
    ready2pick = False
    next_pos_flag = False
    action_flag_left = False
    action_flag_right = False
    set_lift = False
    set_lower = False
    lowered_flag = False
    pos_tolerance = 0.0035
    state = "idle"
    current_arm = "left"
    cube_counter = 0
    gripper_counter = 0

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            print("[INFO] Resetting scene")
            # reset time
            count = 0
            # reset joint state (both arms share the same default pos and vel)
            joint_pos = left_piper_arm.data.default_joint_pos.clone()
            joint_vel = left_piper_arm.data.default_joint_vel.clone()
            left_piper_arm.write_joint_state_to_sim(joint_pos, joint_vel)
            left_piper_arm.reset()

            joint_pos = right_piper_arm.data.default_joint_pos.clone()
            joint_vel = right_piper_arm.data.default_joint_vel.clone()
            right_piper_arm.write_joint_state_to_sim(joint_pos, joint_vel)
            right_piper_arm.reset()
            
            # reset controller
            ik_commands_left = torch.zeros(
                scene.num_envs, left_ik_controller.action_dim, device=left_piper_arm.device
            )
            ik_commands_right = torch.zeros(
                scene.num_envs, right_ik_controller.action_dim, device=right_piper_arm.device
            )

            left_ik_controller.reset()
            left_ik_controller.set_command(ik_commands_left)
            right_ik_controller.reset()
            right_ik_controller.set_command(ik_commands_right)

            # randomize target cube pos on desk
            # rand_x = torch.rand(scene.num_envs, 1, device=sim.device) * 0.30 + 0.05         # [0.05, 0.32)
            # rand_y = (torch.rand(scene.num_envs, 1, device=sim.device) - 0.5) * 0.6         # [-0.3, 0.3)
            # z = torch.full((scene.num_envs, 1), 0.2, device=sim.device)                # fixed 0.2
            # target_offset = torch.cat([rand_x, rand_y, z], dim=-1)  # (num_envs, 3)
            # cube_state = cube.data.default_root_state.clone()
            # cube_state[:, :3] += target_offset
            # cube.write_root_state_to_sim(cube_state)

            gripper_control.open(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
            gripper_control.open(right_piper_arm, right_gripper_entity_cfg, sim.device, scene)
            ready2pick = False
            detections = None
            left_current_goal_idx = 0
            right_current_goal_idx = 0
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
                    left_ee_goals_world = []
                    right_ee_goals_world = []

                    goal_pose = [
                        detections["goal"][0]["x_w"]+0.015,
                        detections["goal"][0]["y_w"]+0.015,
                        detections["goal"][0]["z_w"],
                        0.7071, 0.7071,  0.0000, 0.0000  # default orientation
                    ]

                    for target in detections["target"]:
                        target_pose = [
                            target["x_w"]-0.03, # hard code offset for testing
                            target["y_w"]+0.0,
                            target["z_w"]-0.015, # offset so gripper not touching object
                            0.7071, 0.7071,  0.0000, 0.0000  # default orientation
                        ]

                        # parse target base on the object's y coordinate (left is positivate y)
                        if target["y_w"] >= 0.0:
                            left_ee_goals_world.extend([target_pose, goal_pose])
                            left_ee_goals_world = torch.tensor(left_ee_goals_world, device=sim.device)
                            # left_ee_goals[:, 1] -= 0.32
                        else:
                            right_ee_goals_world.extend([target_pose, goal_pose])
                            right_ee_goals_world = torch.tensor(right_ee_goals_world, device=sim.device)
                            # right_ee_goals[:, 1] += 0.32

                    left_ee_goals_local = left_ee_goals_world.clone()
                    left_ee_goals_local[:, 1] -= 0.32

                    right_ee_goals_local = right_ee_goals_world.clone()
                    right_ee_goals_local[:, 1] += 0.32

                    # Track the given command
                    left_current_goal_idx, right_current_goal_idx = 0, 0               

                    print("left_ee_goals: ", left_ee_goals_world)
                    print("right_ee_goals: ", right_ee_goals_world)


        else:
            # state machine
            if state == "idle":
                if ready2pick:
                    # gripper_control.open(piper_arm, piper_arm_entity_cfg, sim.device, scene)
                    if current_arm == "left":
                        print("left current goal idx: ", left_current_goal_idx, "left_ee_goals_world: ", left_ee_goals_world.shape[0])
                        print("next_pos_flag: ", next_pos_flag)
                        if left_ee_goals_world.shape[0] == 0:
                            current_arm = "right"
                        elif left_current_goal_idx > (left_ee_goals_world.shape[0] - 1): # minus 1 because idx start from 0
                            print("switching to right arm")
                            current_arm = "right"
                        else:
                            next_pos_flag = True

                    if current_arm == "right":
                        if right_ee_goals_world.shape[0] == 0:
                            # home both arms
                            state = "done"
                            print("[STATE]: idle -> done")
                        elif right_current_goal_idx > (right_ee_goals_world.shape[0] - 1):
                            # home both arms
                            state = "done"
                            print("[STATE]: idle -> done")
                        else:
                            next_pos_flag = True

                    if next_pos_flag is True:
                        if current_arm == "left":
                            ik_commands_left[:] = left_ee_goals_local[left_current_goal_idx]
                            left_ik_controller.set_command(ik_commands_left)
                        elif current_arm == "right":
                            ik_commands_right[:] = right_ee_goals_local[right_current_goal_idx]
                            right_ik_controller.set_command(ik_commands_right)
                        next_pos_flag = False
                        state = "to_pose"
                        print("[STATE]: idle -> to_pose")
            elif state == "to_pose":
                # allow arm action to current goal pose
                action_flag_left = True

                left_ee_pose_w = left_piper_arm.data.body_pose_w[
                    :, left_piper_arm_entity_cfg.body_ids[0]
                ]
                # print("current pose: ", left_ee_pose_w)
                # print("ee goal pose: ", ee_goals[left_current_goal_idx, :3])

                # for now only the position difference
                pos_diff = left_ee_goals_world[left_current_goal_idx, :3] - left_ee_pose_w[:, :3]
                pos_err = torch.linalg.norm(pos_diff, dim=-1)
                print("pos error: ", pos_err)
                # print()
                pos_ok = pos_err <= pos_tolerance

                if pos_ok:
                    print(f"    pos_diff: {left_ee_goals_local[left_current_goal_idx, :3]} - {left_ee_pose_w[:, :3]} = {pos_diff}")
                    pos_ok = False
                    action_flag_left = False
                    if left_current_goal_idx % 2 == 0:
                        state = "pick"
                        print("[STATE]: to_pose -> pick")
                    if left_current_goal_idx % 2 == 1:
                        state = "place"
                        print("[STATE]: to_pose -> place")
            elif state == "pick":
                # lower gripper first
                if lowered_flag is False:
                    # ''' temporary debug lines '''
                    # lowered_flag = True
                    # ''''''
                    left_ee_pose_w = left_piper_arm.data.body_pose_w[
                        :, left_piper_arm_entity_cfg.body_ids[0]
                    ]

                    if set_lower is False:
                        
                        lower_offset = torch.tensor([0.0, 0.0, -0.005], device=sim.device, dtype=torch.float32)  # 3cm
                        lower_pos = left_ee_pose_w[:, :3] + lower_offset
                        lower_quat = left_ee_pose_w[:, 3:7]  # keep orientation
                        lower_pose = torch.cat([lower_pos, lower_quat], dim=-1)  # shape (N,7)
                        left_lower_pose = lower_pose.clone()
                        left_lower_pose[:, 1] -= 0.32 # local y axis offset

                        ik_commands_left[:] = left_lower_pose
                        left_ik_controller.set_command(ik_commands_left)
                        set_lower = True
                        action_flag_left = True              

                    print("current pose: ", left_ee_pose_w)
                    print("lower goal pose: ", lower_pose)
                    print("pos error: ", pos_err)
                    print()
                    

                    pos_diff = lower_pose[:, :3] - left_ee_pose_w[:, :3]
                    pos_err = torch.linalg.norm(pos_diff, dim=-1)
                    pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        pos_ok = False
                        lowered_flag = True
                        set_lift = False
                        print("     lowered gripper \n\n\n")
                else:

                    # close gripper and lift for small amount
                    gripper_control.close(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
                    gripper_counter += 1

                    if gripper_counter >= 100:
                        print("Gripper counter: ", gripper_counter)
                        left_ee_pose_w = left_piper_arm.data.body_pose_w[:, left_piper_arm_entity_cfg.body_ids[0]]  # (N,7)
                        if set_lift is False:
                            cube_counter += 1
                            lift_amount = 0.05 * cube_counter
                            lift_offset = torch.tensor([0.0, 0.0, lift_amount], device=sim.device, dtype=torch.float32)  # 3cm
                            lift_pos = left_ee_pose_w[:, :3] + lift_offset
                            lift_quat = left_ee_pose_w[:, 3:7]  # keep orientation
                            lift_pose = torch.cat([lift_pos, lift_quat], dim=-1)  # shape (N,7)
                            left_lift_pose = lift_pose.clone()
                            left_lift_pose[:, 1] -= 0.32 # local y axis offset

                            ik_commands_left[:] = left_lift_pose
                            left_ik_controller.set_command(ik_commands_left)
                            set_lift = True
                            action_flag_left = True

                        pos_diff = lift_pose[:, :3] - left_ee_pose_w[:, :3]
                        pos_err = torch.linalg.norm(pos_diff, dim=-1)
                        print("\n\n lift pos error: ", pos_err)
                        pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        # reset params
                        pos_ok = False
                        set_lift = False
                        lowered_flag = False
                        gripper_counter = 0
                        left_current_goal_idx += 1
                        ik_commands_left[:] = left_ee_goals_local[left_current_goal_idx]
                        left_ik_controller.set_command(ik_commands_left)
                        action_flag_left = True
                        state = "to_pose"
                        print("[STATE]: pick -> to_pose")
            elif state == "place":
                gripper_control.open(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
                gripper_counter += 1

                if gripper_counter >= 100:
                    # increase idx for goal counting
                    if current_arm == "left":
                        left_current_goal_idx += 1
                    elif current_arm == "right":
                        right_current_goal_idx += 1
                    
                    print("Gripper counter: ", gripper_counter)
                    gripper_counter = 0
                    action_flag_left = False
                    state = "home"
                    home_state = {}
                    print("[STATE]: place -> home")
                    
            elif state == "home":
                # reset left arm
                done, q_cmd = arm_control.home(left_piper_arm, left_piper_arm_entity_cfg, sim, home_state, duration_s=1.5)
                if q_cmd is not None:
                    print("q_cmd: ", q_cmd)
                    left_piper_arm.set_joint_position_target(q_cmd, joint_ids=left_piper_arm_entity_cfg.joint_ids)

                if done:
                    state = "idle"
                    action_flag_left = False
                    next_pos_flag = False
                    print("[STATE]: home -> idle")
            elif state == "done":
                print("\n\n[INFO]: Task done.")

            if action_flag_left:
                """ -----------------------Left Arm ---------------------"""
                # obtain quantities from simulation
                left_jacobian = left_piper_arm.root_physx_view.get_jacobians()[
                    :, left_ee_jacobi_idx, :, left_piper_arm_entity_cfg.joint_ids
                ]
                left_ee_pose_w = left_piper_arm.data.body_pose_w[
                    :, left_piper_arm_entity_cfg.body_ids[0]
                ]
                left_root_pose_w = left_piper_arm.data.root_pose_w
                left_joint_pos = left_piper_arm.data.joint_pos[:, left_piper_arm_entity_cfg.joint_ids]
                # compute frame in root frame
                left_ee_pos_b, left_ee_quat_b = subtract_frame_transforms(
                    left_root_pose_w[:, 0:3],
                    left_root_pose_w[:, 3:7],
                    left_ee_pose_w[:, 0:3],
                    left_ee_pose_w[:, 3:7],
                )
                # compute the joint commands
                left_joint_pos_des = left_ik_controller.compute(
                    left_ee_pos_b, left_ee_quat_b, left_jacobian, left_joint_pos
                )

                left_piper_arm.set_joint_position_target(
                    left_joint_pos_des, joint_ids=left_piper_arm_entity_cfg.joint_ids
                )
               
            if action_flag_right:
                """ -----------------------Right Arm ---------------------"""
                # obtain quantities from simulation
                right_jacobian = right_piper_arm.root_physx_view.get_jacobians()[
                    :, right_ee_jacobi_idx, :, right_piper_arm_entity_cfg.joint_ids
                ]
                right_ee_pose_w = right_piper_arm.data.body_pose_w[
                    :, right_piper_arm_entity_cfg.body_ids[0]
                ]
                right_root_pose_w = right_piper_arm.data.root_pose_w
                right_joint_pos = right_piper_arm.data.joint_pos[:, right_piper_arm_entity_cfg.joint_ids]
                # compute frame in root frame
                right_ee_pos_b, right_ee_quat_b = subtract_frame_transforms(
                    right_root_pose_w[:, 0:3],
                    right_root_pose_w[:, 3:7],
                    right_ee_pose_w[:, 0:3],
                    right_ee_pose_w[:, 3:7],
                )
                # compute the joint commands
                right_joint_pos_des = right_ik_controller.compute(
                    right_ee_pos_b, right_ee_quat_b, right_jacobian, right_joint_pos
                )

                right_piper_arm.set_joint_position_target(
                    right_joint_pos_des, joint_ids=right_piper_arm_entity_cfg.joint_ids
                )

        """ Debug statements"""
        # print(piper_arm.data.joint_pos)

        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantitiess from simuation
        left_ee_pose_w = left_piper_arm.data.body_state_w[
            :, left_piper_arm_entity_cfg.body_ids[0], 0:7
        ]
        
        # ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # goal_marker.visualize(
        #     ik_commands_left[:, 0:3] + scene.env_origins, ik_commands_left[:, 3:7]
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
