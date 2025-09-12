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
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.utils.math import subtract_frame_transforms

from piper_grasper import PIPER_CFG, RED_CUBE_CFG, BLUE_CUBE_CFG, DESK_CAM_CFG, HAND_CAM_CFG
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
    cube_1 = RED_CUBE_CFG
    cube_1.init_state.pos = (0.32, 0.25, 0.2)

    cube_2 = BLUE_CUBE_CFG
    cube_2.init_state.pos = (0.44, -0.3, 0.2)

    # desk camera
    desk_cam = DESK_CAM_CFG.replace(prim_path="{ENV_REGEX_NS}/desk_sensor")

    # hand cameras
    hand_cam_left = HAND_CAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Left_piper/link6/hand_cam/left_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.03, 0.0, 0.0),
            rot=(-0.6916548, 0.1470158, 0.1470158, -0.6916548),
        ),
    )

    hand_cam_right = HAND_CAM_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Right_piper/link6/hand_cam/right_camera",
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
    desk_cam = scene["desk_cam"]

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
    pos = np.atleast_2d(np.array([0.31, -0.0, -0.074], dtype=np.float32))  # (1,3)
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
    doneFlag = False
    state = "idle"
    current_arm = "left"
    cube_counter = 0
    gripper_counter = 0
    pos_tolerance = 0.0071

    # local frame offsets
    left_arm_y_offsets = -0.32
    right_arm_y_offsets = 0.38
    right_arm_local_offsets = -0.06

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
            print("     [INFO] Resetting scene")
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

            # default gripper to open state
            gripper_control.open(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
            gripper_control.open(right_piper_arm, right_gripper_entity_cfg, sim.device, scene)
            
            # reset flags and variables
            ready2pick = False
            detections = None
            left_current_goal_idx = 0
            right_current_goal_idx = 0
            state = "idle"
        
        elif count % 15 == 0:
            # rgb to locate object position, depth to estimate height
            camera_color_img = desk_cam.data.output['rgb'][0].clone().cpu().numpy()
            camera_depth_img = desk_cam.data.output['distance_to_image_plane'][0].clone().cpu().numpy()
            annotated, detections = cube_detect.detect_multi_color_cubes(camera_color_img, camera_depth_img)
            cv2.imshow("desk camera annotated", annotated)
            cv2.waitKey(10)

            if ready2pick is False:
                # only start if both target and goal are found in camera frame
                if detections.get("target") and detections.get("goal"):
                    ready2pick = True
                    print("camera detections: ", detections)

                    # Define goals for the arm
                    left_ee_goals_world = []
                    right_ee_goals_world = []

                    goal_pose = [
                        detections["goal"][0]["x_w"]+0.015,
                        detections["goal"][0]["y_w"]+0.015,
                        detections["goal"][0]["z_w"],
                        0.7071, 0.7071,  0.0000, 0.0000  # default gripper down orientation
                    ]

                    for target in detections["target"]:
                        target_pose = [
                            target["x_w"]-0.03,
                            target["y_w"]+0.0,
                            target["z_w"]-0.02, # offset so gripper not touching object
                            0.7071, 0.7071,  0.0000, 0.0000  # default orientation
                        ]

                        # parse target base on the object's y coordinate (left is positivate y)
                        if target["y_w"] >= 0.0:
                            left_ee_goals_world.extend([target_pose, goal_pose])
                            left_ee_goals_world = torch.tensor(left_ee_goals_world, device=sim.device)
                        else:
                            right_ee_goals_world.extend([target_pose, goal_pose])
                            right_ee_goals_world = torch.tensor(right_ee_goals_world, device=sim.device)

                    # save ee goals into local coordinate frame
                    left_ee_goals_local = left_ee_goals_world.clone()
                    left_ee_goals_local[:, 1] += left_arm_y_offsets

                    right_ee_goals_local = right_ee_goals_world.clone()
                    right_ee_goals_local[:, 1] += right_arm_y_offsets

                    # Track the given command
                    left_current_goal_idx, right_current_goal_idx = 0, 0               

                    print("left_ee_goals: ", left_ee_goals_world)
                    print("right_ee_goals: ", right_ee_goals_world)

        else:
            # state machine
            if state == "idle":
                """ IDLE STATE: Check for next availible EE goal"""
                if ready2pick:
                    if current_arm == "left":
                        # check for availible ee goals to move to for left arm 
                        if left_ee_goals_world.shape[0] == 0:
                            current_arm = "right"
                        elif left_current_goal_idx > (left_ee_goals_world.shape[0] - 1): # minus 1 because idx start from 0
                            print("     [INFO]: switching to right arm")
                            current_arm = "right"
                        else:
                            next_pos_flag = True

                    if current_arm == "right":
                        # check for availible ee goals for right arm
                        if right_ee_goals_world.shape[0] == 0:
                            state = "done"
                            print("[STATE]: idle -> done")
                        elif right_current_goal_idx > (right_ee_goals_world.shape[0] - 1):
                            state = "done"
                            print("[STATE]: idle -> done")
                        else:
                            next_pos_flag = True

                    if next_pos_flag is True:
                        if current_arm == "left":
                            ik_commands_left[:] = left_ee_goals_local[left_current_goal_idx]
                            left_ik_controller.set_command(ik_commands_left)
                            # allow arm action to current goal pose
                            action_flag_left = True
                        elif current_arm == "right":
                            ik_commands_right[:] = right_ee_goals_local[right_current_goal_idx]
                            right_ik_controller.set_command(ik_commands_right)
                            action_flag_right = True
                        next_pos_flag = False
                        state = "to_pose"
                        print("[STATE]: idle -> to_pose")

            elif state == "to_pose":
                """ TO_POSE STATE: follow IK command and check if arm is at EE goal"""
                if current_arm == "left":

                    left_ee_pose_w = left_piper_arm.data.body_pose_w[
                        :, left_piper_arm_entity_cfg.body_ids[0]
                    ]
                    pos_diff = left_ee_goals_world[left_current_goal_idx, :3] - left_ee_pose_w[:, :3]

                elif current_arm == "right":
                    right_ee_pose_w = right_piper_arm.data.body_pose_w[
                        :, right_piper_arm_entity_cfg.body_ids[0]
                    ]
                    right_ee_pose_local = right_ee_pose_w.clone()
                    right_ee_pose_local[:, 1] += right_arm_local_offsets
                    pos_diff = right_ee_goals_world[right_current_goal_idx, :3] - right_ee_pose_local[:, :3]

                pos_err = torch.linalg.norm(pos_diff, dim=-1)

                # pos accuracy descrease when height takes in place
                if cube_counter <= 1:
                    pos_ok = pos_err <= pos_tolerance
                else:
                    pos_ok = pos_err <= 0.0537

                if pos_ok:
                    pos_ok = False

                    if current_arm == "left":
                        action_flag_left = False
                        if left_current_goal_idx % 2 == 0:
                            state = "pick"
                            print("[STATE]: (left arm) to_pose -> pick")
                        if left_current_goal_idx % 2 == 1:
                            state = "place"
                            print("[STATE]: (left arm) to_pose -> place")
                    elif current_arm == "right":
                        if right_current_goal_idx % 2 == 0:
                            state = "pick"
                            print("[STATE]: (right arm) to_pose -> pick")
                        if right_current_goal_idx % 2 == 1:
                            state = "place"
                            print("[STATE]: (right arm) to_pose -> place")

            elif state == "pick":
                """ PICK STATE: lower EE, grip on to object, lift back up to a safe height"""
                # lower gripper first
                if lowered_flag is False:
                    left_ee_pose_w = left_piper_arm.data.body_pose_w[:, left_piper_arm_entity_cfg.body_ids[0]]
                    right_ee_pose_w = right_piper_arm.data.body_pose_w[:, right_piper_arm_entity_cfg.body_ids[0]]

                    if set_lower is False:
                        
                        lower_offset = torch.tensor([0.0, 0.0, -0.005], device=sim.device, dtype=torch.float32)
                        if current_arm == "left":
                            # take the current ee pose and add a lower offset
                            lower_pos = left_ee_pose_w[:, :3] + lower_offset
                            lower_quat = left_ee_pose_w[:, 3:7]  # keep orientation
                            lower_pose = torch.cat([lower_pos, lower_quat], dim=-1)  # shape (N,7)
                            left_lower_pose = lower_pose.clone()
                            left_lower_pose[:, 1] += left_arm_y_offsets # left local y axis offset

                            ik_commands_left[:] = left_lower_pose
                            left_ik_controller.set_command(ik_commands_left)
                            set_lower = True
                            action_flag_left = True
                            pos_diff = lower_pose[:, :3] - left_ee_pose_w[:, :3]

                        elif current_arm == "right":
                            lower_pos = right_ee_pose_w[:, :3] + lower_offset
                            lower_quat = right_ee_pose_w[:, 3:7]  # keep orientation
                            lower_pose = torch.cat([lower_pos, lower_quat], dim=-1)  # shape (N,7)
                            right_lower_pose = lower_pose.clone()
                            right_lower_pose[:, 1] += right_arm_y_offsets # right local y axis offset

                            ik_commands_right[:] = right_lower_pose
                            right_ik_controller.set_command(ik_commands_right)
                            set_lower = True
                            action_flag_right = True 
                            pos_diff = lower_pose[:, :3] - right_ee_pose_w[:, :3]

                    pos_err = torch.linalg.norm(pos_diff, dim=-1)
                    pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        pos_ok = False
                        lowered_flag = True
                        set_lift = False
                        print("     [INFO]: lowered gripper")
                else:
                    # close gripper and lift for stacking
                    if current_arm == "left":
                        gripper_control.close(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
                    elif current_arm == "right":
                        gripper_control.close(right_piper_arm, right_gripper_entity_cfg, sim.device, scene)
                    
                    # increment time counter for gripper
                    gripper_counter += 1

                    if gripper_counter >= 50:
                        if current_arm == "left":
                            ee_pose_w = left_piper_arm.data.body_pose_w[:, left_piper_arm_entity_cfg.body_ids[0]]  # (N,7)
                        elif current_arm == "right":
                            ee_pose_w = right_piper_arm.data.body_pose_w[:, right_piper_arm_entity_cfg.body_ids[0]]

                        if set_lift is False:
                            cube_counter += 1
                            lift_amount = 0.03 * cube_counter
                            lift_offset = torch.tensor([0.0, 0.0, lift_amount], device=sim.device, dtype=torch.float32)

                            # take the current ee pose and add a lift offset
                            if current_arm == "left":
                                lift_pos = ee_pose_w[:, :3] + lift_offset
                                lift_quat = ee_pose_w[:, 3:7]  # keep orientation
                                lift_pose = torch.cat([lift_pos, lift_quat], dim=-1)  # shape (N,7)
                                left_lift_pose = lift_pose.clone()
                                left_lift_pose[:, 1] += left_arm_y_offsets # local y axis offset

                                ik_commands_left[:] = left_lift_pose
                                left_ik_controller.set_command(ik_commands_left)
                                set_lift = True
                                action_flag_left = True

                            elif current_arm == "right":
                                lift_pos = ee_pose_w[:, :3] + lift_offset
                                lift_quat = ee_pose_w[:, 3:7]  # keep orientation
                                lift_pose = torch.cat([lift_pos, lift_quat], dim=-1)  # shape (N,7)
                                right_lift_pose = lift_pose.clone()
                                right_lift_pose[:, 1] += right_arm_y_offsets # local y axis offset

                                ik_commands_right[:] = right_lift_pose
                                right_ik_controller.set_command(ik_commands_right)
                                set_lift = True
                                action_flag_right = True

                        pos_diff = lift_pose[:, 2] - ee_pose_w[:, 2]

                        pos_err = torch.linalg.norm(pos_diff, dim=-1)
                        pos_ok = pos_err <= pos_tolerance

                    if pos_ok:
                        # reset params
                        pos_ok = False
                        set_lift = False
                        lowered_flag = False
                        gripper_counter = 0

                        if current_arm == "left":
                            left_current_goal_idx += 1
                            ik_commands_left[:] = left_ee_goals_local[left_current_goal_idx]
                            left_ik_controller.set_command(ik_commands_left)
                            action_flag_left = True
                        elif current_arm == "right":
                            right_current_goal_idx += 1
                            if cube_counter > 1:
                                # reuse the lift offset from above for new stacking height
                                goal_w = right_ee_goals_world[right_current_goal_idx]
                                pos_w = goal_w[:3] + lift_offset
                                quat_w = goal_w[3:7]
                                lift_pose_w = torch.cat((pos_w, quat_w), dim=0)
                                right_ee_goals_world[right_current_goal_idx] = lift_pose_w
                                right_ee_goals_local[right_current_goal_idx] = right_ee_goals_world[right_current_goal_idx].clone()
                                right_ee_goals_local[right_current_goal_idx, 1] += right_arm_y_offsets + right_arm_local_offsets

                            goal_w = right_ee_goals_local[right_current_goal_idx]      # (7,)
                            ik_commands_right.copy_(goal_w.unsqueeze(0).expand(scene.num_envs, -1))
                            right_ik_controller.set_command(ik_commands_right)
                            action_flag_right = True

                        state = "to_pose"
                        print("[STATE]: pick -> to_pose")

            elif state == "place":
                """ PLACE STATE: release the object and increment idex for next ee"""
                if current_arm == "left":
                    gripper_control.open(left_piper_arm, left_gripper_entity_cfg, sim.device, scene)
                elif current_arm == "right":
                    gripper_control.open(right_piper_arm, right_gripper_entity_cfg, sim.device, scene)

                # increment close gripper counter
                gripper_counter += 1

                if gripper_counter >= 50:
                    # increase idx for goal counting
                    if current_arm == "left":
                        left_current_goal_idx += 1
                    elif current_arm == "right":
                        right_current_goal_idx += 1

                    gripper_counter = 0
                    action_flag_left = False
                    state = "home"
                    home_state = {}
                    print("[STATE]: place -> home")

            elif state == "home":
                """ HOME STATE: put arm back in default position"""
                if current_arm == "left":
                    # home left arm
                    done, q_cmd = arm_control.home(left_piper_arm, left_piper_arm_entity_cfg, sim, home_state, duration_s=1)
                    if q_cmd is not None:
                        left_piper_arm.set_joint_position_target(q_cmd, joint_ids=left_piper_arm_entity_cfg.joint_ids)
                elif current_arm == "right":
                    # home right arm
                    done, q_cmd = arm_control.home(right_piper_arm, right_piper_arm_entity_cfg, sim, home_state, duration_s=1)
                    if q_cmd is not None:
                        right_piper_arm.set_joint_position_target(q_cmd, joint_ids=right_piper_arm_entity_cfg.joint_ids)

                if done:
                    state = "idle"
                    action_flag_left = False
                    action_flag_right = False
                    next_pos_flag = False
                    print("[STATE]: home -> idle")
            elif state == "done":
                if doneFlag is False:
                    print("     [INFO]: Task done.")
                    doneFlag = True

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
    print("     [INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close the app
    simulation_app.close()
    cv2.destroyAllWindows()

""" executable command
    ./isaaclab.sh -p ../Piper-Grasper/scripts/two_arms_stacking_cube.py --enable_cameras
"""
