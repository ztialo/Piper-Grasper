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

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.utils.math import subtract_frame_transforms

from piper_grasper import PIPER_CFG

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
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.533, 0.3)
            ),
        ),
    )
    desk.init_state.pos = (0.255, 0.25, -0.145)
    desk.init_state.rot = (0.7071068, 0, 0, 0.7071068)   # 90 deg around z-axis

    # articulation
    piper_arm = PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Piper_arm")

    # cube
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.4)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    cube.init_state.pos = (0.25, 0.24, 0.2)

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
    # cube = scene["cube"]
    desk_cam = scene["desk_cam"]
    hand_cam = scene["hand_cam"]

    # pose the cameras
    desk_cam.set_world_poses(
        torch.as_tensor((0.0, 0.0, 0.75), dtype=torch.float32, device=sim.device).unsqueeze(0),
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
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    left_ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/left_ee_current")
    )
    goal_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")
    )

    # Define goals for the arm
    ee_goals = [
        [0.35,  0.10, 0.2, 0.6533, 0.6533, -0.2706, 0.2706],
        [0.35,  0.00, 0.2, 0.6533, 0.6533, -0.2706, 0.2706],
        [0.25,  0.00, 0.2, 0.7071, 0.7071,  0.0000, 0.0000],
        [0.30,  0.00, 0.3, 0.5000, 0.5000, -0.5000, 0.5000],
        [0.40, -0.10, 0.4, 0.7071, 0.0000,  0.0000, 0.7071],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)

    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(
        scene.num_envs, diff_ik_controller.action_dim, device=piper_arm.device
    )  # create for left arm for now
    ik_commands[:] = ee_goals[current_goal_idx]

    # Specify robot-specifc parameters
    # body names = ["arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8"]
    piper_arm_entity_cfg = SceneEntityCfg(
        "piper_arm", joint_names=["joint.*"], body_names=["link8"]
    )
    # right_piper_entity_cfg = SceneEntityCfg("right_piper", joint_names=["joint.*"], body_names=["link8"])

    # Resolving the scene entities
    piper_arm_entity_cfg.resolve(scene)
    # right_piper_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    print("Left piper body ids", piper_arm_entity_cfg.body_ids)
    print("Left piper body names", piper_arm_entity_cfg.body_names)
    # piper ee is link 8
    left_ee_jacobi_idx = (
        piper_arm_entity_cfg.body_ids[0] - 1
    )  # minus 1 because the jacobian does not include the base
    # right_ee_jacobi_idx = right_piper_entity_cfg.body_ids[0] - 1

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 350 == 0:
            # reset time
            count = 0
            # reset joint state
            left_joint_pos = piper_arm.data.default_joint_pos.clone()
            left_joint_vel = piper_arm.data.default_joint_vel.clone()
            piper_arm.write_joint_state_to_sim(left_joint_pos, left_joint_vel)
            piper_arm.reset()

            # reset actions
            ik_commands[:] = ee_goals[current_goal_idx]
            left_joint_pos_des = left_joint_pos[
                :, piper_arm_entity_cfg.joint_ids
            ].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = piper_arm.root_physx_view.get_jacobians()[
                :, left_ee_jacobi_idx, :, piper_arm_entity_cfg.joint_ids
            ]
            left_ee_pose_w = piper_arm.data.body_pose_w[
                :, piper_arm_entity_cfg.body_ids[0]
            ]
            root_pose_w = piper_arm.data.root_pose_w
            joint_pos = piper_arm.data.joint_pos[:, piper_arm_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                left_ee_pose_w[:, 0:3],
                left_ee_pose_w[:, 3:7],
            )
            # compute the joint commands
            left_joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

        # apply actions
        piper_arm.set_joint_position_target(
            left_joint_pos_des, joint_ids=piper_arm_entity_cfg.joint_ids
        )
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantitiess from simuation
        left_ee_pose_w = piper_arm.data.body_state_w[
            :, piper_arm_entity_cfg.body_ids[0], 0:7
        ]
        # right_ee_pose_w = right_piper.data.body_state_w[:, right_piper_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        left_ee_marker.visualize(left_ee_pose_w[:, 0:3], left_ee_pose_w[:, 3:7])
        # right_ee_marker.visualize(right_ee_pose_w[:, 0:3], right_ee_pose_w[:, 3:7])
        goal_marker.visualize(
            ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7]
        )


def main():
    """Main function"""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([1.5, 1.5, 1.5], [0.0, 0.0, 0.0])
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
