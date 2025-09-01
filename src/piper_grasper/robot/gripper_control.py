""" 
Helper functions to control piper gripper states 
    Gripper joints : piper's joint 7 and 8
    Joint 7 range : lower="0" upper="0.05"
    Joint 8 range : lower="-0.05" upper="0"
"""
import torch


def open(piper_arm, piper_arm_entity_cfg, sim_device, scene):
    open_values = torch.tensor([0.05, -0.05], sim_device).unsqueeze(0).repeat(scene.num_envs, 1)
    gripper_joint_ids = piper_arm_entity_cfg.joint_ids_for(["joint7", "joint8"])  
    piper_arm.set_joint_position_target(
        open_values, joint_ids=gripper_joint_ids
    )


def close(piper_arm, piper_arm_entity_cfg, sim_device, scene, target_contour=None):
    if target_contour is None:
        close_values = torch.tensor([0.0, 0.0], sim_device).unsqueeze(0).repeat(scene.num_envs, 1)
    else:
        # TODO: using contour shape to help decide how much the gripper should close
        pass

    gripper_joint_ids = piper_arm_entity_cfg.joint_ids_for(["joint7", "joint8"])  
    piper_arm.set_joint_position_target(
        close_values, joint_ids=gripper_joint_ids
    )
