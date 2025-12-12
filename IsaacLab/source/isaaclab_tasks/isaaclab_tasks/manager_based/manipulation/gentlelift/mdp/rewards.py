# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_is_slipping(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    velocity_threshold: float = 0.05,
    contact_force_threshold: float = 0.1,
) -> torch.Tensor:
    """Function that return if the object is slipping."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # compute the desired velocity in the world frame
    robot_body_ids, _ = robot.find_bodies("panda_leftfinger")
    robot_vel = robot.data.body_vel_w[:, robot_body_ids[0], 0:3]
    object_vel = object.root_physx_view.get_velocities()[:, 0:3]
    # compute the relative velocity between the robot and the object in the world frame
    assert robot_vel.shape == object_vel.shape, f"Robot velocity shape: {robot_vel.shape}, Object velocity shape: {object_vel.shape}"
    relative_vel = robot_vel - object_vel
    # compute the norm of the relative velocity
    relative_vel_norm = torch.abs(torch.norm(relative_vel, dim=1)).unsqueeze(1)
    
    """The contact forces on the robot's fingers."""
    contact_force_left: ContactSensor = env.scene["contact_force_left"]
    contact_force_right: ContactSensor = env.scene["contact_force_right"]
    contact_force_left_w = torch.norm(torch.mean(contact_force_left.data.net_forces_w_history, dim=1), dim=-1)
    contact_force_right_w = torch.norm(torch.mean(contact_force_right.data.net_forces_w_history, dim=1), dim=-1)
    contact_force_w = torch.mean(torch.cat([contact_force_left_w, contact_force_right_w], dim=1), dim=1, keepdim=True) # (num_envs, 1)
    return torch.logical_and(contact_force_w > contact_force_threshold, relative_vel_norm > velocity_threshold)