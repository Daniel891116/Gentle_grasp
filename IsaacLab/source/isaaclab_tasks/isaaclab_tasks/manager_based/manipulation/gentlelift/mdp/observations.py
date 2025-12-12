# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b
    

def contact_forces(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """The contact forces on the robot's fingers."""
    contact_force_left: ContactSensor = env.scene["contact_force_left"]
    contact_force_right: ContactSensor = env.scene["contact_force_right"]
    # contact_force_test: ContactSensor = env.scene["contact_force_test"]
    contact_force_left_w = torch.norm(torch.mean(contact_force_left.data.net_forces_w_history, dim=1), dim=-1)
    contact_force_right_w = torch.norm(torch.mean(contact_force_right.data.net_forces_w_history, dim=1), dim=-1)
    # contact_force_test_w, _ = torch.max(torch.mean(contact_force_test.data.net_forces_w_history, dim=1), dim=-1)
    # print(f"contact force left: {contact_force_left_w}")
    # print(f"contact force right: {contact_force_right_w}")
    # print(f"contact force test: {contact_force_test_w}")
    return torch.cat([contact_force_left_w, contact_force_right_w], dim=1)


def relative_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compute the relative velocity between the robot and the object."""
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
    return torch.abs(torch.norm(relative_vel, dim=1)).unsqueeze(1)
    
