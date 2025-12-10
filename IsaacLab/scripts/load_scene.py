import argparse
import torch
from isaaclab.app import AppLauncher

# parse CLI args — e.g. device, headless mode, etc.
parser = argparse.ArgumentParser(description="Run Isaac Lab environment.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-GentleLift-Cube-Franka-IK-Rel-v0", help="Name of the Isaac-Lab task (env id)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs")
args_cli = parser.parse_args()

# launch Omniverse / Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# now import gym + Isaac-Lab registration
import gymnasium as gym
import isaaclab_tasks  # make sure tasks are imported so envs are registered
from isaaclab_tasks.utils import parse_env_cfg


# create the env — e.g. for a known valid task:
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg)
robot = env.unwrapped.scene["robot"]
object = env.unwrapped.scene["object"]

lower_stiffness = 150
lower_damping = 15
finger_ids, finger_names = robot.find_joints("panda_finger.*")
robot.write_joint_stiffness_to_sim(lower_stiffness, joint_ids=finger_ids)
robot.write_joint_damping_to_sim(lower_damping, joint_ids=finger_ids)
# import ipdb
# ipdb.set_trace()

# optionally, run a loop
obs = env.reset()
for t in range(1000):
    # import ipdb
    # ipdb.set_trace()
    # action = env.action_space.sample()
    if t < 200:
        action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    elif t < 500:
        action = torch.tensor([[0.0, 0.0, 0.05, 0.0, 0.0, 0.0, -1.0]])
    else:
        action = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    print(f"static_friction: {object.root_physx_view.get_material_properties()[0][0, 1]}")
    print(f"mass: {object.root_physx_view.get_masses()[0]}")
    out = env.step(action)
    # if done:
    #     obs = env.reset()