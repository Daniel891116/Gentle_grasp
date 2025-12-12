import argparse
import torch
import os
from isaaclab.app import AppLauncher

# parse CLI args — e.g. device, headless mode, etc.
parser = argparse.ArgumentParser(description="Run Isaac Lab environment.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-GentleLift-Cube-Franka-IK-Abs-v0", help="Name of the Isaac-Lab task (env id)")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel envs")
parser.add_argument("--out_file", type=str, default="baseline.pt", help="Path to save the data")
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

device = args_cli.device
num_envs = args_cli.num_envs

gain_mean = 300.0
gain_std = 100.0
gain = torch.normal(mean=gain_mean, std=gain_std, size=(num_envs, 1)).to(device)
# gain = 200

gripper_stiffness = gain * torch.ones(num_envs, 1).to(device)
gripper_damping = 0.1 * gain * torch.ones(num_envs, 1).to(device)
finger_ids, finger_names = robot.find_joints("panda_finger.*")
robot.write_joint_stiffness_to_sim(gripper_stiffness, joint_ids=finger_ids)
robot.write_joint_damping_to_sim(gripper_damping, joint_ids=finger_ids)

# --- evaluation buffers ---
slip_distance = torch.zeros(num_envs, device=device)
max_contact_force = torch.zeros(num_envs, device=device)

masses = object.root_physx_view.get_masses()[:, 0]
mat_props = object.root_physx_view.get_material_properties()
static_mu = mat_props[:, 0, 0]

g = 9.81
N_min = masses * g / (2.0 * static_mu)

SLIP_THRESH = 0.025         # meters
RATIO_THRESH = 2.0
LIFT_HEIGHT_THRESH = 0.1

obs, _ = env.reset()
dt_step = env_cfg.sim.dt * env_cfg.decimation
iterations = int(env_cfg.episode_length_s / dt_step)

# --- time-series logging for plotting ---
# We log per step, per env so we can pick examples later.
ts_contact_force = torch.zeros(iterations, num_envs, device=device)
ts_stiffness = torch.zeros(iterations, num_envs, device=device)
ts_slip_speed = torch.zeros(iterations, num_envs, device=device)
ts_slip_flag = torch.zeros(iterations, num_envs, dtype=torch.bool, device=device)
ts_lifted = torch.zeros(iterations, num_envs, dtype=torch.bool, device=device)

def h_t(step: int) -> float:
    start_height = 0.1
    end_height = 0.25
    start_step = 50
    total_steps = 100
    if step < start_step:
        return start_height
    elif step > total_steps + start_step:
        return end_height
    else:
        return start_height + (end_height - start_height) * ((step - start_step) / total_steps)


for t in range(iterations):
    print(f"t: {t}")
    contact_force = torch.mean(torch.abs(obs['policy'][:, -4:-2]), dim=1) # (num_envs, 1)
    slipping = obs['policy'][:, -2:-1].squeeze().bool() # (num_envs, 1)
    rel_vel = obs['policy'][:, -1:] # (num_envs, 1)

    # object height now (for ts_lifted logging)
    obj_states = object.root_physx_view.get_transforms()  # (num_envs, num_bodies, 7)
    obj_pos_z = obj_states[:, 2]                       # <-- fix indexing
    print(f"obj_pos_z: {obj_pos_z[0]}")
    lifted_now = obj_pos_z >= LIFT_HEIGHT_THRESH          # (num_envs,)

    # (1) sliding distance accumulation
    slip_increment = rel_vel.squeeze(-1).abs() * dt_step
    slip_distance[slipping] += slip_increment[slipping]

    # (2) max contact force
    max_contact_force = torch.maximum(max_contact_force, contact_force)

    # ---------- NEW: log time-series at step t ----------
    ts_contact_force[t] = contact_force
    ts_stiffness[t]     = gripper_stiffness.squeeze(-1)
    ts_slip_speed[t]    = rel_vel.squeeze(-1).abs()
    ts_slip_flag[t]     = slipping
    ts_lifted[t]        = lifted_now
    # ----------------------------------------------------
    
    # settling time
    if t < 50:
        action = torch.tensor([[0.55, -0.23, h_t(t), -0.5, -0.5, -0.5, 0.5, 1.0]])
    else:
        action = torch.tensor([[0.55, -0.23, h_t(t), -0.5, -0.5, -0.5, 0.5, -1.0]])
    # print(f"static_friction: {object.root_physx_view.get_material_properties()[:, 0:1]}")
    # print(f"mass: {object.root_physx_view.get_masses()[0]}")
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated.any() or truncated.any():
        # print(f"Done at step {t}, terminated={terminated}, truncated={truncated}")

        # compute metrics
        eps = 1e-6
        gentle_ratio = max_contact_force / (N_min.to(device) + eps)

        success_mask = (
            (slip_distance <= SLIP_THRESH) &
            (gentle_ratio <= RATIO_THRESH) &
            lifted_now
        )

        num_success = success_mask.sum().item()
        success_rate = num_success / num_envs

        print("=== Evaluation ===")
        print("Slip distance:", slip_distance)
        print("Max contact force:", max_contact_force)
        print("N_min:", N_min)
        print("Gentle ratio (F_max / N_min):", gentle_ratio)
        print("Lifted:", lifted_now)
        print("Success mask:", success_mask)
        print(f"Success rate: {success_rate:.3f}")

        # --- SAVE ALL DATA NEEDED FOR PLOTS ---
        out_path = args_cli.out_file
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        data = {
            # per-env scalars
            "slip_distance": slip_distance.detach().cpu(),
            "max_contact_force": max_contact_force.detach().cpu(),
            "N_min": N_min.detach().cpu(),
            "gentle_ratio": gentle_ratio.detach().cpu(),
            "success_mask": success_mask.detach().cpu(),
            "success_rate": success_rate,
            "masses": masses.detach().cpu(),
            "static_mu": static_mu.detach().cpu(),

            # time series (T x num_envs)
            "ts_contact_force": ts_contact_force[: t + 1].detach().cpu(),
            "ts_stiffness": ts_stiffness[: t + 1].detach().cpu(),
            "ts_slip_speed": ts_slip_speed[: t + 1].detach().cpu(),
            "ts_slip_flag": ts_slip_flag[: t + 1].detach().cpu(),
            "ts_lifted": ts_lifted[: t + 1].detach().cpu(),

            # metadata
            "dt_step": dt_step,
            "SLIP_THRESH": SLIP_THRESH,
            "RATIO_THRESH": RATIO_THRESH,
            "LIFT_HEIGHT_THRESH": LIFT_HEIGHT_THRESH,
        }

        torch.save(data, out_path)
        print(f"Saved metrics to: {out_path}")
