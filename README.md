# Franka Gentle Lift (Isaac Lab)

This repository contains a manager-based Isaac Lab environment for a gentle lifting task using a **Franka Emika Panda** mounted on a table and a **YCB can** (e.g., tomato soup can) placed on the same table.

The environment follows Isaac Lab’s manager-based MDP structure (actions, observations, rewards, terminations, events, curriculum).

**Main entry point**
- `load_scene.py`

Use this script to load the scene, validate your configuration, and run simple action tests (arm IK motion, gripper open/close).

---

## Contents

- Overview   
- Installation  
- Quick start   

---

## Overview

The environment includes:

- Franka Panda robot
- Table + ground plane + dome light
- A rigid object loaded from Isaac Nucleus YCB assets via `UsdFileCfg`

---

## Installation

Follow the instructions from this [tutorial](href="https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html")

1. Start the container
2. Enter the container

Because this repo uses Nucleus-hosted assets (`ISAAC_NUCLEUS_DIR`), ensure your system can access the YCB asset paths.

---

## Quick start

Run the main script:

```bash
cd Isaaclab/scripts
python load_scene.py

