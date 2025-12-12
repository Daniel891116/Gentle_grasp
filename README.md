# Franka Gentle Lift (Isaac Lab)

This repository contains a manager-based Isaac Lab environment for a **gentle lifting** task using a **Franka Emika Panda** mounted on a table and a **YCB sugar box** placed on the same table.

The environment follows Isaac Lab’s manager-based MDP structure (actions, observations, rewards, terminations, events, curriculum).

We provide two controllers:

- A **baseline** controller.
- An **adaptive impedance control** controller.

Both controllers generate data for quantitative evaluation, which can then be visualized using the plotting script.

---

## Contents

- [Overview](#overview)  
- [Installation](#installation)  
- [Quick start](#quick-start)  
  - [1. Baseline controller](#1-baseline-controller)  
  - [2. Adaptive impedance controller](#2-adaptive-impedance-controller)  
  - [3. Plotting evaluation metrics](#3-plotting-evaluation-metrics)  
---

## Overview

The environment includes:

- Franka Emika Panda robot arm with gripper.  
- Table, ground plane, and dome light.  
- A rigid object (YCB can) loaded from Isaac Nucleus YCB assets via `UsdFileCfg`.  

The task is a **gentle lift**:

- The robot must lift the object to a target height.  
- The object should not slip or be dropped.  
- Contact forces should stay within a “gentleness” threshold.

---

## Installation

This project is designed to run **inside an Isaac Lab environment**, typically via the official Docker setup.

Please follow the Isaac Lab Docker deployment guide:

- [Isaac Lab Docker docs](href="https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html")

High-level steps:

1. Install and configure **Isaac Lab** and **Isaac Sim** following the official documentation.  
2. Build and start the Isaac Lab Docker container.  
3. Enter the container and ensure you can run Isaac Lab example scripts.  

Because this repository uses Nucleus-hosted assets via `ISAAC_NUCLEUS_DIR`, verify that:

- `ISAAC_NUCLEUS_DIR` is set correctly.
- Your system can access the YCB asset paths used in the environment.

---

## Quick start

Below we assume:

- You are inside the Docker container (or a working Isaac Lab environment).
- You are in the root directory of this repository (where the scripts live).

### 1. Baseline controller

Run the **baseline** method and save data:

```bash
python baseline.py
```

### 2. Adaptive impedance controller
Run the adaptive impedance control method and save data:

```bash
python adaptive.py
```

### 3. Plotting evaluation metrics
After both the baseline and adaptive data have been generated, run:

```bash
python plot_metrics.py
```
This script will:

- Load the saved result files produced by baseline.py and adaptive.py.

- Compute evaluation metrics such as:

    - Success rate.

    - Slip distance.

    - Minimum normal force / gentle contact ratio.

    - Lifted fraction and other task-specific measures.

- Generate comparison plots (e.g., bar plots and time-series curves) and save them as image files.

