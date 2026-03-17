# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AGV (Autonomous Ground Vehicle) IIT KGP Software Task Round. Six independent tasks covering computer vision, autonomous navigation, and robotics. Each task is implemented in its own `Task-N/` directory. All code must run inside the provided Docker container.

## Docker Environment

- **Image:** `agviitkgp/task:24`
- **Windows:** `docker run --name agvdocker -e DISPLAY=host.docker.internal:0 -it agviitkgp/task:24` (requires VcXsrv)
- **macOS:** `docker run --name agvdocker -e DISPLAY=$(ipconfig getifaddr en0):0 -it agviitkgp/task:24` (requires XQuartz)
- **Restart:** `docker start -ai agvdocker`

## Task Summary

| # | Name | Domain | Key Libraries |
|---|------|--------|--------------|
| 1 | Optical Flow | Sparse optical flow (Lucas-Kanade), obstacle avoidance in PyBullet sim | OpenCV, NumPy, PyBullet |
| 2 | Dynamic Obstacle Avoidance | Controller design (PID/Pure Pursuit/Follow the Gap) in C++ simulator | C++ simulator API, LiDAR |
| 3 | Multi-Agent Rendezvous | Localization, map-merging, multi-agent nav in GPS-denied environments | C++ simulator, LiDAR, PRM/RRT |
| 4 | VizDoom | Maze navigation via planning (RRT*) and DFS exploration | ViZDoom (`pip install vizdoom`), .wad files |
| 5 | NDT Localization | Normal Distributions Transform for LiDAR-based localization | OpenCV, NumPy, Newton's Optimization |
| 6 | 3D Reconstruction & Visual Odometry | Sparse 3D reconstruction + monocular VO (PnP, RANSAC) | OpenCV, NumPy |

## Code Conventions

- Prefer OOP design
- Python tasks use OpenCV (`cv2`) and NumPy; visualization via Matplotlib or OpenCV
- Tasks 2 and 3 use a C++ simulator — templates in `main.cpp`, API in `simulation.hpp`
- Task 4 submissions go in the modified ViZDoom repo's `examples/` folder
- Task 6 VO pipeline uses camera intrinsics: fx=517.3, fy=516.5, cx=318.6, cy=255.3

## Important points
- commit changes to github frequently
- Document whatever changes you are doing to keep a clear track of what is going on


## Task-Specific Documentation

- **Task 1 — D4RT Pipeline:** See [`Task-1/d4rt.md`](Task-1/d4rt.md) for full documentation of the D4RT 4D reconstruction and tracking implementation (`Task-1/learning/08_d4rt_deepmind.py`). Covers architecture, API reference, CLI usage, output structure, performance, and design decisions. The input video (`OPTICAL_FLOW.mp4`) is 3 clips concatenated — the pipeline auto-segments and processes each independently.

## Evaluation Criteria

Solutions are judged on: pipeline robustness (graceful failure handling), understanding/originality, generalizability under stress testing, and novelty of approach.
