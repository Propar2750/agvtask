# D4RT: Dynamic 4D Reconstruction and Tracking

## Overview

**File:** `Task-1/learning/08_d4rt_deepmind.py`
**Paper:** "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time" (Google DeepMind, Dec 2025) — [arXiv:2512.08924](https://arxiv.org/abs/2512.08924)
**Video input:** `Task-1/OPTICAL_FLOW.mp4` (3012 frames, 60fps, 1920x1080, 50.2s — 3 videos concatenated)
**Output directory:** `Task-1/d4rt_output/seg1/`, `seg2/`, `seg3/`

This is the flagship implementation for Task-1 (Optical Flow). It implements the D4RT query-based architecture using GPU-accelerated classical vision as a backend (since D4RT's official pretrained weights are not yet publicly released). It goes far beyond basic optical flow — performing point tracking, depth estimation, camera pose estimation, 3D reconstruction, and full 4D (3D + time) scene reconstruction.

## Architecture

```
VIDEO FRAMES
     │
     ▼
┌─────────────┐    Farneback dense optical flow (CUDA if available)
│  D4RTEncoder │──► Precomputes inter-frame flows, cached in D4RTSceneRepresentation
└─────────────┘
     │
     ▼
┌──────────────────────┐
│ D4RTSceneRepresentation │  dense_flows dict + lazy depth cache (Depth Anything V2)
└──────────────────────┘
     │
     ▼
┌──────────────────┐    Batched Lucas-Kanade with forward-backward consistency
│ D4RTQueryDecoder  │──► Answers spatiotemporal queries: "where is pixel (u,v)@t in frame t'?"
└──────────────────┘
     │
     ▼
┌───────┐    Unified interface for all tasks
│  D4RT  │──► track_points, estimate_depth, estimate_camera_pose, compute_flow,
└───────┘    reconstruct_3d, reconstruct_4d, track_with_redetection, detect_segments
```

## Key Classes

| Class | Role |
|-------|------|
| `GPUBackend` | Detects CUDA / NVIDIA OptFlow SDK availability at import time. Global `GPU` instance. |
| `DepthAnythingBackend` | Monocular depth via Depth Anything V2 (ViT-S, 25M params). Auto-downloads weights from HuggingFace. Runs on CUDA. |
| `D4RTSceneRepresentation` | Stores precomputed dense flows + lazy-computed depth maps per frame. |
| `OpticalFlowEngine` | Unified GPU/CPU optical flow: CUDA Farneback, CUDA Sparse LK, or CPU fallback. |
| `D4RTEncoder` | Encodes video by computing all inter-frame dense flows. |
| `D4RTQueryDecoder` | Batched sparse LK tracker with forward-backward consistency check. |
| `D4RT` | **Main class.** Unified model providing all downstream tasks. |

## D4RT Class — Public API

### Loading

| Method | Description |
|--------|-------------|
| `load_video(frames)` | Load from a list of BGR numpy arrays. Encodes immediately. |
| `load_video_from_path(path, max_frames=300, resize=(640,360))` | Load from video file. Default resizes to 640x360 for speed. |

### Core Tasks

| Method | Signature | Returns |
|--------|-----------|---------|
| `track_point(u, v, start_frame, end_frame)` | Single point tracking | `Dict` with positions_2d, positions_3d, confidences, occluded |
| `track_points(points, start_frame, end_frame)` | **Batched** multi-point tracking (200+ pts at once) | `List[Dict]` — same keys per point |
| `estimate_depth(frame_idx)` | Monocular depth (Depth Anything V2 or flow fallback) | `np.ndarray` (H,W) float32 |
| `estimate_camera_pose(frame_a, frame_b, K, num_points)` | Relative pose via Essential matrix + RANSAC | `(R, t)` — 3x3 rotation, 3x1 translation |
| `compute_flow(frame_a, frame_b)` | Optical flow (precomputed for adjacent, chained for distant) | `np.ndarray` (H,W,2) float32 |
| `reconstruct_3d(frame_idx, K, subsample)` | Single-frame 3D point cloud from depth | `(points_3d, colors)` |
| `reconstruct_4d(grid_step, start_frame, end_frame, K, keyframe_interval, chunk_size)` | **Full 4D reconstruction** — dense grid tracked in chunks with fresh re-creation per chunk | `Dict` with keyframe_clouds, alive_per_frame, metadata |
| `track_with_redetection(n_initial, redetect_every, min_alive_ratio, start_frame, end_frame)` | Long-video tracking with periodic feature re-detection | `List[Dict]` of tracklets |
| `track_rolling_window(n_points, start_frame, end_frame, replace_every, min_alive_ratio, trail_length, output_path, show)` | **Rolling window tracker** — maintains ~n_points active tracks by continuously replacing dead ones with fresh features. Each point carries a fading trail. Outputs annotated video with HUD. | `Dict` with per_frame data, total_born, total_died, alive stats |
| `detect_segments(n_segments, min_segment_frames)` | Scene-cut detection via frame difference + non-maximum suppression | `List[Tuple[int,int]]` — (start, end) per segment |

### Visualization

| Method | Description |
|--------|-------------|
| `visualize_tracking(points, ..., precomputed_tracks=)` | Renders tracking video with colored trajectories. Pass `precomputed_tracks` to avoid re-tracking. |
| `visualize_depth(frame_idx)` | Colorized depth map (MAGMA colormap). |
| `visualize_flow(frame_a, frame_b)` | HSV color-wheel flow visualization. |

## Camera Intrinsics

All 3D methods default to `K = [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]]` (same as Task-6 VO pipeline). Override via the `K` parameter.

## Running the Demo

```bash
# Default: 300 frames, 640x360, 3 segments
python Task-1/learning/08_d4rt_deepmind.py --headless

# Full video (all 3012 frames, ~5 min)
python Task-1/learning/08_d4rt_deepmind.py --max-frames 3012 --headless

# Custom segment count / resolution
python Task-1/learning/08_d4rt_deepmind.py --max-frames 3012 --num-segments 3 --resize 640 360 --headless
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `video` | `../OPTICAL_FLOW.mp4` | Path to video file |
| `--headless` | off | Disable GUI windows |
| `--max-frames` | 500 | Max frames to load |
| `--resize W H` | native | Resize resolution (omit for full res) |
| `--num-segments` | 3 | Number of video segments to split into |
| `--output-dir` | `d4rt_output/` | Output directory |

## Demo Pipeline (runs per segment)

1. **Point Tracking (DEMO 1):** 2000 uniform grid points tracked via batched LK across full segment. Saves `tracking_data.npz`, `tracking_video.mp4`, `tracking_trajectories.png`.
2. **Depth Estimation (DEMO 2):** Depth Anything V2 at mid-frame. Saves raw `.npy` + colorized `.png`.
3. **Optical Flow (DEMO 3):** Precomputed flow at multiple gaps. Saves HSV visualizations.
4. **Camera Pose (DEMO 4):** Essential matrix + RANSAC at multiple gaps. Saves `camera_poses.npz`.
5. **3D Reconstruction (DEMO 5):** Back-projected point cloud at mid-frame. Saves `.ply` + `.npz`.
6. **4D Reconstruction (DEMO 6):** 576-point dense grid (18x32, step=20) tracked in 100-frame chunks with fresh grid per chunk. Saves per-keyframe `.ply` sequence, `4d_reconstruction.npz`, `4d_vis.png`.
7. **Rolling Window Tracker (DEMO 7):** Maintains ~150 active points at all times. Dead/occluded points are replaced every 5 frames with fresh features from uncovered regions. Each point draws a fading 30-frame trail. Saves `rolling_window.mp4`, `rolling_window_stats.png`.

## Output Structure

```
d4rt_output/
├── seg1/                          # Segment 1 (frames 0–1326)
│   ├── tracking_data.npz          # 200 point tracks (positions, confidence, occlusion)
│   ├── tracking_video.mp4         # Annotated tracking visualization
│   ├── tracking_trajectories.png  # Matplotlib trajectory + displacement plots
│   ├── depth_frame663_raw.npy     # Raw depth map (float32, HxW)
│   ├── depth_frame663_color.png   # Colorized depth (MAGMA)
│   ├── flow_gap1.png              # Optical flow HSV visualizations
│   ├── flow_gap10.png
│   ├── flow_gap*.png
│   ├── camera_poses.npz           # R, t at multiple frame gaps
│   ├── pointcloud_frame663.ply    # 3D point cloud (viewable in MeshLab)
│   ├── pointcloud_frame663.npz
│   ├── 4d_reconstruction.npz      # Alive-per-frame, grid shape, K matrix
│   ├── 4d_kf_0000.ply             # Keyframe 3D snapshots (temporal sequence)
│   ├── 4d_kf_0132.ply
│   ├── ...
│   ├── 4d_vis.png                 # Track survival + keyframe density plots
│   ├── rolling_window.mp4         # Rolling window tracker with fading trails
│   └── rolling_window_stats.png   # Active point count over time
├── seg2/                          # Segment 2 (frames 1327–2456)
│   └── ... (same structure)
└── seg3/                          # Segment 3 (frames 2457–3011)
    └── ... (same structure)
```

## Performance Characteristics

Optimized for full resolution (1920x1080) with 2000 tracking points:

| Stage | Time (est.) | Notes |
|-------|------------|-------|
| Encoding | **~0s** | Lazy mode — no upfront Farneback computation |
| Segment detection | ~5s | Frame-diff + NMS at full res |
| Point tracking (2000 grid pts) | ~30-100s per segment | Batched LK, FB skip every 3rd frame, 15x15 window |
| Depth Anything V2 | ~0.01s per frame | CUDA ViT-S, cached per frame |
| 4D reconstruction (~5200 pts) | ~40-150s per segment | Chunked, grid_step=20 at full res |
| Rolling window (1500 pts) | ~60-200s per segment | Polyline rendering, no per-segment fading |
| Flow visualization | ~1s per flow | On-demand lazy computation at quarter-res |
| **Total (3 segments, full res)** | **~6-15 min** | Well under the 30-min target |

### Optimization Summary (vs previous version)

| Optimization | Speedup | Impact |
|-------------|---------|--------|
| Lazy flow encoding | ~120-240s saved | No upfront Farneback for 3011 frames |
| Grid points (vs goodFeaturesToTrack) | Better coverage + faster init | Center-of-frame tracked from frame 0 |
| LK window 21→15, iters 30→20 | ~30% per-point | Less compute per LK call |
| FB check every 3rd frame | ~33% LK calls saved | Skip backward tracking most frames |
| Polyline rendering | ~50-120s saved | Single cv2.polylines vs N cv2.line calls |
| Batch PLY writing (numpy) | ~10-20s saved | np.savetxt vs Python loop |

## Key Design Decisions

1. **Lazy flow encoding:** Dense Farneback flows are computed on-demand at quarter-resolution (480x270) only when `compute_flow()` is called. LK tracking uses raw full-res frames directly, so this has zero impact on tracking quality.

2. **Grid-based point initialization:** Uniform grid sampling replaces `cv2.goodFeaturesToTrack`. Guarantees spatial coverage of all regions including frame center from the very first frame. Shi-Tomasi corner detection biases toward textured regions and can miss important objects in uniform areas.

3. **Chunked 4D reconstruction:** Without chunking, tracks die off completely over long sequences (576 → 0 alive after ~750 frames). Chunking with fresh grid re-creation per 100-frame chunk maintains dense coverage (min ~300 alive per chunk).

4. **Scene-cut detection with NMS:** The input video is 3 clips concatenated. Frame-diff detects transitions; non-maximum suppression with minimum gap ensures well-separated cut points. Each segment is processed independently — no tracking across cuts.

5. **Proper 3D back-projection:** Uses camera intrinsics `(fx, fy, cx, cy)` for `x = (u-cx)*z/fx`, not the incorrect `x = u*z`. This matches Task-6 VO conventions.

6. **Precomputed track reuse:** `visualize_tracking` accepts `precomputed_tracks=` to avoid re-running expensive batched LK when tracks are already computed.

7. **Depth Anything V2 over flow-based depth:** Neural monocular depth (ViT-S, CUDA) gives metric-scale depth maps. Flow-based depth (`1/|flow|`) is only used as a fallback when the model is unavailable.

## Extending / Modifying

- **Change grid density:** `reconstruct_4d(grid_step=10)` for denser grid (~2300 points). Increases time proportionally.
- **Change chunk size:** `reconstruct_4d(chunk_size=50)` for more frequent re-creation. Better track survival but more overhead.
- **Add new video:** Just pass the path: `python 08_d4rt_deepmind.py path/to/video.mp4`
- **Use from code:**
  ```python
  from learning.08_d4rt_deepmind import D4RT
  model = D4RT()
  model.load_video_from_path("video.mp4", resize=(640, 360))
  tracks = model.track_points(points_array, start_frame=0, end_frame=100)
  recon = model.reconstruct_4d(grid_step=16)
  ```

## Dependencies

- `opencv-python` (cv2) — core flow, tracking, visualization
- `numpy` — array operations
- `torch` + `depth_anything_v2` — neural depth estimation (auto-downloaded)
- `matplotlib` — trajectory / reconstruction plots (optional, graceful fallback)

## References

- D4RT paper: https://arxiv.org/abs/2512.08924
- D4RT project page: https://d4rt-paper.github.io/
- Depth Anything V2: https://depth-anything-v2.github.io/
- 4DS weights (same team): https://github.com/google-deepmind/representations4d
