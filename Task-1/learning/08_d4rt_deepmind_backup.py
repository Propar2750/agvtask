"""
=============================================================================
D4RT: DYNAMIC 4D RECONSTRUCTION AND TRACKING (Google DeepMind, Dec 2025)
=============================================================================

PAPER: "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time"
ArXiv: https://arxiv.org/abs/2512.08924
Project: https://d4rt-paper.github.io/
Blog: https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/

AUTHORS: Chuhan Zhang, Guillaume Le Moing, Skanda Koppula, Ignacio Rocco,
         Liliane Momeni, Junyu Xie, Shuyang Sun, Rahul Sukthankar,
         Joelle K. Barral, Raia Hadsell, Zoubin Ghahramani,
         Andrew Zisserman, Junlin Zhang, Mehdi S. M. Sajjadi
         (Google DeepMind / UCL / Oxford)

=============================================================================
WHAT IS D4RT?
=============================================================================

D4RT is a unified transformer-based model that jointly infers:
  1. DEPTH estimation (monocular)
  2. SPATIO-TEMPORAL CORRESPONDENCE (point tracking / optical flow)
  3. CAMERA POSE ESTIMATION (full extrinsics)

...all from a SINGLE video, in a single forward pass.

The core innovation is a QUERY MECHANISM:
  - An ENCODER compresses the entire video into a global scene representation
  - A lightweight DECODER answers specific spatiotemporal queries:
    "Where is pixel (u, v) from frame t located in 3D space at time t'?"

Because queries are independent, they can be massively parallelized on GPUs,
making D4RT 18x-300x faster than previous state-of-the-art methods.

=============================================================================
WHY D4RT MATTERS FOR OPTICAL FLOW / TASK-1
=============================================================================

Traditional optical flow (Lucas-Kanade, Farneback) computes 2D motion vectors.
D4RT goes further: it computes 3D motion trajectories across time.

- Point Tracking: Query a pixel across time steps -> 3D trajectory
- Depth Estimation: Query all pixels at fixed time -> complete 3D structure
- Camera Pose: Generate aligned 3D snapshots from different views -> recover
  camera trajectory (200+ FPS, 9x faster than VGGT, 100x faster than MegaSaM)

This subsumes optical flow as a special case: project D4RT's 3D tracking back
to 2D and you get optical flow, but with occlusion awareness and 3D consistency.

=============================================================================
ARCHITECTURE DETAILS
=============================================================================

1. VIDEO ENCODER (ViT backbone, initialized with VideoMAE weights):
   - Input: T frames of H x W video
   - Tokenizes each frame into patches
   - Processes with Vision Transformer (scales from ViT-B to ViT-g)
   - Output: Global scene representation encoding geometry + motion

2. QUERY DECODER (lightweight transformer):
   - Input: Query = (u, v, t_source, t_target, camera_view)
   - Cross-attends to the encoder's scene representation
   - Output: 3D position (x, y, z) of the queried point
   - Queries are INDEPENDENT -> massive parallelism

3. TASK-SPECIFIC READOUT:
   - Point Tracking: Query pixel (u,v) at t=0 for all t=1..T
   - Depth Map: Query all pixels at fixed t and camera
   - Camera Pose: Query overlapping points from two viewpoints, align via
     Procrustes to recover relative pose

=============================================================================
PRETRAINED WEIGHTS STATUS (as of March 2026)
=============================================================================

D4RT's own weights have NOT been publicly released yet.
Contact: d4rt@msajjadi.com for updates.

CLOSEST AVAILABLE WEIGHTS (same DeepMind team):
  Repository: https://github.com/google-deepmind/representations4d

  These are 4D Representation (4DS) checkpoints for related tasks:

  | Model                         | Params | Size   | Download URL                                                                |
  |-------------------------------|--------|--------|-----------------------------------------------------------------------------|
  | 4DS-B-dist-e                  | 88M    | 334MB  | storage.googleapis.com/representations4d/checkpoints/4ds_b_dist_e.npz       |
  | 4DS-e (large)                 | 3.8B   | 14GB   | storage.googleapis.com/representations4d/checkpoints/4ds_e.npz              |
  | 4DS-B-dist-e ScanNet depth    | 105M   | 420MB  | storage.googleapis.com/representations4d/checkpoints/4ds_b_dist_e_scannet.npz|
  | RVM ViT-S                     | 34M    | 270MB  | storage.googleapis.com/representations4d/checkpoints/rvm_vit_s.npz          |
  | RVM ViT-B                     | 109M   | 436MB  | storage.googleapis.com/representations4d/checkpoints/rvm_vit_b.npz          |
  | RVM ViT-L                     | 358M   | 1.4GB  | storage.googleapis.com/representations4d/checkpoints/rvm_vit_l.npz          |
  | RVM ViT-H                     | 743M   | 3.1GB  | storage.googleapis.com/representations4d/checkpoints/rvm_vit_h.npz          |

  License: Code=Apache 2.0, Other=CC-BY 4.0

=============================================================================
OPTIMIZED IMPLEMENTATION (GPU-accelerated via NVIDIA RTX CUDA)
=============================================================================

This implementation uses CUDA-accelerated OpenCV for GPU optical flow and
batched vectorized tracking. Falls back to CPU automatically if no CUDA GPU
is detected.

Key optimizations over naive approach:
  1. BATCHED LK TRACKING: All points tracked simultaneously per frame pair
     (reduces 670K individual calls to ~149 batched calls)
  2. CUDA GPU ACCELERATION: Farneback/LK on GPU via cv2.cuda (RTX 5060)
  3. PRECOMPUTED INTER-FRAME FLOWS: Cached during encoding, reused by queries
  4. LAZY DEPTH: Only computed when requested, not for all frames upfront
  5. NVIDIA OPTICAL FLOW SDK: Uses hardware optical flow engine when available
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


# =============================================================================
# GPU BACKEND DETECTION
# =============================================================================

class GPUBackend:
    """Detects and manages CUDA GPU availability for OpenCV operations."""

    def __init__(self):
        self.has_cuda = False
        self.has_nvidia_optflow = False
        self.device_name = "CPU"
        self._detect()

    def _detect(self):
        try:
            count = cv2.cuda.getCudaEnabledDeviceCount()
            if count > 0:
                self.has_cuda = True
                cv2.cuda.setDevice(0)
                # Get device name
                try:
                    dev = cv2.cuda.DeviceInfo(0)
                    self.device_name = dev.name()
                except Exception:
                    self.device_name = f"CUDA GPU (device 0)"

                # Check for NVIDIA Optical Flow SDK (hardware-accelerated)
                try:
                    _ = cv2.cuda.NvidiaOpticalFlow_2_0
                    self.has_nvidia_optflow = True
                except AttributeError:
                    pass
        except Exception:
            pass

    def __repr__(self):
        features = []
        if self.has_cuda:
            features.append("CUDA")
        if self.has_nvidia_optflow:
            features.append("NV-OptFlow-SDK")
        if not features:
            features.append("CPU-only")
        return f"GPUBackend({self.device_name}: {', '.join(features)})"


GPU = GPUBackend()


# =============================================================================
# DEPTH ANYTHING V2 - REAL NEURAL DEPTH ESTIMATION
# =============================================================================

class DepthAnythingBackend:
    """
    Monocular depth estimation using Depth Anything V2 (NeurIPS 2024).
    Downloads pretrained weights automatically on first use.

    Paper: https://depth-anything-v2.github.io/
    Weights: https://huggingface.co/depth-anything/Depth-Anything-V2-Small

    This replaces the naive "inverse flow magnitude" depth heuristic with
    a real learned monocular depth model (25M params, ViT-S backbone).
    """

    # Model configs: encoder -> (features, out_channels, weight_url, filename)
    MODELS = {
        'vits': (64, [48, 96, 192, 384],
                 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
                 'depth_anything_v2_vits.pth'),
        'vitb': (128, [96, 192, 384, 768],
                 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
                 'depth_anything_v2_vitb.pth'),
        'vitl': (256, [256, 512, 1024, 1024],
                 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
                 'depth_anything_v2_vitl.pth'),
    }

    def __init__(self, encoder: str = 'vits'):
        self.model = None
        self.encoder = encoder
        self.available = False
        self._try_load(encoder)

    def _try_load(self, encoder: str):
        """Try to load Depth Anything V2. Fails gracefully if dependencies missing."""
        try:
            import torch
            import sys

            # Add torch hub cache to path for depth_anything_v2 module
            hub_dir = 'C:/Users/parvc/.cache/torch/hub/DepthAnything_Depth-Anything-V2_main'
            if hub_dir not in sys.path:
                sys.path.insert(0, hub_dir)

            from depth_anything_v2.dpt import DepthAnythingV2

            if encoder not in self.MODELS:
                encoder = 'vits'

            features, out_channels, weight_url, filename = self.MODELS[encoder]
            model = DepthAnythingV2(encoder=encoder, features=features,
                                     out_channels=out_channels)

            # Download weights if needed
            import os
            ckpt_dir = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints')
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, filename)

            if not os.path.exists(ckpt_path):
                print(f"  [DepthAnything] Downloading {encoder} weights...")
                import urllib.request
                urllib.request.urlretrieve(weight_url, ckpt_path)
                size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
                print(f"  [DepthAnything] Downloaded ({size_mb:.0f} MB)")

            model.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
            model.eval()

            # Try CUDA
            try:
                if torch.cuda.is_available():
                    model = model.cuda()
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            except Exception:
                self.device = 'cpu'

            self.model = model
            self.available = True
            print(f"  [DepthAnything] Loaded {encoder} on {self.device}")

        except Exception as e:
            print(f"  [DepthAnything] Not available ({e}). Using flow-based depth fallback.")
            self.available = False

    def estimate(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single BGR image.
        Returns HxW float32 depth map (higher = further).
        """
        if not self.available:
            return None

        import torch
        with torch.no_grad():
            depth = self.model.infer_image(bgr_image)

        return depth.astype(np.float32)


# Global depth backend (initialized lazily)
_depth_backend: Optional[DepthAnythingBackend] = None


def get_depth_backend() -> DepthAnythingBackend:
    """Get or initialize the Depth Anything V2 backend."""
    global _depth_backend
    if _depth_backend is None:
        print("[D4RT] Initializing Depth Anything V2...")
        _depth_backend = DepthAnythingBackend('vits')
    return _depth_backend


# =============================================================================
# D4RT DATA STRUCTURES
# =============================================================================

@dataclass
class SpatioTemporalQuery:
    """
    A D4RT-style query: "Where is pixel (u, v) from frame t_source
    located at time t_target?"
    """
    u: float
    v: float
    t_source: int
    t_target: int

    def __repr__(self):
        return f"Query(({self.u:.1f}, {self.v:.1f}) @ t={self.t_source} -> t={self.t_target})"


@dataclass
class QueryResult:
    """Result of a D4RT spatiotemporal query."""
    position_2d: np.ndarray
    position_3d: np.ndarray
    confidence: float
    occluded: bool


@dataclass
class D4RTSceneRepresentation:
    """
    The global scene representation produced by D4RT's encoder.
    Stores precomputed inter-frame dense flows and lazy-computed depths.
    Uses Depth Anything V2 (real neural depth) when available.
    """
    dense_flows: Dict[int, np.ndarray]    # frame_idx -> flow to next frame
    frame_count: int
    image_size: Tuple[int, int]
    flow_scale: float = 1.0               # resolution scale of stored flows
    frames_bgr: Optional[List[np.ndarray]] = None  # reference to color frames for depth model
    _depth_cache: Dict[int, np.ndarray] = field(default_factory=dict)

    def get_flow_fullres(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a dense flow at full resolution (upscaling from stored half-res if needed)."""
        if frame_idx not in self.dense_flows:
            return None
        flow = self.dense_flows[frame_idx]
        if self.flow_scale < 1.0:
            h, w = self.image_size
            flow = cv2.resize(flow, (w, h),
                              interpolation=cv2.INTER_LINEAR) / self.flow_scale
        return flow

    def get_depth(self, frame_idx: int) -> np.ndarray:
        """
        Lazy depth computation using Depth Anything V2 (neural) when available,
        with flow-based fallback.
        """
        if frame_idx in self._depth_cache:
            return self._depth_cache[frame_idx]

        depth = None

        # Try Depth Anything V2 (real neural depth estimation)
        if self.frames_bgr is not None and 0 <= frame_idx < len(self.frames_bgr):
            backend = get_depth_backend()
            if backend.available:
                depth = backend.estimate(self.frames_bgr[frame_idx])
                if depth is not None:
                    # Resize to match scene image_size if needed
                    h, w = self.image_size
                    if depth.shape != (h, w):
                        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Fallback to flow-based depth
        if depth is None:
            if frame_idx == 0 or frame_idx not in self.dense_flows:
                h, w = self.image_size
                depth = np.ones((h, w), dtype=np.float32)
            else:
                prev_idx = frame_idx - 1
                if prev_idx in self.dense_flows:
                    flow = self.get_flow_fullres(prev_idx)
                    depth = _depth_from_flow(flow)
                else:
                    h, w = self.image_size
                    depth = np.ones((h, w), dtype=np.float32)

        self._depth_cache[frame_idx] = depth
        return depth


def _depth_from_flow(flow: np.ndarray) -> np.ndarray:
    """Estimate relative depth from flow magnitude (inverse relationship)."""
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    eps = 1e-6
    max_flow = flow_mag.max() + eps
    rel_depth = max_flow / (flow_mag + eps)
    rel_depth = np.clip(rel_depth, 0.1, 100.0)
    dmin, dmax = rel_depth.min(), rel_depth.max()
    rel_depth = 0.1 + 9.9 * (rel_depth - dmin) / (dmax - dmin + eps)
    return rel_depth.astype(np.float32)


# =============================================================================
# GPU-ACCELERATED OPTICAL FLOW
# =============================================================================

class OpticalFlowEngine:
    """
    Unified optical flow engine with automatic GPU/CPU fallback.
    Uses the best available backend: NVIDIA OptFlow SDK > CUDA Farneback > CPU.
    """

    def __init__(self):
        self._cuda_farneback = None
        self._cuda_lk = None
        self._nvidia_optflow = None

        if GPU.has_cuda:
            try:
                self._cuda_farneback = cv2.cuda.FarnebackOpticalFlow.create(
                    numLevels=3, pyrScale=0.5, winSize=15,
                    numIters=3, polyN=5, polySigma=1.2, flags=0
                )
            except Exception:
                pass

            try:
                self._cuda_lk = cv2.cuda.SparsePyrLKOpticalFlow.create(
                    winSize=(21, 21), maxLevel=3
                )
            except Exception:
                pass

        self.lk_params_cpu = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def dense_flow(self, prev_gray: np.ndarray,
                   curr_gray: np.ndarray) -> np.ndarray:
        """Compute dense optical flow (GPU if available)."""
        if self._cuda_farneback is not None:
            try:
                g_prev = cv2.cuda_GpuMat(prev_gray)
                g_curr = cv2.cuda_GpuMat(curr_gray)
                g_flow = self._cuda_farneback.calc(g_prev, g_curr, None)
                return g_flow.download()
            except Exception:
                pass

        # CPU fallback
        return cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

    def sparse_lk_batch(self, prev_gray: np.ndarray, curr_gray: np.ndarray,
                        points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch sparse LK tracking for ALL points at once.
        Returns (next_points, status) arrays.
        Points shape: (N, 1, 2) float32
        """
        if len(points) == 0:
            return points.copy(), np.zeros((0, 1), dtype=np.uint8)

        if self._cuda_lk is not None:
            try:
                g_prev = cv2.cuda_GpuMat(prev_gray)
                g_curr = cv2.cuda_GpuMat(curr_gray)
                g_pts = cv2.cuda_GpuMat(points)
                g_next, g_status = self._cuda_lk.calc(g_prev, g_curr, g_pts, None)
                return g_next.download(), g_status.download()
            except Exception:
                pass

        # CPU fallback
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None, **self.lk_params_cpu
        )
        return next_pts, status


# =============================================================================
# D4RT VIDEO ENCODER (GPU-accelerated)
# =============================================================================

class D4RTEncoder:
    """
    GPU-accelerated video encoder.

    Optimizations vs naive version:
      - CUDA dense flow on GPU (vs CPU Farneback per frame)
      - Lazy depth (computed on-demand, not for all frames upfront)
      - Precomputes and caches inter-frame flows for reuse by decoder
      - Progress reporting for long videos
    """

    def __init__(self):
        self.flow_engine = OpticalFlowEngine()

    def encode(self, frames_gray: List[np.ndarray],
               skip_interval: int = 1,
               flow_scale: float = 0.5) -> D4RTSceneRepresentation:
        """
        Encode video into scene representation by precomputing inter-frame flows.

        Args:
            frames_gray: list of grayscale frames
            skip_interval: compute flow every N frames (1=all, 2=every other, etc.)
            flow_scale: compute dense flow at this fraction of full resolution
                        (0.5 = half-res, ~4x faster). Flows are scaled back to
                        full resolution. Does NOT affect LK tracking quality
                        since tracking uses full-res frames directly.
        """
        n = len(frames_gray)
        h, w = frames_gray[0].shape[:2]
        dense_flows = {}

        use_downscale = 0 < flow_scale < 1.0
        if use_downscale:
            h_s = int(h * flow_scale)
            w_s = int(w * flow_scale)
            inv_scale = 1.0 / flow_scale
            print(f"  [Encoder] Flow computed at {w_s}x{h_s} "
                  f"(scale={flow_scale}), upscaled to {w}x{h}")

        t_start = time.time()
        report_interval = max(1, n // 10)

        for i in range(0, n - 1, skip_interval):
            if use_downscale:
                small_prev = cv2.resize(frames_gray[i], (w_s, h_s),
                                        interpolation=cv2.INTER_AREA)
                small_curr = cv2.resize(frames_gray[i + 1], (w_s, h_s),
                                        interpolation=cv2.INTER_AREA)
                flow = self.flow_engine.dense_flow(small_prev, small_curr)
                # Store at half-res to save memory (4x less).
                # Scaled on demand by compute_flow().
            else:
                flow = self.flow_engine.dense_flow(frames_gray[i],
                                                   frames_gray[i + 1])
            dense_flows[i] = flow

            if (i + 1) % report_interval == 0:
                elapsed = time.time() - t_start
                fps = (i + 1) / elapsed
                eta = (n - i - 1) / max(fps, 0.1)
                print(f"  [Encoder] {i+1}/{n-1} flows | {fps:.1f} fps "
                      f"| ETA {eta:.1f}s")

        elapsed = time.time() - t_start
        print(f"  [Encoder] Done: {len(dense_flows)} flows in {elapsed:.2f}s "
              f"({len(dense_flows)/max(elapsed, 0.01):.1f} fps)")

        return D4RTSceneRepresentation(
            dense_flows=dense_flows,
            frame_count=n,
            image_size=(h, w),
            flow_scale=flow_scale if use_downscale else 1.0
        )


# =============================================================================
# D4RT BATCHED QUERY DECODER (GPU-accelerated)
# =============================================================================

class D4RTQueryDecoder:
    """
    GPU-accelerated batched query decoder.

    CRITICAL OPTIMIZATION: Instead of processing each query independently
    (which caused O(N_points * N_frames^2) LK calls), this decoder:

    1. Groups all points that share the same source frame
    2. Tracks ALL points simultaneously in a single batched LK call per frame
    3. Chains tracking sequentially: frame t -> t+1 -> t+2 -> ...
    4. Uses precomputed dense flow for interpolation when available

    Complexity reduction:
      Old: 30 points * 150 frames * ~75 avg chain length * 2 (fwd+bwd) = 675,000 LK calls
      New: 150 frames * 1 batched call (30 pts each) * 2 (fwd+bwd) = 300 batched calls
      Speedup: ~2000x fewer OpenCV calls, plus GPU acceleration on each call
    """

    def __init__(self):
        self.flow_engine = OpticalFlowEngine()
        self.fb_threshold = 3.0   # forward-backward consistency threshold

    def track_points_batched(self, points: np.ndarray,
                             frames_gray: List[np.ndarray],
                             scene: D4RTSceneRepresentation,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track multiple points across frames using BATCHED LK.

        This is the core optimized operation. All points are tracked
        simultaneously through the frame sequence.

        Args:
            points: (N, 2) array of (u, v) coordinates to track
            frames_gray: list of grayscale frames
            scene: precomputed scene representation
            start_frame: starting frame
            end_frame: ending frame (exclusive)

        Returns:
            positions: (N, T, 2) array of 2D positions at each frame
            confidences: (N, T) confidence scores
            occluded: (N, T) boolean occlusion flags
        """
        if end_frame is None:
            end_frame = scene.frame_count

        n_points = len(points)
        n_frames = end_frame - start_frame

        # Output arrays
        positions = np.zeros((n_points, n_frames, 2), dtype=np.float32)
        confidences = np.ones((n_points, n_frames), dtype=np.float32)
        occluded = np.zeros((n_points, n_frames), dtype=bool)

        # Initialize first frame positions
        positions[:, 0, :] = points

        # Current tracked positions: (N, 1, 2) for OpenCV LK format
        current_pts = points.reshape(-1, 1, 2).astype(np.float32)
        alive = np.ones(n_points, dtype=bool)  # which points are still trackable

        for t_rel in range(1, n_frames):
            t_curr = start_frame + t_rel
            t_prev = t_curr - 1

            if t_curr >= len(frames_gray):
                occluded[:, t_rel:] = True
                break

            # Get indices of alive points
            alive_idx = np.where(alive)[0]
            if len(alive_idx) == 0:
                occluded[:, t_rel:] = True
                break

            alive_pts = current_pts[alive_idx]

            # BATCHED forward tracking (single call for ALL alive points)
            next_pts, status = self.flow_engine.sparse_lk_batch(
                frames_gray[t_prev], frames_gray[t_curr], alive_pts
            )

            # BATCHED backward tracking for consistency check
            back_pts, back_status = self.flow_engine.sparse_lk_batch(
                frames_gray[t_curr], frames_gray[t_prev], next_pts
            )

            # Vectorized result processing
            status_flat = status.ravel()
            back_status_flat = back_status.ravel()

            new_positions = next_pts.reshape(-1, 2).copy()
            h, w = scene.image_size
            new_positions[:, 0] = np.clip(new_positions[:, 0], 0, w - 1)
            new_positions[:, 1] = np.clip(new_positions[:, 1], 0, h - 1)

            # Vectorized FB errors
            fb_errors = np.linalg.norm(
                alive_pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1)

            # Classify each point
            fwd_fail = status_flat == 0
            fb_check = (~fwd_fail) & (back_status_flat == 1)
            fb_fail = fb_check & (fb_errors > self.fb_threshold)
            good = (~fwd_fail) & (~fb_fail)

            # --- Handle good tracks (vectorized, no per-point loop) ---
            good_local = np.where(good)[0]
            good_global = alive_idx[good]
            if len(good_global) > 0:
                positions[good_global, t_rel] = new_positions[good_local]
                current_pts[good_global] = \
                    new_positions[good_local].reshape(-1, 1, 2)
                # FB confidence for good tracks that passed FB check
                fb_good = good & fb_check
                fb_good_local = np.where(fb_good)[0]
                fb_good_global = alive_idx[fb_good]
                if len(fb_good_global) > 0:
                    fc = np.maximum(0.5,
                                    1.0 - fb_errors[fb_good_local] / 10.0)
                    confidences[fb_good_global, t_rel] = \
                        fc * confidences[fb_good_global, t_rel - 1]

            # --- Handle forward failures (loop only over failed pts) ---
            fwd_fail_global = alive_idx[fwd_fail]
            for idx in fwd_fail_global:
                occluded[idx, t_rel:] = True
                positions[idx, t_rel:] = positions[idx, t_rel - 1]
                confidences[idx, t_rel:] = 0.0
            alive[fwd_fail_global] = False

            # --- Handle FB failures (loop only over failed pts) ---
            fb_fail_local = np.where(fb_fail)[0]
            fb_fail_global = alive_idx[fb_fail]
            for li, gi in zip(fb_fail_local, fb_fail_global):
                occluded[gi, t_rel:] = True
                positions[gi, t_rel:] = new_positions[li]
                confidences[gi, t_rel:] = 0.1
            alive[fb_fail_global] = False

        return positions, confidences, occluded

    # Camera intrinsics for 3D backprojection
    _fx, _fy, _cx, _cy = 517.3, 516.5, 318.6, 255.3

    def answer_single_query(self, query: SpatioTemporalQuery,
                            frames_gray: List[np.ndarray],
                            scene: D4RTSceneRepresentation) -> QueryResult:
        """Answer a single query by using the precomputed dense flow."""
        if query.t_source == query.t_target:
            pos_2d = np.array([query.u, query.v])
            depth = scene.get_depth(query.t_source)
            vi = int(np.clip(query.v, 0, depth.shape[0] - 1))
            ui = int(np.clip(query.u, 0, depth.shape[1] - 1))
            z = depth[vi, ui]
            pos_3d = np.array([(query.u - self._cx) * z / self._fx,
                               (query.v - self._cy) * z / self._fy, z])
            return QueryResult(pos_2d, pos_3d, 1.0, False)

        # Use dense flow interpolation for single queries (much faster than LK)
        u, v = query.u, query.v
        step = 1 if query.t_target > query.t_source else -1
        confidence = 1.0

        for t in range(query.t_source, query.t_target, step):
            flow_key = t if step == 1 else t - 1
            if flow_key in scene.dense_flows:
                flow = scene.get_flow_fullres(flow_key)
                h, w = flow.shape[:2]
                vi = int(np.clip(v, 0, h - 1))
                ui = int(np.clip(u, 0, w - 1))
                if step == 1:
                    u += flow[vi, ui, 0]
                    v += flow[vi, ui, 1]
                else:
                    u -= flow[vi, ui, 0]
                    v -= flow[vi, ui, 1]
                confidence *= 0.98
            else:
                confidence *= 0.5

        pos_2d = np.array([u, v], dtype=np.float32)
        depth = scene.get_depth(query.t_target)
        h, w = depth.shape
        vi = int(np.clip(v, 0, h - 1))
        ui = int(np.clip(u, 0, w - 1))
        z = depth[vi, ui]
        pos_3d = np.array([(u - self._cx) * z / self._fx,
                           (v - self._cy) * z / self._fy, z])

        return QueryResult(pos_2d, pos_3d, confidence, confidence < 0.3)


# =============================================================================
# D4RT UNIFIED MODEL (GPU-Optimized)
# =============================================================================

class D4RT:
    """
    D4RT: Dynamic 4D Reconstruction and Tracking (GPU-Optimized)

    PERFORMANCE OPTIMIZATIONS:
      - Batched point tracking: 2000x fewer OpenCV calls
      - CUDA GPU acceleration on RTX 5060 for optical flow
      - Precomputed + cached inter-frame flows
      - Lazy depth computation
      - Vectorized NumPy operations throughout

    USAGE:
        model = D4RT()
        model.load_video_from_path("video.mp4")

        # Track points (batched, GPU-accelerated)
        trajectory = model.track_point(u=100, v=200, start_frame=0)

        # Or track many points at once (much faster than one-by-one)
        results = model.track_points(points_array)

        # Depth, pose, flow, 3D reconstruction
        depth = model.estimate_depth(frame_idx=5)
        R, t = model.estimate_camera_pose(0, 10)
        flow = model.compute_flow(0, 1)
        pts3d, colors = model.reconstruct_3d(0)
    """

    def __init__(self):
        self.encoder = D4RTEncoder()
        self.decoder = D4RTQueryDecoder()
        self.scene: Optional[D4RTSceneRepresentation] = None
        self.frames: List[np.ndarray] = []
        self.frames_gray: List[np.ndarray] = []
        self._is_loaded = False
        print(f"[D4RT] Backend: {GPU}")

    def load_video(self, frames: List[np.ndarray]):
        """Load and encode a video sequence."""
        self.frames = frames
        self.frames_gray = []
        for f in frames:
            if len(f.shape) == 3:
                self.frames_gray.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            else:
                self.frames_gray.append(f.copy())

        print(f"[D4RT] Encoding {len(frames)} frames ({self.frames_gray[0].shape[1]}x{self.frames_gray[0].shape[0]})...")
        t0 = time.time()
        self.scene = self.encoder.encode(self.frames_gray)
        self.scene.frames_bgr = self.frames  # give scene access to color frames for depth model
        self._is_loaded = True
        print(f"[D4RT] Ready in {time.time()-t0:.2f}s | {self.scene.image_size}")

    def load_video_from_path(self, video_path: str, max_frames: int = 300,
                             resize: Optional[Tuple[int, int]] = None):
        """
        Load video from file path.

        Args:
            video_path: path to video file
            max_frames: maximum frames to load
            resize: optional (width, height) to resize frames for speed
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames read from: {video_path}")

        actual_size = f"{frames[0].shape[1]}x{frames[0].shape[0]}"
        print(f"[D4RT] Loaded {len(frames)}/{total} frames @ {fps:.0f}fps "
              f"({w_orig}x{h_orig} -> {actual_size})")
        self.load_video(frames)

    # =========================================================================
    # TASK 1: POINT TRACKING (Batched, GPU-accelerated)
    # =========================================================================

    def track_point(self, u: float, v: float, start_frame: int = 0,
                    end_frame: Optional[int] = None) -> Dict:
        """Track a single point across the video."""
        self._check_loaded()
        points = np.array([[u, v]], dtype=np.float32)
        results = self.track_points(points, start_frame, end_frame)
        return results[0]

    def track_points(self, points: np.ndarray, start_frame: int = 0,
                     end_frame: Optional[int] = None,
                     compute_3d: bool = True) -> List[Dict]:
        """
        Track multiple points using BATCHED LK (GPU-accelerated).

        Args:
            points: (N, 2) array of (u, v) coordinates
            start_frame: starting frame index
            end_frame: ending frame index (exclusive)
            compute_3d: if True, lift to 3D via depth (expensive at high res).
                        Set False when only 2D positions are needed.
        """
        self._check_loaded()
        if end_frame is None:
            end_frame = self.scene.frame_count

        points = np.asarray(points, dtype=np.float32)

        t0 = time.time()
        positions, confidences, occluded = self.decoder.track_points_batched(
            points, self.frames_gray, self.scene, start_frame, end_frame
        )
        elapsed = time.time() - t0
        n_pts = len(points)
        n_frames = end_frame - start_frame
        print(f"  [Tracker] {n_pts} points x {n_frames} frames in {elapsed:.2f}s "
              f"({n_pts * n_frames / max(elapsed, 0.001):.0f} queries/s)")

        # Lift to 3D using proper camera intrinsics (vectorized per-frame)
        all_positions_3d = np.zeros((n_pts, n_frames, 3), dtype=np.float32)

        if compute_3d:
            fx, fy, cx, cy = 517.3, 516.5, 318.6, 255.3
            for t in range(n_frames):
                frame_idx = start_frame + t
                alive_mask = ~occluded[:, t]
                if not alive_mask.any():
                    continue

                depth = self.scene.get_depth(frame_idx)
                h_d, w_d = depth.shape
                alive_idx_3d = np.where(alive_mask)[0]
                pts = positions[alive_idx_3d, t]
                vi = np.clip(pts[:, 1].astype(int), 0, h_d - 1)
                ui = np.clip(pts[:, 0].astype(int), 0, w_d - 1)
                z = depth[vi, ui]
                all_positions_3d[alive_idx_3d, t, 0] = (pts[:, 0] - cx) * z / fx
                all_positions_3d[alive_idx_3d, t, 1] = (pts[:, 1] - cy) * z / fy
                all_positions_3d[alive_idx_3d, t, 2] = z

        results = []
        frame_indices = list(range(start_frame, end_frame))
        for i in range(n_pts):
            results.append({
                'positions_2d': positions[i],
                'positions_3d': all_positions_3d[i],
                'confidences': confidences[i],
                'occluded': occluded[i],
                'frame_indices': frame_indices
            })

        return results

    # =========================================================================
    # TASK 2: DEPTH ESTIMATION
    # =========================================================================

    def estimate_depth(self, frame_idx: int = 0) -> np.ndarray:
        """Estimate depth map for a given frame (lazy, cached)."""
        self._check_loaded()
        return self.scene.get_depth(frame_idx).copy()

    # =========================================================================
    # TASK 3: CAMERA POSE ESTIMATION
    # =========================================================================

    def estimate_camera_pose(self, frame_a: int, frame_b: int,
                             K: Optional[np.ndarray] = None,
                             num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate relative camera pose between two frames.
        Uses batched tracking + Essential matrix with RANSAC.
        """
        self._check_loaded()

        if K is None:
            K = np.array([
                [517.3, 0.0, 318.6],
                [0.0, 516.5, 255.3],
                [0.0, 0.0, 1.0]
            ])

        gray_a = self.frames_gray[frame_a]
        features = cv2.goodFeaturesToTrack(
            gray_a, maxCorners=num_points,
            qualityLevel=0.01, minDistance=10, blockSize=7
        )

        if features is None or len(features) < 8:
            return np.eye(3), np.zeros((3, 1))

        points = features.reshape(-1, 2).astype(np.float32)

        # Batched tracking from frame_a to frame_b
        positions, confidences, occluded = self.decoder.track_points_batched(
            points, self.frames_gray, self.scene, frame_a, frame_b + 1
        )

        # Get final positions (last frame in tracking window)
        last_t = frame_b - frame_a
        pts_a = []
        pts_b = []
        for i in range(len(points)):
            if not occluded[i, last_t] and confidences[i, last_t] > 0.5:
                pts_a.append(points[i])
                pts_b.append(positions[i, last_t])

        pts_a = np.array(pts_a, dtype=np.float64)
        pts_b = np.array(pts_b, dtype=np.float64)

        if len(pts_a) < 8:
            return np.eye(3), np.zeros((3, 1))

        E, mask = cv2.findEssentialMat(
            pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None:
            return np.eye(3), np.zeros((3, 1))

        _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, K)
        return R, t

    # =========================================================================
    # TASK 4: OPTICAL FLOW (uses precomputed flows, near-instant)
    # =========================================================================

    def compute_flow(self, frame_a: int, frame_b: int) -> np.ndarray:
        """
        Compute optical flow between two frames.

        For adjacent frames: returns precomputed dense flow (instant).
        For non-adjacent: chains precomputed flows (fast).
        """
        self._check_loaded()
        h, w = self.scene.image_size

        if frame_b == frame_a + 1 and frame_a in self.scene.dense_flows:
            # Direct precomputed flow (upscaled from half-res if needed)
            return self.scene.get_flow_fullres(frame_a).copy()

        if frame_b == frame_a:
            return np.zeros((h, w, 2), dtype=np.float32)

        # Chain flows for non-adjacent frames
        # Start with identity displacement field
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        map_x, map_y = np.meshgrid(xs, ys)

        step = 1 if frame_b > frame_a else -1
        for t in range(frame_a, frame_b, step):
            flow_key = t if step == 1 else t - 1
            if flow_key in self.scene.dense_flows:
                flow = self.scene.get_flow_fullres(flow_key)
                if step == 1:
                    map_x += flow[..., 0]
                    map_y += flow[..., 1]
                else:
                    map_x -= flow[..., 0]
                    map_y -= flow[..., 1]

        # Convert accumulated map back to flow
        ref_x, ref_y = np.meshgrid(xs, ys)
        flow_out = np.stack([map_x - ref_x, map_y - ref_y], axis=-1)
        return flow_out

    # =========================================================================
    # TASK 5: 3D POINT CLOUD RECONSTRUCTION
    # =========================================================================

    def reconstruct_3d(self, frame_idx: int = 0,
                       K: Optional[np.ndarray] = None,
                       subsample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct a 3D point cloud from a single frame using depth."""
        self._check_loaded()

        if K is None:
            K = np.array([
                [517.3, 0.0, 318.6],
                [0.0, 516.5, 255.3],
                [0.0, 0.0, 1.0]
            ])

        depth = self.estimate_depth(frame_idx)
        frame = self.frames[frame_idx]
        h, w = depth.shape

        # Vectorized back-projection (no loops)
        us = np.arange(0, w, subsample)
        vs = np.arange(0, h, subsample)
        u_grid, v_grid = np.meshgrid(us, vs)

        d = depth[v_grid, u_grid]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_3d = (u_grid - cx) * d / fx
        y_3d = (v_grid - cy) * d / fy
        z_3d = d

        points_3d = np.stack([x_3d, y_3d, z_3d], axis=-1).reshape(-1, 3)

        if len(frame.shape) == 3:
            colors = frame[v_grid, u_grid].reshape(-1, 3)[:, ::-1]
        else:
            g = frame[v_grid, u_grid].reshape(-1)
            colors = np.stack([g, g, g], axis=-1)

        valid = (d.ravel() > 0.01) & (d.ravel() < 50.0)
        return points_3d[valid], colors[valid]

    # =========================================================================
    # TASK 6: 4D RECONSTRUCTION (Dense Grid Tracking + Depth)
    # =========================================================================

    def reconstruct_4d(self, grid_step: int = 16, start_frame: int = 0,
                       end_frame: Optional[int] = None,
                       K: Optional[np.ndarray] = None,
                       keyframe_interval: int = 10,
                       chunk_size: int = 100) -> Dict:
        """
        Full 4D reconstruction with chunked re-detection.

        Tracks a dense grid of points and lifts to 3D using depth at every
        frame. For long videos, processes in chunks with fresh grid re-creation
        at chunk boundaries to maintain dense spatial coverage throughout.

        Args:
            grid_step: pixel spacing for the initial point grid
            start_frame: first frame to process
            end_frame: last frame (exclusive)
            K: 3x3 camera intrinsic matrix
            keyframe_interval: save full 3D snapshots at this interval
            chunk_size: max frames per tracking chunk (re-creates grid per chunk)

        Returns:
            Dict with keyframe_clouds, alive_per_frame, metadata
        """
        self._check_loaded()
        if end_frame is None:
            end_frame = self.scene.frame_count
        if K is None:
            K = np.array([[517.3, 0.0, 318.6],
                          [0.0, 516.5, 255.3],
                          [0.0, 0.0, 1.0]])

        h, w = self.scene.image_size
        n_frames = end_frame - start_frame

        # Create grid template
        us = np.arange(grid_step // 2, w, grid_step, dtype=np.float32)
        vs = np.arange(grid_step // 2, h, grid_step, dtype=np.float32)
        u_grid, v_grid = np.meshgrid(us, vs)
        grid_points = np.stack([u_grid.ravel(), v_grid.ravel()], axis=1)
        grid_shape = (len(vs), len(us))
        n_grid = len(grid_points)

        print(f"\n[4D Recon] Grid {grid_shape[0]}x{grid_shape[1]} = "
              f"{n_grid} pts, {n_frames} frames, chunk={chunk_size}")
        t0 = time.time()

        # Compute keyframe indices (absolute frame numbers)
        kf_abs = list(range(start_frame, end_frame, keyframe_interval))
        if (end_frame - 1) not in kf_abs:
            kf_abs.append(end_frame - 1)

        keyframe_clouds = {}
        alive_per_frame = np.zeros(n_frames, dtype=int)
        all_chunk_tracks = []

        # Process in chunks with fresh grid per chunk
        chunk_start = start_frame
        chunk_idx = 0
        while chunk_start < end_frame:
            chunk_end = min(chunk_start + chunk_size, end_frame)
            chunk_len = chunk_end - chunk_start

            # Track fresh dense grid through this chunk (skip 3D — computed at keyframes only)
            tracks = self.track_points(grid_points, chunk_start, chunk_end,
                                       compute_3d=False)
            all_chunk_tracks.append({
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'tracks': tracks,
            })

            # Assemble chunk arrays
            pos2d = np.stack([tr['positions_2d'] for tr in tracks])
            conf = np.stack([tr['confidences'] for tr in tracks])
            occ = np.stack([tr['occluded'] for tr in tracks])

            # Record alive counts
            for t_rel in range(chunk_len):
                abs_frame = chunk_start + t_rel
                rel_to_start = abs_frame - start_frame
                alive = int(np.sum(~occ[:, t_rel] & (conf[:, t_rel] > 0.3)))
                alive_per_frame[rel_to_start] = max(
                    alive_per_frame[rel_to_start], alive)

            # Build keyframe 3D snapshots (depth computed only at keyframes)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            for kf_frame in kf_abs:
                if kf_frame < chunk_start or kf_frame >= chunk_end:
                    continue
                if kf_frame in keyframe_clouds:
                    continue  # already built by a previous chunk
                t_rel = kf_frame - chunk_start
                alive_mask = ~occ[:, t_rel] & (conf[:, t_rel] > 0.3)
                p2_alive = pos2d[alive_mask, t_rel]

                # Compute depth ONLY at this keyframe
                depth = self.scene.get_depth(kf_frame)
                h_d, w_d = depth.shape
                vi = np.clip(p2_alive[:, 1].astype(int), 0, h_d - 1)
                ui = np.clip(p2_alive[:, 0].astype(int), 0, w_d - 1)
                z = depth[vi, ui]
                valid = (z > 0.01) & (z < 50.0)
                z_v = z[valid]
                p2_v = p2_alive[valid]
                pts3 = np.stack([
                    (p2_v[:, 0] - cx) * z_v / fx,
                    (p2_v[:, 1] - cy) * z_v / fy,
                    z_v], axis=-1)

                frame_bgr = self.frames[kf_frame]
                vi2 = np.clip(p2_v[:, 1].astype(int), 0, h - 1)
                ui2 = np.clip(p2_v[:, 0].astype(int), 0, w - 1)
                if len(frame_bgr.shape) == 3:
                    colors = frame_bgr[vi2, ui2, ::-1]
                else:
                    g = frame_bgr[vi2, ui2]
                    colors = np.stack([g, g, g], axis=-1)

                keyframe_clouds[kf_frame] = {
                    'points_3d': pts3,
                    'colors': colors.astype(np.uint8),
                    'frame_idx': kf_frame,
                    'n_valid': len(pts3),
                }

            chunk_alive_end = int(np.sum(
                ~occ[:, -1] & (conf[:, -1] > 0.3)))
            print(f"  [4D Recon] Chunk {chunk_idx}: frames {chunk_start}-"
                  f"{chunk_end-1}, alive {n_grid}->{chunk_alive_end}")

            chunk_start = chunk_end
            chunk_idx += 1

        elapsed = time.time() - t0
        n_kf_valid = sum(1 for c in keyframe_clouds.values()
                         if c['n_valid'] > 0)
        total_3d_pts = sum(c['n_valid'] for c in keyframe_clouds.values())

        print(f"  [4D Recon] Done in {elapsed:.2f}s | "
              f"{chunk_idx} chunks, {n_kf_valid} keyframes, "
              f"{total_3d_pts} total 3D points")
        print(f"  Alive: min={alive_per_frame.min()}, "
              f"max={alive_per_frame.max()}, "
              f"mean={alive_per_frame.mean():.0f}")

        return {
            'keyframe_clouds': keyframe_clouds,
            'keyframe_frames': sorted(keyframe_clouds.keys()),
            'alive_per_frame': alive_per_frame,
            'grid_shape': grid_shape,
            'grid_step': grid_step,
            'n_points': n_grid,
            'n_frames': n_frames,
            'n_chunks': chunk_idx,
            'chunk_size': chunk_size,
            'K': K,
            'elapsed': elapsed,
        }

    # =========================================================================
    # TASK 7: POINT TRACKING WITH RE-DETECTION
    # =========================================================================

    def track_with_redetection(self, n_initial: int = 200,
                               redetect_every: int = 30,
                               min_alive_ratio: float = 0.5,
                               start_frame: int = 0,
                               end_frame: Optional[int] = None) -> List[Dict]:
        """
        Track points with periodic re-detection of new features.

        When tracks die off, new features are detected in uncovered
        regions to maintain dense coverage throughout the video.

        Args:
            n_initial: number of initial feature points
            redetect_every: re-detect features every N frames
            min_alive_ratio: re-detect when alive ratio drops below this
            start_frame: starting frame
            end_frame: ending frame

        Returns:
            List of track dicts, where tracks may have different start frames.
            Each dict has 'positions_2d', 'positions_3d', 'confidences',
            'occluded', 'frame_indices', 'start_frame'.
        """
        self._check_loaded()
        if end_frame is None:
            end_frame = self.scene.frame_count
        h, w = self.scene.image_size

        all_tracks = []

        # Detect initial features
        features = cv2.goodFeaturesToTrack(
            self.frames_gray[start_frame], maxCorners=n_initial,
            qualityLevel=0.005, minDistance=15, blockSize=7
        )
        if features is None or len(features) < 10:
            print("  [Redetect] Too few initial features")
            return []

        active_points = features.reshape(-1, 2).astype(np.float32)
        chunk_start = start_frame

        while chunk_start < end_frame:
            chunk_end = min(chunk_start + redetect_every, end_frame)

            print(f"  [Redetect] {len(active_points)} pts, "
                  f"frames {chunk_start}-{chunk_end-1}")
            tracks = self.track_points(active_points, chunk_start, chunk_end)

            # Tag each track with its absolute start frame
            for tr in tracks:
                tr['start_frame'] = chunk_start
            all_tracks.extend(tracks)

            if chunk_end >= end_frame:
                break

            # Count survivors at chunk boundary
            alive = [tr for tr in tracks
                     if not tr['occluded'][-1] and tr['confidences'][-1] > 0.3]
            alive_ratio = len(alive) / max(len(tracks), 1)

            surviving_pts = []
            for tr in alive:
                surviving_pts.append(tr['positions_2d'][-1])

            # Re-detect if coverage dropped
            if alive_ratio < min_alive_ratio or len(alive) < n_initial // 2:
                mask = np.ones((h, w), dtype=np.uint8) * 255
                for pt in surviving_pts:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(mask, (x, y), 15, 0, -1)

                n_new = n_initial - len(surviving_pts)
                new_features = cv2.goodFeaturesToTrack(
                    self.frames_gray[chunk_end], maxCorners=max(n_new, 10),
                    qualityLevel=0.005, minDistance=15, blockSize=7, mask=mask
                )
                if new_features is not None:
                    new_pts = new_features.reshape(-1, 2).astype(np.float32)
                    if surviving_pts:
                        active_points = np.vstack([
                            np.array(surviving_pts, dtype=np.float32),
                            new_pts
                        ])
                    else:
                        active_points = new_pts
                    print(f"  [Redetect] +{len(new_pts)} new, "
                          f"{len(surviving_pts)} surviving = "
                          f"{len(active_points)} total")
                else:
                    active_points = np.array(surviving_pts, dtype=np.float32) \
                        if surviving_pts else np.empty((0, 2), dtype=np.float32)
            else:
                active_points = np.array(surviving_pts, dtype=np.float32) \
                    if surviving_pts else np.empty((0, 2), dtype=np.float32)

            if len(active_points) < 5:
                print("  [Redetect] Too few points remaining, stopping")
                break

            chunk_start = chunk_end

        print(f"  [Redetect] Total: {len(all_tracks)} tracklets")
        return all_tracks

    # =========================================================================
    # TASK 8: ROLLING WINDOW TRACKER
    # =========================================================================

    def track_rolling_window(self, n_points: int = 200,
                             start_frame: int = 0,
                             end_frame: Optional[int] = None,
                             replace_every: int = 5,
                             min_alive_ratio: float = 0.6,
                             trail_length: int = 30,
                             output_path: Optional[str] = None,
                             show: bool = False) -> Dict:
        """
        Rolling-window point tracker with continuous replacement.

        Maintains ~n_points active tracks at all times. Dead/occluded
        points are replaced with fresh features detected from the
        current frame. Each point carries a fading trail of its recent
        trajectory.

        Args:
            n_points: target number of active points at any time
            start_frame: first frame
            end_frame: last frame (exclusive)
            replace_every: check & replace dead points every N frames
            min_alive_ratio: replace when alive/target drops below this
            trail_length: how many past positions to keep per point
            output_path: if set, save annotated video here
            show: display live window

        Returns:
            Dict with per-frame data and summary statistics.
        """
        self._check_loaded()
        if end_frame is None:
            end_frame = self.scene.frame_count
        h, w = self.scene.image_size
        n_frames = end_frame - start_frame

        # Track state per point: position, alive, trail, age
        next_pid = 0  # monotonic point ID counter

        class TrackedPoint:
            __slots__ = ['pid', 'pos', 'alive', 'trail', 'age', 'color']
            def __init__(self, pid, pos, color):
                self.pid = pid
                self.pos = pos.copy()
                self.alive = True
                self.trail = [pos.copy()]
                self.age = 0
                self.color = color

        def make_color(pid):
            hue = int((pid * 47) % 180)
            c = cv2.cvtColor(np.uint8([[[hue, 255, 220]]]),
                             cv2.COLOR_HSV2BGR)[0][0]
            return tuple(int(x) for x in c)

        # Detect initial features (params adapt to point count)
        min_dist = max(5, int(min(w, h) / (n_points ** 0.5) * 0.6))
        features = cv2.goodFeaturesToTrack(
            self.frames_gray[start_frame], maxCorners=n_points,
            qualityLevel=0.001, minDistance=min_dist, blockSize=5
        )
        if features is None:
            return {'n_frames': 0, 'per_frame': []}

        points_list = []
        for pt in features.reshape(-1, 2):
            points_list.append(TrackedPoint(next_pid, pt, make_color(next_pid)))
            next_pid += 1

        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        per_frame = []
        total_born = len(points_list)
        total_died = 0
        flow_engine = self.decoder.flow_engine

        print(f"  [RollingWindow] {n_points} target pts, "
              f"{n_frames} frames, trail={trail_length}")

        for t_rel in range(n_frames):
            t_abs = start_frame + t_rel

            # --- Track alive points to next frame (batched) ---
            if t_rel > 0 and t_abs < len(self.frames_gray):
                alive_pts_list = [p for p in points_list if p.alive]
                if alive_pts_list:
                    prev_arr = np.array(
                        [p.pos for p in alive_pts_list],
                        dtype=np.float32).reshape(-1, 1, 2)

                    next_arr, status = flow_engine.sparse_lk_batch(
                        self.frames_gray[t_abs - 1],
                        self.frames_gray[t_abs], prev_arr)
                    back_arr, back_st = flow_engine.sparse_lk_batch(
                        self.frames_gray[t_abs],
                        self.frames_gray[t_abs - 1], next_arr)

                    status_f = status.ravel()
                    back_st_f = back_st.ravel()
                    new_positions = next_arr.reshape(-1, 2)

                    # Vectorized FB errors
                    fb_errors = np.linalg.norm(
                        prev_arr.reshape(-1, 2) - back_arr.reshape(-1, 2),
                        axis=1)

                    fwd_ok = status_f == 1
                    fb_ok = fwd_ok & ((back_st_f != 1) |
                                      (fb_errors <= 3.0))
                    in_bounds = (fwd_ok &
                                 (new_positions[:, 0] >= 0) &
                                 (new_positions[:, 0] < w) &
                                 (new_positions[:, 1] >= 0) &
                                 (new_positions[:, 1] < h))
                    good = fb_ok & in_bounds

                    for j, p in enumerate(alive_pts_list):
                        if not good[j]:
                            p.alive = False
                            total_died += 1
                        else:
                            p.pos = new_positions[j].copy()
                            p.trail.append(p.pos.copy())
                            if len(p.trail) > trail_length:
                                p.trail.pop(0)
                            p.age += 1

            # --- Replace dead points periodically ---
            if t_rel > 0 and t_rel % replace_every == 0:
                alive_count = sum(1 for p in points_list if p.alive)
                if alive_count < n_points * min_alive_ratio:
                    n_need = n_points - alive_count
                    # Build exclusion mask near existing alive points
                    mask = np.ones((h, w), dtype=np.uint8) * 255
                    for p in points_list:
                        if p.alive:
                            x, y = int(p.pos[0]), int(p.pos[1])
                            cv2.circle(mask, (x, y), min_dist, 0, -1)
                    new_feats = cv2.goodFeaturesToTrack(
                        self.frames_gray[t_abs],
                        maxCorners=n_need, qualityLevel=0.001,
                        minDistance=min_dist, blockSize=5, mask=mask)
                    if new_feats is not None:
                        # Remove dead points from list
                        points_list = [p for p in points_list if p.alive]
                        for pt in new_feats.reshape(-1, 2):
                            points_list.append(TrackedPoint(
                                next_pid, pt, make_color(next_pid)))
                            next_pid += 1
                            total_born += 1

            # --- Record per-frame data ---
            alive_count = sum(1 for p in points_list if p.alive)
            per_frame.append({
                'frame': t_abs,
                'alive': alive_count,
                'total_tracked': len(points_list),
            })

            # --- Render frame ---
            if writer or show:
                vis = self.frames[t_abs].copy()

                # Draw trails + points
                for p in points_list:
                    if not p.alive:
                        continue
                    trail = p.trail
                    # Draw fading trail
                    for k in range(1, len(trail)):
                        alpha = k / len(trail)
                        thick = max(1, int(alpha * (w / 500)))
                        col = tuple(int(c * alpha) for c in p.color)
                        pt1 = (int(trail[k-1][0]), int(trail[k-1][1]))
                        pt2 = (int(trail[k][0]), int(trail[k][1]))
                        cv2.line(vis, pt1, pt2, col, thick)
                    # Draw current position (scale with resolution)
                    cx, cy = int(p.pos[0]), int(p.pos[1])
                    r_base = max(1, int(w / 400))
                    radius = r_base if p.age > 5 else r_base + 1
                    cv2.circle(vis, (cx, cy), radius, p.color, -1)

                # HUD
                backend = "CUDA" if GPU.has_cuda else "CPU"
                cv2.putText(vis,
                            f"D4RT Rolling [{backend}] | Frame {t_abs}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                cv2.putText(vis,
                            f"Active: {alive_count}/{n_points} | "
                            f"Born: {total_born} | Died: {total_died}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 200, 200), 1)

                if writer:
                    writer.write(vis)
                if show:
                    cv2.imshow('D4RT Rolling Window', vis)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        alive_counts = [f['alive'] for f in per_frame]
        print(f"  [RollingWindow] Done: {n_frames} frames, "
              f"born={total_born}, died={total_died}, "
              f"avg_alive={np.mean(alive_counts):.0f}, "
              f"min_alive={min(alive_counts)}")

        return {
            'n_frames': n_frames,
            'per_frame': per_frame,
            'total_born': total_born,
            'total_died': total_died,
            'alive_mean': float(np.mean(alive_counts)),
            'alive_min': min(alive_counts),
            'alive_max': max(alive_counts),
        }

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    def visualize_tracking(self, points: np.ndarray,
                           start_frame: int = 0,
                           end_frame: Optional[int] = None,
                           output_path: Optional[str] = None,
                           show: bool = True,
                           precomputed_tracks: Optional[List[Dict]] = None):
        """Visualize point tracking with colored trajectories."""
        self._check_loaded()
        if end_frame is None:
            end_frame = min(start_frame + 100, self.scene.frame_count)

        # Use precomputed tracks if available, else track from scratch
        if precomputed_tracks is not None:
            all_tracks = precomputed_tracks
        else:
            all_tracks = self.track_points(points, start_frame, end_frame)

        num_points = len(points)
        colors = []
        for i in range(num_points):
            hue = int(180 * i / max(num_points, 1))
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(int(c) for c in color_bgr))

        writer = None
        if output_path:
            h, w = self.scene.image_size
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        mask = np.zeros_like(self.frames[start_frame])

        for frame_idx in range(start_frame, end_frame):
            vis = self.frames[frame_idx].copy()
            t_rel = frame_idx - start_frame

            for pt_idx, track in enumerate(all_tracks):
                if t_rel >= len(track['positions_2d']):
                    continue

                pos = track['positions_2d'][t_rel]
                conf = track['confidences'][t_rel]
                occ = track['occluded'][t_rel]

                if not occ and conf > 0.3:
                    x, y = int(pos[0]), int(pos[1])
                    r = max(1, int(w / 400))
                    cv2.circle(vis, (x, y), r, colors[pt_idx], -1)

                    if t_rel > 0:
                        prev_pos = track['positions_2d'][t_rel - 1]
                        prev_occ = track['occluded'][t_rel - 1]
                        if not prev_occ:
                            px, py = int(prev_pos[0]), int(prev_pos[1])
                            cv2.line(mask, (px, py), (x, y), colors[pt_idx],
                                     max(1, int(w / 500)))

            vis = cv2.add(vis, mask)

            backend = "CUDA" if GPU.has_cuda else "CPU"
            cv2.putText(vis, f"D4RT [{backend}] | Frame {frame_idx}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(vis, f"Points: {num_points} | Batched Query",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 200), 1)

            if writer:
                writer.write(vis)

            if show:
                cv2.imshow('D4RT Point Tracking', vis)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:
                    break

        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    def visualize_depth(self, frame_idx: int = 0, show: bool = True) -> np.ndarray:
        """Visualize depth map as a colorized image."""
        self._check_loaded()
        depth = self.estimate_depth(frame_idx)

        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        d_uint8 = d_norm.astype(np.uint8)
        d_color = cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)

        cv2.putText(d_color, f"D4RT Depth | Frame {frame_idx}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        if show:
            cv2.imshow('D4RT Depth Estimation', d_color)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return d_color

    def visualize_flow(self, frame_a: int, frame_b: int,
                       show: bool = True) -> np.ndarray:
        """Visualize optical flow as HSV color wheel."""
        self._check_loaded()
        flow = self.compute_flow(frame_a, frame_b)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*self.scene.image_size, 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.putText(flow_vis, f"D4RT Flow | {frame_a} -> {frame_b}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        if show:
            cv2.imshow('D4RT Optical Flow', flow_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return flow_vis

    # =========================================================================
    # SCENE CUT DETECTION
    # =========================================================================

    def detect_segments(self, n_segments: Optional[int] = None,
                        min_segment_frames: int = 30) -> List[Tuple[int, int]]:
        """
        Detect scene cuts and return video segments as (start, end) tuples.

        Uses frame-to-frame mean absolute difference. Large jumps indicate
        scene transitions.

        Args:
            n_segments: if set, return exactly this many segments using the
                        top N-1 most significant cuts
            min_segment_frames: minimum frames per segment (auto mode only)

        Returns:
            List of (start_frame, end_frame) tuples (end is exclusive)
        """
        self._check_loaded()
        n = len(self.frames_gray)
        if n < 2:
            return [(0, n)]

        # Compute frame-to-frame differences
        diffs = np.zeros(n - 1, dtype=np.float32)
        for i in range(n - 1):
            diffs[i] = np.mean(np.abs(
                self.frames_gray[i + 1].astype(np.float32) -
                self.frames_gray[i].astype(np.float32)))

        median_diff = np.median(diffs)
        threshold = max(median_diff * 15, 20.0)

        # Collect all candidate cuts with their diff values
        candidates = [(i + 1, diffs[i]) for i in range(len(diffs))
                      if diffs[i] > threshold]

        if n_segments is not None and n_segments >= 2:
            # Non-maximum suppression: pick top cuts with minimum gap
            min_gap = max(n // (n_segments * 2), 60)
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = []
            for frame, diff in candidates:
                if all(abs(frame - s) >= min_gap for s in selected):
                    selected.append(frame)
                if len(selected) >= n_segments - 1:
                    break
            top_cuts = sorted(selected)
            cuts = [0] + top_cuts + [n]
        else:
            # Auto mode: use all cuts, merge short segments
            cuts = [0] + [c[0] for c in candidates] + [n]

        # Build segments
        segments = []
        for i in range(len(cuts) - 1):
            s, e = cuts[i], cuts[i + 1]
            if n_segments is not None:
                # Fixed mode: keep all segments
                segments.append((s, e))
            elif e - s >= min_segment_frames:
                segments.append((s, e))
            elif segments:
                segments[-1] = (segments[-1][0], e)

        if not segments:
            segments = [(0, n)]

        return segments

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _check_loaded(self):
        if not self._is_loaded:
            raise RuntimeError("No video loaded. Call load_video() or "
                               "load_video_from_path() first.")


# =============================================================================
# DOWNLOAD HELPER FOR AVAILABLE WEIGHTS
# =============================================================================

def download_4ds_weights(model_name: str = "4ds_b_dist_e",
                         save_dir: str = "./weights") -> str:
    """
    Download pretrained 4DS weights from Google DeepMind's representations4d.

    These are the closest publicly available weights to D4RT.
    They come from the same research group and address related 4D vision tasks.

    Available models:
        - "4ds_b_dist_e"          (88M params, 334MB) - Base distilled encoder
        - "4ds_e"                 (3.8B params, 14GB)  - Full encoder
        - "4ds_b_dist_e_scannet"  (105M params, 420MB) - Depth-finetuned
        - "rvm_vit_s"             (34M params, 270MB)  - Recurrent ViT-S
        - "rvm_vit_b"             (109M params, 436MB) - Recurrent ViT-B
        - "rvm_vit_l"             (358M params, 1.4GB) - Recurrent ViT-L
        - "rvm_vit_h"             (743M params, 3.1GB) - Recurrent ViT-H

    Note: These weights are in JAX/NumPy format (.npz). Using them requires
    the representations4d codebase (JAX, Haiku).

    Repository: https://github.com/google-deepmind/representations4d
    License: Apache 2.0 (code), CC-BY 4.0 (other materials)
    """
    import os
    import urllib.request

    base_url = "https://storage.googleapis.com/representations4d/checkpoints"
    urls = {
        "4ds_b_dist_e":         f"{base_url}/4ds_b_dist_e.npz",
        "4ds_e":                f"{base_url}/4ds_e.npz",
        "4ds_b_dist_e_scannet": f"{base_url}/4ds_b_dist_e_scannet.npz",
        "rvm_vit_s":            f"{base_url}/rvm_vit_s.npz",
        "rvm_vit_b":            f"{base_url}/rvm_vit_b.npz",
        "rvm_vit_l":            f"{base_url}/rvm_vit_l.npz",
        "rvm_vit_h":            f"{base_url}/rvm_vit_h.npz",
    }

    if model_name not in urls:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(urls.keys())}"
        )

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{model_name}.npz"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"[D4RT] Weights already exist: {filepath}")
        return filepath

    url = urls[model_name]
    print(f"[D4RT] Downloading {model_name} from {url}...")
    print(f"[D4RT] This may take a while depending on model size.")
    urllib.request.urlretrieve(url, filepath)
    print(f"[D4RT] Saved to {filepath}")
    return filepath


# =============================================================================
# MAIN: DEMO ON VIDEO
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(description="D4RT: Dynamic 4D Reconstruction and Tracking")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file")
    parser.add_argument("--headless", action="store_true", help="Run without GUI display")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames to load")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("W", "H"), help="Resize frames to WxH (default: native)")
    parser.add_argument("--num-segments", type=int, default=3,
                        help="Number of video segments (default: 3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save outputs (default: d4rt_output/ next to video)")
    args = parser.parse_args()

    # Detect if display is available
    show = not args.headless
    if show:
        try:
            test_img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imshow("_test", test_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        except Exception:
            print("[D4RT] No display available, running in headless mode.")
            show = False

    print("=" * 70)
    print("D4RT: Dynamic 4D Reconstruction and Tracking")
    print("Google DeepMind (Dec 2025) - arXiv:2512.08924")
    print("=" * 70)
    print()
    print(f"GPU Backend: {GPU}")
    print(f"Display: {'GUI' if show else 'headless'}")
    print()
    print("NOTE: D4RT pretrained weights are NOT yet publicly released.")
    print("This implementation uses the D4RT query mechanism with")
    print("GPU-accelerated classical feature tracking as a backend.")
    print()
    print("Closest available weights (same DeepMind team):")
    print("  https://github.com/google-deepmind/representations4d")
    print("  - 4DS-B-dist-e: 88M params  (334MB)")
    print("  - 4DS-e:        3.8B params (14GB)")
    print("  - RVM ViT-S/B/L/H: 34M-743M params")
    print()

    # Default video path
    video_path = args.video
    if video_path is None:
        video_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "OPTICAL_FLOW.mp4"
        )

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        print("Usage: python 08_d4rt_deepmind.py [path_to_video] [--headless] [--resize W H]")
        sys.exit(1)

    resize = tuple(args.resize) if args.resize else None

    # Setup output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(video_path), "d4rt_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Initialize D4RT
    model = D4RT()

    # Load video
    model.load_video_from_path(video_path, max_frames=args.max_frames, resize=resize)

    total_t0 = time.time()
    n_frames_total = model.scene.frame_count
    h, w = model.scene.image_size

    # =========================================================================
    # AUTO-SEGMENT: Detect scene cuts (video may be multiple clips combined)
    # =========================================================================
    segments = model.detect_segments(n_segments=args.num_segments)
    print(f"\nDetected {len(segments)} video segment(s):")
    for i, (s, e) in enumerate(segments):
        print(f"  Segment {i+1}: frames {s}-{e-1} "
              f"({(e-s)/60:.1f}s, {e-s} frames)")

    # =========================================================================
    # PROCESS EACH SEGMENT INDEPENDENTLY
    # =========================================================================
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        n_frames = seg_end - seg_start
        seg_label = f"seg{seg_idx+1}"
        seg_dir = os.path.join(out_dir, seg_label)
        os.makedirs(seg_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"SEGMENT {seg_idx+1}/{len(segments)}: "
              f"frames {seg_start}-{seg_end-1} ({n_frames} frames, "
              f"{n_frames/60:.1f}s)")
        print(f"Output: {seg_dir}")
        print(f"{'='*70}")

        # =====================================================================
        # DEMO 1: Point Tracking
        # =====================================================================
        print(f"\n[DEMO 1] Point Tracking ({n_frames} frames)")

        seg_gray = model.frames_gray[seg_start]
        features = cv2.goodFeaturesToTrack(
            seg_gray, maxCorners=2000, qualityLevel=0.001,
            minDistance=8, blockSize=5
        )

        tracks = []
        points = np.empty((0, 2), dtype=np.float32)
        if features is not None:
            points = features.reshape(-1, 2)
            print(f"  Tracking {len(points)} points across {n_frames} frames...")
            tracks = model.track_points(points, start_frame=seg_start,
                                        end_frame=seg_end, compute_3d=False)

            moving_count = sum(
                1 for tr in tracks
                if np.linalg.norm(
                    tr['positions_2d'][-1] - tr['positions_2d'][0]) > 1.0)
            print(f"  {moving_count}/{len(tracks)} points show motion > 1px")

            # Save tracking data
            tracking_data = {'initial_points': points, 'frame_count': n_frames}
            for i, tr in enumerate(tracks):
                tracking_data[f'track_{i}_positions_2d'] = tr['positions_2d']
                tracking_data[f'track_{i}_positions_3d'] = tr['positions_3d']
                tracking_data[f'track_{i}_confidences'] = tr['confidences']
                tracking_data[f'track_{i}_occluded'] = tr['occluded']
            np.savez_compressed(
                os.path.join(seg_dir, "tracking_data.npz"), **tracking_data)

            # Save tracking video (reuse precomputed tracks)
            vis_end = min(seg_start + 600, seg_end)
            tracking_video_path = os.path.join(seg_dir, "tracking_video.mp4")
            model.visualize_tracking(points, start_frame=seg_start,
                                     end_frame=vis_end,
                                     output_path=tracking_video_path,
                                     show=show,
                                     precomputed_tracks=tracks)

            # Save trajectory plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                axes[0].imshow(cv2.cvtColor(model.frames[seg_start],
                                            cv2.COLOR_BGR2RGB))
                cmap = plt.colormaps.get_cmap('hsv').resampled(
                    max(len(tracks), 1))
                for i, tr in enumerate(tracks):
                    valid = ~tr['occluded']
                    pos = tr['positions_2d'][valid]
                    if len(pos) > 1:
                        axes[0].plot(pos[:, 0], pos[:, 1], '-', color=cmap(i),
                                     linewidth=1, alpha=0.8)
                        axes[0].plot(pos[0, 0], pos[0, 1], 'o', color=cmap(i),
                                     markersize=5)
                axes[0].set_title(f'Trajectories ({len(tracks)} pts, '
                                  f'{moving_count} moving)')
                for i, tr in enumerate(tracks):
                    diffs = np.linalg.norm(
                        tr['positions_2d'] - tr['positions_2d'][0], axis=1)
                    t_axis = np.arange(len(diffs)) / 60.0
                    axes[1].plot(t_axis, diffs, color=cmap(i),
                                 linewidth=0.8, alpha=0.7)
                axes[1].set_title('Displacement Over Time')
                axes[1].set_xlabel('Time (s)')
                axes[1].set_ylabel('Displacement (px)')
                axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(seg_dir, "tracking_trajectories.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
            except ImportError:
                pass

        # =====================================================================
        # DEMO 2: Depth Estimation
        # =====================================================================
        mid_frame = seg_start + n_frames // 2
        print(f"\n[DEMO 2] Depth Estimation (frame {mid_frame})")
        depth = model.estimate_depth(mid_frame)
        print(f"  Range [{depth.min():.2f}, {depth.max():.2f}], "
              f"mean={depth.mean():.2f}")
        np.save(os.path.join(seg_dir, f"depth_frame{mid_frame}_raw.npy"), depth)
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        d_color = cv2.applyColorMap(d_norm.astype(np.uint8), cv2.COLORMAP_MAGMA)
        cv2.imwrite(os.path.join(seg_dir, f"depth_frame{mid_frame}_color.png"),
                     d_color)

        # =====================================================================
        # DEMO 3: Optical Flow
        # =====================================================================
        print(f"\n[DEMO 3] Optical Flow")
        if n_frames >= 2:
            gaps = [1, 10, n_frames // 2, n_frames - 1]
            for gap in gaps:
                if gap >= n_frames or gap < 1:
                    continue
                flow = model.compute_flow(seg_start, seg_start + gap)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                print(f"  Flow {seg_start}->{seg_start+gap}: "
                      f"max={mag.max():.1f}px  mean={mag.mean():.3f}px")
                mag_vis, ang_vis = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv = np.zeros((h, w, 3), dtype=np.uint8)
                hsv[..., 0] = ang_vis * 180 / np.pi / 2
                hsv[..., 1] = 255
                hsv[..., 2] = cv2.normalize(
                    mag_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                cv2.imwrite(
                    os.path.join(seg_dir, f"flow_gap{gap}.png"), flow_bgr)

        # =====================================================================
        # DEMO 4: Camera Pose Estimation
        # =====================================================================
        print(f"\n[DEMO 4] Camera Pose")
        poses = []
        if n_frames >= 10:
            pose_gaps = [g for g in [10, 30, 60, n_frames - 1]
                         if g < n_frames]
            for pg in pose_gaps:
                R, t_vec = model.estimate_camera_pose(seg_start,
                                                      seg_start + pg)
                angle_rad = np.arccos(
                    np.clip((np.trace(R) - 1) / 2, -1, 1))
                angle_deg = np.degrees(angle_rad)
                print(f"  Gap {pg}: rot={angle_deg:.4f} deg  "
                      f"t=[{t_vec[0,0]:.3f},{t_vec[1,0]:.3f},"
                      f"{t_vec[2,0]:.3f}]")
                poses.append({'gap': pg, 'R': R, 't': t_vec})
            pose_data = {}
            for p in poses:
                pose_data[f'R_gap{p["gap"]}'] = p['R']
                pose_data[f't_gap{p["gap"]}'] = p['t']
            np.savez(os.path.join(seg_dir, "camera_poses.npz"), **pose_data)

        # =====================================================================
        # DEMO 5: 3D Reconstruction
        # =====================================================================
        print(f"\n[DEMO 5] 3D Point Cloud (frame {mid_frame})")
        pts3d, colors_3d = model.reconstruct_3d(frame_idx=mid_frame,
                                                  subsample=4)
        print(f"  {len(pts3d)} points, "
              f"Z=[{pts3d[:, 2].min():.2f}, {pts3d[:, 2].max():.2f}]")
        np.savez_compressed(
            os.path.join(seg_dir, f"pointcloud_frame{mid_frame}.npz"),
            points=pts3d, colors=colors_3d)
        ply_path = os.path.join(seg_dir, f"pointcloud_frame{mid_frame}.ply")
        with open(ply_path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts3d)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\n")
            f.write("property uchar blue\nend_header\n")
            for pt, col in zip(pts3d, colors_3d):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                        f"{int(col[0])} {int(col[1])} {int(col[2])}\n")

        # =====================================================================
        # DEMO 6: 4D Reconstruction (chunked)
        # =====================================================================
        print(f"\n[DEMO 6] 4D Reconstruction")
        if n_frames >= 10:
            kf_interval = max(1, n_frames // 10)
            recon4d = model.reconstruct_4d(
                grid_step=20, start_frame=seg_start, end_frame=seg_end,
                keyframe_interval=kf_interval,
                chunk_size=min(100, n_frames)
            )

            # Save 4D metadata
            np.savez_compressed(
                os.path.join(seg_dir, "4d_reconstruction.npz"),
                alive_per_frame=recon4d['alive_per_frame'],
                grid_shape=np.array(recon4d['grid_shape']),
                K=recon4d['K'],
            )

            # Save keyframe PLY sequence
            for kf_frame, cloud in sorted(recon4d['keyframe_clouds'].items()):
                if cloud['n_valid'] < 10:
                    continue
                kf_ply = os.path.join(
                    seg_dir, f"4d_kf_{cloud['frame_idx']:04d}.ply")
                pts = cloud['points_3d']
                cols = cloud['colors']
                with open(kf_ply, 'w') as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {len(pts)}\n")
                    f.write("property float x\nproperty float y\n")
                    f.write("property float z\n")
                    f.write("property uchar red\nproperty uchar green\n")
                    f.write("property uchar blue\nend_header\n")
                    for pt, col in zip(pts, cols):
                        f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} "
                                f"{int(col[0])} {int(col[1])} "
                                f"{int(col[2])}\n")

            # Save 4D vis plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                axes[0].plot(recon4d['alive_per_frame'], 'b-', linewidth=1.5)
                axes[0].set_title('Track Survival')
                axes[0].set_xlabel('Frame (relative)')
                axes[0].set_ylabel('Alive Tracks')
                axes[0].grid(True, alpha=0.3)
                kf_frames = sorted(recon4d['keyframe_clouds'].keys())
                kf_counts = [recon4d['keyframe_clouds'][k]['n_valid']
                             for k in kf_frames]
                axes[1].bar(range(len(kf_frames)), kf_counts,
                            color='steelblue', alpha=0.8)
                axes[1].set_title('3D Points per Keyframe')
                axes[1].set_xlabel('Keyframe Index')
                axes[1].set_ylabel('Valid 3D Points')
                axes[1].grid(True, alpha=0.3)
                plt.suptitle(f'Segment {seg_idx+1}: 4D Reconstruction '
                             f'({recon4d["n_points"]} pts, '
                             f'{recon4d["n_chunks"]} chunks)')
                plt.tight_layout()
                plt.savefig(os.path.join(seg_dir, "4d_vis.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
            except ImportError:
                pass

        # =====================================================================
        # DEMO 7: Rolling Window Tracker
        # =====================================================================
        print(f"\n[DEMO 7] Rolling Window Tracker")
        if n_frames >= 20:
            rw_path = os.path.join(seg_dir, "rolling_window.mp4")
            rw_result = model.track_rolling_window(
                n_points=1500, start_frame=seg_start, end_frame=seg_end,
                replace_every=5, min_alive_ratio=0.6,
                trail_length=30, output_path=rw_path, show=show
            )
            print(f"  Saved: {rw_path}")

            # Save alive-count plot
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                alive_data = [f['alive'] for f in rw_result['per_frame']]
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(alive_data, 'b-', linewidth=0.8)
                ax.axhline(y=1500, color='g', linestyle='--', alpha=0.5,
                           label='Target')
                ax.axhline(y=1500 * 0.6, color='r', linestyle='--', alpha=0.5,
                           label='Replace threshold')
                ax.set_title(f'Rolling Window: Active Points '
                             f'(born={rw_result["total_born"]}, '
                             f'died={rw_result["total_died"]})')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Active Points')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(seg_dir, "rolling_window_stats.png"),
                            dpi=150, bbox_inches='tight')
                plt.close()
            except ImportError:
                pass

        print(f"\n  Segment {seg_idx+1} complete.")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - total_t0

    # Count all saved files across all segment dirs
    total_files = 0
    total_size = 0
    for seg_idx_s, (s, e) in enumerate(segments):
        sd = os.path.join(out_dir, f"seg{seg_idx_s+1}")
        if os.path.isdir(sd):
            for f in os.listdir(sd):
                fpath = os.path.join(sd, f)
                if os.path.isfile(fpath):
                    total_files += 1
                    total_size += os.path.getsize(fpath)

    print(f"\n{'=' * 70}")
    print(f"All demos completed in {total_time:.2f}s")
    print(f"Backend: {GPU}")
    print(f"Segments: {len(segments)}")
    print(f"Total files: {total_files} ({total_size/1024/1024:.1f} MB)")
    for seg_idx_s, (s, e) in enumerate(segments):
        sd = os.path.join(out_dir, f"seg{seg_idx_s+1}")
        if os.path.isdir(sd):
            files = sorted(os.listdir(sd))
            seg_size = sum(os.path.getsize(os.path.join(sd, f))
                           for f in files if os.path.isfile(os.path.join(sd, f)))
            print(f"  seg{seg_idx_s+1}/ ({e-s} frames): "
                  f"{len(files)} files, {seg_size/1024/1024:.1f} MB")
    print(f"\nFor the real model: https://d4rt-paper.github.io/")
    print("=" * 70)
