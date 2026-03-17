"""
=============================================================================
ALGORITHM 2: FARNEBACK DENSE OPTICAL FLOW
=============================================================================

DENSE vs SPARSE RECAP
---------------------
Unlike Lucas-Kanade which tracks selected points, DENSE optical flow computes
a flow vector (u, v) for EVERY SINGLE PIXEL. This gives a complete motion
map but is computationally more expensive.


THE FARNEBACK METHOD (2003)
---------------------------
Gunnar Farneback's algorithm approximates the neighborhood of each pixel
using POLYNOMIAL EXPANSION rather than simple intensity gradients.

Key idea: Model each pixel neighborhood as a quadratic polynomial:
    f(x) ≈ x^T * A * x + b^T * x + c

Where:
    A = 2x2 matrix (captures the "shape" of the neighborhood)
    b = 2x1 vector (captures the "tilt"/gradient)
    c = scalar (average intensity)

HOW IT WORKS — Step by Step:
1. For frame 1, compute polynomial coefficients (A1, b1, c1) at each pixel
2. For frame 2, compute polynomial coefficients (A2, b2, c2) at each pixel
3. When a neighborhood moves by displacement d, the polynomials relate as:
   f2(x) = f1(x - d)
4. Through algebraic manipulation, we can derive d from A1, b1 and A2, b2

The displacement d at each pixel is computed by solving:
    A * d = -Δb/2
Where A is the average of A1 and A2, and Δb = b2 - b1.

PYRAMID EXTENSION:
Like Lucas-Kanade, Farneback uses image pyramids for large motion.
Start at coarsest level, estimate rough flow, then refine at each level.

PARAMETERS EXPLAINED:
    pyr_scale  = pyramid downscale factor (0.5 means each level is half size)
    levels     = number of pyramid levels
    winsize    = averaging window size (larger = smoother, more robust)
    iterations = iterations at each pyramid level
    poly_n     = size of pixel neighborhood for polynomial expansion
                 (5 = fast, 7 = more accurate)
    poly_sigma = std dev of Gaussian for polynomial expansion smoothing
    flags      = optional flags for using previous flow as initial guess


VISUALIZATION WITH HSV
----------------------
Dense flow gives a (u, v) vector at every pixel. We visualize this as:
    HUE        = direction of motion (angle of the flow vector)
    SATURATION = 255 (full color)
    VALUE      = magnitude of motion (how fast the pixel moves)

This creates an intuitive color map where:
    - RED    = rightward motion
    - GREEN  = downward motion
    - BLUE   = leftward motion
    - YELLOW = upward motion
    - BRIGHT = fast motion
    - DARK   = slow/no motion

=============================================================================
"""

import cv2
import numpy as np
import time
import os

# ─── Configuration ───────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 400  # ~6.5 seconds at 60fps
START_FRAME = 1800  # Skip to where there's actual motion

class DebugLogger:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
    def log(self, msg, level="INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] [{level:5s}] [{self.name}] {msg}")

log = DebugLogger("Farneback-Dense")


def flow_to_hsv(flow):
    """
    Convert a dense optical flow field to an HSV color image.

    flow: (H, W, 2) array where flow[...,0] = u (horizontal)
                                  flow[...,1] = v (vertical)

    Returns: BGR image for display
    """
    # Compute magnitude and angle of flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create HSV image
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2   # Hue = direction (0-180 in OpenCV)
    hsv[..., 1] = 255                         # Saturation = full
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value = speed

    # Convert to BGR for display/saving
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def run_farneback_dense():
    """
    Run Farneback dense optical flow on the sample video.
    Computes per-pixel flow and visualizes using HSV color coding.
    """

    # ── Step 1: Open Video ───────────────────────────────────────────────
    log.log(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log.log("FAILED to open video!", "ERROR")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.log(f"Video: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    # Downscale for speed — dense flow is expensive
    scale = 0.4
    new_w, new_h = int(width * scale), int(height * scale)
    log.log(f"Working at {scale}x resolution: {new_w}x{new_h} ({new_w*new_h} pixels to compute flow for)")

    # ── Step 2: Setup outputs ────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "02_farneback_dense.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Output is side-by-side: original + flow visualization
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w * 2, new_h))
    log.log(f"Output (side-by-side): {out_path}")

    # ── Step 3: Read first frame ─────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    log.log(f"Skipping to frame {START_FRAME}")
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 4: Main loop ────────────────────────────────────────────────
    frame_count = 0
    log.log("Starting dense flow computation...")
    log.log("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.log("End of video.")
            break

        frame_count += 1
        if MAX_FRAMES and frame_count > MAX_FRAMES:
            log.log(f"Reached max frames ({MAX_FRAMES}).")
            break

        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 4a: Compute Farneback dense optical flow ────────────────
        t_start = time.time()
        flow = cv2.calcOpticalFlowFarneback(
            old_gray,       # Previous frame (grayscale)
            frame_gray,     # Current frame (grayscale)
            None,           # Initial flow estimate (None = compute from scratch)
            pyr_scale=0.5,  # Pyramid scale: each level is half the previous
            levels=3,       # 3 pyramid levels
            winsize=15,     # Averaging window: 15x15
            iterations=3,   # 3 iterations at each pyramid level
            poly_n=5,       # 5x5 neighborhood for polynomial expansion
            poly_sigma=1.2, # Gaussian sigma for polynomial smoothing
            flags=0         # No special flags
        )
        t_flow = time.time() - t_start
        # flow shape: (new_h, new_w, 2) — a (u,v) vector at every pixel!

        # ── Step 4b: Analyze the flow field ──────────────────────────────
        u = flow[..., 0]  # horizontal displacement at each pixel
        v = flow[..., 1]  # vertical displacement at each pixel
        magnitude = np.sqrt(u**2 + v**2)

        avg_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)

        # What percentage of pixels are actually moving?
        moving_threshold = 1.0  # pixels moving > 1px per frame
        pct_moving = np.mean(magnitude > moving_threshold) * 100

        # Dominant direction (average flow vector of moving pixels)
        moving_mask = magnitude > moving_threshold
        if np.any(moving_mask):
            avg_u = np.mean(u[moving_mask])
            avg_v = np.mean(v[moving_mask])
            dom_angle = np.arctan2(avg_v, avg_u) * 180 / np.pi
            dom_mag = np.sqrt(avg_u**2 + avg_v**2)
        else:
            avg_u = avg_v = dom_angle = dom_mag = 0

        if frame_count % 15 == 0:
            log.log(
                f"Frame {frame_count:4d} | "
                f"Flow: avg={avg_mag:.2f}px, max={max_mag:.2f}px | "
                f"Moving: {pct_moving:.1f}% pixels | "
                f"Dominant: {dom_mag:.1f}px @ {dom_angle:.0f}° | "
                f"Time: {t_flow*1000:.1f}ms"
            )

        # ── Step 4c: Visualize ───────────────────────────────────────────
        # Convert flow to HSV color visualization
        flow_vis = flow_to_hsv(flow)

        # Draw some flow arrows on the original frame for intuition
        vis = frame.copy()
        step = 20  # Sample every 20 pixels for arrows
        y_coords, x_coords = np.mgrid[step//2:new_h:step, step//2:new_w:step]
        for y, x in zip(y_coords.flatten(), x_coords.flatten()):
            fx, fy = flow[y, x]
            mag = np.sqrt(fx**2 + fy**2)
            if mag > 0.5:  # Only draw if there's meaningful motion
                end_x = int(x + fx * 3)  # Scale up arrows for visibility
                end_y = int(y + fy * 3)
                # Color by magnitude: green=slow, red=fast
                color = (0, max(0, 255 - int(mag * 30)), min(255, int(mag * 30)))
                cv2.arrowedLine(vis, (int(x), int(y)), (end_x, end_y), color, 1, tipLength=0.3)

        # Add text overlay
        info_lines = [
            f"Farneback Dense Optical Flow",
            f"Frame: {frame_count} | Moving: {pct_moving:.0f}% pixels",
            f"Avg flow: {avg_mag:.1f}px | Max: {max_mag:.1f}px",
            f"Compute time: {t_flow*1000:.1f}ms",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 22 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(flow_vis, "HSV Flow Map", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Side-by-side output
        combined = np.hstack([vis, flow_vis])
        out.write(combined)

        # Update previous frame
        old_gray = frame_gray.copy()

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    out.release()
    log.log("=" * 70)
    log.log(f"Processed {frame_count} frames")
    log.log(f"Output saved: {out_path}")
    log.log("Done!")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" FARNEBACK DENSE OPTICAL FLOW — LEARNING IMPLEMENTATION")
    print("=" * 70)
    print()
    run_farneback_dense()
