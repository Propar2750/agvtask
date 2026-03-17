"""
=============================================================================
ALGORITHM 4: HORN-SCHUNCK DENSE OPTICAL FLOW (From Scratch)
=============================================================================

Horn-Schunck (1981) is a CLASSIC dense optical flow method. Unlike Lucas-Kanade
which is local (window-based), Horn-Schunck is GLOBAL — it formulates optical
flow as an optimization problem over the ENTIRE image.

THE PROBLEM
-----------
Same starting point as Lucas-Kanade: the optical flow constraint equation
    Ix*u + Iy*v + It = 0

This is one equation with two unknowns at each pixel. Lucas-Kanade's solution
was spatial coherence in a local window. Horn-Schunck takes a different path:

HORN-SCHUNCK'S APPROACH: GLOBAL SMOOTHNESS
-------------------------------------------
Add a REGULARIZATION TERM that penalizes flow fields that aren't smooth.

Minimize the energy functional:
    E(u, v) = ∫∫ [(Ix*u + Iy*v + It)² + α²(|∇u|² + |∇v|²)] dx dy
              ─────────────────────────  ───────────────────────
              Data term: flow should     Smoothness term: flow
              satisfy brightness         should be smooth
              constancy                  (no sudden jumps)

Where:
    α = regularization parameter (VERY IMPORTANT!)
    - Small α: Trust the data more → noisy but detailed flow
    - Large α: Enforce smoothness more → smooth but may blur boundaries
    ∇u = gradient of u-flow field → penalizes sharp changes in flow

SOLVING WITH EULER-LAGRANGE EQUATIONS
-------------------------------------
Setting the variational derivatives to zero gives:
    Ix*(Ix*u + Iy*v + It) - α²∇²u = 0
    Iy*(Ix*u + Iy*v + It) - α²∇²v = 0

Where ∇² is the Laplacian operator (approximated by the difference between
a pixel and the average of its neighbors).

Rearranging for an ITERATIVE SOLUTION:
    u_new = ū - Ix * (Ix*ū + Iy*v̄ + It) / (α² + Ix² + Iy²)
    v_new = v̄ - Iy * (Ix*ū + Iy*v̄ + It) / (α² + Ix² + Iy²)

Where ū, v̄ are the LOCAL AVERAGES (Laplacian approximation) of u, v.

This is a JACOBI-STYLE ITERATION: keep updating u and v using neighbors'
values until convergence.

COMPARISON WITH LUCAS-KANADE
-----------------------------
| Feature           | Lucas-Kanade          | Horn-Schunck            |
|-------------------|-----------------------|-------------------------|
| Type              | Local (window-based)  | Global (whole image)    |
| Output            | Sparse or semi-dense  | Dense (every pixel)     |
| Speed             | Fast                  | Slower (iterative)      |
| Motion boundaries | Preserves edges       | Smooths across edges    |
| Noise handling    | Good (overdetermined) | Good (regularization)   |
| Large motion      | Needs pyramids        | Needs pyramids too      |
| Key parameter     | Window size           | α (smoothness weight)   |

=============================================================================
"""

import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 200  # Fewer frames since HS is slower
START_FRAME = 1800

class DebugLogger:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
    def log(self, msg, level="INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] [{level:5s}] [{self.name}] {msg}")

log = DebugLogger("Horn-Schunck")


def horn_schunck(img1, img2, alpha=1.0, num_iterations=100,
                  convergence_threshold=1e-4):
    """
    Compute dense optical flow using the Horn-Schunck method.

    Parameters:
        img1, img2: Consecutive frames (grayscale, float64)
        alpha: Regularization parameter
               - Small (0.1): More detailed, noisier flow
               - Large (10.0): Smoother flow, may miss fine details
               Typical range: 0.5 to 5.0
        num_iterations: Maximum Jacobi iterations
        convergence_threshold: Stop if max change < this value

    Returns:
        u, v: Dense flow fields (same size as input images)
    """
    h, w = img1.shape

    # ── Step 1: Compute image gradients ──────────────────────────────────
    # Spatial gradients (average of both frames for better accuracy)
    # Using simple finite differences here for clarity:
    #   Ix ≈ (I(x+1) - I(x-1)) / 2  (central difference)

    # Averaged spatial gradients (average between frame 1 and 2)
    Ix = (cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3) +
          cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)) / 2
    Iy = (cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3) +
          cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)) / 2

    # Temporal gradient
    It = img2 - img1

    # ── Step 2: Initialize flow fields to zero ───────────────────────────
    u = np.zeros((h, w), dtype=np.float64)
    v = np.zeros((h, w), dtype=np.float64)

    # Precompute denominator (constant across iterations)
    # denominator = α² + Ix² + Iy²
    alpha_sq = alpha ** 2
    denom = alpha_sq + Ix**2 + Iy**2

    # Laplacian kernel for computing local average
    # This 4-neighbor averaging kernel approximates ∇²:
    #   [0    1/4  0  ]
    #   [1/4  0    1/4]
    #   [0    1/4  0  ]
    laplacian_kernel = np.array([
        [0,    1/4, 0],
        [1/4,  0,   1/4],
        [0,    1/4, 0]
    ], dtype=np.float64)

    # ── Step 3: Iterative refinement ─────────────────────────────────────
    for iteration in range(num_iterations):
        # Compute local averages ū and v̄ using the Laplacian kernel
        u_avg = cv2.filter2D(u, cv2.CV_64F, laplacian_kernel)
        v_avg = cv2.filter2D(v, cv2.CV_64F, laplacian_kernel)

        # Compute the "brightness constancy residual" using averaged flow
        # P = (Ix*ū + Iy*v̄ + It)
        P = Ix * u_avg + Iy * v_avg + It

        # Update flow:
        # u_new = ū - Ix * P / denom
        # v_new = v̄ - Iy * P / denom
        u_new = u_avg - Ix * P / denom
        v_new = v_avg - Iy * P / denom

        # Check convergence: how much did the flow change?
        max_change = max(np.max(np.abs(u_new - u)), np.max(np.abs(v_new - v)))

        u = u_new
        v = v_new

        if max_change < convergence_threshold:
            log.log(f"    Converged at iteration {iteration+1} "
                    f"(max_change={max_change:.6f})", "DEBUG")
            break

    return u, v


def flow_to_hsv(u, v):
    """Convert flow (u, v) to HSV visualization."""
    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(v, u)

    hsv = np.zeros((*u.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle * 180 / np.pi / 2 + 180) % 180
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def run_horn_schunck():
    """
    Run Horn-Schunck dense optical flow on the sample video.
    Shows how the regularization parameter α affects results.
    """

    log.log(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log.log("FAILED!", "ERROR")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use small resolution — Horn-Schunck is iterative and slow
    scale = 0.25
    new_w, new_h = int(width * scale), int(height * scale)
    log.log(f"Working at {scale}x: {new_w}x{new_h} ({new_w*new_h} pixels)")
    log.log("NOTE: Horn-Schunck is slow due to iterative solving. Be patient!")

    # Output
    out_path = os.path.join(OUTPUT_DIR, "04_horn_schunck.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w * 2, new_h))

    # Skip to interesting part
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    log.log(f"Skipping to frame {START_FRAME}")

    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    frame_count = 0
    alpha = 2.0  # Regularization weight — experiment with this!
    num_iter = 50  # Max iterations per frame

    log.log(f"Parameters: alpha={alpha}, max_iterations={num_iter}")
    log.log("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES and frame_count >= MAX_FRAMES):
            break

        frame_count += 1
        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # ── Compute Horn-Schunck flow ────────────────────────────────────
        t_start = time.time()
        u, v = horn_schunck(old_gray, frame_gray, alpha=alpha,
                            num_iterations=num_iter)
        t_flow = time.time() - t_start

        # Flow statistics
        magnitude = np.sqrt(u**2 + v**2)
        avg_mag = np.mean(magnitude)
        max_mag = np.max(magnitude)
        pct_moving = np.mean(magnitude > 0.5) * 100

        if frame_count % 10 == 0:
            log.log(
                f"Frame {frame_count:4d} | "
                f"Flow: avg={avg_mag:.2f}px, max={max_mag:.2f}px | "
                f"Moving: {pct_moving:.0f}% | "
                f"Time: {t_flow*1000:.0f}ms"
            )

        # Visualize
        flow_vis = flow_to_hsv(u, v)

        # Draw arrows on original
        vis = frame.copy()
        step = 15
        for y in range(step//2, new_h, step):
            for x in range(step//2, new_w, step):
                fx, fy = u[y, x], v[y, x]
                mag = np.sqrt(fx**2 + fy**2)
                if mag > 0.3:
                    end_x = int(x + fx * 3)
                    end_y = int(y + fy * 3)
                    color = (0, max(0, 255-int(mag*40)), min(255, int(mag*40)))
                    cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 1, tipLength=0.3)

        info = [
            f"Horn-Schunck Dense Flow (alpha={alpha})",
            f"Frame: {frame_count} | Moving: {pct_moving:.0f}%",
            f"Time: {t_flow*1000:.0f}ms",
        ]
        for i, line in enumerate(info):
            cv2.putText(vis, line, (10, 18+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1, cv2.LINE_AA)

        combined = np.hstack([vis, flow_vis])
        out.write(combined)

        old_gray = frame_gray.copy()

    cap.release()
    out.release()
    log.log("=" * 70)
    log.log(f"Processed {frame_count} frames. Output: {out_path}")
    log.log("Done!")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" HORN-SCHUNCK DENSE OPTICAL FLOW — FROM SCRATCH")
    print("=" * 70)
    print()
    print("A global variational method — computes flow at every pixel")
    print("by minimizing a combined data + smoothness energy function.")
    print()
    run_horn_schunck()
