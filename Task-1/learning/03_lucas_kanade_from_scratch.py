"""
=============================================================================
ALGORITHM 3: LUCAS-KANADE FROM SCRATCH (No cv2.calcOpticalFlowPyrLK)
=============================================================================

This is the MOST IMPORTANT file for Task 1 — you need to UNDERSTAND and
IMPLEMENT Lucas-Kanade yourself. This implementation builds it step-by-step
from raw math, using only basic NumPy/OpenCV operations.

THE MATH IN DETAIL
------------------

Step 1: THE OPTICAL FLOW CONSTRAINT EQUATION

Starting from brightness constancy:
    I(x, y, t) = I(x + u, y + v, t + 1)

Taylor expand the right side:
    I(x + u, y + v, t+1) ≈ I(x,y,t) + Ix*u + Iy*v + It

Where Ix = ∂I/∂x, Iy = ∂I/∂y, It = ∂I/∂t (partial derivatives)

Substituting back (and canceling I(x,y,t)):
    Ix*u + Iy*v + It = 0

    Or equivalently: ∇I · [u, v]^T = -It

This is ONE equation with TWO unknowns (u, v). Unsolvable alone!


Step 2: LUCAS-KANADE ASSUMPTION — SPATIAL COHERENCE

Assume all pixels in a window W around (x, y) share the same flow (u, v).
For a window of n pixels, we get n equations:

    Ix_1 * u + Iy_1 * v = -It_1
    Ix_2 * u + Iy_2 * v = -It_2
    ...
    Ix_n * u + Iy_n * v = -It_n

In matrix form: A * d = b

Where:
    A = | Ix_1  Iy_1 |      d = | u |      b = | -It_1 |
        | Ix_2  Iy_2 |          | v |          | -It_2 |
        | ...   ...  |                         | ...   |
        | Ix_n  Iy_n |                         | -It_n |


Step 3: LEAST SQUARES SOLUTION

This overdetermined system is solved via least squares:
    d = (A^T * A)^(-1) * A^T * b

Let's expand A^T * A (a 2x2 matrix):
    A^T * A = | Σ(Ix²)    Σ(Ix*Iy) |
              | Σ(Ix*Iy)  Σ(Iy²)   |

And A^T * b (a 2x1 vector):
    A^T * b = | -Σ(Ix*It) |
              | -Σ(Iy*It) |

So the solution is:
    | u |   | Σ(Ix²)    Σ(Ix*Iy) |^(-1)   | -Σ(Ix*It) |
    | v | = | Σ(Ix*Iy)  Σ(Iy²)   |      * | -Σ(Iy*It) |

For a 2x2 matrix M = [[a,b],[c,d]], the inverse is:
    M^(-1) = (1/det) * [[d, -b], [-c, a]]
    where det = a*d - b*c

WHEN DOES THIS FAIL?
- When det ≈ 0: The matrix is singular. This happens in:
  - Flat regions (no texture): Ix ≈ Iy ≈ 0
  - Edges: Strong gradient in one direction only
- Works best at CORNERS: Strong gradients in both directions!
  (Same insight as Harris corner detector)


Step 4: IMAGE PYRAMIDS FOR LARGE MOTION

Problem: Taylor expansion assumes small motion. For fast motion, the
linear approximation breaks down.

Solution: Image pyramids!
1. Build pyramid: repeatedly downsample by 2x with Gaussian blur
2. At coarsest level (smallest image), motion appears small → LK works
3. Compute flow at coarse level
4. Upsample flow to next finer level as initial guess
5. Refine at each level by computing residual flow
6. Final flow at original resolution

=============================================================================
"""

import cv2
import numpy as np
import time
import os

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 400
START_FRAME = 1800

class DebugLogger:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
    def log(self, msg, level="INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] [{level:5s}] [{self.name}] {msg}")

log = DebugLogger("LK-Scratch")


# =============================================================================
# STEP-BY-STEP IMPLEMENTATION
# =============================================================================

def compute_image_gradients(img):
    """
    Compute spatial gradients Ix and Iy using Sobel operators.

    The Sobel operator is a discrete approximation of the derivative.
    It combines Gaussian smoothing with differentiation:

    Sobel-x kernel:        Sobel-y kernel:
    [-1  0  1]             [-1 -2 -1]
    [-2  0  2]             [ 0  0  0]
    [-1  0  1]             [ 1  2  1]

    The Gaussian weighting (center row/col weighted 2x) reduces noise
    compared to simple finite differences like [-1, 0, 1].
    """
    # cv2.Sobel(src, ddepth, dx, dy, ksize)
    # ddepth=cv2.CV_64F ensures we get float output (negative gradients matter!)
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # ∂I/∂x
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # ∂I/∂y
    return Ix, Iy


def compute_temporal_gradient(img1, img2):
    """
    Compute temporal gradient It = I(t+1) - I(t).

    This is simply the pixel-wise difference between consecutive frames.
    """
    return img2.astype(np.float64) - img1.astype(np.float64)


def lucas_kanade_point(Ix, Iy, It, x, y, window_size=15):
    """
    Compute optical flow (u, v) at a SINGLE point (x, y) using Lucas-Kanade.

    This is the core algorithm — pure math, no OpenCV optical flow functions.

    Parameters:
        Ix, Iy, It: Gradient images (spatial x, spatial y, temporal)
        x, y: Point coordinates
        window_size: Size of the local window (must be odd)

    Returns:
        (u, v): Flow vector, or (0, 0) if the system is ill-conditioned
        eigenvalues: For debugging — tells us about point quality
    """
    half_w = window_size // 2
    h, w = Ix.shape

    # Boundary check
    if (y - half_w < 0 or y + half_w >= h or
        x - half_w < 0 or x + half_w >= w):
        return (0, 0), (0, 0)

    # Extract windows around the point
    Ix_w = Ix[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
    Iy_w = Iy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
    It_w = It[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()

    # Build the A^T * A matrix (2x2)
    #
    #   A^T*A = | Σ(Ix²)    Σ(Ix*Iy) |
    #           | Σ(Ix*Iy)  Σ(Iy²)   |
    #
    sum_IxIx = np.sum(Ix_w * Ix_w)
    sum_IxIy = np.sum(Ix_w * Iy_w)
    sum_IyIy = np.sum(Iy_w * Iy_w)

    ATA = np.array([
        [sum_IxIx, sum_IxIy],
        [sum_IxIy, sum_IyIy]
    ])

    # Build the A^T * b vector (2x1)
    #
    #   A^T*b = | -Σ(Ix*It) |
    #           | -Σ(Iy*It) |
    #
    ATb = np.array([
        [-np.sum(Ix_w * It_w)],
        [-np.sum(Iy_w * It_w)]
    ])

    # Check if the system is well-conditioned by looking at eigenvalues
    eigenvalues = np.linalg.eigvalsh(ATA)
    min_eigen = min(eigenvalues)

    # If eigenvalues are too small, the point is in a flat region or edge
    # (not a corner), and the solution would be unreliable
    EIGEN_THRESHOLD = 1e-4
    if min_eigen < EIGEN_THRESHOLD:
        return (0, 0), eigenvalues

    # Solve: d = (A^T * A)^(-1) * A^T * b
    # Using np.linalg.solve is more numerically stable than computing inverse
    try:
        d = np.linalg.solve(ATA, ATb)
        u, v = d[0, 0], d[1, 0]
    except np.linalg.LinAlgError:
        return (0, 0), eigenvalues

    return (u, v), eigenvalues


def build_pyramid(img, levels):
    """
    Build a Gaussian image pyramid.

    Level 0 = original image
    Level 1 = half size (after Gaussian blur + downsample)
    Level 2 = quarter size
    ...

    The Gaussian blur before downsampling prevents aliasing artifacts.
    """
    pyramid = [img.astype(np.float64)]
    for i in range(levels):
        # Blur then downsample by 2x
        blurred = cv2.GaussianBlur(pyramid[-1], (5, 5), 1.0)
        downsampled = blurred[::2, ::2]  # Take every other pixel
        pyramid.append(downsampled)
    return pyramid


def lucas_kanade_pyramidal(img1, img2, points, window_size=15, num_levels=3):
    """
    Pyramidal Lucas-Kanade optical flow — handles larger motions.

    Algorithm:
    1. Build pyramids for both images
    2. At coarsest level, compute LK flow (large motion appears small here)
    3. Propagate flow estimate to next finer level:
       a. Upsample flow by 2x (multiply values by 2)
       b. Warp img1 using current flow estimate
       c. Compute RESIDUAL flow with LK
       d. Add residual to current flow estimate
    4. Return final flow at original resolution

    Parameters:
        img1, img2: Consecutive frames (grayscale, uint8)
        points: Nx2 array of (x, y) coordinates to track
        window_size: LK window size
        num_levels: Number of pyramid levels

    Returns:
        new_points: Nx2 array of new positions
        status: Nx1 array (1=tracked, 0=lost)
    """
    # Build pyramids
    pyr1 = build_pyramid(img1, num_levels)
    pyr2 = build_pyramid(img2, num_levels)

    # Scale points to coarsest level
    scale_factor = 2 ** num_levels
    scaled_points = points.astype(np.float64) / scale_factor

    # Initialize flow as zero at coarsest level
    flow = np.zeros_like(scaled_points)

    # Process from coarsest to finest level
    for level in range(num_levels, -1, -1):
        level_img1 = pyr1[level]
        level_img2 = pyr2[level]

        # Scale points to this level
        level_points = points.astype(np.float64) / (2 ** level)

        # Compute gradients at this level
        Ix, Iy = compute_image_gradients(level_img1)
        It = compute_temporal_gradient(level_img1, level_img2)

        # Compute flow at each point
        for i, (x, y) in enumerate(level_points.astype(int)):
            uv, eigenvals = lucas_kanade_point(Ix, Iy, It, x, y, window_size)
            flow[i, 0] += uv[0]
            flow[i, 1] += uv[1]

        # Scale flow up for next (finer) level
        if level > 0:
            flow *= 2

    # Compute new positions
    new_points = points.astype(np.float64) + flow

    # Determine status — mark as lost if flow is too large or point goes OOB
    h, w = img1.shape
    status = np.ones(len(points), dtype=np.uint8)
    for i in range(len(points)):
        nx, ny = new_points[i]
        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            status[i] = 0
        elif np.sqrt(flow[i, 0]**2 + flow[i, 1]**2) > 50:
            status[i] = 0  # Unreasonably large flow

    return new_points, status


def run_lk_from_scratch():
    """
    Run our from-scratch Lucas-Kanade implementation.
    """

    log.log(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log.log("FAILED to open video!", "ERROR")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Use smaller resolution since our implementation is slower
    scale = 0.35
    new_w, new_h = int(width * scale), int(height * scale)
    log.log(f"Working at {scale}x: {new_w}x{new_h}")

    # Output
    out_path = os.path.join(OUTPUT_DIR, "03_lk_from_scratch.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))

    # Skip to interesting part
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    log.log(f"Skipping to frame {START_FRAME}")

    # Read first frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect initial features (we use OpenCV for this — allowed per task rules)
    corners = cv2.goodFeaturesToTrack(old_gray, maxCorners=100, qualityLevel=0.05,
                                      minDistance=15, blockSize=7)
    points = corners.reshape(-1, 2)
    log.log(f"Detected {len(points)} initial feature points")

    # Colors for visualization
    colors = np.random.randint(0, 255, (len(points), 3)).tolist()
    trail_mask = np.zeros_like(frame)

    frame_count = 0
    log.log("Starting from-scratch LK tracking...")
    log.log("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES and frame_count >= MAX_FRAMES):
            break

        frame_count += 1
        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(points) == 0:
            corners = cv2.goodFeaturesToTrack(frame_gray, maxCorners=100,
                                              qualityLevel=0.05, minDistance=15)
            if corners is not None:
                points = corners.reshape(-1, 2)
                colors = np.random.randint(0, 255, (len(points), 3)).tolist()
                trail_mask = np.zeros_like(frame)
            old_gray = frame_gray.copy()
            continue

        # ── Run OUR Lucas-Kanade implementation ──────────────────────────
        t_start = time.time()
        new_points, status = lucas_kanade_pyramidal(
            old_gray, frame_gray, points,
            window_size=15,
            num_levels=3
        )
        t_flow = time.time() - t_start

        # Filter to good points
        good_mask = status == 1
        good_new = new_points[good_mask]
        good_old = points[good_mask]
        good_colors = [colors[i] for i in range(len(status)) if status[i] == 1]

        tracked = len(good_new)
        lost = len(points) - tracked

        # Flow statistics
        if tracked > 0:
            flow_vecs = good_new - good_old
            mags = np.sqrt(flow_vecs[:, 0]**2 + flow_vecs[:, 1]**2)
            avg_mag = np.mean(mags)
            max_mag = np.max(mags)
        else:
            avg_mag = max_mag = 0

        if frame_count % 15 == 0:
            log.log(
                f"Frame {frame_count:4d} | "
                f"Tracked: {tracked:3d} | Lost: {lost:2d} | "
                f"Flow: avg={avg_mag:.2f}px, max={max_mag:.2f}px | "
                f"Time: {t_flow*1000:.1f}ms"
            )

        # Draw results
        vis = frame.copy()
        for i, (new, old) in enumerate(zip(good_new.astype(int), good_old.astype(int))):
            a, b = new
            c, d = old
            color = good_colors[i]
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), color, 2)
            vis = cv2.circle(vis, (a, b), 4, color, -1)

        vis = cv2.add(vis, trail_mask)

        info_lines = [
            f"LK FROM SCRATCH (no cv2.calcOpticalFlowPyrLK!)",
            f"Frame: {frame_count} | Tracking: {tracked} points",
            f"Avg flow: {avg_mag:.1f}px | Compute: {t_flow*1000:.0f}ms",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 22 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out.write(vis)

        # Update
        old_gray = frame_gray.copy()
        points = good_new
        colors = good_colors

        # Re-detect if too few points
        if frame_count % 30 == 0 or tracked < 30:
            corners = cv2.goodFeaturesToTrack(frame_gray, maxCorners=100,
                                              qualityLevel=0.05, minDistance=15)
            if corners is not None:
                new_pts = corners.reshape(-1, 2)
                points = np.vstack([points, new_pts]) if len(points) > 0 else new_pts
                points = points[:100]
                colors = np.random.randint(0, 255, (len(points), 3)).tolist()
                trail_mask = np.zeros_like(frame)

    cap.release()
    out.release()
    log.log("=" * 70)
    log.log(f"Processed {frame_count} frames")
    log.log(f"Output: {out_path}")
    log.log("Done!")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" LUCAS-KANADE FROM SCRATCH — LEARNING IMPLEMENTATION")
    print("=" * 70)
    print()
    print("This implements Lucas-Kanade using ONLY basic NumPy/OpenCV operations.")
    print("No cv2.calcOpticalFlowPyrLK — just Sobel gradients + linear algebra.")
    print()
    run_lk_from_scratch()
