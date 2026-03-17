"""
=============================================================================
ALGORITHM 1: LUCAS-KANADE SPARSE OPTICAL FLOW
=============================================================================

WHAT IS OPTICAL FLOW?
---------------------
Optical flow is the pattern of apparent motion of objects in a visual scene
caused by relative motion between the observer (camera) and the scene.
Think of it as: "where did each pixel move between frame N and frame N+1?"

It's represented as a 2D vector field where each vector (u, v) tells you
the displacement of a pixel from one frame to the next.


WHAT IS "SPARSE" vs "DENSE"?
-----------------------------
- SPARSE: Compute flow only at selected "interesting" points (corners, edges).
  Faster, but you only know motion at a few hundred points.
- DENSE: Compute flow for EVERY pixel in the image.
  Slower, but gives you a complete motion map.


THE LUCAS-KANADE METHOD (1981)
------------------------------
Core idea: It assumes that the flow is essentially constant in a local
neighborhood of the pixel under consideration.

It relies on THREE ASSUMPTIONS:
  1. BRIGHTNESS CONSTANCY: A pixel's intensity doesn't change between frames.
     I(x, y, t) = I(x + dx, y + dy, t + dt)

  2. TEMPORAL PERSISTENCE (SMALL MOTION): Motion between frames is small
     enough that we can use a Taylor series approximation.

  3. SPATIAL COHERENCE: Neighboring pixels have similar motion.

From assumption 1 + Taylor expansion, we get the OPTICAL FLOW EQUATION:
    Ix * u + Iy * v + It = 0

Where:
    Ix = dI/dx (spatial gradient in x)
    Iy = dI/dy (spatial gradient in y)
    It = dI/dt (temporal gradient — change over time)
    u  = dx/dt (horizontal velocity — what we want)
    v  = dy/dt (vertical velocity — what we want)

PROBLEM: One equation, two unknowns (u, v). Can't solve!

LUCAS-KANADE SOLUTION: Use assumption 3 (spatial coherence).
Take a window of pixels (e.g., 15x15) around each point. ALL pixels in
that window share the same flow (u, v). Now we have MANY equations:

    Ix1 * u + Iy1 * v = -It1
    Ix2 * u + Iy2 * v = -It2
    ...
    Ixn * u + Iyn * v = -Itn

This is an overdetermined system: A * [u, v]^T = b
Where:
    A = [[Ix1, Iy1], [Ix2, Iy2], ...] (n x 2 matrix)
    b = [[-It1], [-It2], ...]          (n x 1 vector)

Solve using LEAST SQUARES (pseudoinverse):
    [u, v]^T = (A^T * A)^(-1) * A^T * b

The matrix A^T * A is a 2x2 matrix:
    [[sum(Ix^2),    sum(Ix*Iy)],
     [sum(Ix*Iy),   sum(Iy^2) ]]

This is the same structure as the Harris corner detector's matrix!
That's WHY Lucas-Kanade works best on CORNERS — corners have strong
gradients in both x and y, making A^T*A well-conditioned (invertible).


PYRAMID (MULTI-SCALE) EXTENSION
--------------------------------
The "small motion" assumption breaks for fast-moving objects. Solution:
Build an image pyramid (downscale by 2x at each level). At the coarsest
level, large motions appear small. Compute flow there, then refine at
each finer level. This is what cv2.calcOpticalFlowPyrLK does.


WHAT ARE "GOOD FEATURES TO TRACK"?
-----------------------------------
cv2.goodFeaturesToTrack finds corners using the Shi-Tomasi method:
- Compute Harris-like matrix M for each pixel
- Take min(eigenvalue1, eigenvalue2)
- If min eigenvalue > threshold → it's a good corner
- Apply non-maximum suppression to avoid clustering

Corners are ideal for LK because they have texture in BOTH directions,
making the A^T*A matrix invertible.

=============================================================================
"""

import cv2
import numpy as np
import time
import os
import sys

# ─── Configuration ───────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# How many frames to process (None = all)
MAX_FRAMES = 400  # ~6.5 seconds at 60fps, enough to learn from
START_FRAME = 1800  # Skip to where there's actual motion in the video

# ─── Debug Logger ────────────────────────────────────────────────────────────
class DebugLogger:
    """Simple logger that prints timestamped debug info."""
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()

    def log(self, msg, level="INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] [{level:5s}] [{self.name}] {msg}")

log = DebugLogger("LK-Sparse")


def run_lucas_kanade_sparse():
    """
    Run Lucas-Kanade sparse optical flow on the sample video.
    Tracks corner features across frames and draws their motion trails.
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
    log.log(f"Video properties: {width}x{height}, {fps:.1f}fps, {total_frames} frames")

    # We'll work at half resolution for speed
    scale = 0.5
    new_w, new_h = int(width * scale), int(height * scale)
    log.log(f"Working at {scale}x resolution: {new_w}x{new_h}")

    # ── Step 2: Setup output video writer ────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "01_lk_sparse.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))
    log.log(f"Output will be saved to: {out_path}")

    # ── Step 3: Configure feature detection parameters ───────────────────
    #
    # cv2.goodFeaturesToTrack parameters:
    #   maxCorners  = max number of corners to detect
    #   qualityLevel = minimum quality (fraction of best corner's quality)
    #   minDistance  = minimum euclidean distance between detected corners
    #   blockSize   = size of neighborhood for corner computation
    #
    feature_params = dict(
        maxCorners=200,       # Track up to 200 points
        qualityLevel=0.05,    # Accept corners with quality >= 5% of best
        minDistance=15,        # Corners must be >= 15px apart
        blockSize=7           # 7x7 neighborhood for corner detection
    )
    log.log(f"Feature detection params: {feature_params}")

    # ── Step 4: Configure Lucas-Kanade parameters ────────────────────────
    #
    # cv2.calcOpticalFlowPyrLK parameters:
    #   winSize  = size of the search window at each pyramid level
    #              Larger = more robust to noise, but assumes larger uniform motion
    #   maxLevel = max pyramid level (0 = no pyramid, 3 = 4 levels)
    #              Each level halves resolution. More levels = handles faster motion
    #   criteria = when to stop the iterative search
    #              (type, max_iterations, epsilon)
    #
    lk_params = dict(
        winSize=(21, 21),     # 21x21 window — the "neighborhood" in LK theory
        maxLevel=3,           # 4 pyramid levels (0,1,2,3) — handles motion up to ~16px
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,     # max 30 iterations
            0.01    # or stop when flow changes by < 0.01 px
        )
    )
    log.log(f"LK params: winSize={lk_params['winSize']}, maxLevel={lk_params['maxLevel']}")
    log.log(f"  Pyramid allows tracking motion up to ~{2**lk_params['maxLevel'] * lk_params['winSize'][0]//2}px")

    # ── Step 5: Read first frame and detect initial features ─────────────
    # Skip to where there's action in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    log.log(f"Skipping to frame {START_FRAME} (where motion begins)")

    ret, frame = cap.read()
    if not ret:
        log.log("Failed to read first frame!", "ERROR")
        return

    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect initial corners
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    log.log(f"Detected {len(p0)} initial feature points (corners)")

    # Create a mask image for drawing flow trails (accumulated over frames)
    trail_mask = np.zeros_like(frame)

    # Generate random colors for each tracked point
    colors = np.random.randint(0, 255, (200, 3)).tolist()

    # ── Step 6: Main tracking loop ───────────────────────────────────────
    frame_count = 0
    redetect_interval = 30  # Re-detect features every 30 frames

    log.log("Starting tracking loop...")
    log.log("=" * 70)

    while True:
        ret, frame = cap.read()
        if not ret:
            log.log("End of video reached.")
            break

        frame_count += 1
        if MAX_FRAMES and frame_count > MAX_FRAMES:
            log.log(f"Reached max frames limit ({MAX_FRAMES}).")
            break

        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is None or len(p0) == 0:
            log.log(f"Frame {frame_count}: No points to track, re-detecting...", "WARN")
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            trail_mask = np.zeros_like(frame)
            continue

        # ── Step 6a: Calculate optical flow ──────────────────────────────
        #
        # calcOpticalFlowPyrLK returns:
        #   p1     = new positions of tracked points (Nx1x2 float32)
        #   status = 1 if flow for corresponding point was found, 0 otherwise
        #   err    = error for each point (lower = more confident)
        #
        t_start = time.time()
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        t_flow = time.time() - t_start

        # ── Step 6b: Filter — keep only successfully tracked points ──────
        if p1 is not None:
            good_new = p1[status.flatten() == 1].reshape(-1, 2)
            good_old = p0[status.flatten() == 1].reshape(-1, 2)
        else:
            good_new = np.empty((0, 2), dtype=np.float32)
            good_old = np.empty((0, 2), dtype=np.float32)

        tracked = len(good_new)
        lost = len(p0) - tracked

        # ── Step 6c: Compute flow statistics ─────────────────────────────
        if len(good_new) > 0 and len(good_old) > 0:
            # Flow vectors = new_position - old_position
            flow_vectors = good_new - good_old
            magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
            angles = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0]) * 180 / np.pi

            avg_mag = np.mean(magnitudes)
            max_mag = np.max(magnitudes)
            avg_angle = np.mean(angles)

            # Determine dominant motion direction
            if avg_mag < 0.5:
                direction = "STATIONARY"
            elif -45 <= avg_angle <= 45:
                direction = "RIGHT"
            elif 45 < avg_angle <= 135:
                direction = "DOWN"
            elif -135 <= avg_angle < -45:
                direction = "UP"
            else:
                direction = "LEFT"
        else:
            avg_mag = max_mag = avg_angle = 0
            direction = "N/A"

        # Debug log every 15 frames
        if frame_count % 15 == 0:
            log.log(
                f"Frame {frame_count:4d} | "
                f"Tracked: {tracked:3d} pts | "
                f"Lost: {lost:2d} | "
                f"Flow: avg={avg_mag:.2f}px, max={max_mag:.2f}px | "
                f"Dir: {direction:10s} | "
                f"Time: {t_flow*1000:.1f}ms"
            )

        # ── Step 6d: Draw the results ────────────────────────────────────
        vis = frame.copy()

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.astype(int)
            c, d = old.astype(int)
            color = colors[i % len(colors)]

            # Draw trail line (accumulated on mask)
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), color, 2)
            # Draw current point position
            vis = cv2.circle(vis, (a, b), 4, color, -1)

        # Overlay trails on current frame
        vis = cv2.add(vis, trail_mask)

        # Add text overlay with debug info
        info_lines = [
            f"Lucas-Kanade Sparse Optical Flow",
            f"Frame: {frame_count}/{total_frames}",
            f"Tracking: {tracked} points",
            f"Avg motion: {avg_mag:.1f}px | Direction: {direction}",
            f"Flow computation: {t_flow*1000:.1f}ms",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(vis, line, (10, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        out.write(vis)

        # ── Step 6e: Update for next iteration ───────────────────────────
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # ── Step 6f: Re-detect features periodically ─────────────────────
        # Points get lost over time (go off-screen, get occluded, tracking fails).
        # Re-detect fresh corners periodically to maintain good coverage.
        if frame_count % redetect_interval == 0 or tracked < 50:
            new_points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            if new_points is not None:
                if len(p0) > 0:
                    p0 = np.vstack([p0, new_points])
                    # Remove duplicates (points too close together)
                    # Simple approach: keep first N unique points
                    if len(p0) > feature_params['maxCorners']:
                        p0 = p0[:feature_params['maxCorners']]
                else:
                    p0 = new_points
                log.log(f"  Re-detected features. Now tracking {len(p0)} points.", "DEBUG")
                # Reset trails to avoid visual clutter
                trail_mask = np.zeros_like(frame)
                colors = np.random.randint(0, 255, (len(p0), 3)).tolist()

    # ── Step 7: Cleanup ──────────────────────────────────────────────────
    cap.release()
    out.release()

    log.log("=" * 70)
    log.log(f"Processed {frame_count} frames")
    log.log(f"Output saved to: {out_path}")
    log.log("Done!")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" LUCAS-KANADE SPARSE OPTICAL FLOW — LEARNING IMPLEMENTATION")
    print("=" * 70)
    print()
    run_lucas_kanade_sparse()
