"""
=============================================================================
EXPERIMENT: CANNY EDGES vs HARRIS CORNERS for Lucas-Kanade Tracking
=============================================================================

THE QUESTION:
What happens if we feed edge points (from Canny) into Lucas-Kanade
instead of corner points (from goodFeaturesToTrack / Harris)?

HYPOTHESIS:
Corners should track better because LK needs the A^T*A matrix to be
well-conditioned (both eigenvalues large). Edges have the "aperture
problem" — ambiguous motion along the edge direction.

BUT edges might give:
- Better coverage along object boundaries
- More points in texture-poor regions
- Useful motion info perpendicular to edges

Let's measure and compare!

WHAT WE MEASURE:
1. Tracking survival rate (what % of points are still tracked after N frames)
2. Flow confidence (eigenvalue ratio of A^T*A at each point)
3. Flow stability (how consistent is flow between consecutive frames)
4. Visual comparison (side-by-side video)

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
NUM_POINTS = 150  # Same number of points for fair comparison


class DebugLogger:
    def __init__(self, name):
        self.name = name
        self.start_time = time.time()
    def log(self, msg, level="INFO"):
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.2f}s] [{level:5s}] [{self.name}] {msg}")

log = DebugLogger("Canny-vs-Harris")


def detect_corner_points(gray, max_points=NUM_POINTS):
    """
    Detect points using Shi-Tomasi corners (goodFeaturesToTrack).

    Shi-Tomasi criterion: min(λ1, λ2) > threshold
    Where λ1, λ2 are eigenvalues of the structure tensor (A^T*A).

    This GUARANTEES both eigenvalues are large → LK will work well.
    """
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_points,
        qualityLevel=0.05,
        minDistance=10,
        blockSize=7
    )
    if corners is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return corners


def detect_canny_points(gray, max_points=NUM_POINTS):
    """
    Detect points using Canny edge detection.

    Canny pipeline:
    1. Gaussian blur (reduce noise)
    2. Sobel gradients → magnitude & direction
    3. Non-maximum suppression (thin edges to 1px wide)
    4. Hysteresis thresholding (strong edges + connected weak edges)

    The output is a binary edge map. We sample points FROM these edges
    to use as LK tracking points.

    KEY INSIGHT: These are EDGE points, not corners. They have strong
    gradient in one direction but not necessarily both. This means the
    A^T*A matrix may be ill-conditioned at these points.
    """
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Get coordinates of all edge pixels
    edge_coords = np.argwhere(edges > 0)  # Returns (row, col) = (y, x)

    if len(edge_coords) == 0:
        return np.empty((0, 1, 2), dtype=np.float32)

    # Subsample edge points (evenly spaced along the edge map)
    if len(edge_coords) > max_points:
        indices = np.linspace(0, len(edge_coords) - 1, max_points, dtype=int)
        edge_coords = edge_coords[indices]

    # Convert to the format LK expects: (N, 1, 2) with (x, y) order
    points = edge_coords[:, ::-1].astype(np.float32).reshape(-1, 1, 2)
    return points


def compute_eigenvalue_quality(gray, points, block_size=7):
    """
    Compute the eigenvalue ratio (condition number) of A^T*A at each point.

    This tells us HOW WELL Lucas-Kanade can track each point:
    - Ratio ≈ 1: Both eigenvalues similar → CORNER → excellent for LK
    - Ratio >> 1: One eigenvalue dominates → EDGE → aperture problem
    - Both small: Flat region → nothing to track

    Returns:
        min_eigenvalues: min(λ1, λ2) at each point (higher = better)
        ratios: max(λ1, λ2) / min(λ1, λ2) at each point (closer to 1 = better)
    """
    # Compute structure tensor components using Sobel
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # For each point, compute A^T*A in a local window
    half_w = block_size // 2
    h, w = gray.shape
    min_eigenvalues = []
    ratios = []

    for pt in points.reshape(-1, 2):
        x, y = int(pt[0]), int(pt[1])

        if (y - half_w < 0 or y + half_w >= h or
            x - half_w < 0 or x + half_w >= w):
            min_eigenvalues.append(0)
            ratios.append(float('inf'))
            continue

        # Extract window
        ix_w = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
        iy_w = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1]

        # Build A^T*A
        ATA = np.array([
            [np.sum(ix_w**2),      np.sum(ix_w * iy_w)],
            [np.sum(ix_w * iy_w),  np.sum(iy_w**2)]
        ])

        # Compute eigenvalues
        eigvals = np.linalg.eigvalsh(ATA)
        eigvals = np.sort(np.abs(eigvals))

        min_eigen = eigvals[0]
        max_eigen = eigvals[1]
        min_eigenvalues.append(min_eigen)

        if min_eigen > 1e-6:
            ratios.append(max_eigen / min_eigen)
        else:
            ratios.append(float('inf'))

    return np.array(min_eigenvalues), np.array(ratios)


def run_experiment():
    """
    Run the Canny vs Harris comparison experiment.
    """

    log.log("=" * 70)
    log.log("EXPERIMENT: Canny Edge Points vs Harris Corner Points for LK")
    log.log("=" * 70)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log.log("Failed to open video!", "ERROR")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    scale = 0.5
    new_w, new_h = int(width * scale), int(height * scale)
    log.log(f"Resolution: {new_w}x{new_h}")

    # Output: side-by-side comparison video
    out_path = os.path.join(OUTPUT_DIR, "06_canny_vs_harris.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w * 2, new_h))

    # LK params (same for both)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # Skip to interesting frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Detect initial points with both methods ──────────────────────────
    harris_pts = detect_corner_points(old_gray)
    canny_pts = detect_canny_points(old_gray)

    log.log(f"Initial Harris corners: {len(harris_pts)}")
    log.log(f"Initial Canny edges:    {len(canny_pts)}")

    # ── Analyze initial point quality ────────────────────────────────────
    h_min_eig, h_ratios = compute_eigenvalue_quality(old_gray, harris_pts)
    c_min_eig, c_ratios = compute_eigenvalue_quality(old_gray, canny_pts)

    # Filter out inf ratios for stats
    h_valid = h_ratios[h_ratios < 1e6]
    c_valid = c_ratios[c_ratios < 1e6]

    log.log("")
    log.log("INITIAL POINT QUALITY ANALYSIS:")
    log.log(f"  Harris corners:")
    log.log(f"    Min eigenvalue:  mean={np.mean(h_min_eig):.0f}, median={np.median(h_min_eig):.0f}")
    if len(h_valid) > 0:
        log.log(f"    Eigenvalue ratio: mean={np.mean(h_valid):.1f}, median={np.median(h_valid):.1f}")
    log.log(f"    (Ratio closer to 1 = more corner-like = better for LK)")
    log.log(f"  Canny edges:")
    log.log(f"    Min eigenvalue:  mean={np.mean(c_min_eig):.0f}, median={np.median(c_min_eig):.0f}")
    if len(c_valid) > 0:
        log.log(f"    Eigenvalue ratio: mean={np.mean(c_valid):.1f}, median={np.median(c_valid):.1f}")
    log.log(f"    (Higher ratio = more edge-like = aperture problem)")
    log.log("")

    # Create trail masks
    harris_trail = np.zeros_like(frame)
    canny_trail = np.zeros_like(frame)
    harris_colors = np.random.randint(0, 255, (500, 3)).tolist()
    canny_colors = np.random.randint(0, 255, (500, 3)).tolist()

    # Stats accumulators
    harris_stats = {"tracked": [], "lost": [], "avg_flow": []}
    canny_stats = {"tracked": [], "lost": [], "avg_flow": []}

    frame_count = 0
    redetect_interval = 60  # Re-detect less frequently to see survival rates

    log.log("Starting tracking comparison...")
    log.log("-" * 70)

    while True:
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES and frame_count >= MAX_FRAMES):
            break

        frame_count += 1
        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Track Harris points ──────────────────────────────────────────
        if harris_pts is not None and len(harris_pts) > 0:
            h_new, h_status, h_err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, harris_pts, None, **lk_params
            )
            if h_new is not None:
                h_good_new = h_new[h_status.flatten() == 1].reshape(-1, 1, 2)
                h_good_old = harris_pts[h_status.flatten() == 1].reshape(-1, 1, 2)
                h_tracked = len(h_good_new)
                h_lost = len(harris_pts) - h_tracked
                if h_tracked > 0:
                    h_flow = np.sqrt(np.sum((h_good_new - h_good_old)**2, axis=2))
                    h_avg_flow = np.mean(h_flow)
                else:
                    h_avg_flow = 0
            else:
                h_good_new = np.empty((0, 1, 2), dtype=np.float32)
                h_good_old = np.empty((0, 1, 2), dtype=np.float32)
                h_tracked = 0
                h_lost = len(harris_pts)
                h_avg_flow = 0
        else:
            h_good_new = np.empty((0, 1, 2), dtype=np.float32)
            h_good_old = np.empty((0, 1, 2), dtype=np.float32)
            h_tracked = h_lost = 0
            h_avg_flow = 0

        # ── Track Canny points ───────────────────────────────────────────
        if canny_pts is not None and len(canny_pts) > 0:
            c_new, c_status, c_err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, canny_pts, None, **lk_params
            )
            if c_new is not None:
                c_good_new = c_new[c_status.flatten() == 1].reshape(-1, 1, 2)
                c_good_old = canny_pts[c_status.flatten() == 1].reshape(-1, 1, 2)
                c_tracked = len(c_good_new)
                c_lost = len(canny_pts) - c_tracked
                if c_tracked > 0:
                    c_flow = np.sqrt(np.sum((c_good_new - c_good_old)**2, axis=2))
                    c_avg_flow = np.mean(c_flow)
                else:
                    c_avg_flow = 0
            else:
                c_good_new = np.empty((0, 1, 2), dtype=np.float32)
                c_good_old = np.empty((0, 1, 2), dtype=np.float32)
                c_tracked = 0
                c_lost = len(canny_pts)
                c_avg_flow = 0
        else:
            c_good_new = np.empty((0, 1, 2), dtype=np.float32)
            c_good_old = np.empty((0, 1, 2), dtype=np.float32)
            c_tracked = c_lost = 0
            c_avg_flow = 0

        # Record stats
        harris_stats["tracked"].append(h_tracked)
        harris_stats["lost"].append(h_lost)
        harris_stats["avg_flow"].append(h_avg_flow)
        canny_stats["tracked"].append(c_tracked)
        canny_stats["lost"].append(c_lost)
        canny_stats["avg_flow"].append(c_avg_flow)

        # Debug log
        if frame_count % 20 == 0:
            log.log(
                f"Frame {frame_count:4d} | "
                f"Harris: {h_tracked:3d} tracked, {h_lost:2d} lost, flow={h_avg_flow:.2f}px | "
                f"Canny: {c_tracked:3d} tracked, {c_lost:2d} lost, flow={c_avg_flow:.2f}px"
            )

        # ── Draw Harris side ─────────────────────────────────────────────
        vis_harris = frame.copy()
        for i in range(len(h_good_new)):
            a, b = h_good_new[i].ravel().astype(int)
            c, d = h_good_old[i].ravel().astype(int)
            color = harris_colors[i % len(harris_colors)]
            harris_trail = cv2.line(harris_trail, (a, b), (c, d), color, 2)
            vis_harris = cv2.circle(vis_harris, (a, b), 4, color, -1)
        vis_harris = cv2.add(vis_harris, harris_trail)

        # ── Draw Canny side ──────────────────────────────────────────────
        vis_canny = frame.copy()
        for i in range(len(c_good_new)):
            a, b = c_good_new[i].ravel().astype(int)
            c, d = c_good_old[i].ravel().astype(int)
            color = canny_colors[i % len(canny_colors)]
            canny_trail = cv2.line(canny_trail, (a, b), (c, d), color, 2)
            vis_canny = cv2.circle(vis_canny, (a, b), 4, color, -1)
        vis_canny = cv2.add(vis_canny, canny_trail)

        # Labels
        h_info = [
            f"HARRIS CORNERS (Shi-Tomasi)",
            f"Tracking: {h_tracked}/{NUM_POINTS} | Flow: {h_avg_flow:.1f}px",
        ]
        c_info = [
            f"CANNY EDGE POINTS",
            f"Tracking: {c_tracked}/{NUM_POINTS} | Flow: {c_avg_flow:.1f}px",
        ]
        for i, line in enumerate(h_info):
            cv2.putText(vis_harris, line, (10, 22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        for i, line in enumerate(c_info):
            cv2.putText(vis_canny, line, (10, 22+i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1, cv2.LINE_AA)

        combined = np.hstack([vis_harris, vis_canny])
        out.write(combined)

        # Update points
        harris_pts = h_good_new if len(h_good_new) > 0 else None
        canny_pts = c_good_new if len(c_good_new) > 0 else None
        old_gray = frame_gray.copy()

        # Re-detect periodically
        if frame_count % redetect_interval == 0:
            harris_pts = detect_corner_points(frame_gray)
            canny_pts = detect_canny_points(frame_gray)
            harris_trail = np.zeros_like(frame)
            canny_trail = np.zeros_like(frame)
            harris_colors = np.random.randint(0, 255, (500, 3)).tolist()
            canny_colors = np.random.randint(0, 255, (500, 3)).tolist()
            log.log(f"  Re-detected: Harris={len(harris_pts)}, Canny={len(canny_pts)}", "DEBUG")

    cap.release()
    out.release()

    # ── FINAL ANALYSIS ───────────────────────────────────────────────────
    log.log("")
    log.log("=" * 70)
    log.log("FINAL RESULTS")
    log.log("=" * 70)

    h_tracked_arr = np.array(harris_stats["tracked"])
    c_tracked_arr = np.array(canny_stats["tracked"])
    h_lost_arr = np.array(harris_stats["lost"])
    c_lost_arr = np.array(canny_stats["lost"])

    log.log("")
    log.log("TRACKING SURVIVAL:")
    log.log(f"  Harris: avg {np.mean(h_tracked_arr):.0f} points tracked per frame")
    log.log(f"  Canny:  avg {np.mean(c_tracked_arr):.0f} points tracked per frame")
    log.log(f"  Harris total lost: {np.sum(h_lost_arr):.0f}")
    log.log(f"  Canny  total lost: {np.sum(c_lost_arr):.0f}")

    log.log("")
    log.log("FLOW MAGNITUDE:")
    log.log(f"  Harris avg flow: {np.mean(harris_stats['avg_flow']):.3f} px/frame")
    log.log(f"  Canny  avg flow: {np.mean(canny_stats['avg_flow']):.3f} px/frame")

    log.log("")
    log.log("INTERPRETATION:")
    if np.mean(h_tracked_arr) > np.mean(c_tracked_arr):
        log.log("  >> Harris corners survived BETTER (as predicted by theory)")
        log.log("  >> Corners have well-conditioned A^T*A -> reliable LK solutions")
    else:
        log.log("  >> Canny edges survived better (surprising!)")
        log.log("  >> This may happen with very textured scenes or slow motion")

    if np.sum(c_lost_arr) > np.sum(h_lost_arr):
        pct_more = (np.sum(c_lost_arr) - np.sum(h_lost_arr)) / max(np.sum(h_lost_arr), 1) * 100
        log.log(f"  >> Canny lost {pct_more:.0f}% more points than Harris")
        log.log(f"  >> This is the APERTURE PROBLEM in action!")
    else:
        pct_more = (np.sum(h_lost_arr) - np.sum(c_lost_arr)) / max(np.sum(c_lost_arr), 1) * 100
        log.log(f"  >> Harris lost {pct_more:.0f}% more points (unusual)")

    log.log("")
    log.log("WHY THIS MATTERS FOR TASK 1:")
    log.log("  In the PyBullet simulator, you need RELIABLE flow vectors to")
    log.log("  compute the Focus of Expansion (FOE). Unreliable edge-based")
    log.log("  flow could give noisy FOE estimates -> bad navigation.")
    log.log("  HOWEVER: you could use Canny to find obstacle BOUNDARIES,")
    log.log("  then use Harris corners ON those boundaries for tracking.")
    log.log("  That is a creative hybrid approach worth exploring!")

    log.log("")
    log.log(f"Output: {out_path}")
    log.log("Done!")


if __name__ == "__main__":
    print()
    print("=" * 70)
    print(" EXPERIMENT: CANNY EDGES vs HARRIS CORNERS for Lucas-Kanade")
    print("=" * 70)
    print()
    run_experiment()
