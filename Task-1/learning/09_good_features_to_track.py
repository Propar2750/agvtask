"""
=============================================================================
GOOD FEATURES TO TRACK — cv2 usage + full manual implementation
=============================================================================

WHAT THIS FILE DOES:
  1. Loads OPTICAL_FLOW.mp4 and runs cv2.goodFeaturesToTrack on every frame,
     printing the detected corner points and overlaying them visually.

  2. Implements the EXACT same algorithm from scratch:
       - Compute image gradients (Ix, Iy) via Sobel
       - Build the structure tensor (auto-correlation matrix) M per pixel
       - Compute corner response  R = min(λ1, λ2)  [Shi-Tomasi criterion]
       - Threshold R at  qualityLevel * max(R)
       - Non-maximum suppression using pure numpy (fast, no Python loops)
       - Return top-N corners

  3. Shows and saves 3 panels side-by-side:
       LEFT   — Manual Shi-Tomasi implementation        (blue dots)
       CENTRE — cv2.goodFeaturesToTrack  (Shi-Tomasi)   (green dots)
       RIGHT  — cv2.goodFeaturesToTrack  after 5×5 Gaussian blur  (red dots)

WHY IS IT STILL SOMEWHAT SLOW?
  The video is 1920×1080 @ 60 fps (3012 frames).  Even the C-backed cv2
  calls process ~2 MP per frame.  The manual implementation adds:
    - Sobel (fast, C)  →  structure tensor (fast, vectorised numpy)
    - NMS:  now fully vectorised with numpy reshape+argmax, no Python loops
  Most of the remaining time is I/O (disk read + video encode) and the
  Python overhead of 3× goodFeaturesToTrack calls per frame.

THEORY — WHY "GOOD FEATURES"?
  Harris (1988) defined a corner as a point where R = det(M) - k*trace(M)^2
  is large.  Shi & Tomasi (1994) showed that tracking quality is better
  predicted by R = min(λ1, λ2):  a point is trackable iff both eigenvalues
  are large, i.e. intensity changes strongly in ALL directions.

  The structure tensor at pixel (x, y) is:
        ⎡ ΣIx²   ΣIxIy ⎤
    M = ⎢               ⎥   summed over a blockSize × blockSize window
        ⎣ ΣIxIy  ΣIy²  ⎦

  For a 2×2 symmetric matrix  [[a, b], [b, c]]:
    λ1,2 = ( (a+c) ± sqrt((a-c)² + 4b²) ) / 2
    min(λ1, λ2) = ( (a+c) - sqrt((a-c)² + 4b²) ) / 2

  Gaussian blur before detection smooths noise → fewer spurious corners,
  but also suppresses fine texture → fewer total corners detected.

OUTPUT FILES (saved next to this script):
  output/09_panel_manual.avi       — manual Shi-Tomasi channel
  output/09_panel_cv2.avi          — cv2 Shi-Tomasi channel
  output/09_panel_cv2_blur.avi     — cv2 Shi-Tomasi + Gaussian blur channel
  output/09_combined.avi           — all 3 panels side-by-side (3×width)

=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH  = os.path.join(SCRIPT_DIR, "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "output")

# ---------------------------------------------------------------------------
# PARAMETERS  (mirror cv2.goodFeaturesToTrack defaults where possible)
# ---------------------------------------------------------------------------
MAX_CORNERS    = 200     # max points to return
QUALITY_LEVEL  = 0.01   # fraction of best response to use as threshold
MIN_DISTANCE   = 10     # minimum pixel distance between returned corners
BLOCK_SIZE     = 3      # neighbourhood for structure tensor
USE_HARRIS     = False  # False → Shi-Tomasi, True → Harris
HARRIS_K       = 0.04   # Harris free parameter  (only used if USE_HARRIS)
BLUR_KSIZE     = 5      # Gaussian blur kernel size applied before the 3rd detector

# Display
SHOW_WINDOW    = True   # set False to run headless / faster
WAIT_MS        = 1      # ms per frame in imshow  (0 = wait for keypress)

# ---------------------------------------------------------------------------
# MANUAL IMPLEMENTATION
# ---------------------------------------------------------------------------

def compute_gradients(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial image gradients using Sobel kernels (same as OpenCV default).
    Returns Ix, Iy as float32 arrays.
    """
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return Ix, Iy


def compute_structure_tensor(
    Ix: np.ndarray,
    Iy: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the per-pixel structure tensor components A, B, C where:
        M = [[A, B],
             [B, C]]
        A = ΣIx²,  B = ΣIxIy,  C = ΣIy²
    The sum is taken over a block_size × block_size window via box filter.
    """
    ksize = (block_size, block_size)
    A = cv2.boxFilter(Ix * Ix, -1, ksize, normalize=False)
    B = cv2.boxFilter(Ix * Iy, -1, ksize, normalize=False)
    C = cv2.boxFilter(Iy * Iy, -1, ksize, normalize=False)
    return A, B, C


def shi_tomasi_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    """
    R = min(λ1, λ2)  for every pixel.

    For 2×2 symmetric [[A,B],[B,C]]:
        λ_min = ( (A+C) - sqrt((A-C)² + 4B²) ) / 2
    """
    disc    = np.sqrt((A - C) ** 2 + 4.0 * B ** 2)
    lam_min = ((A + C) - disc) * 0.5
    return lam_min.astype(np.float32)


def harris_response(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    k: float = 0.04,
) -> np.ndarray:
    """
    R = det(M) - k * trace(M)²
        det  = A*C - B²
        trace = A + C
    """
    det   = A * C - B * B
    trace = A + C
    return (det - k * trace * trace).astype(np.float32)


def non_max_suppression_numpy(
    response: np.ndarray,
    min_distance: int,
    max_corners: int,
) -> np.ndarray:
    """
    Fully vectorised NMS — no Python loops.

    Algorithm:
      1. Reshape the response map into a grid of (cell × cell) tiles.
      2. Use numpy argmax over the last two axes to find the best pixel
         in each tile simultaneously.
      3. Convert tile-local indices back to global (row, col).
      4. Filter out zero-valued cells and return top-N sorted by response.

    Returns array of shape (N, 2) with columns [col, row]  (x, y order).
    """
    h, w = response.shape
    cell = max(1, min_distance)

    # Pad so dimensions are multiples of cell
    pad_h = (cell - h % cell) % cell
    pad_w = (cell - w % cell) % cell
    R_pad = np.pad(response, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    ph, pw = R_pad.shape
    rows_tiles = ph // cell
    cols_tiles = pw // cell

    # Reshape to (rows_tiles, cell, cols_tiles, cell) then swap axes
    # → (rows_tiles, cols_tiles, cell, cell)
    tiles = R_pad.reshape(rows_tiles, cell, cols_tiles, cell) \
                 .transpose(0, 2, 1, 3)                       # (RT, CT, cell, cell)

    # Best value and flat index within each cell
    flat_vals = tiles.reshape(rows_tiles, cols_tiles, -1)
    best_flat = flat_vals.argmax(axis=-1)                     # (RT, CT)
    best_vals = flat_vals.max(axis=-1)                        # (RT, CT)

    # Convert flat cell index → (local_row, local_col)
    local_r = best_flat // cell
    local_c = best_flat  % cell

    # Tile origins in global coordinates
    tile_r = np.arange(rows_tiles)[:, None] * cell            # (RT, 1)
    tile_c = np.arange(cols_tiles)[None, :] * cell            # (1, CT)

    global_r = (tile_r + local_r).ravel()                     # (RT*CT,)
    global_c = (tile_c + local_c).ravel()
    vals     = best_vals.ravel()

    # Filter out border padding and zero-response cells
    valid = (vals > 0) & (global_r < h) & (global_c < w)
    global_r, global_c, vals = global_r[valid], global_c[valid], vals[valid]

    # Sort descending by response, keep top-N
    order    = np.argsort(vals)[::-1][:max_corners]
    global_r = global_r[order]
    global_c = global_c[order]

    # Return as (N, 2) in [x=col, y=row] order to match cv2
    return np.stack([global_c, global_r], axis=1).astype(np.float32)


def manual_good_features_to_track(
    gray: np.ndarray,
    max_corners: int     = MAX_CORNERS,
    quality_level: float = QUALITY_LEVEL,
    min_distance: int    = MIN_DISTANCE,
    block_size: int      = BLOCK_SIZE,
    use_harris: bool     = USE_HARRIS,
    harris_k: float      = HARRIS_K,
) -> Optional[np.ndarray]:
    """
    Manual re-implementation of cv2.goodFeaturesToTrack (Shi-Tomasi variant).

    Steps:
      1. Compute Ix, Iy (Sobel gradients)
      2. Build structure tensor M per pixel (summed over block_size window)
      3. Compute corner response R for every pixel
      4. Threshold: keep only pixels where R >= quality_level * max(R)
      5. Vectorised numpy NMS with min_distance grid
      6. Return top max_corners points as (N,1,2) float32 array

    Returns None if no corners found, else shape (N,1,2) matching OpenCV.
    """
    Ix, Iy  = compute_gradients(gray)
    A, B, C = compute_structure_tensor(Ix, Iy, block_size)

    R = harris_response(A, B, C, harris_k) if use_harris else shi_tomasi_response(A, B, C)

    R_max = float(R.max())
    if R_max <= 0:
        return None

    R[R < quality_level * R_max] = 0.0

    pts_xy = non_max_suppression_numpy(R, min_distance, max_corners)

    if pts_xy.shape[0] == 0:
        return None

    # Shape (N,1,2) to match cv2 output
    return pts_xy[:, np.newaxis, :]


# ---------------------------------------------------------------------------
# VISUALISATION HELPERS
# ---------------------------------------------------------------------------

def draw_points(
    frame: np.ndarray,
    points: Optional[np.ndarray],
    color: Tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:
    out = frame.copy()
    if points is None:
        return out
    for pt in points:
        x, y = int(pt[0][0]), int(pt[0][1])
        cv2.circle(out, (x, y), radius, color, -1)
        cv2.circle(out, (x, y), radius + 1, (0, 0, 0), 1)
    return out


def add_label(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color,   2)


def add_frame_info(img: np.ndarray, frame_idx: int, fps: float) -> None:
    txt = f"frame {frame_idx:04d}  |  {fps:.1f} fps"
    cv2.putText(img, txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, txt, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)


def print_frame_points(
    frame_idx: int,
    manual_pts: Optional[np.ndarray],
    cv_pts: Optional[np.ndarray],
    blur_pts: Optional[np.ndarray],
) -> None:
    n_m = 0 if manual_pts is None else len(manual_pts)
    n_c = 0 if cv_pts     is None else len(cv_pts)
    n_b = 0 if blur_pts   is None else len(blur_pts)
    print(f"\n--- Frame {frame_idx:04d} ---")
    for label, pts, n in [
        ("manual Shi-Tomasi ", manual_pts, n_m),
        ("cv2 Shi-Tomasi    ", cv_pts,     n_c),
        ("cv2 + Gaussian blur", blur_pts,   n_b),
    ]:
        print(f"  {label}: {n} corners")
        if pts is not None:
            for i, pt in enumerate(pts[:5]):
                print(f"    [{i:3d}]  x={pt[0][0]:7.2f}  y={pt[0][1]:7.2f}")
            if n > 5:
                print(f"    ... ({n - 5} more)")


# ---------------------------------------------------------------------------
# VIDEO WRITER HELPER
# ---------------------------------------------------------------------------

def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not writer.isOpened():
        print(f"[WARN] Could not open writer for {path}")
    return writer


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Individual channel writers
    w_manual   = make_writer(os.path.join(OUTPUT_DIR, "09_panel_manual.avi"),    src_fps, src_w, src_h)
    w_cv2      = make_writer(os.path.join(OUTPUT_DIR, "09_panel_cv2.avi"),       src_fps, src_w, src_h)
    w_blur     = make_writer(os.path.join(OUTPUT_DIR, "09_panel_cv2_blur.avi"),  src_fps, src_w, src_h)
    # Combined (3 panels side by side)
    w_combined = make_writer(os.path.join(OUTPUT_DIR, "09_combined.avi"),        src_fps, src_w * 3, src_h)

    print("=" * 72)
    print("  GOOD FEATURES TO TRACK  —  3 variants")
    print(f"  Source : {src_w}x{src_h} @ {src_fps:.0f} fps  ({total} frames)")
    print(f"  Output : {OUTPUT_DIR}/")
    print("  LEFT   blue  = Manual Shi-Tomasi")
    print("  CENTRE green = cv2.goodFeaturesToTrack (Shi-Tomasi)")
    print("  RIGHT  red   = cv2.goodFeaturesToTrack + Gaussian blur")
    print("  Press  Q  to quit early.")
    print("=" * 72)

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)

        # ---- 1. Manual Shi-Tomasi -------------------------------------------
        manual_pts = manual_good_features_to_track(
            gray,
            max_corners   = MAX_CORNERS,
            quality_level = QUALITY_LEVEL,
            min_distance  = MIN_DISTANCE,
            block_size    = BLOCK_SIZE,
            use_harris    = USE_HARRIS,
            harris_k      = HARRIS_K,
        )

        # ---- 2. cv2 Shi-Tomasi (no blur) ------------------------------------
        cv_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners        = MAX_CORNERS,
            qualityLevel      = QUALITY_LEVEL,
            minDistance       = MIN_DISTANCE,
            blockSize         = BLOCK_SIZE,
            useHarrisDetector = USE_HARRIS,
            k                 = HARRIS_K,
        )

        # ---- 3. cv2 Shi-Tomasi after Gaussian blur --------------------------
        blur_pts = cv2.goodFeaturesToTrack(
            gray_blur,
            maxCorners        = MAX_CORNERS,
            qualityLevel      = QUALITY_LEVEL,
            minDistance       = MIN_DISTANCE,
            blockSize         = BLOCK_SIZE,
            useHarrisDetector = USE_HARRIS,
            k                 = HARRIS_K,
        )

        # ---- Console output --------------------------------------------------
        print_frame_points(frame_idx, manual_pts, cv_pts, blur_pts)

        # ---- Build panel frames ---------------------------------------------
        elapsed = time.time() - t0
        cur_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        vis_manual = draw_points(frame, manual_pts, (255, 100,   0))  # blue
        vis_cv2    = draw_points(frame, cv_pts,     (  0, 220,   0))  # green
        vis_blur   = draw_points(frame, blur_pts,   (  0,   0, 220))  # red

        n_m = 0 if manual_pts is None else len(manual_pts)
        n_c = 0 if cv_pts     is None else len(cv_pts)
        n_b = 0 if blur_pts   is None else len(blur_pts)

        add_label(vis_manual, f"Manual Shi-Tomasi  ({n_m} pts)", (255, 100, 0))
        add_label(vis_cv2,    f"cv2 Shi-Tomasi     ({n_c} pts)", (  0, 200, 0))
        add_label(vis_blur,   f"cv2 + Gauss blur   ({n_b} pts)", (  0,   0, 200))

        for vis in (vis_manual, vis_cv2, vis_blur):
            add_frame_info(vis, frame_idx, cur_fps)

        combined = np.hstack([vis_manual, vis_cv2, vis_blur])

        # ---- Write to files -------------------------------------------------
        w_manual.write(vis_manual)
        w_cv2.write(vis_cv2)
        w_blur.write(vis_blur)
        w_combined.write(combined)

        # ---- Display --------------------------------------------------------
        if SHOW_WINDOW:
            # Scale down for display (3× wide is too big for most monitors)
            display = cv2.resize(combined, (src_w, src_h // 3))
            cv2.imshow("Good Features [manual | cv2 | cv2+blur]  —  Q to quit", display)
            if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
                print("\n[INFO] Quit requested.")
                break

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"[INFO] {frame_idx}/{total} frames  ({cur_fps:.1f} fps processing)")

    # ---- Cleanup ------------------------------------------------------------
    cap.release()
    for w in (w_manual, w_cv2, w_blur, w_combined):
        w.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n[INFO] Done. {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps average)")
    print(f"[INFO] Saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
