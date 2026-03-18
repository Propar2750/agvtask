"""
=============================================================================
CONTINUOUS BACKGROUND REMOVAL — 4 methods compared
=============================================================================

WHAT THIS FILE DOES:
  Runs 4 background subtraction methods simultaneously on OPTICAL_FLOW.mp4
  and displays / saves them side-by-side so you can see exactly how each
  model builds up its understanding of the "background" over time.

  ┌────────────────┬────────────────┬────────────────┬────────────────┐
  │  Frame Diff    │  Running Avg   │  MOG2 (cv2)    │  KNN  (cv2)    │
  │  (manual)      │  (manual)      │                │                │
  └────────────────┴────────────────┴────────────────┴────────────────┘

  Each panel shows the FOREGROUND MASK (white = moving / changed,
  black = background).  The original frame is shown in a 5th strip at
  the top to give context.

METHODS EXPLAINED:

  1. FRAME DIFFERENCING  (manual, simplest possible)
     ─────────────────────────────────────────────
     bg  = previous frame
     diff = |current_gray - previous_gray|
     mask = diff > threshold
     Pros : zero latency, no model state.
     Cons : detects BOTH edges of a moving object (ghost double-edge),
            cannot distinguish slow vs fast movement well,
            sensitive to any camera shake.

  2. RUNNING AVERAGE  (manual, adaptive background)
     ──────────────────────────────────────────────
     bg  = α * current + (1-α) * bg      (exponential moving average)
     mask = |current - bg| > threshold
     α controls how fast the background adapts.
       α=0.01  → slow to adapt (static camera, slow changes)
       α=0.1   → fast to adapt (moving camera, lighting changes)
     Pros : slowly absorbs static objects into background.
     Cons : ghost trails for fast objects, single Gaussian per pixel
            (bad for waving trees, water).

  3. MOG2 — Mixture of Gaussians v2  (Zivkovic 2004)
     ──────────────────────────────────────────────────
     Each pixel is modelled as a mixture of K Gaussians (default K=5).
     The algorithm decides which Gaussians are "background" based on
     weight × (1/variance):  a Gaussian is background if it is:
       - high weight  (appears often)
       - low variance  (consistent value)
     On every frame it updates the matching Gaussian's mean/variance.
     Non-matching pixels → new Gaussian or replace lowest-weight one.
     Also detects shadows (grey pixels in mask = shadow, white = foreground).
     Pros : handles multi-modal backgrounds (water, trees), shadow detection.
     Cons : needs ~100–500 frames warm-up, more compute.

  4. KNN — K-Nearest Neighbours  (Zivkovic & van der Heijden 2006)
     ──────────────────────────────────────────────────────────────
     Instead of Gaussians, stores the last N background samples per pixel.
     A new pixel value is "background" if at least K of its nearest stored
     samples are within a distance threshold (dist2Threshold).
     Pros : very robust to sudden illumination changes, no Gaussian
            assumption → works well for non-unimodal distributions.
     Cons : higher memory usage (stores raw samples), slightly slower.

OUTPUT FILES:
  output/10_frame_diff.avi
  output/10_running_avg.avi
  output/10_mog2.avi
  output/10_knn.avi
  output/10_combined.avi      ← all 4 masks + original, stacked

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

# --- Frame differencing ---
DIFF_THRESHOLD   = 25        # pixel diff value to count as foreground

# --- Running average ---
RUNNING_ALPHA    = 0.05      # blend rate: higher = adapts faster
RUNNING_THRESH   = 30        # abs difference threshold for foreground

# --- MOG2 ---
MOG2_HISTORY     = 500       # number of frames to model background from
MOG2_VAR_THRESH  = 16        # Mahalanobis distance² threshold (default 16)
MOG2_SHADOWS     = True      # detect and mark shadows

# --- KNN ---
KNN_HISTORY      = 500
KNN_DIST2THRESH  = 400.0     # squared distance threshold per pixel
KNN_SHADOWS      = True

# --- Post-processing applied to all masks ---
MORPH_OPEN_KSIZE  = 3        # removes speckle noise (erosion then dilation)
MORPH_CLOSE_KSIZE = 7        # fills small holes in foreground blobs

# --- Display / output ---
SHOW_WINDOW  = True
WAIT_MS      = 1
PANEL_W      = 480           # each panel scaled to this width for display

# ---------------------------------------------------------------------------
# MANUAL METHOD 1 — FRAME DIFFERENCING
# ---------------------------------------------------------------------------

class FrameDifferencer:
    """
    Background = the previous frame.
    Foreground = pixels that changed more than DIFF_THRESHOLD.
    """
    def __init__(self, threshold: int = DIFF_THRESHOLD):
        self.threshold  = threshold
        self.prev_gray: Optional[np.ndarray] = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return np.zeros_like(gray)

        diff = cv2.absdiff(gray, self.prev_gray)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        self.prev_gray = gray.copy()
        return mask


# ---------------------------------------------------------------------------
# MANUAL METHOD 2 — RUNNING AVERAGE
# ---------------------------------------------------------------------------

class RunningAverageSubtractor:
    """
    Background model = exponential moving average of past frames.
      bg_new = alpha * frame + (1-alpha) * bg_old
    Foreground = pixels where |frame - bg| > threshold.
    """
    def __init__(self, alpha: float = RUNNING_ALPHA, threshold: int = RUNNING_THRESH):
        self.alpha     = alpha
        self.threshold = threshold
        self.bg: Optional[np.ndarray] = None   # float32 background model

    def apply(self, gray: np.ndarray) -> np.ndarray:
        frame_f = gray.astype(np.float32)

        if self.bg is None:
            self.bg = frame_f.copy()
            return np.zeros_like(gray)

        # Update background model
        self.bg = self.alpha * frame_f + (1.0 - self.alpha) * self.bg

        # Foreground mask
        diff = np.abs(frame_f - self.bg).astype(np.uint8)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        return mask

    @property
    def background(self) -> np.ndarray:
        """Return current background model as uint8."""
        return np.clip(self.bg, 0, 255).astype(np.uint8) if self.bg is not None else None


# ---------------------------------------------------------------------------
# MORPHOLOGICAL POST-PROCESSING
# ---------------------------------------------------------------------------

_OPEN_KERNEL  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_KSIZE,  MORPH_OPEN_KSIZE))
_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Two-step morphological cleanup:
      1. Opening  (erode → dilate): removes isolated noise pixels
      2. Closing  (dilate → erode): fills small gaps inside foreground blobs
    """
    # MOG2/KNN return 3-value masks (0=bg, 127=shadow, 255=fg) — binarise first
    binary = np.where(mask == 255, 255, 0).astype(np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _OPEN_KERNEL)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return closed


# ---------------------------------------------------------------------------
# VISUALISATION HELPERS
# ---------------------------------------------------------------------------

def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert a greyscale mask to BGR for display / stacking."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def overlay_mask_on_frame(
    frame: np.ndarray,
    mask: np.ndarray,
    fg_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay foreground mask on the original frame.
    Foreground pixels are tinted with fg_color at opacity alpha.
    """
    out   = frame.copy()
    fg    = mask == 255
    tint  = np.zeros_like(frame)
    tint[fg] = fg_color
    out   = cv2.addWeighted(out, 1.0,    tint, alpha, 0)
    return out


def add_label(img: np.ndarray, text: str, color: Tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,   2)


def add_stats(img: np.ndarray, mask: np.ndarray, frame_idx: int, fps: float) -> None:
    fg_pct = 100.0 * np.count_nonzero(mask) / mask.size
    lines  = [
        f"frame {frame_idx:04d}  {fps:.1f}fps",
        f"fg: {fg_pct:.1f}%",
    ]
    for i, line in enumerate(lines):
        y = 52 + i * 22
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0),       2)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)


def scale_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = int(h * width / w)
    return cv2.resize(img, (width, new_h))


def make_writer(path: str, fps: float, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    wr     = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not wr.isOpened():
        print(f"[WARN] Could not open writer: {path}")
    return wr


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Initialise subtractors --------------------------------------------
    fd  = FrameDifferencer(threshold=DIFF_THRESHOLD)
    ra  = RunningAverageSubtractor(alpha=RUNNING_ALPHA, threshold=RUNNING_THRESH)
    mog = cv2.createBackgroundSubtractorMOG2(
              history      = MOG2_HISTORY,
              varThreshold = MOG2_VAR_THRESH,
              detectShadows = MOG2_SHADOWS)
    knn = cv2.createBackgroundSubtractorKNN(
              history       = KNN_HISTORY,
              dist2Threshold = KNN_DIST2THRESH,
              detectShadows  = KNN_SHADOWS)

    # ---- Video writers (full resolution) -----------------------------------
    w_diff  = make_writer(os.path.join(OUTPUT_DIR, "10_frame_diff.avi"),    src_fps, src_w, src_h)
    w_ra    = make_writer(os.path.join(OUTPUT_DIR, "10_running_avg.avi"),   src_fps, src_w, src_h)
    w_mog   = make_writer(os.path.join(OUTPUT_DIR, "10_mog2.avi"),          src_fps, src_w, src_h)
    w_knn   = make_writer(os.path.join(OUTPUT_DIR, "10_knn.avi"),           src_fps, src_w, src_h)
    # Combined: 4 panels side-by-side at PANEL_W each
    panel_h = int(src_h * PANEL_W / src_w)
    w_comb  = make_writer(os.path.join(OUTPUT_DIR, "10_combined.avi"),      src_fps, PANEL_W * 4, panel_h)

    print("=" * 72)
    print("  CONTINUOUS BACKGROUND REMOVAL — 4 methods")
    print(f"  Source  : {src_w}x{src_h} @ {src_fps:.0f}fps  ({total} frames)")
    print(f"  Output  : {OUTPUT_DIR}/")
    print("  Panel 1 : Frame Differencing      (manual)")
    print("  Panel 2 : Running Average model   (manual, alpha={})".format(RUNNING_ALPHA))
    print("  Panel 3 : MOG2 — Gaussian mixture (cv2)")
    print("  Panel 4 : KNN  — nearest neighbour(cv2)")
    print("  White = foreground   Black = background")
    print("  Press Q to quit.")
    print("=" * 72)

    frame_idx = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Run all 4 subtractors -----------------------------------------
        mask_diff = fd.apply(gray)
        mask_ra   = ra.apply(gray)
        mask_mog  = mog.apply(gray)          # returns 0/127/255
        mask_knn  = knn.apply(gray)          # returns 0/127/255

        # ---- Post-process: morphological cleanup ----------------------------
        mask_diff = clean_mask(mask_diff)
        mask_ra   = clean_mask(mask_ra)
        mask_mog  = clean_mask(mask_mog)
        mask_knn  = clean_mask(mask_knn)

        # ---- Build annotated BGR panels (full-res for file output) ----------
        elapsed = time.time() - t0
        cur_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        def make_panel(mask, label, color):
            vis = overlay_mask_on_frame(frame, mask, fg_color=color)
            add_label(vis, label, color)
            add_stats(vis, mask, frame_idx, cur_fps)
            return vis

        p_diff = make_panel(mask_diff, "Frame Diff (manual)",        (80,  200, 255))  # yellow
        p_ra   = make_panel(mask_ra,   f"Running Avg a={RUNNING_ALPHA}", (0,  255, 120))  # green
        p_mog  = make_panel(mask_mog,  "MOG2 (cv2)",                 (255, 120,   0))  # blue
        p_knn  = make_panel(mask_knn,  "KNN  (cv2)",                 (180,  60, 255))  # purple

        # ---- Write full-res panels to individual files ----------------------
        w_diff.write(p_diff)
        w_ra.write(p_ra)
        w_mog.write(p_mog)
        w_knn.write(p_knn)

        # ---- Build scaled combined panel (4 × PANEL_W wide) ----------------
        comb = np.hstack([
            scale_to_width(p_diff, PANEL_W),
            scale_to_width(p_ra,   PANEL_W),
            scale_to_width(p_mog,  PANEL_W),
            scale_to_width(p_knn,  PANEL_W),
        ])
        w_comb.write(comb)

        # ---- Console stats every 60 frames ----------------------------------
        if frame_idx % 60 == 0:
            fg_pcts = [
                100.0 * np.count_nonzero(m) / m.size
                for m in (mask_diff, mask_ra, mask_mog, mask_knn)
            ]
            print(
                f"[{frame_idx:04d}/{total}]  {cur_fps:.1f}fps  |  "
                f"diff={fg_pcts[0]:.1f}%  ra={fg_pcts[1]:.1f}%  "
                f"mog={fg_pcts[2]:.1f}%  knn={fg_pcts[3]:.1f}%  fg"
            )

        # ---- Display --------------------------------------------------------
        if SHOW_WINDOW:
            cv2.imshow("Background Removal  [diff | running-avg | MOG2 | KNN]  Q=quit", comb)
            if cv2.waitKey(WAIT_MS) & 0xFF == ord('q'):
                print("\n[INFO] Quit requested.")
                break

        frame_idx += 1

    # ---- Cleanup ------------------------------------------------------------
    cap.release()
    for w in (w_diff, w_ra, w_mog, w_knn, w_comb):
        w.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n[DONE] {frame_idx} frames in {elapsed:.1f}s  ({frame_idx/elapsed:.1f}fps avg)")
    print(f"[DONE] Files saved to: {OUTPUT_DIR}/")
    print()
    print("  Observations to look for:")
    print("  - Frame Diff : fast response but ghosting at object edges")
    print("  - Running Avg: smoother mask, slow objects get absorbed into bg")
    print("  - MOG2       : best overall after ~200 frame warm-up")
    print("  - KNN        : handles sudden illumination shifts well")


if __name__ == "__main__":
    main()
