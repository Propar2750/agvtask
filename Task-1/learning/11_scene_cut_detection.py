"""
=============================================================================
SCENE CUT DETECTION — 4 methods compared
=============================================================================

WHAT THIS FILE DOES:
  Detects hard cuts (shot boundaries) in OPTICAL_FLOW.mp4 using four
  independent methods, then reports every detected cut and produces outputs
  you can inspect to judge accuracy.

  No assumption is made about how many cuts exist — they are all auto-detected.

METHODS (increasing sophistication):

  1. PIXEL MAD — Mean Absolute Difference  (manual, simplest)
     ─────────────────────────────────────────────────────────
     signal[i] = mean( |gray[i] - gray[i-1]| )

     A hard cut replaces every pixel at once → enormous spike in MAD.
     Normal motion causes small, localised changes → low MAD.

     Threshold: median(signal) + STD_MULT * std(signal)
     This adaptive threshold handles videos with very different noise levels.

  2. BGR HISTOGRAM CORRELATION  (cv2, per-channel)
     ──────────────────────────────────────────────
     For each channel (B, G, R):
       hist = cv2.calcHist(frame, [c], None, [256], [0,256])
       score = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)
     signal[i] = 1 - mean(score_B, score_G, score_R)

     Correlation ranges [−1, 1]; identical frames → 1.0.
     Inverted so spikes = cuts (high signal = dissimilar frames).
     More robust than MAD to gradual illumination changes.

  3. HSV CHI-SQUARED  (cv2, colour-aware)
     ──────────────────────────────────────
     Convert to HSV.  Use only Hue + Saturation (ignore Value = brightness).
     Compute a 2D H×S histogram per frame, normalize it.
     Compare: cv2.compareHist(..., cv2.HISTCMP_CHISQR_ALT)

     Symmetric χ² is invariant to brightness → handles shadows, exposure
     changes.  Best at distinguishing scenes with different colour palettes.

  4. EDGE DENSITY CHANGE  (manual, content-aware)
     ──────────────────────────────────────────────
     edge_density[i] = nonzero_pixels(Canny(gray[i])) / total_pixels
     signal[i] = |edge_density[i] - edge_density[i-1]|

     Content structure (edges) changes abruptly at cuts even when the
     overall brightness/colour is similar.  Complements histogram methods.

POST-PROCESSING (all methods):
  Non-maximum suppression → keep only the highest peak within any window
  of NMS_WINDOW frames.  Then enforce MIN_CUT_GAP between retained cuts.

TWO-PASS DESIGN:
  Pass 1 — reads every frame once, computes all 4 signals, stores as arrays.
            No RAM spike: only two consecutive frames in memory at once.
  Pass 2 — re-reads video, writes annotated output and segment clips.

OUTPUT FILES (saved to output/):
  11_diff_signals.png  — 4-panel plot of signals with cut markers
  11_annotated.avi     — original video with red "CUT DETECTED" banner
  11_seg1.avi … N.avi  — video split at detected cut points
  11_cuts.txt          — all cuts from all methods (frame, time, score)

=============================================================================
"""

import cv2
import numpy as np
import os
import sys
import time
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------

# Adaptive threshold: flag frame i as a cut if
#   signal[i] > median(signal) + STD_MULT * std(signal)
STD_MULT = 5.0

# Non-maximum suppression half-window (frames).
# Only the single highest peak within a ±NMS_WINDOW band is kept.
NMS_WINDOW = 30

# Minimum gap between two accepted cuts (frames).
# 60 frames = 1 second at 60 fps.
MIN_CUT_GAP = 60

# HSV histogram bins
H_BINS = 32
S_BINS = 32

# Canny thresholds for edge detection
CANNY_LO = 50
CANNY_HI = 150

# Video writer codec
FOURCC = cv2.VideoWriter_fourcc(*"XVID")

# Which method's cuts to use for splitting the video into segments
# Options: "mad", "hist_corr", "hsv_chi2", "edge"
SPLIT_METHOD = "mad"

# ---------------------------------------------------------------------------
# PASS 1 — SIGNAL COMPUTATION
# ---------------------------------------------------------------------------

def compute_all_signals(
    video_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    Single-pass signal computation — reads each frame exactly once.

    Returns:
        sig_mad      : float32 array of length (n_frames - 1)
        sig_corr     : float32 array, 1 - BGR histogram correlation
        sig_chi2     : float32 array, HSV chi-squared distance
        sig_edge     : float32 array, edge density delta
        fps          : source video fps
        total_frames : total number of frames read
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}", file=sys.stderr)
        sys.exit(1)

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Pass 1] Analysing {total} frames  ({src_w}x{src_h} @ {fps:.0f}fps)")

    sig_mad  = np.zeros(max(total - 1, 0), dtype=np.float32)
    sig_corr = np.zeros(max(total - 1, 0), dtype=np.float32)
    sig_chi2 = np.zeros(max(total - 1, 0), dtype=np.float32)
    sig_edge = np.zeros(max(total - 1, 0), dtype=np.float32)

    prev_gray     = None
    prev_bgr_hist = None   # list of 3 channel histograms
    prev_hs_hist  = None   # 2D H-S histogram
    prev_edge_den = None

    frame_idx = 0
    t0        = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Build histograms for current frame ----------------------------
        bgr_hists = [
            cv2.calcHist([frame], [c], None, [256], [0, 256])
            for c in range(3)
        ]
        hs_hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [H_BINS, S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(hs_hist, hs_hist, alpha=1.0, norm_type=cv2.NORM_L1)

        edges     = cv2.Canny(gray, CANNY_LO, CANNY_HI)
        edge_den  = float(np.count_nonzero(edges)) / edges.size

        # --- Compute signals (only from frame 1 onward) -------------------
        if frame_idx > 0:
            i = frame_idx - 1   # signal index

            # 1. Pixel MAD
            sig_mad[i] = float(np.mean(
                np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))
            ))

            # 2. BGR Histogram Correlation (inverted)
            corrs = [
                cv2.compareHist(prev_bgr_hist[c], bgr_hists[c], cv2.HISTCMP_CORREL)
                for c in range(3)
            ]
            sig_corr[i] = 1.0 - float(np.mean(corrs))

            # 3. HSV Chi-Squared (symmetric alternative)
            sig_chi2[i] = float(
                cv2.compareHist(prev_hs_hist, hs_hist, cv2.HISTCMP_CHISQR_ALT)
            )

            # 4. Edge density change
            sig_edge[i] = abs(edge_den - prev_edge_den)

        # --- Store previous state -----------------------------------------
        prev_gray     = gray
        prev_bgr_hist = bgr_hists
        prev_hs_hist  = hs_hist
        prev_edge_den = edge_den

        frame_idx += 1
        if frame_idx % 300 == 0:
            elapsed = time.time() - t0
            print(f"  {frame_idx}/{total}  ({frame_idx/elapsed:.1f}fps)")

    cap.release()
    elapsed = time.time() - t0
    print(f"[Pass 1] Done. {frame_idx} frames in {elapsed:.1f}s  ({frame_idx/elapsed:.1f}fps)")
    return sig_mad, sig_corr, sig_chi2, sig_edge, fps, frame_idx


# ---------------------------------------------------------------------------
# CUT DETECTION FROM A SIGNAL
# ---------------------------------------------------------------------------

def detect_cuts(
    signal: np.ndarray,
    std_mult: float  = STD_MULT,
    nms_window: int  = NMS_WINDOW,
    min_gap: int     = MIN_CUT_GAP,
) -> List[Tuple[int, float]]:
    """
    Detect cut positions in a 1-D difference signal.

    Steps:
      1. Adaptive threshold = median + std_mult * std
      2. Flag all positions above threshold as candidates
      3. Non-maximum suppression: within each ±nms_window band keep the peak
      4. Enforce min_gap between accepted cuts

    Returns list of (frame_index, score) sorted by frame_index.
    Note: frame_index here refers to the *second* frame of the pair
          (i.e. signal[i] corresponds to the transition from frame i to i+1,
          and the cut is at frame i+1).
    """
    if signal.size == 0:
        return []

    med       = float(np.median(signal))
    std       = float(np.std(signal))
    threshold = med + std_mult * std

    # Candidates: positions where signal exceeds threshold
    cand_idx = np.where(signal > threshold)[0]
    if cand_idx.size == 0:
        return []

    # NMS: for each candidate keep it only if it is the local max
    # within [max(0, i-nms_window) : i+nms_window+1]
    kept = []
    for i in cand_idx:
        lo  = max(0, int(i) - nms_window)
        hi  = min(len(signal), int(i) + nms_window + 1)
        if signal[i] == signal[lo:hi].max():
            kept.append(int(i))

    # Enforce minimum gap between consecutive cuts
    cuts = []
    last_cut = -min_gap - 1
    for i in sorted(kept):
        if i - last_cut >= min_gap:
            cuts.append((i + 1, float(signal[i])))   # +1: cut at next frame
            last_cut = i

    return cuts


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_signals(
    signals: List[Tuple[str, np.ndarray, List[Tuple[int, float]]]],
    fps: float,
    out_path: str,
) -> None:
    """
    Draw a 4-panel figure: one row per method.
    Each panel shows the signal over time with vertical red lines at cuts.
    """
    n     = len(signals)
    times = None

    fig, axes = plt.subplots(n, 1, figsize=(18, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (label, signal, cuts) in zip(axes, signals):
        t = np.arange(len(signal)) / fps
        if times is None:
            times = t

        ax.plot(t, signal, linewidth=0.6, color="#4488ff", label=label)

        # Adaptive threshold line
        med       = float(np.median(signal))
        std       = float(np.std(signal))
        threshold = med + STD_MULT * std
        ax.axhline(threshold, color="orange", linewidth=1.0,
                   linestyle="--", label=f"threshold ({threshold:.2f})")

        # Cut markers
        for frame, score in cuts:
            t_cut = frame / fps
            ax.axvline(t_cut, color="red", linewidth=1.5, alpha=0.8)
            ax.text(t_cut, ax.get_ylim()[1] * 0.9,
                    f" {t_cut:.1f}s", color="red", fontsize=7, va="top")

        ax.set_title(f"{label}  —  {len(cuts)} cut(s) detected", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved: {out_path}")


# ---------------------------------------------------------------------------
# PASS 2 — ANNOTATE + SEGMENT WRITING
# ---------------------------------------------------------------------------

def write_annotated_and_segments(
    video_path: str,
    cut_frames: List[int],
    fps: float,
    src_w: int,
    src_h: int,
    out_dir: str,
) -> None:
    """
    Re-read the video and:
    1. Write a single annotated video with red banner at every cut frame.
    2. Write individual segment clips between consecutive cut points.
    """
    # Build segment boundaries: [0, cut1, cut2, ..., total]
    cap       = cv2.VideoCapture(video_path)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    boundaries = [0] + sorted(set(cut_frames)) + [total]

    # Open annotated writer
    ann_path = os.path.join(out_dir, "11_annotated.avi")
    w_ann    = cv2.VideoWriter(ann_path, FOURCC, fps, (src_w, src_h))

    # Open all segment writers upfront
    seg_writers = []
    for seg_idx in range(len(boundaries) - 1):
        path = os.path.join(out_dir, f"11_seg{seg_idx + 1}.avi")
        seg_writers.append(cv2.VideoWriter(path, FOURCC, fps, (src_w, src_h)))

    cut_set = set(cut_frames)
    frame_idx = 0

    print(f"[Pass 2] Writing annotated video + {len(seg_writers)} segment(s)...")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Determine which segment this frame belongs to
        seg_idx = 0
        for b_idx in range(len(boundaries) - 1):
            if boundaries[b_idx] <= frame_idx < boundaries[b_idx + 1]:
                seg_idx = b_idx
                break

        seg_writers[seg_idx].write(frame)

        # Annotate
        ann_frame = frame.copy()
        if frame_idx in cut_set:
            # Red banner
            cv2.rectangle(ann_frame, (0, 0), (src_w, 80), (0, 0, 200), -1)
            cv2.putText(
                ann_frame,
                f"CUT DETECTED  —  frame {frame_idx}  |  {frame_idx / fps:.2f}s",
                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3,
            )
        else:
            # Segment label
            cv2.putText(
                ann_frame,
                f"seg {seg_idx + 1}  |  frame {frame_idx}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3,
            )
            cv2.putText(
                ann_frame,
                f"seg {seg_idx + 1}  |  frame {frame_idx}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1,
            )

        w_ann.write(ann_frame)

        frame_idx += 1
        if frame_idx % 300 == 0:
            elapsed = time.time() - t0
            print(f"  {frame_idx}/{total}  ({frame_idx/elapsed:.1f}fps)")

    cap.release()
    w_ann.release()
    for w in seg_writers:
        w.release()

    elapsed = time.time() - t0
    print(f"[Pass 2] Done. {frame_idx} frames in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# CONSOLE REPORTING
# ---------------------------------------------------------------------------

def report_cuts(
    label: str,
    cuts: List[Tuple[int, float]],
    fps: float,
) -> None:
    print(f"\n  [{label}]  {len(cuts)} cut(s) detected:")
    if not cuts:
        print("    (none)")
        return
    for i, (frame, score) in enumerate(cuts, 1):
        print(f"    Cut #{i:2d}  frame {frame:5d}  time {frame/fps:7.2f}s  score {score:.4f}")


def save_cuts_txt(
    signals: List[Tuple[str, np.ndarray, List[Tuple[int, float]]]],
    fps: float,
    out_path: str,
) -> None:
    with open(out_path, "w") as f:
        f.write("method,cut_number,frame,time_s,score\n")
        for label, _, cuts in signals:
            for i, (frame, score) in enumerate(cuts, 1):
                f.write(f"{label},{i},{frame},{frame/fps:.4f},{score:.6f}\n")
    print(f"[Cuts] Saved: {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Pass 1: compute signals ----------------------------------------
    sig_mad, sig_corr, sig_chi2, sig_edge, fps, n_frames = compute_all_signals(VIDEO_PATH)

    src_cap = cv2.VideoCapture(VIDEO_PATH)
    src_w   = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_cap.release()

    # ---- Detect cuts from each signal -----------------------------------
    cuts_mad  = detect_cuts(sig_mad)
    cuts_corr = detect_cuts(sig_corr)
    cuts_chi2 = detect_cuts(sig_chi2)
    cuts_edge = detect_cuts(sig_edge)

    # ---- Console report -------------------------------------------------
    print("\n" + "=" * 68)
    print("  SCENE CUT DETECTION RESULTS")
    print("=" * 68)
    report_cuts("Pixel MAD",              cuts_mad,  fps)
    report_cuts("BGR Hist Correlation",   cuts_corr, fps)
    report_cuts("HSV Chi-Squared",        cuts_chi2, fps)
    report_cuts("Edge Density Change",    cuts_edge, fps)

    # ---- Save signal plot -----------------------------------------------
    signals_info = [
        ("Pixel MAD",            sig_mad,  cuts_mad),
        ("BGR Hist Correlation",  sig_corr, cuts_corr),
        ("HSV Chi-Squared",       sig_chi2, cuts_chi2),
        ("Edge Density Change",   sig_edge, cuts_edge),
    ]
    plot_signals(
        signals_info, fps,
        os.path.join(OUTPUT_DIR, "11_diff_signals.png"),
    )

    # ---- Save cuts.txt --------------------------------------------------
    save_cuts_txt(signals_info, fps, os.path.join(OUTPUT_DIR, "11_cuts.txt"))

    # ---- Pick cuts from the chosen method for video splitting -----------
    method_cuts = {
        "mad":       cuts_mad,
        "hist_corr": cuts_corr,
        "hsv_chi2":  cuts_chi2,
        "edge":      cuts_edge,
    }
    chosen_cuts = method_cuts.get(SPLIT_METHOD, cuts_mad)
    cut_frames  = [f for f, _ in chosen_cuts]

    print(f"\n[Segments] Using '{SPLIT_METHOD}' method cuts for video splitting.")
    print(f"           {len(cut_frames)} cut(s) -> {len(cut_frames)+1} segment(s)")

    # ---- Pass 2: annotate + segment ------------------------------------
    write_annotated_and_segments(VIDEO_PATH, cut_frames, fps, src_w, src_h, OUTPUT_DIR)

    print(f"\n[DONE] All outputs saved to: {OUTPUT_DIR}/")
    print("  11_diff_signals.png  — signal plots with cut markers")
    print("  11_annotated.avi     — annotated video")
    print("  11_cuts.txt          — all detected cuts (CSV)")
    for i in range(len(cut_frames) + 1):
        print(f"  11_seg{i+1}.avi")


if __name__ == "__main__":
    main()
