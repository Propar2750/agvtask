"""
=============================================================================
COMPARISON RUNNER — Run all algorithms and compare results
=============================================================================

This script runs all optical flow algorithms on the same video and produces
a comparison summary. Run this to see how they differ in:
    - Speed (computation time per frame)
    - Quality (how well they track motion)
    - Coverage (sparse vs dense)
    - Robustness (how many points are lost)

ALGORITHM SUMMARY
-----------------

┌──────────────────────┬───────────┬────────────┬─────────────┬──────────────┐
│ Algorithm            │ Type      │ Speed      │ Best For    │ Key Param    │
├──────────────────────┼───────────┼────────────┼─────────────┼──────────────┤
│ Lucas-Kanade (OpenCV)│ Sparse    │ Very Fast  │ Feature     │ winSize,     │
│                      │           │            │ tracking    │ maxLevel     │
├──────────────────────┼───────────┼────────────┼─────────────┼──────────────┤
│ Farneback            │ Dense     │ Medium     │ Full motion │ winsize,     │
│                      │           │            │ maps        │ poly_n       │
├──────────────────────┼───────────┼────────────┼─────────────┼──────────────┤
│ LK From Scratch      │ Sparse    │ Slow       │ Learning!   │ window_size, │
│                      │           │            │             │ num_levels   │
├──────────────────────┼───────────┼────────────┼─────────────┼──────────────┤
│ Horn-Schunck         │ Dense     │ Very Slow  │ Smooth flow │ alpha,       │
│                      │           │            │ fields      │ iterations   │
└──────────────────────┴───────────┴────────────┴─────────────┴──────────────┘


FOR TASK 1 SPECIFICALLY:
-----------------------
You MUST implement Lucas-Kanade (Algorithm 1/3 above).
The pipeline for Subtask 2 requires:
    1. Camera feed → frames
    2. Lucas-Kanade sparse flow → motion vectors
    3. Focus of Expansion (FOE) → where is the agent heading
    4. Visual Potential Field → repulsive (obstacles) + attractive (target)
    5. Gradient control → steer the robot

Farneback/Horn-Schunck are good to KNOW (bonus: dense flow implementation)
but the core deliverable is Lucas-Kanade.

=============================================================================
"""

import subprocess
import sys
import os
import time


def run_script(script_name, description):
    """Run a script and capture its output."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"\n{'='*70}")
    print(f" RUNNING: {description}")
    print(f" Script:  {script_name}")
    print(f"{'='*70}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    elapsed = time.time() - start

    print(f"\n--- Completed in {elapsed:.1f}s (exit code: {result.returncode}) ---\n")
    return result.returncode, elapsed


def main():
    print()
    print("=" * 70)
    print(" OPTICAL FLOW ALGORITHM COMPARISON")
    print("=" * 70)
    print()
    print("This will run all 4 algorithms on OPTICAL_FLOW.mp4 and save")
    print("output videos in the 'output/' directory.")
    print()
    print("Algorithms to run:")
    print("  1. Lucas-Kanade Sparse (OpenCV)    — Fast, tracks corners")
    print("  2. Farneback Dense (OpenCV)         — Medium, per-pixel flow")
    print("  3. Lucas-Kanade From Scratch        — Slow, educational")
    print("  4. Horn-Schunck Dense (From Scratch) — Slowest, global method")
    print()

    scripts = [
        ("01_lucas_kanade_sparse.py",    "Algorithm 1: Lucas-Kanade Sparse (OpenCV)"),
        ("02_farneback_dense.py",        "Algorithm 2: Farneback Dense (OpenCV)"),
        ("03_lucas_kanade_from_scratch.py", "Algorithm 3: Lucas-Kanade From Scratch"),
        ("04_horn_schunck_dense.py",     "Algorithm 4: Horn-Schunck Dense (From Scratch)"),
    ]

    results = []
    for script, desc in scripts:
        code, elapsed = run_script(script, desc)
        results.append((desc, code, elapsed))

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    for desc, code, elapsed in results:
        status = "OK" if code == 0 else "FAILED"
        print(f"  [{status:6s}] {desc}: {elapsed:.1f}s")

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    print(f"\nOutput videos saved in: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Watch the output videos to compare results")
    print("  2. Experiment with parameters in each script")
    print("  3. Read the docstrings to understand the math")
    print("  4. Start implementing Task 1 Subtask 2 (PyBullet navigation)")


if __name__ == "__main__":
    main()
