import sys, os, time, traceback
sys.path.insert(0, 'learning')
spec = __import__('learning.08_d4rt_deepmind', fromlist=['D4RT', 'SpatioTemporalQuery'])
D4RT = spec.D4RT

print("=== D4RT SANITY CHECK ===\n")
errors = []

# 1. Load a small portion of video
try:
    model = D4RT()
    model.load_video_from_path('OPTICAL_FLOW.mp4', max_frames=60, resize=(320, 180))
    print("[PASS] Video loading (60 frames, 320x180)")
except Exception as e:
    errors.append(f"Video loading: {e}")
    print(f"[FAIL] Video loading: {e}")
    sys.exit(1)

import numpy as np
import cv2
h, w = model.scene.image_size

# 2. track_point
try:
    result = model.track_point(160, 90, start_frame=0, end_frame=30)
    assert 'positions_2d' in result
    assert 'positions_3d' in result
    assert result['positions_2d'].shape == (30, 2)
    assert result['positions_3d'].shape == (30, 3)
    print("[PASS] track_point: shapes correct")
except Exception as e:
    errors.append(f"track_point: {e}")
    print(f"[FAIL] track_point: {e}")

# 3. track_points (batched)
try:
    pts = np.array([[50, 50], [100, 100], [200, 150]], dtype=np.float32)
    results = model.track_points(pts, start_frame=0, end_frame=30)
    assert len(results) == 3
    for r in results:
        assert r['positions_2d'].shape == (30, 2)
        assert r['positions_3d'].shape == (30, 3)
        assert r['confidences'].shape == (30,)
        assert r['occluded'].shape == (30,)
    print("[PASS] track_points: 3 points, shapes correct")
except Exception as e:
    errors.append(f"track_points: {e}")
    print(f"[FAIL] track_points: {e}")

# 4. Check 3D backprojection uses intrinsics (not pos*z)
try:
    pts = np.array([[160, 90]], dtype=np.float32)  # center pixel
    results = model.track_points(pts, 0, 2)
    p3d = results[0]['positions_3d'][0]
    # With intrinsics: x = (160-318.6)*z/517.3, y = (90-255.3)*z/516.5
    # x should be NEGATIVE (160 < cx=318.6), y should be NEGATIVE (90 < cy=255.3)
    z = p3d[2]
    if z > 0:
        expected_x = (160 - 318.6) * z / 517.3
        expected_y = (90 - 255.3) * z / 516.5
        assert abs(p3d[0] - expected_x) < 0.01, f"x={p3d[0]}, expected={expected_x}"
        assert abs(p3d[1] - expected_y) < 0.01, f"y={p3d[1]}, expected={expected_y}"
        assert p3d[0] < 0, f"x should be negative for u<cx, got {p3d[0]}"
        assert p3d[1] < 0, f"y should be negative for v<cy, got {p3d[1]}"
        print(f"[PASS] 3D backprojection uses intrinsics (x={p3d[0]:.2f}, y={p3d[1]:.2f}, z={z:.2f})")
    else:
        print(f"[WARN] z={z}, skipping 3D check")
except Exception as e:
    errors.append(f"3D backprojection: {e}")
    print(f"[FAIL] 3D backprojection: {e}")
    traceback.print_exc()

# 5. estimate_depth
try:
    depth = model.estimate_depth(10)
    assert depth.shape == (h, w), f"Expected ({h},{w}), got {depth.shape}"
    assert depth.dtype == np.float32
    assert depth.min() >= 0
    print(f"[PASS] estimate_depth: shape={depth.shape}, range=[{depth.min():.2f}, {depth.max():.2f}]")
except Exception as e:
    errors.append(f"estimate_depth: {e}")
    print(f"[FAIL] estimate_depth: {e}")

# 6. compute_flow
try:
    flow = model.compute_flow(0, 1)
    assert flow.shape == (h, w, 2), f"Expected ({h},{w},2), got {flow.shape}"
    flow_10 = model.compute_flow(0, 10)
    assert flow_10.shape == (h, w, 2)
    flow_same = model.compute_flow(5, 5)
    assert np.allclose(flow_same, 0), "Same-frame flow should be zero"
    print(f"[PASS] compute_flow: shapes correct, same-frame=zero")
except Exception as e:
    errors.append(f"compute_flow: {e}")
    print(f"[FAIL] compute_flow: {e}")

# 7. estimate_camera_pose
try:
    R, t = model.estimate_camera_pose(0, 30)
    assert R.shape == (3, 3), f"R shape {R.shape}"
    assert t.shape == (3, 1), f"t shape {t.shape}"
    det = np.linalg.det(R)
    assert abs(det - 1.0) < 0.01, f"R det={det}, should be ~1"
    print(f"[PASS] estimate_camera_pose: R det={det:.6f}")
except Exception as e:
    errors.append(f"estimate_camera_pose: {e}")
    print(f"[FAIL] estimate_camera_pose: {e}")

# 8. reconstruct_3d
try:
    pts3d, colors = model.reconstruct_3d(frame_idx=10, subsample=8)
    assert pts3d.shape[1] == 3
    assert colors.shape[1] == 3
    assert len(pts3d) == len(colors)
    assert len(pts3d) > 100
    print(f"[PASS] reconstruct_3d: {len(pts3d)} points")
except Exception as e:
    errors.append(f"reconstruct_3d: {e}")
    print(f"[FAIL] reconstruct_3d: {e}")

# 9. reconstruct_4d
try:
    recon = model.reconstruct_4d(grid_step=40, start_frame=0, end_frame=60,
                                  keyframe_interval=20, chunk_size=30)
    assert 'keyframe_clouds' in recon
    assert 'alive_per_frame' in recon
    assert len(recon['alive_per_frame']) == 60
    assert recon['n_chunks'] >= 2
    n_kf = len(recon['keyframe_clouds'])
    assert n_kf > 0
    print(f"[PASS] reconstruct_4d: {recon['n_chunks']} chunks, {n_kf} keyframes, {recon['n_points']} pts")
except Exception as e:
    errors.append(f"reconstruct_4d: {e}")
    print(f"[FAIL] reconstruct_4d: {e}")
    traceback.print_exc()

# 10. detect_segments
try:
    segs = model.detect_segments(n_segments=2)
    assert len(segs) == 2 or len(segs) == 1  # might be 1 if no cut found in 60 frames
    for s, e in segs:
        assert 0 <= s < e <= 60
    print(f"[PASS] detect_segments: {len(segs)} segments")
except Exception as e:
    errors.append(f"detect_segments: {e}")
    print(f"[FAIL] detect_segments: {e}")

# 11. track_with_redetection
try:
    tracks = model.track_with_redetection(n_initial=30, redetect_every=20,
                                           start_frame=0, end_frame=60)
    assert len(tracks) > 0
    for tr in tracks:
        assert 'positions_2d' in tr
        assert 'start_frame' in tr
    print(f"[PASS] track_with_redetection: {len(tracks)} tracklets")
except Exception as e:
    errors.append(f"track_with_redetection: {e}")
    print(f"[FAIL] track_with_redetection: {e}")

# 12. track_rolling_window (no video output)
try:
    rw = model.track_rolling_window(n_points=30, start_frame=0, end_frame=60,
                                     replace_every=10, trail_length=10,
                                     output_path=None, show=False)
    assert rw['n_frames'] == 60
    assert len(rw['per_frame']) == 60
    assert rw['alive_min'] > 0
    print(f"[PASS] track_rolling_window: born={rw['total_born']}, died={rw['total_died']}, avg={rw['alive_mean']:.0f}")
except Exception as e:
    errors.append(f"track_rolling_window: {e}")
    print(f"[FAIL] track_rolling_window: {e}")

# 13. visualize_tracking (to file, no display)
try:
    pts = np.array([[50, 50], [150, 100]], dtype=np.float32)
    model.visualize_tracking(pts, start_frame=0, end_frame=10,
                              output_path='d4rt_output/_test_vis.mp4', show=False)
    assert os.path.exists('d4rt_output/_test_vis.mp4')
    os.remove('d4rt_output/_test_vis.mp4')
    print("[PASS] visualize_tracking: video created")
except Exception as e:
    errors.append(f"visualize_tracking: {e}")
    print(f"[FAIL] visualize_tracking: {e}")

# 14. Check answer_single_query 3D backprojection
try:
    query = spec.SpatioTemporalQuery(u=160, v=90, t_source=5, t_target=5)
    result = model.decoder.answer_single_query(query, model.frames_gray, model.scene)
    # For same-frame query, pos_3d should use intrinsics
    # Currently it uses pos*z which is WRONG. Flag this.
    z = result.position_3d[2]
    x = result.position_3d[0]
    if z > 0 and abs(x - 160 * z) < 0.01:
        print(f"[BUG] answer_single_query: uses u*z={x:.2f} instead of (u-cx)*z/fx={(160-318.6)*z/517.3:.2f}")
        errors.append("answer_single_query: wrong 3D backprojection (u*z instead of (u-cx)*z/fx)")
    elif z > 0:
        print(f"[PASS] answer_single_query: 3D={result.position_3d}")
except Exception as e:
    errors.append(f"answer_single_query: {e}")
    print(f"[FAIL] answer_single_query: {e}")

# 15. Check unused grid_step parameter in compute_flow
import inspect
sig = inspect.signature(model.compute_flow)
src = inspect.getsource(model.compute_flow)
if 'grid_step' in sig.parameters and 'grid_step' not in src.split('def compute_flow')[1].split('return')[0].replace('grid_step: int = 8', ''):
    print("[WARN] compute_flow: 'grid_step' parameter is declared but unused")
else:
    print("[OK] compute_flow parameter check")

# 16. Check _track_cache is used
src_class = inspect.getsource(spec.D4RT)
cache_uses = src_class.count('_track_cache')
if cache_uses <= 2:  # only __init__ and load_video clear
    print(f"[WARN] _track_cache defined but never read (only {cache_uses} references)")

print(f"\n=== SUMMARY: {len(errors)} issue(s) found ===")
for i, e in enumerate(errors):
    print(f"  {i+1}. {e}")
if not errors:
    print("  All checks passed!")
