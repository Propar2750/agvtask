"""
=============================================================================
MODULAR LUCAS-KANADE DEEP DIVE
=============================================================================

Every stage of Lucas-Kanade is broken into a swappable component.
Each component prints debug output so you can see exactly what's happening
at every step. Swap any component to experiment with different approaches.

COMPONENTS (each is a class you can swap):
  1. FeatureDetector    -- which points to track
  2. GradientComputer   -- spatial gradients Ix, Iy
  3. TemporalGradient   -- temporal gradient It
  4. FlowSolver         -- solve the 2x2 linear system for (u,v)
  5. PointValidator     -- is this point still good to track?
  6. PyramidBuilder     -- multi-scale image pyramid
  7. RedetectionPolicy  -- when/how to replenish lost points

HOW TO EXPERIMENT:
  Scroll to the bottom (CONFIGURATION SECTION) and swap any component.
  For example, change `GradientSobel()` to `GradientScharr()` to see
  how Scharr kernels affect tracking quality.

=============================================================================
"""

import cv2
import numpy as np
import time
import os
from abc import ABC, abstractmethod

VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "OPTICAL_FLOW.mp4")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FRAMES = 400
START_FRAME = 1800


# =============================================================================
# DEBUG INFRASTRUCTURE
# =============================================================================

class StageDebugger:
    """
    Collects and prints debug info for each pipeline stage.
    Every component gets one of these so you can see what it's doing.
    """
    def __init__(self, stage_name, enabled=True, verbose=False):
        self.stage_name = stage_name
        self.enabled = enabled
        self.verbose = verbose  # True = print every frame, False = periodic
        self._frame_count = 0
        self._stats = {}

    def log(self, msg, level="INFO"):
        if self.enabled:
            print(f"  [{self.stage_name:20s}] [{level:5s}] {msg}")

    def log_periodic(self, msg, period=15, level="INFO"):
        """Only print every `period` frames to avoid flooding."""
        if self.enabled and (self.verbose or self._frame_count % period == 0):
            self.log(msg, level)

    def set_frame(self, n):
        self._frame_count = n

    def record_stat(self, key, value):
        """Record a stat for the current frame (accessible for visualization)."""
        self._stats[key] = value

    def get_stat(self, key, default=None):
        return self._stats.get(key, default)

    def get_all_stats(self):
        return dict(self._stats)


# =============================================================================
# COMPONENT 1: FEATURE DETECTOR
# =============================================================================
# Decides WHICH points in the image are good to track.
# Good points = corners (strong gradients in 2 directions).
# Bad points = flat regions or edges.

class FeatureDetector(ABC):
    """Base class for feature detectors. Swap to change what points get tracked."""
    def __init__(self):
        self.debug = StageDebugger("FeatureDetector")

    @abstractmethod
    def detect(self, gray_image, max_points=100):
        """
        Detect trackable feature points.

        Args:
            gray_image: Grayscale uint8 image
            max_points: Maximum number of points to return

        Returns:
            Nx2 numpy array of (x, y) coordinates
        """
        pass


class ShiTomasiDetector(FeatureDetector):
    """
    Shi-Tomasi corner detection (cv2.goodFeaturesToTrack).

    HOW IT WORKS:
    - Computes the Harris matrix M = [[Ix^2, Ix*Iy], [Ix*Iy, Iy^2]] at each pixel
    - Finds eigenvalues lambda1, lambda2 of M
    - Shi-Tomasi criterion: quality = min(lambda1, lambda2)
    - A point is a corner if min(lambda1, lambda2) > threshold
    - Unlike Harris which uses det(M) - k*trace(M)^2, Shi-Tomasi directly
      uses the smaller eigenvalue, which is more stable

    PARAMETERS YOU CAN TUNE:
    - quality_level: Reject corners with quality < quality_level * best_quality
      Lower = more points but noisier. Higher = fewer but stronger corners.
    - min_distance: Minimum pixels between detected corners (suppression radius)
    - block_size: Window size for computing the Harris matrix
    """
    def __init__(self, quality_level=0.05, min_distance=15, block_size=7):
        super().__init__()
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.debug = StageDebugger("ShiTomasi")

    def detect(self, gray_image, max_points=100):
        corners = cv2.goodFeaturesToTrack(
            gray_image,
            maxCorners=max_points,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )

        if corners is None:
            self.debug.log("No corners found!", "WARN")
            return np.empty((0, 2), dtype=np.float32)

        points = corners.reshape(-1, 2)

        # Debug: analyze the quality of detected points
        # Compute eigenvalues at each detected point to show quality distribution
        eigenvalues = []
        h, w = gray_image.shape
        half = self.block_size // 2
        for (x, y) in points.astype(int):
            if y-half < 0 or y+half+1 > h or x-half < 0 or x+half+1 > w:
                continue
            patch = gray_image[y-half:y+half+1, x-half:x+half+1].astype(np.float64)
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue  # Too small for Sobel
            Ix = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            Iy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            M = np.array([
                [np.sum(Ix*Ix), np.sum(Ix*Iy)],
                [np.sum(Ix*Iy), np.sum(Iy*Iy)]
            ])
            eigs = np.linalg.eigvalsh(M)
            eigenvalues.append(min(eigs))

        if eigenvalues:
            eig_arr = np.array(eigenvalues)
            self.debug.log(
                f"Detected {len(points)} corners | "
                f"Eigenvalue range: [{eig_arr.min():.1f}, {eig_arr.max():.1f}] | "
                f"Median: {np.median(eig_arr):.1f}"
            )
            self.debug.record_stat("num_detected", len(points))
            self.debug.record_stat("eigenvalue_median", np.median(eig_arr))
            self.debug.record_stat("eigenvalue_min", eig_arr.min())

        return points


class HarrisDetector(FeatureDetector):
    """
    Harris corner detection -- alternative to Shi-Tomasi.

    HOW IT DIFFERS:
    - Harris score: R = det(M) - k * trace(M)^2
      where det = lambda1*lambda2, trace = lambda1+lambda2
    - k is typically 0.04-0.06
    - Corner when R > threshold (both eigenvalues large)
    - Edge when R < 0 (one eigenvalue much larger)
    - Flat when |R| is small (both eigenvalues small)

    TRY THIS: Compare with ShiTomasi -- Harris sometimes picks different corners,
    especially near edges. Some corners Harris likes, ShiTomasi doesn't, and vice versa.
    """
    def __init__(self, k=0.04, threshold_ratio=0.01, min_distance=15, block_size=7):
        super().__init__()
        self.k = k
        self.threshold_ratio = threshold_ratio
        self.min_distance = min_distance
        self.block_size = block_size
        self.debug = StageDebugger("Harris")

    def detect(self, gray_image, max_points=100):
        harris_response = cv2.cornerHarris(
            gray_image.astype(np.float32),
            blockSize=self.block_size,
            ksize=3,
            k=self.k
        )

        # Threshold: keep points with response > ratio * max_response
        max_response = harris_response.max()
        if max_response <= 0:
            self.debug.log("No positive Harris responses!", "WARN")
            return np.empty((0, 2), dtype=np.float32)
        threshold = self.threshold_ratio * max_response
        corner_mask = harris_response > threshold

        # Non-maximum suppression via dilation
        # cv2.dilate requires float32
        harris_f32 = harris_response.astype(np.float32)
        dilated = cv2.dilate(harris_f32, None)
        local_max = (harris_f32 == dilated) & corner_mask

        ys, xs = np.where(local_max)

        if len(xs) == 0:
            self.debug.log("No Harris corners found!", "WARN")
            return np.empty((0, 2), dtype=np.float32)

        # Sort by response strength and take top max_points
        responses = harris_response[ys, xs]
        order = np.argsort(-responses)[:max_points]
        points = np.column_stack([xs[order], ys[order]]).astype(np.float32)

        self.debug.log(
            f"Detected {len(points)} Harris corners | "
            f"Response range: [{responses[order].min():.2e}, {responses[order].max():.2e}] | "
            f"k={self.k}"
        )
        self.debug.record_stat("num_detected", len(points))
        self.debug.record_stat("harris_max_response", float(responses[order].max()))

        return points


class FASTDetector(FeatureDetector):
    """
    FAST (Features from Accelerated Segment Test) corner detection.

    HOW IT WORKS:
    - Looks at a circle of 16 pixels around each candidate point
    - A point is a corner if N contiguous pixels on the circle are all
      brighter (or all darker) than the center pixel by some threshold
    - Much faster than Harris/Shi-Tomasi but less discriminative
    - No eigenvalue analysis -- purely intensity-based

    TRY THIS: FAST finds MANY more points (especially on textured surfaces)
    but some may be poor for tracking. Compare tracking survival rates.
    """
    def __init__(self, threshold=20, non_max_suppression=True):
        super().__init__()
        self.threshold = threshold
        self.non_max_suppression = non_max_suppression
        self.debug = StageDebugger("FAST")

    def detect(self, gray_image, max_points=100):
        fast = cv2.FastFeatureDetector_create(
            threshold=self.threshold,
            nonmaxSuppression=self.non_max_suppression
        )
        keypoints = fast.detect(gray_image, None)

        if not keypoints:
            self.debug.log("No FAST keypoints found!", "WARN")
            return np.empty((0, 2), dtype=np.float32)

        # Sort by response and take top
        keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
        keypoints = keypoints[:max_points]

        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        responses = [kp.response for kp in keypoints]
        self.debug.log(
            f"Detected {len(points)} FAST keypoints | "
            f"Response range: [{min(responses):.1f}, {max(responses):.1f}] | "
            f"Threshold: {self.threshold}"
        )
        self.debug.record_stat("num_detected", len(points))

        return points


# =============================================================================
# COMPONENT 2: GRADIENT COMPUTER (Spatial: Ix, Iy)
# =============================================================================
# Computes how the image intensity changes in x and y directions.
# This is the dI/dx and dI/dy in the optical flow equation.

class GradientComputer(ABC):
    """Base class for spatial gradient computation."""
    def __init__(self):
        self.debug = StageDebugger("GradientComputer")

    @abstractmethod
    def compute(self, gray_image):
        """
        Compute spatial gradients.

        Args:
            gray_image: Grayscale float64 image

        Returns:
            (Ix, Iy): Tuple of gradient images, same shape as input
        """
        pass


class GradientSobel(GradientComputer):
    """
    Sobel operator for spatial gradients.

    KERNEL (3x3):
        Sobel-x:          Sobel-y:
        [-1  0  1]        [-1 -2 -1]
        [-2  0  2]        [ 0  0  0]
        [-1  0  1]        [ 1  2  1]

    WHY SOBEL:
    - Combines smoothing (Gaussian weights in perpendicular direction)
      with differentiation. The [-1, 0, 1] part differentiates, the
      [1, 2, 1] part smooths.
    - ksize=3 is the smallest; ksize=5 gives wider smoothing
    - ksize=1 gives [-1, 0, 1] without smoothing (noisier)

    TRY: Change ksize to 5 or 7 -- more smoothing means less noise but
    also less precise localization of gradients.
    """
    def __init__(self, ksize=3):
        super().__init__()
        self.ksize = ksize
        self.debug = StageDebugger("Sobel")

    def compute(self, gray_image):
        Ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=self.ksize)
        Iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=self.ksize)

        self.debug.log_periodic(
            f"Gradient stats - Ix: [{Ix.min():.1f}, {Ix.max():.1f}], "
            f"Iy: [{Iy.min():.1f}, {Iy.max():.1f}] | "
            f"Mean |Ix|: {np.mean(np.abs(Ix)):.2f}, Mean |Iy|: {np.mean(np.abs(Iy)):.2f}"
        )
        self.debug.record_stat("Ix_range", (Ix.min(), Ix.max()))
        self.debug.record_stat("Iy_range", (Iy.min(), Iy.max()))
        self.debug.record_stat("Ix_mean_abs", np.mean(np.abs(Ix)))
        self.debug.record_stat("Iy_mean_abs", np.mean(np.abs(Iy)))

        return Ix, Iy


class GradientScharr(GradientComputer):
    """
    Scharr operator -- more rotationally accurate than Sobel.

    KERNEL (3x3):
        Scharr-x:         Scharr-y:
        [-3   0   3]      [-3  -10  -3]
        [-10  0  10]      [ 0    0   0]
        [-3   0   3]      [ 3   10   3]

    WHY SCHARR:
    - Better rotational symmetry than Sobel
    - The weights [3, 10, 3] approximate the ideal gradient better
      than Sobel's [1, 2, 1]
    - Particularly noticeable for diagonal edges

    TRY: Compare Scharr vs Sobel -- look at the gradient magnitudes and
    how they affect flow accuracy, especially for diagonal motion.
    """
    def __init__(self):
        super().__init__()
        self.debug = StageDebugger("Scharr")

    def compute(self, gray_image):
        Ix = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
        Iy = cv2.Scharr(gray_image, cv2.CV_64F, 0, 1)

        self.debug.log_periodic(
            f"Gradient stats - Ix: [{Ix.min():.1f}, {Ix.max():.1f}], "
            f"Iy: [{Iy.min():.1f}, {Iy.max():.1f}] | "
            f"Mean |Ix|: {np.mean(np.abs(Ix)):.2f}, Mean |Iy|: {np.mean(np.abs(Iy)):.2f}"
        )
        self.debug.record_stat("Ix_range", (Ix.min(), Ix.max()))
        self.debug.record_stat("Iy_range", (Iy.min(), Iy.max()))

        return Ix, Iy


class GradientCentralDiff(GradientComputer):
    """
    Simple central difference for gradients -- the most basic approach.

    KERNEL:
        x: [-1, 0, 1] (no smoothing!)
        y: [-1, 0, 1]^T

    WHY TRY THIS:
    - Simplest possible gradient -- helps you see how much the Sobel/Scharr
      smoothing actually matters
    - More noise-sensitive but more precise localization
    - Good baseline to compare against

    WARNING: Noisy images will produce noisy gradients with this method.
    """
    def __init__(self):
        super().__init__()
        self.debug = StageDebugger("CentralDiff")

    def compute(self, gray_image):
        # Manual central difference: I(x+1) - I(x-1)
        Ix = np.zeros_like(gray_image, dtype=np.float64)
        Iy = np.zeros_like(gray_image, dtype=np.float64)
        Ix[:, 1:-1] = (gray_image[:, 2:] - gray_image[:, :-2]) / 2.0
        Iy[1:-1, :] = (gray_image[2:, :] - gray_image[:-2, :]) / 2.0

        self.debug.log_periodic(
            f"Gradient stats -- Ix: [{Ix.min():.1f}, {Ix.max():.1f}], "
            f"Iy: [{Iy.min():.1f}, {Iy.max():.1f}] | "
            f"Mean |Ix|: {np.mean(np.abs(Ix)):.2f}, Mean |Iy|: {np.mean(np.abs(Iy)):.2f}"
        )
        self.debug.record_stat("Ix_range", (Ix.min(), Ix.max()))
        self.debug.record_stat("Iy_range", (Iy.min(), Iy.max()))

        return Ix, Iy


# =============================================================================
# COMPONENT 3: TEMPORAL GRADIENT (It)
# =============================================================================
# Computes how the image intensity changes between frames.
# This is dI/dt -- the time derivative.

class TemporalGradient(ABC):
    """Base class for temporal gradient computation."""
    def __init__(self):
        self.debug = StageDebugger("TemporalGradient")

    @abstractmethod
    def compute(self, img1, img2):
        """
        Compute temporal gradient between two frames.

        Args:
            img1, img2: Consecutive grayscale frames (float64)

        Returns:
            It: Temporal gradient image
        """
        pass


class TemporalSimpleDiff(TemporalGradient):
    """
    Simplest temporal gradient: It = I(t+1) - I(t).

    This is just pixel-wise subtraction. Fast and straightforward.

    ASSUMPTION: Frames are close enough in time that the linear
    approximation of brightness change is valid.
    """
    def __init__(self):
        super().__init__()
        self.debug = StageDebugger("SimpleDiff")

    def compute(self, img1, img2):
        It = img2.astype(np.float64) - img1.astype(np.float64)

        self.debug.log_periodic(
            f"It range: [{It.min():.1f}, {It.max():.1f}] | "
            f"Mean |It|: {np.mean(np.abs(It)):.2f} | "
            f"Pixels with |It|>30: {np.sum(np.abs(It) > 30)} "
            f"({100*np.mean(np.abs(It) > 30):.1f}%)"
        )
        self.debug.record_stat("It_mean_abs", np.mean(np.abs(It)))
        self.debug.record_stat("It_range", (It.min(), It.max()))
        self.debug.record_stat("motion_pixel_pct", 100*np.mean(np.abs(It) > 30))

        return It


class TemporalAveraged(TemporalGradient):
    """
    Averaged temporal gradient -- smooths both spatially before differencing.

    It = GaussianBlur(I2) - GaussianBlur(I1)

    WHY: Reduces noise in the temporal gradient. The flow equation
    involves products of Ix, Iy, It -- if any of these are noisy,
    the noise gets amplified in the ATA matrix.

    TRY: Compare with SimpleDiff -- averaged version produces smoother
    flow at the cost of slightly less precise localization.
    """
    def __init__(self, kernel_size=3, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.debug = StageDebugger("AvgTemporal")

    def compute(self, img1, img2):
        blur1 = cv2.GaussianBlur(img1.astype(np.float64),
                                  (self.kernel_size, self.kernel_size), self.sigma)
        blur2 = cv2.GaussianBlur(img2.astype(np.float64),
                                  (self.kernel_size, self.kernel_size), self.sigma)
        It = blur2 - blur1

        self.debug.log_periodic(
            f"It range: [{It.min():.1f}, {It.max():.1f}] | "
            f"Mean |It|: {np.mean(np.abs(It)):.2f} | "
            f"Smoothing: kernel={self.kernel_size}, sigma={self.sigma}"
        )
        self.debug.record_stat("It_mean_abs", np.mean(np.abs(It)))

        return It


class TemporalCrossFrame(TemporalGradient):
    """
    Cross-frame temporal gradient using average of both frames' spatial info.

    It = 0.25 * ( I2[y,x] + I2[y,x+1] + I2[y+1,x] + I2[y+1,x+1]
                 - I1[y,x] - I1[y,x+1] - I1[y+1,x] - I1[y+1,x+1] )

    WHY: This is the classic Horn-Schunck style temporal derivative that
    averages over a 2x2 spatial neighborhood in both frames. Reduces noise
    by using more samples per estimate.

    TRY: This pairs well with the CentralDiff gradient computer since
    both operate at the same spatial scale.
    """
    def __init__(self):
        super().__init__()
        self.debug = StageDebugger("CrossFrame")

    def compute(self, img1, img2):
        f1 = img1.astype(np.float64)
        f2 = img2.astype(np.float64)
        It = np.zeros_like(f1)
        # Average over 2x2 block in both frames
        It[:-1, :-1] = 0.25 * (
            f2[:-1, :-1] + f2[:-1, 1:] + f2[1:, :-1] + f2[1:, 1:]
            - f1[:-1, :-1] - f1[:-1, 1:] - f1[1:, :-1] - f1[1:, 1:]
        )

        self.debug.log_periodic(
            f"It range: [{It.min():.1f}, {It.max():.1f}] | "
            f"Mean |It|: {np.mean(np.abs(It)):.2f} | "
            f"(2x2 spatial averaging)"
        )
        self.debug.record_stat("It_mean_abs", np.mean(np.abs(It)))

        return It


# =============================================================================
# COMPONENT 4: FLOW SOLVER
# =============================================================================
# Given Ix, Iy, It at a point's window, solve for the flow vector (u, v).
# This is the core math of Lucas-Kanade.

class FlowSolver(ABC):
    """Base class for solving the optical flow at a single point."""
    def __init__(self):
        self.debug = StageDebugger("FlowSolver")

    @abstractmethod
    def solve(self, Ix, Iy, It, x, y, window_size=15):
        """
        Solve for flow (u, v) at point (x, y).

        Args:
            Ix, Iy, It: Full gradient images
            x, y: Point coordinates (integers)
            window_size: Local window size

        Returns:
            (u, v): Flow vector
            debug_info: Dict with solver internals (eigenvalues, condition, etc.)
        """
        pass


class FlowSolverLeastSquares(FlowSolver):
    """
    Standard least-squares solver: d = (A^T*A)^(-1) * A^T*b

    This is the classic Lucas-Kanade formulation. All pixels in the
    window contribute equally to the solution.

    DEBUG OUTPUT:
    - Eigenvalues of A^T*A (tells you corner quality)
    - Condition number (ratio of eigenvalues -- high = unreliable)
    - The actual (u, v) solution
    """
    def __init__(self, eigen_threshold=1e-4):
        super().__init__()
        self.eigen_threshold = eigen_threshold
        self.debug = StageDebugger("LeastSquares")

    def solve(self, Ix, Iy, It, x, y, window_size=15):
        half_w = window_size // 2
        h, w = Ix.shape

        debug_info = {"method": "least_squares", "x": x, "y": y}

        # Boundary check
        if (y - half_w < 0 or y + half_w >= h or
            x - half_w < 0 or x + half_w >= w):
            debug_info["status"] = "out_of_bounds"
            return (0, 0), debug_info

        # Extract local windows
        Ix_w = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        Iy_w = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
        It_w = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

        # Build A^T*A (the structure tensor / Harris matrix)
        sum_IxIx = np.sum(Ix_w * Ix_w)
        sum_IxIy = np.sum(Ix_w * Iy_w)
        sum_IyIy = np.sum(Iy_w * Iy_w)

        ATA = np.array([[sum_IxIx, sum_IxIy],
                        [sum_IxIy, sum_IyIy]])

        ATb = np.array([[-np.sum(Ix_w * It_w)],
                        [-np.sum(Iy_w * It_w)]])

        # Eigenvalue analysis
        eigenvalues = np.linalg.eigvalsh(ATA)
        min_eigen = min(eigenvalues)
        max_eigen = max(eigenvalues)
        condition = max_eigen / min_eigen if min_eigen > 1e-10 else float('inf')

        debug_info["eigenvalues"] = (float(min_eigen), float(max_eigen))
        debug_info["condition_number"] = condition
        debug_info["ATA"] = ATA.tolist()

        if min_eigen < self.eigen_threshold:
            debug_info["status"] = "ill_conditioned"
            debug_info["reason"] = (
                "flat_region" if max_eigen < self.eigen_threshold
                else "edge_aperture_problem"
            )
            return (0, 0), debug_info

        # Solve
        try:
            d = np.linalg.solve(ATA, ATb)
            u, v = float(d[0, 0]), float(d[1, 0])
            debug_info["status"] = "solved"
            debug_info["flow"] = (u, v)
            debug_info["flow_magnitude"] = np.sqrt(u**2 + v**2)
        except np.linalg.LinAlgError:
            debug_info["status"] = "singular_matrix"
            return (0, 0), debug_info

        return (u, v), debug_info


class FlowSolverWeighted(FlowSolver):
    """
    Weighted least-squares -- pixels closer to center contribute more.

    Instead of d = (A^T*A)^(-1) * A^T*b, we solve:
        d = (A^T*W*A)^(-1) * A^T*W*b

    where W is a diagonal matrix of Gaussian weights.

    WHY WEIGHTING:
    - Pixels near the center of the window are more likely to share
      the same motion as the center point
    - Reduces the effect of pixels at the window edge that might have
      different motion (violating the spatial coherence assumption)

    TRY: Compare with unweighted -- weighted usually gives smoother
    flow but might miss large motions at the edges of objects.
    """
    def __init__(self, eigen_threshold=1e-4, sigma_factor=0.3):
        super().__init__()
        self.eigen_threshold = eigen_threshold
        self.sigma_factor = sigma_factor  # sigma = sigma_factor * window_size
        self.debug = StageDebugger("WeightedLS")

    def solve(self, Ix, Iy, It, x, y, window_size=15):
        half_w = window_size // 2
        h, w = Ix.shape

        debug_info = {"method": "weighted_least_squares", "x": x, "y": y}

        if (y - half_w < 0 or y + half_w >= h or
            x - half_w < 0 or x + half_w >= w):
            debug_info["status"] = "out_of_bounds"
            return (0, 0), debug_info

        Ix_w = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
        Iy_w = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
        It_w = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1]

        # Build Gaussian weight matrix
        sigma = self.sigma_factor * window_size
        yy, xx = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
        weights = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        weights /= weights.sum()  # Normalize

        debug_info["weight_center"] = float(weights[half_w, half_w])
        debug_info["weight_corner"] = float(weights[0, 0])
        debug_info["weight_ratio"] = float(weights[half_w, half_w] / weights[0, 0])

        # Weighted sums
        wIx = weights * Ix_w
        wIy = weights * Iy_w
        wIt = weights * It_w

        ATA = np.array([
            [np.sum(wIx * Ix_w), np.sum(wIx * Iy_w)],
            [np.sum(wIy * Ix_w), np.sum(wIy * Iy_w)]
        ])
        ATb = np.array([[-np.sum(wIx * It_w)],
                        [-np.sum(wIy * It_w)]])

        eigenvalues = np.linalg.eigvalsh(ATA)
        min_eigen = min(eigenvalues)
        max_eigen = max(eigenvalues)

        debug_info["eigenvalues"] = (float(min_eigen), float(max_eigen))

        if min_eigen < self.eigen_threshold:
            debug_info["status"] = "ill_conditioned"
            return (0, 0), debug_info

        try:
            d = np.linalg.solve(ATA, ATb)
            u, v = float(d[0, 0]), float(d[1, 0])
            debug_info["status"] = "solved"
            debug_info["flow"] = (u, v)
            debug_info["flow_magnitude"] = np.sqrt(u**2 + v**2)
        except np.linalg.LinAlgError:
            debug_info["status"] = "singular_matrix"
            return (0, 0), debug_info

        return (u, v), debug_info


class FlowSolverIterative(FlowSolver):
    """
    Iterative refinement solver -- runs LK multiple times, warping each time.

    Instead of solving once, we:
    1. Solve LK to get initial (u, v)
    2. Warp img1 by (u, v) toward img2
    3. Recompute It using warped image
    4. Solve again for residual (du, dv)
    5. Update: u += du, v += dv
    6. Repeat until convergence or max iterations

    WHY ITERATE:
    - The Taylor expansion is only accurate for small displacements
    - By warping and re-solving, we can handle larger motions
    - This is essentially Newton's method applied to the brightness constancy

    TRY: Compare iteration counts -- usually 3-5 iterations suffice.
    More iterations help with fast motion but cost more compute.
    """
    def __init__(self, max_iters=5, convergence_threshold=0.01, eigen_threshold=1e-4):
        super().__init__()
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.eigen_threshold = eigen_threshold
        self.debug = StageDebugger("IterativeLK")
        # We need the full images for warping, stored per-frame
        self._img1 = None
        self._img2 = None

    def set_images(self, img1, img2):
        """Must be called before solve() with the full frame pair."""
        self._img1 = img1.astype(np.float64)
        self._img2 = img2.astype(np.float64)

    def solve(self, Ix, Iy, It, x, y, window_size=15):
        half_w = window_size // 2
        h, w = Ix.shape

        debug_info = {"method": "iterative", "x": x, "y": y, "iterations": []}

        if (y - half_w < 0 or y + half_w >= h or
            x - half_w < 0 or x + half_w >= w):
            debug_info["status"] = "out_of_bounds"
            return (0, 0), debug_info

        u_total, v_total = 0.0, 0.0

        for iteration in range(self.max_iters):
            # On first iteration, use provided gradients
            # On subsequent iterations, warp and recompute
            if iteration == 0 or self._img1 is None or self._img2 is None:
                It_local = It
                if iteration > 0:
                    # Can't refine without full images
                    break
            else:
                # Recompute temporal gradient using current flow estimate.
                # Flow means: point at (x,y) in img1 moved to (x+u, y+v) in img2.
                # So residual It = img2(x+u, y+v) - img1(x, y).
                shifted_x = int(round(x + u_total))
                shifted_y = int(round(y + v_total))
                if (shifted_y - half_w < 0 or shifted_y + half_w >= h or
                    shifted_x - half_w < 0 or shifted_x + half_w >= w):
                    break
                # img1 patch at ORIGINAL position, img2 patch at SHIFTED position
                patch1 = self._img1[y-half_w:y+half_w+1, x-half_w:x+half_w+1]
                patch2 = self._img2[shifted_y-half_w:shifted_y+half_w+1,
                                     shifted_x-half_w:shifted_x+half_w+1]
                if patch1.shape != patch2.shape:
                    break
                It_local_patch = patch2 - patch1
                # Need full-size It_local for indexing -- create a copy
                It_local = It.copy()
                It_local[y-half_w:y+half_w+1, x-half_w:x+half_w+1] = It_local_patch

            Ix_w = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            Iy_w = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            It_w = It_local[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

            ATA = np.array([
                [np.sum(Ix_w * Ix_w), np.sum(Ix_w * Iy_w)],
                [np.sum(Ix_w * Iy_w), np.sum(Iy_w * Iy_w)]
            ])
            ATb = np.array([[-np.sum(Ix_w * It_w)],
                            [-np.sum(Iy_w * It_w)]])

            eigenvalues = np.linalg.eigvalsh(ATA)
            min_eigen = min(eigenvalues)

            if min_eigen < self.eigen_threshold:
                debug_info["status"] = "ill_conditioned"
                debug_info["stopped_at_iter"] = iteration
                return (u_total, v_total), debug_info

            try:
                d = np.linalg.solve(ATA, ATb)
                du, dv = float(d[0, 0]), float(d[1, 0])
            except np.linalg.LinAlgError:
                break

            u_total += du
            v_total += dv

            iter_info = {
                "iter": iteration,
                "du": du, "dv": dv,
                "u_total": u_total, "v_total": v_total,
                "residual": np.sqrt(du**2 + dv**2)
            }
            debug_info["iterations"].append(iter_info)

            # Check convergence
            if np.sqrt(du**2 + dv**2) < self.convergence_threshold:
                debug_info["converged"] = True
                break

        debug_info["status"] = "solved"
        debug_info["flow"] = (u_total, v_total)
        debug_info["flow_magnitude"] = np.sqrt(u_total**2 + v_total**2)
        debug_info["num_iterations"] = len(debug_info["iterations"])

        return (u_total, v_total), debug_info


# =============================================================================
# COMPONENT 5: POINT VALIDATOR
# =============================================================================
# After computing flow, decide if the tracked point is still reliable.

class PointValidator(ABC):
    """Base class for validating tracked points."""
    def __init__(self):
        self.debug = StageDebugger("PointValidator")

    @abstractmethod
    def validate(self, old_point, new_point, flow, debug_info, img_shape):
        """
        Decide if a tracked point should be kept.

        Args:
            old_point: (x, y) original position
            new_point: (x, y) after flow
            flow: (u, v) flow vector
            debug_info: Dict from flow solver
            img_shape: (h, w) of the image

        Returns:
            (is_valid, reason): bool and string explanation
        """
        pass


class ValidatorEigenvalue(PointValidator):
    """
    Validate based on eigenvalue quality from the flow solver.

    A point is good if:
    1. The solver succeeded (not ill-conditioned)
    2. The new position is within image bounds
    3. The flow magnitude isn't unreasonably large

    This is the simplest validator -- just checks the solver's own diagnostics.
    """
    def __init__(self, max_flow=50):
        super().__init__()
        self.max_flow = max_flow
        self.debug = StageDebugger("EigenVal")

    def validate(self, old_point, new_point, flow, debug_info, img_shape):
        h, w = img_shape
        nx, ny = new_point

        if debug_info.get("status") != "solved":
            return False, f"solver_failed: {debug_info.get('status', 'unknown')}"

        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False, f"out_of_bounds: ({nx:.1f}, {ny:.1f})"

        mag = np.sqrt(flow[0]**2 + flow[1]**2)
        if mag > self.max_flow:
            return False, f"flow_too_large: {mag:.1f} > {self.max_flow}"

        return True, "ok"


class ValidatorForwardBackward(PointValidator):
    """
    Forward-backward consistency check.

    1. Track point forward: p1 -> p2 (using flow)
    2. Track point backward: p2 -> p1' (reverse flow)
    3. If |p1 - p1'| > threshold, the tracking is unreliable

    WHY: If tracking is accurate in both directions, the point should
    return to approximately the same position. Large forward-backward
    error indicates the point is near an occlusion boundary or the
    flow estimate is unreliable.

    NOTE: This validator is more expensive since it needs the reverse
    flow. In this simplified version, we approximate by checking flow
    magnitude and solver quality since full reverse flow needs another
    LK solve. For full FB check, you'd need to run the pipeline twice.
    """
    def __init__(self, max_flow=50, max_condition=1000):
        super().__init__()
        self.max_flow = max_flow
        self.max_condition = max_condition
        self.debug = StageDebugger("FwdBwd")

    def validate(self, old_point, new_point, flow, debug_info, img_shape):
        h, w = img_shape
        nx, ny = new_point

        if debug_info.get("status") != "solved":
            return False, f"solver_failed: {debug_info.get('status')}"

        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False, "out_of_bounds"

        mag = np.sqrt(flow[0]**2 + flow[1]**2)
        if mag > self.max_flow:
            return False, f"flow_too_large: {mag:.1f}"

        # Check condition number as proxy for reliability
        cond = debug_info.get("condition_number", float('inf'))
        if cond > self.max_condition:
            return False, f"poorly_conditioned: {cond:.0f}"

        return True, "ok"


class ValidatorNCC(PointValidator):
    """
    Normalized Cross-Correlation (NCC) validation.

    After tracking p1 -> p2, extract patches around both positions and
    compute NCC. If the patches don't look similar, tracking failed.

    NCC = Sigma((I1 - mu1)(I2 - mu2)) / (sigma1 * sigma2 * N)
    Range: [-1, 1], where 1 = perfect match

    WHY: Even if the math says the flow is (u,v), the actual appearance
    at the new location might not match. NCC catches drift, occlusions,
    and lighting changes that eigenvalue checks miss.
    """
    def __init__(self, ncc_threshold=0.7, patch_size=11, max_flow=50):
        super().__init__()
        self.ncc_threshold = ncc_threshold
        self.patch_size = patch_size
        self.max_flow = max_flow
        self.debug = StageDebugger("NCC")
        self._img1 = None
        self._img2 = None

    def set_images(self, img1, img2):
        """Set frame pair for NCC computation."""
        self._img1 = img1.astype(np.float64)
        self._img2 = img2.astype(np.float64)

    def validate(self, old_point, new_point, flow, debug_info, img_shape):
        h, w = img_shape
        nx, ny = new_point
        ox, oy = old_point

        if debug_info.get("status") != "solved":
            return False, f"solver_failed: {debug_info.get('status')}"

        if nx < 0 or nx >= w or ny < 0 or ny >= h:
            return False, "out_of_bounds"

        mag = np.sqrt(flow[0]**2 + flow[1]**2)
        if mag > self.max_flow:
            return False, f"flow_too_large: {mag:.1f}"

        # NCC check
        if self._img1 is not None and self._img2 is not None:
            half = self.patch_size // 2
            ox_i, oy_i = int(round(ox)), int(round(oy))
            nx_i, ny_i = int(round(nx)), int(round(ny))

            if (oy_i-half >= 0 and oy_i+half < h and ox_i-half >= 0 and ox_i+half < w and
                ny_i-half >= 0 and ny_i+half < h and nx_i-half >= 0 and nx_i+half < w):

                patch1 = self._img1[oy_i-half:oy_i+half+1, ox_i-half:ox_i+half+1]
                patch2 = self._img2[ny_i-half:ny_i+half+1, nx_i-half:nx_i+half+1]

                # NCC
                p1 = patch1 - patch1.mean()
                p2 = patch2 - patch2.mean()
                std1 = np.std(patch1)
                std2 = np.std(patch2)

                if std1 > 1e-6 and std2 > 1e-6:
                    ncc = np.sum(p1 * p2) / (std1 * std2 * p1.size)
                    debug_info["ncc"] = float(ncc)

                    if ncc < self.ncc_threshold:
                        return False, f"ncc_low: {ncc:.3f} < {self.ncc_threshold}"

        return True, "ok"


# =============================================================================
# COMPONENT 6: PYRAMID BUILDER
# =============================================================================
# Builds multi-scale image representations for handling large motions.

class PyramidBuilder(ABC):
    """Base class for image pyramid construction."""
    def __init__(self):
        self.debug = StageDebugger("PyramidBuilder")

    @abstractmethod
    def build(self, img, num_levels):
        """
        Build an image pyramid.

        Args:
            img: Grayscale image (uint8 or float64)
            num_levels: Number of additional levels (0 = just original)

        Returns:
            List of images, index 0 = original, index N = coarsest
        """
        pass


class GaussianPyramid(PyramidBuilder):
    """
    Standard Gaussian pyramid -- blur then downsample by 2x.

    Level 0: Original (e.g., 640x480)
    Level 1: 320x240 (Gaussian blur + subsample)
    Level 2: 160x120
    Level 3: 80x60

    The Gaussian blur before downsampling is CRITICAL -- without it,
    you get aliasing artifacts that corrupt the gradient computation.

    PARAMETERS:
    - blur_ksize: Gaussian kernel size (default 5)
    - blur_sigma: Gaussian sigma (default 1.0)
    """
    def __init__(self, blur_ksize=5, blur_sigma=1.0):
        super().__init__()
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.debug = StageDebugger("GaussPyramid")
        self._build_count = 0

    def build(self, img, num_levels):
        self._build_count += 1
        pyramid = [img.astype(np.float64)]

        # Only log on first build (to show structure) -- not every frame
        should_log = (self._build_count <= 2)  # first img1 + img2

        if should_log:
            self.debug.log(f"Building {num_levels}-level Gaussian pyramid:")
            self.debug.log(f"  Level 0: {img.shape[1]}x{img.shape[0]} (original)")

        for i in range(num_levels):
            blurred = cv2.GaussianBlur(pyramid[-1],
                                        (self.blur_ksize, self.blur_ksize),
                                        self.blur_sigma)
            downsampled = blurred[::2, ::2]
            pyramid.append(downsampled)

            if should_log:
                self.debug.log(
                    f"  Level {i+1}: {downsampled.shape[1]}x{downsampled.shape[0]} | "
                    f"Intensity range: [{downsampled.min():.0f}, {downsampled.max():.0f}]"
                )

        self.debug.record_stat("num_levels", num_levels + 1)
        self.debug.record_stat("coarsest_shape", pyramid[-1].shape)

        return pyramid


class LaplacianPyramid(PyramidBuilder):
    """
    Laplacian pyramid -- stores difference between levels.

    While the Gaussian pyramid just stores blurred/downsampled versions,
    the Laplacian stores the DETAIL lost at each level:
        L[i] = G[i] - upsample(G[i+1])

    This preserves high-frequency detail that can be used for more
    precise flow computation at each level.

    NOTE: For LK flow, we still track on the Gaussian pyramid, but
    the Laplacian is useful for understanding what information
    each level captures.

    TRY: Enable the Laplacian visualization to see edge details at
    each scale.
    """
    def __init__(self, blur_ksize=5, blur_sigma=1.0):
        super().__init__()
        self.blur_ksize = blur_ksize
        self.blur_sigma = blur_sigma
        self.debug = StageDebugger("LaplacPyramid")
        self._build_count = 0

    def build(self, img, num_levels):
        self._build_count += 1
        # First build Gaussian pyramid
        gaussian = [img.astype(np.float64)]
        for i in range(num_levels):
            blurred = cv2.GaussianBlur(gaussian[-1],
                                        (self.blur_ksize, self.blur_ksize),
                                        self.blur_sigma)
            downsampled = blurred[::2, ::2]
            gaussian.append(downsampled)

        # Only log Laplacian detail on first build
        should_log = (self._build_count <= 2)
        if should_log:
            for i in range(num_levels):
                upsampled = cv2.resize(gaussian[i+1],
                                        (gaussian[i].shape[1], gaussian[i].shape[0]))
                laplacian = gaussian[i] - upsampled
                detail_energy = np.mean(laplacian**2)
                self.debug.log(
                    f"  Level {i}: {gaussian[i].shape[1]}x{gaussian[i].shape[0]} | "
                    f"Detail energy: {detail_energy:.1f}"
                )

        # Return Gaussian pyramid (that's what LK operates on)
        return gaussian


# =============================================================================
# COMPONENT 7: REDETECTION POLICY
# =============================================================================
# Decides when and how to add new feature points.

class RedetectionPolicy(ABC):
    """Base class for point redetection strategy."""
    def __init__(self):
        self.debug = StageDebugger("Redetection")

    @abstractmethod
    def should_redetect(self, frame_count, num_tracked, num_original):
        """Should we detect new points this frame?"""
        pass

    @abstractmethod
    def merge_points(self, existing_points, new_points, max_total):
        """Combine existing and newly detected points."""
        pass


class RedetectPeriodic(RedetectionPolicy):
    """
    Redetect every N frames OR when too few points remain.

    This is the simplest policy -- just redetect on a schedule.
    When merging, keep all existing points and fill up to max_total
    with new detections.
    """
    def __init__(self, period=30, min_points=30):
        super().__init__()
        self.period = period
        self.min_points = min_points
        self.debug = StageDebugger("Periodic")

    def should_redetect(self, frame_count, num_tracked, num_original):
        triggered_by_count = num_tracked < self.min_points
        triggered_by_period = frame_count % self.period == 0

        if triggered_by_count:
            self.debug.log(f"Redetecting: only {num_tracked} points left (min={self.min_points})")
        elif triggered_by_period:
            self.debug.log(f"Redetecting: periodic refresh (every {self.period} frames)")

        return triggered_by_count or triggered_by_period

    def merge_points(self, existing_points, new_points, max_total):
        if len(existing_points) == 0:
            merged = new_points[:max_total]
        else:
            merged = np.vstack([existing_points, new_points])
            merged = merged[:max_total]

        self.debug.log(
            f"Merged: {len(existing_points)} existing + {len(new_points)} new "
            f"-> {len(merged)} total (max={max_total})"
        )
        return merged


class RedetectAdaptive(RedetectionPolicy):
    """
    Adaptive redetection -- redetect based on tracking quality.

    Tracks the survival rate (how many points survive each frame)
    and triggers redetection when the rate drops below a threshold.
    This avoids unnecessary redetection when tracking is stable
    and triggers quickly when things go wrong.
    """
    def __init__(self, survival_threshold=0.7, min_points=20, cooldown=10):
        super().__init__()
        self.survival_threshold = survival_threshold
        self.min_points = min_points
        self.cooldown = cooldown
        self._last_redetect = -cooldown
        self.debug = StageDebugger("Adaptive")

    def should_redetect(self, frame_count, num_tracked, num_original):
        if num_original == 0:
            return True

        survival_rate = num_tracked / num_original
        too_few = num_tracked < self.min_points
        poor_survival = survival_rate < self.survival_threshold
        cooled_down = (frame_count - self._last_redetect) >= self.cooldown

        if (too_few or poor_survival) and cooled_down:
            reason = f"survival={survival_rate:.1%}" if poor_survival else f"count={num_tracked}"
            self.debug.log(f"Redetecting: {reason} (frame {frame_count})")
            self._last_redetect = frame_count
            return True

        return False

    def merge_points(self, existing_points, new_points, max_total):
        if len(existing_points) == 0:
            merged = new_points[:max_total]
        else:
            # Filter new points that are too close to existing ones
            min_dist = 10
            keep = []
            for pt in new_points:
                dists = np.sqrt(np.sum((existing_points - pt)**2, axis=1))
                if np.min(dists) > min_dist:
                    keep.append(pt)
            if keep:
                merged = np.vstack([existing_points, np.array(keep)])
            else:
                merged = existing_points
            merged = merged[:max_total]

        self.debug.log(f"Merged -> {len(merged)} points")
        return merged


# =============================================================================
# THE MODULAR PIPELINE
# =============================================================================

class ModularLucasKanade:
    """
    Fully modular Lucas-Kanade pipeline.

    Every component can be swapped independently. Debug output at every stage.
    """
    def __init__(
        self,
        feature_detector,
        gradient_computer,
        temporal_gradient,
        flow_solver,
        point_validator,
        pyramid_builder,
        redetection_policy,
        window_size=15,
        num_pyramid_levels=3,
        max_points=100,
        verbose=False,
    ):
        self.detector = feature_detector
        self.gradient = gradient_computer
        self.temporal = temporal_gradient
        self.solver = flow_solver
        self.validator = point_validator
        self.pyramid = pyramid_builder
        self.redetection = redetection_policy

        self.window_size = window_size
        self.num_levels = num_pyramid_levels
        self.max_points = max_points
        self.verbose = verbose

        # Set verbose on all debuggers
        if verbose:
            for component in [self.detector, self.gradient, self.temporal,
                              self.solver, self.validator, self.pyramid,
                              self.redetection]:
                component.debug.verbose = True

    def track(self, img1, img2, points, frame_num=0):
        """
        Track points from img1 to img2.

        This is the pyramidal LK pipeline with full debug output.

        Returns:
            new_points: Nx2 array of tracked positions
            status: Nx1 array (1=good, 0=lost)
            frame_debug: Dict with all debug info from all components
        """
        frame_debug = {"stages": {}}

        # --- STAGE 1: Build pyramids ------------------------------------
        t0 = time.time()
        pyr1 = self.pyramid.build(img1, self.num_levels)
        pyr2 = self.pyramid.build(img2, self.num_levels)
        t_pyramid = time.time() - t0
        frame_debug["stages"]["pyramid"] = {
            "time_ms": t_pyramid * 1000,
            "num_levels": self.num_levels + 1,
            "stats": self.pyramid.debug.get_all_stats()
        }

        # --- STAGE 2: Initialize flow ----------------------------------
        flow = np.zeros((len(points), 2), dtype=np.float64)
        all_solver_debug = []

        # --- STAGE 3: Coarse-to-fine tracking --------------------------
        t0 = time.time()
        for level in range(self.num_levels, -1, -1):
            level_img1 = pyr1[level]
            level_img2 = pyr2[level]

            # Scale points to this level
            level_points = points.astype(np.float64) / (2 ** level)

            # STAGE 3a: Compute spatial gradients
            # Only set frame for periodic logging at finest level to avoid 4x spam
            self.gradient.debug.set_frame(frame_num if level == 0 else -1)
            Ix, Iy = self.gradient.compute(level_img1)

            # STAGE 3b: Compute temporal gradient
            self.temporal.debug.set_frame(frame_num if level == 0 else -1)
            It = self.temporal.compute(level_img1, level_img2)

            # Set images for solvers/validators that need them
            if hasattr(self.solver, 'set_images'):
                self.solver.set_images(level_img1, level_img2)
            if hasattr(self.validator, 'set_images'):
                self.validator.set_images(level_img1, level_img2)

            # STAGE 3c: Solve flow at each point
            level_debug = []
            for i, (x, y) in enumerate(level_points.astype(int)):
                uv, debug_info = self.solver.solve(
                    Ix, Iy, It, x, y, self.window_size
                )
                debug_info["pyramid_level"] = level
                flow[i, 0] += uv[0]
                flow[i, 1] += uv[1]
                level_debug.append(debug_info)

            all_solver_debug.append({
                "level": level,
                "shape": level_img1.shape,
                "points_solved": len(level_points),
                "per_point": level_debug
            })

            # Scale flow up for next finer level
            if level > 0:
                flow *= 2

        t_flow = time.time() - t0
        frame_debug["stages"]["flow"] = {
            "time_ms": t_flow * 1000,
            "per_level": all_solver_debug
        }

        # --- STAGE 4: Compute new positions ----------------------------
        new_points = points.astype(np.float64) + flow

        # --- STAGE 5: Validate each point ------------------------------
        t0 = time.time()
        status = np.ones(len(points), dtype=np.uint8)
        validation_debug = []

        for i in range(len(points)):
            is_valid, reason = self.validator.validate(
                old_point=points[i],
                new_point=new_points[i],
                flow=flow[i],
                debug_info=all_solver_debug[-1]["per_point"][i],  # finest level
                img_shape=img1.shape
            )
            if not is_valid:
                status[i] = 0
            validation_debug.append({
                "point_idx": i,
                "valid": is_valid,
                "reason": reason,
                "flow_mag": float(np.sqrt(flow[i,0]**2 + flow[i,1]**2))
            })

        t_validate = time.time() - t0
        frame_debug["stages"]["validation"] = {
            "time_ms": t_validate * 1000,
            "total": len(points),
            "kept": int(status.sum()),
            "lost": int(len(points) - status.sum()),
            "per_point": validation_debug
        }

        frame_debug["total_time_ms"] = (t_pyramid + t_flow + t_validate) * 1000

        return new_points, status, frame_debug


# =============================================================================
# DEBUG VISUALIZATION
# =============================================================================

def print_frame_summary(frame_num, frame_debug, tracked, lost, avg_flow, max_flow):
    """Print a concise per-frame summary with key stats from each stage."""
    total_ms = frame_debug.get("total_time_ms", 0)
    pyr_ms = frame_debug["stages"]["pyramid"]["time_ms"]
    flow_ms = frame_debug["stages"]["flow"]["time_ms"]
    val_ms = frame_debug["stages"]["validation"]["time_ms"]

    # Count solver outcomes at finest level
    finest_level_data = frame_debug["stages"]["flow"]["per_level"][-1]
    solved = sum(1 for p in finest_level_data["per_point"] if p.get("status") == "solved")
    failed = len(finest_level_data["per_point"]) - solved

    print(f"\n{'='*80}")
    print(f"FRAME {frame_num}")
    print(f"{'='*80}")
    print(f"  PIPELINE TIMING:  Pyramid={pyr_ms:.1f}ms | Flow={flow_ms:.1f}ms | "
          f"Validate={val_ms:.1f}ms | TOTAL={total_ms:.1f}ms")
    print(f"  TRACKING:         Tracked={tracked} | Lost={lost} | "
          f"Solver: {solved} solved, {failed} failed")
    print(f"  FLOW:             Avg={avg_flow:.2f}px | Max={max_flow:.2f}px")

    # Validation reasons
    val_data = frame_debug["stages"]["validation"]
    reasons = {}
    for p in val_data["per_point"]:
        if not p["valid"]:
            r = p["reason"].split(":")[0]
            reasons[r] = reasons.get(r, 0) + 1
    if reasons:
        reason_str = ", ".join(f"{k}={v}" for k, v in reasons.items())
        print(f"  LOST REASONS:     {reason_str}")

    # Sample eigenvalues from solved points (finest level)
    eigs = [p.get("eigenvalues", (0,0)) for p in finest_level_data["per_point"]
            if p.get("status") == "solved"]
    if eigs:
        min_eigs = [e[0] for e in eigs]
        print(f"  EIGENVALUES:      min(eig_min)={min(min_eigs):.2e} | "
              f"median(eig_min)={np.median(min_eigs):.2e} | max(eig_min)={max(min_eigs):.2e}")


def print_detailed_point_debug(frame_debug, point_indices=None):
    """Print detailed per-point debug info for selected points."""
    finest = frame_debug["stages"]["flow"]["per_level"][-1]
    val = frame_debug["stages"]["validation"]

    if point_indices is None:
        point_indices = range(min(5, len(finest["per_point"])))

    for idx in point_indices:
        if idx >= len(finest["per_point"]):
            break

        solver_info = finest["per_point"][idx]
        val_info = val["per_point"][idx]

        print(f"\n  Point #{idx} at ({solver_info.get('x')}, {solver_info.get('y')})")
        print(f"    Solver: {solver_info.get('method')} -> {solver_info.get('status')}")
        if "eigenvalues" in solver_info:
            e = solver_info["eigenvalues"]
            print(f"    Eigenvalues: eig_min={e[0]:.2e}, eig_max={e[1]:.2e}")
        if "condition_number" in solver_info:
            print(f"    Condition #: {solver_info['condition_number']:.1f}")
        if "flow" in solver_info:
            u, v = solver_info["flow"]
            print(f"    Flow: u={u:.3f}, v={v:.3f} (mag={solver_info.get('flow_magnitude', 0):.3f})")
        if "iterations" in solver_info:
            for it in solver_info["iterations"]:
                print(f"      Iter {it['iter']}: du={it['du']:.4f} dv={it['dv']:.4f} "
                      f"residual={it['residual']:.4f}")
        if "ncc" in solver_info:
            print(f"    NCC: {solver_info['ncc']:.3f}")
        print(f"    Validation: {'PASS' if val_info['valid'] else 'FAIL'} ({val_info['reason']})")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_modular_lk(pipeline, scale=0.35, print_period=15, detail_points=3):
    """
    Run the modular pipeline on the video.

    Args:
        pipeline: ModularLucasKanade instance
        scale: Video downscale factor
        print_period: How often to print full debug (every N frames)
        detail_points: How many points to show detailed debug for
    """
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Failed to open video!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    new_w, new_h = int(width * scale), int(height * scale)
    print(f"Working at {scale}x: {new_w}x{new_h}")

    out_path = os.path.join(OUTPUT_DIR, "07_lk_modular.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (new_w, new_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    print(f"Skipping to frame {START_FRAME}")

    ret, frame = cap.read()
    frame = cv2.resize(frame, (new_w, new_h))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initial detection
    print("\n" + "="*80)
    print("INITIAL FEATURE DETECTION")
    print("="*80)
    points = pipeline.detector.detect(old_gray, pipeline.max_points)
    num_original = len(points)
    print(f"Starting with {num_original} points\n")

    colors = np.random.randint(0, 255, (len(points), 3)).tolist()
    trail_mask = np.zeros_like(frame)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or (MAX_FRAMES and frame_count >= MAX_FRAMES):
            break

        frame_count += 1
        frame = cv2.resize(frame, (new_w, new_h))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(points) == 0:
            points = pipeline.detector.detect(frame_gray, pipeline.max_points)
            colors = np.random.randint(0, 255, (len(points), 3)).tolist()
            trail_mask = np.zeros_like(frame)
            old_gray = frame_gray.copy()
            num_original = len(points)
            continue

        # -- Run the modular pipeline ------------------------------------
        new_points, status, frame_debug = pipeline.track(
            old_gray, frame_gray, points, frame_num=frame_count
        )

        # Filter to good points
        good_mask = status == 1
        good_new = new_points[good_mask]
        good_old = points[good_mask]
        good_colors = [colors[i] for i in range(len(status)) if status[i] == 1]

        tracked = len(good_new)
        lost = len(points) - tracked

        # Flow stats
        if tracked > 0:
            flow_vecs = good_new - good_old
            mags = np.sqrt(flow_vecs[:, 0]**2 + flow_vecs[:, 1]**2)
            avg_mag = float(np.mean(mags))
            max_mag = float(np.max(mags))
        else:
            avg_mag = max_mag = 0.0

        # -- Debug output ------------------------------------------------
        if frame_count % print_period == 0:
            print_frame_summary(frame_count, frame_debug, tracked, lost, avg_mag, max_mag)
            if detail_points > 0:
                print_detailed_point_debug(frame_debug, range(detail_points))

        # -- Visualization -----------------------------------------------
        vis = frame.copy()
        for i, (new, old) in enumerate(zip(good_new.astype(int), good_old.astype(int))):
            a, b = new
            c, d = old
            color = good_colors[i]
            trail_mask = cv2.line(trail_mask, (a, b), (c, d), color, 2)
            vis = cv2.circle(vis, (a, b), 4, color, -1)

        vis = cv2.add(vis, trail_mask)

        # HUD
        total_ms = frame_debug.get("total_time_ms", 0)
        info = [
            f"MODULAR LK | Frame: {frame_count} | Points: {tracked}",
            f"Flow: avg={avg_mag:.1f}px max={max_mag:.1f}px | {total_ms:.0f}ms",
        ]
        for i, line in enumerate(info):
            cv2.putText(vis, line, (10, 22 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        out.write(vis)

        # -- Update state ------------------------------------------------
        old_gray = frame_gray.copy()
        points = good_new
        colors = good_colors

        # -- Redetection -------------------------------------------------
        if pipeline.redetection.should_redetect(frame_count, tracked, num_original):
            new_detections = pipeline.detector.detect(frame_gray, pipeline.max_points)
            if len(new_detections) > 0:
                points = pipeline.redetection.merge_points(
                    points, new_detections, pipeline.max_points
                )
                colors = np.random.randint(0, 255, (len(points), 3)).tolist()
                trail_mask = np.zeros_like(frame)
                num_original = len(points)

    cap.release()
    out.release()
    print(f"\n{'='*80}")
    print(f"DONE -- Processed {frame_count} frames")
    print(f"Output saved to: {out_path}")
    print(f"{'='*80}")


# =============================================================================
# +===========================================================================+
# |  CONFIGURATION -- SWAP COMPONENTS HERE!                                   |
# +===========================================================================+
#
#  To experiment, just change which class is used for each component.
#  Every combination works. Here are the options:
#
#  FeatureDetector:   ShiTomasiDetector() | HarrisDetector() | FASTDetector()
#  GradientComputer:  GradientSobel()     | GradientScharr() | GradientCentralDiff()
#  TemporalGradient:  TemporalSimpleDiff()| TemporalAveraged()| TemporalCrossFrame()
#  FlowSolver:        FlowSolverLeastSquares() | FlowSolverWeighted() | FlowSolverIterative()
#  PointValidator:    ValidatorEigenvalue()| ValidatorForwardBackward() | ValidatorNCC()
#  PyramidBuilder:    GaussianPyramid()   | LaplacianPyramid()
#  RedetectionPolicy: RedetectPeriodic()  | RedetectAdaptive()
#
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 80)
    print(" MODULAR LUCAS-KANADE DEEP DIVE")
    print(" Every component is swappable -- experiment freely!")
    print("=" * 80)
    print()

    # -- BUILD YOUR PIPELINE ---------------------------------------------
    pipeline = ModularLucasKanade(
        # COMPONENT 1: How to find points to track
        # Try: HarrisDetector(k=0.04) or FASTDetector(threshold=20)
        feature_detector=ShiTomasiDetector(quality_level=0.05, min_distance=15),

        # COMPONENT 2: How to compute spatial gradients (Ix, Iy)
        # Try: GradientScharr() or GradientCentralDiff()
        gradient_computer=GradientSobel(ksize=3),

        # COMPONENT 3: How to compute temporal gradient (It)
        # Try: TemporalAveraged(kernel_size=5) or TemporalCrossFrame()
        temporal_gradient=TemporalSimpleDiff(),

        # COMPONENT 4: How to solve for flow (u, v) at each point
        # Try: FlowSolverWeighted(sigma_factor=0.3) or FlowSolverIterative(max_iters=5)
        flow_solver=FlowSolverLeastSquares(eigen_threshold=1e-4),

        # COMPONENT 5: How to validate if a point is still good
        # Try: ValidatorForwardBackward() or ValidatorNCC(ncc_threshold=0.7)
        point_validator=ValidatorEigenvalue(max_flow=50),

        # COMPONENT 6: How to build the image pyramid
        # Try: LaplacianPyramid()
        pyramid_builder=GaussianPyramid(blur_ksize=5, blur_sigma=1.0),

        # COMPONENT 7: When to detect new points
        # Try: RedetectAdaptive(survival_threshold=0.7)
        redetection_policy=RedetectPeriodic(period=30, min_points=30),

        # Pipeline parameters
        window_size=15,
        num_pyramid_levels=3,
        max_points=100,
        verbose=False,  # Set True for per-frame debug from every component
    )

    run_modular_lk(
        pipeline,
        scale=0.35,
        print_period=15,     # Print full debug every N frames
        detail_points=3,     # Show detailed per-point debug for first N points
    )
