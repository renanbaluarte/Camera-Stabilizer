"""Frame-by-frame video stabilizer (v2).

Pipeline per frame:
  1. Motion estimation: persistent grid-distributed feature tracks (Lucas-Kanade
     with a forward-backward consistency check), then a RANSAC similarity fit
     (translation + rotation + scale). Moving people are rejected as outliers
     instead of just being "outvoted" by a median.
  2. Path smoothing, two stages:
       a. Optional look-ahead: a short frame delay lets us smooth with a
          centered Gaussian (zero phase lag), the trick LiveVisionKit uses.
          It removes estimation noise without trailing behind the camera.
       b. High-pass "home position" baseline (exponential), so a camera bolted
          to a vibrating structure is held locked while slow / intentional
          pan-tilt still passes through.
  3. Warp: one single warpAffine that applies the correction AND the crop zoom
     (no second resize, so it is sharper and cheaper). The correction is
     scaled back whenever it would expose a border, with anti-windup.

Feed frames via process_frame(). Stateful: one instance per stream. Tuning can
be changed live via set_tuning(). With lookahead_frames=N the returned frame
is the one captured N frames earlier (set OBS audio sync offset accordingly).
"""
from collections import deque

import cv2
import numpy as np

DEFAULT_SMOOTHING_S = 1.0
INTENTIONAL_SMOOTHING_S = 0.12
DEFAULT_STRENGTH = 1.0

DEFAULT_MAX_KEEP = 0.98
DEFAULT_MIN_KEEP = 0.90
CROP_WINDOW_S = 3.0
CROP_SLEW_STEP = 0.001    # 0.1%/frame: adaptive zoom changes are imperceptible
CROP_SAFETY_MARGIN = 1.25

DEFAULT_MOTION_WINDOW_S = 0.5
DEFAULT_LOOKAHEAD = 4          # frames (~133ms at 30fps); 0 = no delay
DEFAULT_TRACKING_WIDTH = 640

GRID_COLS, GRID_ROWS = 8, 6
MAX_PER_CELL = 8
MIN_TRACKS = 120
FB_MAX_ERR = 0.8               # px at tracking scale
RANSAC_THRESH = 1.0            # px at tracking scale
MIN_INLIERS = 12

FLOW_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Path vector layout: [x, y, angle(rad), log(scale)]
IX, IY, IA, IS = range(4)


def sensitivity_to_strength(sensitivity):
    """Map the 0-100 'sensibilidade' slider to a correction fraction."""
    return float(np.clip(sensitivity, 0, 100) / 100.0)


class VideoStabilizer:
    def __init__(self, frame_w, frame_h, fps=30,
                 strength=DEFAULT_STRENGTH,
                 smoothing_s=DEFAULT_SMOOTHING_S,
                 max_keep_ratio=DEFAULT_MAX_KEEP,
                 min_keep_ratio=DEFAULT_MIN_KEEP,
                 detect_intentional_motion=True,
                 stabilize_rotation=True,
                 stabilize_zoom=False,
                 adaptive_crop=False,
                 lookahead_frames=DEFAULT_LOOKAHEAD,
                 tracking_width=DEFAULT_TRACKING_WIDTH,
                 motion_window_s=DEFAULT_MOTION_WINDOW_S,
                 **_ignored):
        self.frame_w, self.frame_h, self.fps = frame_w, frame_h, fps

        tracking_width = min(tracking_width, frame_w)
        self._track_scale = tracking_width / frame_w
        self._track_size = (tracking_width, round(frame_h * self._track_scale))
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._motion_window = deque(maxlen=max(3, round(fps * motion_window_s)))
        self._need_window = deque(maxlen=max(3, round(fps * CROP_WINDOW_S)))

        self._prev_gray = None
        self._prev_pts = None
        self._traj = np.zeros(4)
        self._baseline = None
        self._frames = deque()        # delayed frames waiting for look-ahead
        self._traj_hist = deque()     # trajectory samples (newest last)

        # Live-readable state for the UI / debug overlay.
        self.last_motion = (0.0, 0.0)
        self.last_correction = (0.0, 0.0)
        self.last_rotation_deg = 0.0
        self.intentional_ratio = 0.0
        self.tracked_points = 0
        self.crop_ratio_now = min_keep_ratio

        self.lookahead = 0
        self.set_tuning(strength=strength, smoothing_s=smoothing_s,
                        max_keep_ratio=max_keep_ratio, min_keep_ratio=min_keep_ratio,
                        detect_intentional_motion=detect_intentional_motion,
                        stabilize_rotation=stabilize_rotation, stabilize_zoom=stabilize_zoom,
                        adaptive_crop=adaptive_crop, lookahead_frames=lookahead_frames)
        self.crop_ratio_now = self._min_keep if not self.adaptive_crop else self._max_keep

    # ------------------------------------------------------------------ tuning
    def set_tuning(self, strength=None, smoothing_s=None, max_keep_ratio=None,
                   min_keep_ratio=None, detect_intentional_motion=None,
                   stabilize_rotation=None, stabilize_zoom=None, adaptive_crop=None,
                   lookahead_frames=None, **_ignored):
        """Change tuning while running (safe from the UI thread: plain scalars)."""
        if strength is not None:
            self.strength = float(np.clip(strength, 0.0, 1.0))
        if smoothing_s is not None:
            self.smoothing_s = max(0.05, float(smoothing_s))
        if max_keep_ratio is not None:
            self._max_keep = float(np.clip(max_keep_ratio, 0.50, 1.0))
        if min_keep_ratio is not None:
            self._min_keep = float(np.clip(min_keep_ratio, 0.50, 1.0))
        if detect_intentional_motion is not None:
            self.detect_intentional_motion = bool(detect_intentional_motion)
        if stabilize_rotation is not None:
            self.stabilize_rotation = bool(stabilize_rotation)
        if stabilize_zoom is not None:
            self.stabilize_zoom = bool(stabilize_zoom)
        if adaptive_crop is not None:
            self.adaptive_crop = bool(adaptive_crop)
        if lookahead_frames is not None:
            self.lookahead = int(np.clip(lookahead_frames, 0, 30))
        self._min_keep = min(self._min_keep, self._max_keep)

    # ------------------------------------------------------- motion estimation
    def _prepare(self, frame):
        small = cv2.resize(frame, self._track_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gray)  # dark, low-contrast churches still give features

    def _detect(self, gray, existing):
        """Top up features so they stay spread over a grid (a single textured
        corner of the picture must not dominate the estimate)."""
        h, w = gray.shape
        cw, ch = w / GRID_COLS, h / GRID_ROWS
        counts = np.zeros((GRID_ROWS, GRID_COLS), int)
        mask = np.full((h, w), 255, np.uint8)
        if existing is not None and len(existing):
            for x, y in existing.reshape(-1, 2):
                counts[min(int(y / ch), GRID_ROWS - 1), min(int(x / cw), GRID_COLS - 1)] += 1
                cv2.circle(mask, (int(x), int(y)), 8, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=GRID_COLS * GRID_ROWS * MAX_PER_CELL * 2,
                                      qualityLevel=0.005, minDistance=8, blockSize=5, mask=mask)
        keep = [] if existing is None else [existing.reshape(-1, 2)]
        if new is not None:
            sel = []
            for x, y in new.reshape(-1, 2):
                r, c = min(int(y / ch), GRID_ROWS - 1), min(int(x / cw), GRID_COLS - 1)
                if counts[r, c] < MAX_PER_CELL:
                    counts[r, c] += 1
                    sel.append((x, y))
            if sel:
                keep.append(np.array(sel, np.float32))
        if not keep:
            return None
        return np.concatenate(keep).astype(np.float32).reshape(-1, 1, 2)

    def _estimate_motion(self, prev_gray, gray):
        """Return inter-frame motion [dx, dy, da, dlogs] in full-res pixels."""
        if self._prev_pts is None or len(self._prev_pts) < MIN_TRACKS:
            self._prev_pts = self._detect(prev_gray, self._prev_pts)
        p0 = self._prev_pts
        if p0 is None or len(p0) < MIN_INLIERS:
            self._prev_pts = None
            return np.zeros(4)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **FLOW_PARAMS)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **FLOW_PARAMS)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < FB_MAX_ERR)
        h, w = gray.shape
        q = p1.reshape(-1, 2)
        ok &= (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
        self.tracked_points = int(len(a))
        if len(a) < MIN_INLIERS:
            self._prev_pts = None
            return np.zeros(4)

        M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                 ransacReprojThreshold=RANSAC_THRESH,
                                                 maxIters=500, confidence=0.995, refineIters=10)
        if M is None or inliers is None or inliers.sum() < MIN_INLIERS:
            d = np.median(b - a, axis=0)
            self._prev_pts = b.reshape(-1, 1, 2)
            return np.array([d[0] / self._track_scale, d[1] / self._track_scale, 0.0, 0.0])

        inl = inliers.reshape(-1).astype(bool)
        self.tracked_points = int(inl.sum())
        # Carry only inlier tracks forward: points on moving people drop out.
        self._prev_pts = b[inl].reshape(-1, 1, 2)

        scale = float(np.hypot(M[0, 0], M[1, 0]))
        angle = float(np.arctan2(M[1, 0], M[0, 0]))
        # Express translation about the image centre so rotation and
        # translation don't leak into each other.
        c = np.array([w / 2, h / 2])
        t = M[:, :2] @ c + M[:, 2] - c
        return np.array([t[0] / self._track_scale, t[1] / self._track_scale,
                         angle, np.log(max(scale, 1e-6))])

    def _intentional_motion_ratio(self):
        """0 = pure back-and-forth vibration, 1 = clean sustained pan/tilt."""
        if not self.detect_intentional_motion or len(self._motion_window) < 3:
            return 0.0
        deltas = np.array(self._motion_window)
        net = np.linalg.norm(deltas.sum(axis=0))
        total = np.sum(np.linalg.norm(deltas, axis=1)) + 1e-6
        # Below ~0.3px/frame on average it is just noise, not a pan.
        if total / len(deltas) < 0.3:
            return 0.0
        return float(np.clip((net / total - 0.25) / 0.6, 0.0, 1.0))

    # --------------------------------------------------------------- smoothing
    def _gaussian_target(self, center_idx):
        hist = np.array(self._traj_hist)
        n = len(hist)
        radius = self.lookahead
        if radius == 0:
            return hist[center_idx]
        sigma = max(radius / 2.0, 0.8)
        idx = np.arange(max(0, center_idx - radius), min(n, center_idx + radius + 1))
        wts = np.exp(-0.5 * ((idx - center_idx) / sigma) ** 2)
        return (hist[idx] * wts[:, None]).sum(axis=0) / wts.sum()

    # -------------------------------------------------------------------- warp
    def _build_matrix(self, corr, keep):
        cx, cy = self.frame_w / 2, self.frame_h / 2
        s = np.exp(corr[IS]) / keep
        ca, sa = np.cos(corr[IA]) * s, np.sin(corr[IA]) * s
        # rotate/scale about centre, then translate
        return np.array([[ca, -sa, cx - ca * cx + sa * cy + corr[IX] / keep],
                         [sa, ca, cy - sa * cx - ca * cy + corr[IY] / keep]], np.float64)

    def _fits(self, M):
        """True if every output pixel samples from inside the source frame."""
        inv = cv2.invertAffineTransform(M)
        w, h = self.frame_w, self.frame_h
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float)
        src = corners @ inv.T
        return bool((src[:, 0] >= -0.5).all() and (src[:, 0] <= w - 0.5).all()
                    and (src[:, 1] >= -0.5).all() and (src[:, 1] <= h - 0.5).all())

    def _limit(self, corr, keep):
        """Scale the correction down (binary search) until no border shows."""
        if self._fits(self._build_matrix(corr, keep)):
            return corr
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            if self._fits(self._build_matrix(corr * mid, keep)):
                lo = mid
            else:
                hi = mid
        return corr * lo

    def _update_crop(self, corr):
        if not self.adaptive_crop:
            self.crop_ratio_now = self._min_keep
            return
        diag = np.hypot(self.frame_w, self.frame_h) / 2
        rot_px = abs(corr[IA]) * diag
        need = (np.abs(corr[:2]) + rot_px) * CROP_SAFETY_MARGIN
        self._need_window.append(need)
        need = np.max(self._need_window, axis=0)
        required = np.min(1 - 2 * need / np.array([self.frame_w, self.frame_h]))
        target = float(np.clip(required, self._min_keep, self._max_keep))
        step = CROP_SLEW_STEP * (4 if target < self.crop_ratio_now else 1)  # grow faster
        delta = np.clip(target - self.crop_ratio_now, -step, step)
        self.crop_ratio_now = float(np.clip(self.crop_ratio_now + delta,
                                            self._min_keep, self._max_keep))

    # ----------------------------------------------------------------- process
    def process_frame(self, frame):
        gray = self._prepare(frame)
        if self._prev_gray is not None:
            motion = self._estimate_motion(self._prev_gray, gray)
        else:
            motion = np.zeros(4)
        self._prev_gray = gray

        self._motion_window.append(motion[:2].copy())
        self.last_motion = (float(motion[IX]), float(motion[IY]))
        self._traj = self._traj + motion
        self._traj_hist.append(self._traj.copy())
        self._frames.append(frame)

        # Keep enough history for a centred window around the output frame.
        while len(self._frames) > self.lookahead + 1:
            self._frames.popleft()
        while len(self._traj_hist) > 2 * self.lookahead + 1:
            self._traj_hist.popleft()

        out_frame = self._frames[0]
        out_idx = len(self._traj_hist) - len(self._frames)
        out_traj = self._traj_hist[out_idx]
        target = self._gaussian_target(out_idx)

        # High-pass "home position": slow baseline, snaps faster on real pans.
        self.intentional_ratio = self._intentional_motion_ratio()
        tau = (self.smoothing_s * (1 - self.intentional_ratio)
               + INTENTIONAL_SMOOTHING_S * self.intentional_ratio)
        alpha = 1.0 - np.exp(-1.0 / max(self.fps * tau, 1e-6))
        if self._baseline is None:
            self._baseline = target.copy()
        self._baseline += alpha * (target - self._baseline)

        corr = self.strength * (self._baseline - out_traj)
        if not self.stabilize_rotation:
            corr[IA] = 0.0
        if not self.stabilize_zoom:
            corr[IS] = 0.0

        self._update_crop(corr)
        keep = self.crop_ratio_now
        applied = self._limit(corr, keep)

        # Anti-windup: never park the picture against the crop edge.
        if self.strength > 0 and not np.allclose(applied, corr):
            self._baseline = out_traj + applied / self.strength
            if not self.stabilize_rotation:
                self._baseline[IA] = out_traj[IA]
            if not self.stabilize_zoom:
                self._baseline[IS] = out_traj[IS]

        self.last_correction = (float(applied[IX]), float(applied[IY]))
        self.last_rotation_deg = float(np.degrees(applied[IA]))

        M = self._build_matrix(applied, keep)
        return cv2.warpAffine(out_frame, M, (self.frame_w, self.frame_h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
