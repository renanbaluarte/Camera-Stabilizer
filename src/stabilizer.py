"""Frame-by-frame video stabilizer (v3).

Pipeline per frame:
  1. Motion estimation, rolling-shutter aware. Persistent grid-distributed
     feature tracks (Lucas-Kanade + forward-backward check) are fitted with a
     robust (Tukey IRLS) model:
         dx = (s-1)*x + fx(y)        dy = theta(y)*x + gy(y)
     where fx, gy, theta are smooth piecewise-linear curves over the image
     height. Each row of a rolling-shutter sensor is read at a different
     instant, so under vibration every row has its own shift and angle: the
     variation of fx/gy/theta along the height (after removing the rigid,
     frame-wide part) is the "jelly" between the two frames.
     A plain similarity fit mistakes that shear for rotation and *adds* wobble.
  2. Path smoothing: optional look-ahead (frame delay) enabling a centered
     zero-lag Gaussian, followed by a high-pass "home position" baseline. The
     jelly profile is integrated into the same path and high-passed too, so
     its absolute per-row distortion is recovered and cancelled.
  3. Warp: one resampling pass doing correction + jelly removal + crop zoom
     (warpAffine, or remap when jelly correction is active). The correction is
     scaled back whenever it would expose a border, with anti-windup.

Feed frames via process_frame(). Stateful: one instance per stream. Tuning can
be changed live via set_tuning(). With lookahead_frames=N the returned frame
is the one captured N frames earlier: audio_delay_ms() tells how much audio
delay to set in OBS.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

DEFAULT_SMOOTHING_S = 1.0
INTENTIONAL_SMOOTHING_S = 0.12
JELLY_SMOOTHING_S = 0.3        # high-pass for the jelly profile (it has no slow part)
DEFAULT_STRENGTH = 1.0
DEFAULT_JELLY_STRENGTH = 1.0

DEFAULT_MAX_KEEP = 0.98
DEFAULT_MIN_KEEP = 0.90
CROP_WINDOW_S = 3.0
CROP_SLEW_STEP = 0.001
CROP_SAFETY_MARGIN = 1.25

DEFAULT_MOTION_WINDOW_S = 0.5
DEFAULT_LOOKAHEAD = 8
DEFAULT_TRACKING_WIDTH = 640

GRID_COLS, GRID_ROWS = 8, 6
MAX_PER_CELL = 8
MIN_TRACKS = 120
FB_MAX_ERR = 0.8               # px at tracking scale
TUKEY_C = 2.5                  # px at tracking scale
MIN_INLIERS = 12
JELLY_KNOTS = 7                # control points of the per-row profile
JELLY_SMOOTH = 2.0             # curvature penalty of the profile fit

FLOW_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Path vector: [x, y, angle(rad), log(scale), jx_0..K-1, jy_0..K-1, ja_0..K-1]
IX, IY, IA, IS = range(4)
J0 = 4
NDIM = 4 + 3 * JELLY_KNOTS


def sensitivity_to_strength(sensitivity):
    """Map the 0-100 'sensibilidade' slider to a correction fraction."""
    return float(np.clip(sensitivity, 0, 100) / 100.0)


def _hat(y, knots):
    """Piecewise-linear (hat) basis evaluated at y. Returns len(y) x K."""
    step = knots[1] - knots[0]
    return np.clip(1.0 - np.abs(y[:, None] - knots[None, :]) / step, 0.0, 1.0)


class VideoStabilizer:
    def __init__(self, frame_w, frame_h, fps=30,
                 strength=DEFAULT_STRENGTH,
                 smoothing_s=DEFAULT_SMOOTHING_S,
                 max_keep_ratio=DEFAULT_MAX_KEEP,
                 min_keep_ratio=DEFAULT_MIN_KEEP,
                 detect_intentional_motion=True,
                 stabilize_rotation=True,
                 stabilize_zoom=False,
                 jelly_strength=DEFAULT_JELLY_STRENGTH,
                 adaptive_crop=False,
                 lookahead_frames=DEFAULT_LOOKAHEAD,
                 tracking_width=DEFAULT_TRACKING_WIDTH,
                 motion_window_s=DEFAULT_MOTION_WINDOW_S,
                 **_ignored):
        self.frame_w, self.frame_h, self.fps = frame_w, frame_h, fps

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._prev_gray = None
        self._prev_pts = None
        self._tracking_width = None
        self._set_tracking_width(tracking_width)
        self._knots_full = np.linspace(0, frame_h - 1, JELLY_KNOTS)  # full-res px
        k = JELLY_KNOTS
        d2 = np.zeros((k - 2, k))
        for i in range(k - 2):
            d2[i, i:i + 3] = (1, -2, 1)
        self._d2 = d2

        # remap maps are built per output row: map = a(y)*x + b(y)
        self._xrow = np.stack([np.arange(frame_w, dtype=np.float32),
                               np.ones(frame_w, np.float32)])
        self._rows = np.arange(frame_h, dtype=np.float64)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._hat_rows = _hat(self._rows, self._knots_full)
        w, h = frame_w, frame_h
        t = np.linspace(0, 1, 17)
        self._border = np.concatenate([
            np.stack([t * (w - 1), np.zeros_like(t)], 1), np.stack([t * (w - 1), np.full_like(t, h - 1)], 1),
            np.stack([np.zeros_like(t), t * (h - 1)], 1), np.stack([np.full_like(t, w - 1), t * (h - 1)], 1)])

        self._motion_window = deque(maxlen=max(3, round(fps * motion_window_s)))
        self._need_window = deque(maxlen=max(3, round(fps * CROP_WINDOW_S)))

        self._traj = np.zeros(NDIM)
        self._baseline = None
        self._frames = deque()
        self._traj_hist = deque()
        self._frame_index = -1
        self._applied = np.zeros(NDIM)
        self._applied_keep = 1.0

        # Live-readable state for the UI / debug overlay.
        self.last_motion = (0.0, 0.0)
        self.last_correction = (0.0, 0.0)
        self.last_rotation_deg = 0.0
        self.last_jelly_px = 0.0
        self.intentional_ratio = 0.0
        self.tracked_points = 0
        self.output_index = 0          # index of the input frame just returned
        self.crop_ratio_now = min_keep_ratio

        self.lookahead = 0
        self.set_tuning(strength=strength, smoothing_s=smoothing_s,
                        max_keep_ratio=max_keep_ratio, min_keep_ratio=min_keep_ratio,
                        detect_intentional_motion=detect_intentional_motion,
                        stabilize_rotation=stabilize_rotation, stabilize_zoom=stabilize_zoom,
                        jelly_strength=jelly_strength, adaptive_crop=adaptive_crop,
                        lookahead_frames=lookahead_frames)
        self.crop_ratio_now = self._min_keep if not self.adaptive_crop else self._max_keep

    # ------------------------------------------------------------------ tuning
    def _set_tracking_width(self, tracking_width):
        tracking_width = int(min(max(tracking_width, 160), self.frame_w))
        if tracking_width == self._tracking_width:
            return
        self._tracking_width = tracking_width
        self._track_scale = tracking_width / self.frame_w
        self._track_size = (tracking_width, round(self.frame_h * self._track_scale))
        self._knots_track = np.linspace(0, self._track_size[1] - 1, JELLY_KNOTS)
        # restart tracking at the new resolution (the path itself is kept)
        self._prev_gray = None
        self._prev_pts = None

    def set_tuning(self, strength=None, smoothing_s=None, max_keep_ratio=None,
                   min_keep_ratio=None, detect_intentional_motion=None,
                   stabilize_rotation=None, stabilize_zoom=None, jelly_strength=None,
                   adaptive_crop=None, lookahead_frames=None, tracking_width=None, **_ignored):
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
        if jelly_strength is not None:
            self.jelly_strength = float(np.clip(jelly_strength, 0.0, 1.0))
        if adaptive_crop is not None:
            self.adaptive_crop = bool(adaptive_crop)
        if lookahead_frames is not None:
            self.lookahead = int(np.clip(lookahead_frames, 0, 30))
        if tracking_width is not None:
            self._set_tracking_width(tracking_width)
        self._min_keep = min(self._min_keep, self._max_keep)

    def audio_delay_ms(self, processing_ms=0.0):
        """Audio delay to set in OBS so voice matches the stabilized picture:
        the look-ahead frames plus the time spent processing one frame."""
        return self.lookahead * 1000.0 / self.fps + processing_ms

    # ------------------------------------------------------- motion estimation
    def _prepare(self, frame):
        small = cv2.resize(frame, self._track_size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gray)

    def _detect(self, gray, existing):
        """Top up features so they stay spread over a grid."""
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

    def _fit_model(self, a, b, w, h):
        """Robust fit of dx = (s-1)x + fx(y), dy = theta(y)*x + gy(y).
        Returns (params, weights) or (None, None)."""
        d = b - a
        xc = a[:, 0] - w / 2
        H = _hat(a[:, 1], self._knots_track)
        n, k = len(a), JELLY_KNOTS

        # Coarse outlier pre-filter: per height band, drop points far from the
        # band median (a person walking by is a minority in its band).
        band = np.argmax(H, axis=1)
        wts = np.ones(n)
        for j in range(k):
            sel = band == j
            if sel.sum() >= 3:
                med = np.median(d[sel], axis=0)
                wts[sel] = (np.linalg.norm(d[sel] - med, axis=1) < 4 * TUKEY_C).astype(float)

        # params: [s-1, fx(K), gy(K), theta(K)]; theta columns scaled by
        # half the width so every block is in comparable pixel units.
        xs = w / 2
        P = 1 + 3 * k
        A = np.zeros((2 * n, P))
        A[:n, 0] = xc
        A[:n, 1:1 + k] = H
        A[n:, 1 + k:1 + 2 * k] = H
        A[n:, 1 + 2 * k:] = H * (xc / xs)[:, None]
        rhs = np.concatenate([d[:, 0], d[:, 1]])
        reg = np.zeros((3 * (k - 2) + P, P))
        for j in range(3):
            reg[j * (k - 2):(j + 1) * (k - 2), 1 + j * k:1 + (j + 1) * k] = self._d2 * JELLY_SMOOTH
        reg[3 * (k - 2):] = np.eye(P) * 1e-3
        RtR = reg.T @ reg

        params = None
        for _ in range(5):
            if (wts > 0).sum() < MIN_INLIERS:
                return None, None, None
            ww = np.concatenate([wts, wts])
            Aw = A * ww[:, None]
            params = np.linalg.solve(A.T @ Aw + RtR, Aw.T @ rhs)
            r = (A @ params - rhs).reshape(2, n)
            u = np.linalg.norm(r, axis=0) / TUKEY_C
            wts = np.where(u < 1, (1 - u ** 2) ** 2, 0.0) * (wts > 0)
        params = params.copy()
        params[1 + 2 * k:] /= xs
        return params, wts, None

    def _estimate_motion(self, prev_gray, gray):
        """Return the inter-frame path increment (NDIM), full-res units."""
        if self._prev_pts is None or len(self._prev_pts) < MIN_TRACKS:
            self._prev_pts = self._detect(prev_gray, self._prev_pts)
        p0 = self._prev_pts
        if p0 is None or len(p0) < MIN_INLIERS:
            self._prev_pts = None
            return np.zeros(NDIM)

        p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **FLOW_PARAMS)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **FLOW_PARAMS)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < FB_MAX_ERR)
        h, w = gray.shape
        q = p1.reshape(-1, 2)
        ok &= (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        a = p0[ok].reshape(-1, 2).astype(np.float64)
        b = p1[ok].reshape(-1, 2).astype(np.float64)
        self.tracked_points = int(len(a))
        if len(a) < MIN_INLIERS:
            self._prev_pts = None
            return np.zeros(NDIM)

        params, wts, _ = self._fit_model(a, b, w, h)
        if params is None:
            self._prev_pts = None
            return np.zeros(NDIM)
        inl = wts > 0
        self.tracked_points = int(inl.sum())
        self._prev_pts = b[inl].astype(np.float32).reshape(-1, 1, 2)

        k = JELLY_KNOTS
        sm1 = params[0]
        fx, gy, th = params[1:1 + k], params[1 + k:1 + 2 * k], params[1 + 2 * k:]
        yk = self._knots_track - h / 2
        fx_c = np.interp(h / 2, self._knots_track, fx)
        gy_c = np.interp(h / 2, self._knots_track, gy)
        th_c = np.interp(h / 2, self._knots_track, th)
        # Rigid part over height: dx = tx - theta(y)*y,  dy = ty + (s-1)*y
        jx = fx - fx_c + th * yk
        jy = gy - gy_c - sm1 * yk
        s = 1.0 / self._track_scale
        out = np.zeros(NDIM)
        out[IX], out[IY] = fx_c * s, gy_c * s
        out[IA] = th_c
        out[IS] = np.log(max(1.0 + sm1, 1e-6))
        out[J0:J0 + k] = jx * s
        out[J0 + k:J0 + 2 * k] = jy * s
        out[J0 + 2 * k:] = th - th_c
        return out

    def _intentional_motion_ratio(self):
        """0 = pure back-and-forth vibration, 1 = clean sustained pan/tilt."""
        if not self.detect_intentional_motion or len(self._motion_window) < 3:
            return 0.0
        deltas = np.array(self._motion_window)
        net = np.linalg.norm(deltas.sum(axis=0))
        total = np.sum(np.linalg.norm(deltas, axis=1)) + 1e-6
        if total / len(deltas) < 0.3:
            return 0.0
        return float(np.clip((net / total - 0.25) / 0.6, 0.0, 1.0))

    # --------------------------------------------------------------- smoothing
    def _gaussian_target(self, center_idx):
        hist = np.array(self._traj_hist)
        radius = self.lookahead
        if radius == 0:
            return hist[center_idx]
        sigma = max(radius / 2.0, 0.8)
        idx = np.arange(max(0, center_idx - radius), min(len(hist), center_idx + radius + 1))
        wts = np.exp(-0.5 * ((idx - center_idx) / sigma) ** 2)
        return (hist[idx] * wts[:, None]).sum(axis=0) / wts.sum()

    # -------------------------------------------------------------------- warp
    def _similarity(self, corr, keep):
        """Forward matrix input->output (correction + crop zoom about centre)."""
        cx, cy = self.frame_w / 2, self.frame_h / 2
        s = np.exp(corr[IS]) / keep
        ca, sa = np.cos(corr[IA]) * s, np.sin(corr[IA]) * s
        return np.array([[ca, -sa, cx - ca * cx + sa * cy + corr[IX] / keep],
                         [sa, ca, cy - sa * cx - ca * cy + corr[IY] / keep]], np.float64)

    def _jelly_px(self, corr):
        k = JELLY_KNOTS
        rot = np.abs(corr[J0 + 2 * k:]).max() * np.hypot(self.frame_w, self.frame_h) / 2
        return max(float(np.abs(corr[J0:J0 + 2 * k]).max()), float(rot))

    def _has_jelly(self, corr):
        return self._jelly_px(corr) > 0.05

    def map_output_to_input(self, pts, corr=None, keep=None):
        """Where in the input frame each output pixel is sampled from."""
        corr = self._applied if corr is None else corr
        keep = self._applied_keep if keep is None else keep
        inv = cv2.invertAffineTransform(self._similarity(corr, keep))
        p = pts @ inv[:, :2].T + inv[:, 2]
        if self._has_jelly(corr):
            hy = _hat(np.clip(p[:, 1], 0, self.frame_h - 1), self._knots_full)
            k = JELLY_KNOTS
            cx_, cy_, ca_ = hy @ corr[J0:J0 + k], hy @ corr[J0 + k:J0 + 2 * k], hy @ corr[J0 + 2 * k:]
            xc, yc = p[:, 0] - self.frame_w / 2, p[:, 1] - self.frame_h / 2
            # correction profile = -(row distortion): sample where the row
            # really is (its own shift + its own small rotation about centre)
            p = p - np.stack([cx_ - ca_ * yc, cy_ + ca_ * xc], axis=1)
        return p

    def _fits(self, corr, keep):
        src = self.map_output_to_input(self._border, corr, keep)
        w, h = self.frame_w, self.frame_h
        return bool((src[:, 0] >= -0.5).all() and (src[:, 0] <= w - 0.5).all()
                    and (src[:, 1] >= -0.5).all() and (src[:, 1] <= h - 0.5).all())

    def _limit(self, corr, keep):
        """Scale the correction down (binary search) until no border shows."""
        if self._fits(corr, keep):
            return corr
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid = (lo + hi) / 2
            if self._fits(corr * mid, keep):
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
        need = (np.abs(corr[:2]) + rot_px + self._jelly_px(corr)) * CROP_SAFETY_MARGIN
        self._need_window.append(need)
        need = np.max(self._need_window, axis=0)
        required = np.min(1 - 2 * need / np.array([self.frame_w, self.frame_h]))
        target = float(np.clip(required, self._min_keep, self._max_keep))
        step = CROP_SLEW_STEP * (4 if target < self.crop_ratio_now else 1)
        delta = np.clip(target - self.crop_ratio_now, -step, step)
        self.crop_ratio_now = float(np.clip(self.crop_ratio_now + delta,
                                            self._min_keep, self._max_keep))

    def _warp(self, frame, corr, keep):
        if not self._has_jelly(corr):
            return cv2.warpAffine(frame, self._similarity(corr, keep), (self.frame_w, self.frame_h),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # For output row y, p = inv @ (x, y, 1) is affine in x. The jelly
        # profile is looked up at the row's centre (it varies over ~150px, the
        # tilt across one row is a few px), so each map row is a(y)*x + b(y),
        # built for the whole frame with one small matrix product per map.
        inv = cv2.invertAffineTransform(self._similarity(corr, keep))
        y = self._rows
        py_mid = inv[1, 0] * (self.frame_w / 2) + inv[1, 1] * y + inv[1, 2]
        hy = _hat(np.clip(py_mid, 0, self.frame_h - 1), self._knots_full)
        k = JELLY_KNOTS
        cx_, cy_, ca_ = hy @ corr[J0:J0 + k], hy @ corr[J0 + k:J0 + 2 * k], hy @ corr[J0 + 2 * k:]
        hw, hh = self.frame_w / 2, self.frame_h / 2
        # px = i00 x + (i01 y + i02);  py = i10 x + (i11 y + i12)
        bx = inv[0, 1] * y + inv[0, 2]
        by = inv[1, 1] * y + inv[1, 2]
        # src = p - (c_x - c_a*(py - hh), c_y + c_a*(px - hw))
        ax_ = inv[0, 0] + ca_ * inv[1, 0]
        bx_ = bx - cx_ + ca_ * (by - hh)
        ay_ = inv[1, 0] - ca_ * inv[0, 0]
        by_ = by - cy_ - ca_ * (bx - hw)
        cx_coef = np.stack([ax_, bx_], 1).astype(np.float32)
        cy_coef = np.stack([ay_, by_], 1).astype(np.float32)
        fut = self._pool.submit(np.matmul, cy_coef, self._xrow)  # both maps in parallel
        mx = cx_coef @ self._xrow
        my = fut.result()
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ----------------------------------------------------------------- process
    def process_frame(self, frame):
        self._frame_index += 1
        gray = self._prepare(frame)
        motion = (self._estimate_motion(self._prev_gray, gray)
                  if self._prev_gray is not None else np.zeros(NDIM))
        self._prev_gray = gray

        self._motion_window.append(motion[:2].copy())
        self.last_motion = (float(motion[IX]), float(motion[IY]))
        self._traj = self._traj + motion
        self._traj_hist.append(self._traj.copy())
        self._frames.append((self._frame_index, frame))

        while len(self._frames) > self.lookahead + 1:
            self._frames.popleft()
        while len(self._traj_hist) > 2 * self.lookahead + 1:
            self._traj_hist.popleft()

        self.output_index, out_frame = self._frames[0]
        out_idx = len(self._traj_hist) - len(self._frames)
        out_traj = self._traj_hist[out_idx]
        target = self._gaussian_target(out_idx)

        self.intentional_ratio = self._intentional_motion_ratio()
        tau = (self.smoothing_s * (1 - self.intentional_ratio)
               + INTENTIONAL_SMOOTHING_S * self.intentional_ratio)
        tau_vec = np.full(NDIM, tau)
        tau_vec[J0:] = min(JELLY_SMOOTHING_S, self.smoothing_s)
        alpha = 1.0 - np.exp(-1.0 / np.maximum(self.fps * tau_vec, 1e-6))
        if self._baseline is None:
            self._baseline = target.copy()
        self._baseline += alpha * (target - self._baseline)

        gain = np.full(NDIM, self.strength)
        gain[J0:] = self.jelly_strength
        if not self.stabilize_rotation:
            gain[IA] = 0.0
        if not self.stabilize_zoom:
            gain[IS] = 0.0
        corr = gain * (self._baseline - out_traj)

        self._update_crop(corr)
        keep = self.crop_ratio_now
        applied = self._limit(corr, keep)

        # Anti-windup: never park the picture against the crop edge.
        if not np.allclose(applied, corr):
            g = gain > 0
            self._baseline[g] = out_traj[g] + applied[g] / gain[g]

        self._applied, self._applied_keep = applied, keep
        self.last_correction = (float(applied[IX]), float(applied[IY]))
        self.last_rotation_deg = float(np.degrees(applied[IA]))
        self.last_jelly_px = self._jelly_px(applied)
        return self._warp(out_frame, applied, keep)
