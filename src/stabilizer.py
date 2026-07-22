"""Frame-by-frame video stabilizer: optical flow -> Kalman smoothing -> adaptive crop.

Feed frames in one at a time via process_frame(). Stateful (tracks previous
frame, cumulative trajectory, current crop level), so use one instance per
video stream.
"""
from collections import deque

import cv2
import numpy as np

# Tuning defaults from the church "earthquake" vibration case (spec section
# "Parametros de Tuning"). Increase kalman_q for a snappier response, decrease
# for smoother/laggier stabilization. Use sensitivity_to_kalman_q() to drive
# this from a 1-100 user-facing slider instead.
DEFAULT_KALMAN_Q = 0.001
DEFAULT_KALMAN_R = 0.1

# Crop never goes better (less cropped) than the ceiling, never worse (more
# cropped) than the floor -- floor matches the spec's hard "never below 80%"
# rule. The ratio actually used each frame adapts between them, see _update_crop.
DEFAULT_CROP_CEILING = 0.97
DEFAULT_CROP_FLOOR = 0.80
CROP_ATTACK_STEP = 0.02    # crop grows fast when motion needs more margin
CROP_RELEASE_STEP = 0.004  # crop eases back slowly once things calm down (avoids pumping)
CROP_SAFETY_MARGIN = 1.2   # headroom so the correction isn't clipped right at the edge

# Vibration (metal structure, 5-20Hz) oscillates back and forth, so its
# displacement cancels out over a short window. Sustained pan/tilt/zoom
# doesn't cancel: it keeps a consistent direction. This ratio (net
# displacement / total path length over the window) tells them apart without
# needing an FFT.
DEFAULT_MOTION_WINDOW_S = 0.5
DEFAULT_MIN_R_SCALE = 0.05  # how much to trust raw motion once it looks intentional


def sensitivity_to_kalman_q(sensitivity, q_min=1e-4, q_max=1e-1):
    """Map a 1-100 'sensibilidade' slider to Kalman process noise, log scale.
    Higher sensitivity = more motion removed = lower Q."""
    t = (np.clip(sensitivity, 1, 100) - 1) / 99.0
    log_q = np.log10(q_max) + t * (np.log10(q_min) - np.log10(q_max))
    return float(10 ** log_q)


class VideoStabilizer:
    def __init__(self, frame_w, frame_h, crop_ratio=DEFAULT_CROP_CEILING,
                 min_crop_ratio=DEFAULT_CROP_FLOOR,
                 kalman_q=DEFAULT_KALMAN_Q, kalman_r=DEFAULT_KALMAN_R,
                 tracking_width=480, fps=30,
                 motion_window_s=DEFAULT_MOTION_WINDOW_S, min_r_scale=DEFAULT_MIN_R_SCALE,
                 detect_intentional_motion=True):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self._crop_ceiling = crop_ratio
        self._crop_floor = min_crop_ratio
        self.crop_ratio_now = crop_ratio

        # Downscale for optical flow tracking only; corrections are scaled
        # back up to full resolution. Keeps per-frame cost low.
        self._track_scale = tracking_width / frame_w
        self._track_size = (tracking_width, round(frame_h * self._track_scale))

        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * kalman_q
        self._base_measurement_noise_cov = np.eye(2, dtype=np.float32) * kalman_r
        self._kf.measurementNoiseCov = self._base_measurement_noise_cov.copy()

        self._min_r_scale = min_r_scale if detect_intentional_motion else 1.0
        window_frames = max(3, round(fps * motion_window_s))
        self._motion_window = deque(maxlen=window_frames)

        self._prev_track_gray = None
        self._traj_x = 0.0
        self._traj_y = 0.0
        self.last_motion = (0.0, 0.0)

        self._max_shift_x = self._max_shift_y = 0.0
        self._crop_origin = (0, 0)
        self._crop_size = (frame_w, frame_h)
        self._apply_crop_ratio(crop_ratio)

    def _apply_crop_ratio(self, ratio):
        self._max_shift_x = self.frame_w * (1 - ratio) / 2
        self._max_shift_y = self.frame_h * (1 - ratio) / 2
        crop_w = round(self.frame_w * ratio)
        crop_h = round(self.frame_h * ratio)
        self._crop_origin = ((self.frame_w - crop_w) // 2, (self.frame_h - crop_h) // 2)
        self._crop_size = (crop_w, crop_h)

    def _update_crop(self, raw_dx, raw_dy):
        """Grow the crop margin fast when motion needs it, ease it back slowly
        once things calm down. Never leaves [floor, ceiling]."""
        needed_x = abs(raw_dx) * CROP_SAFETY_MARGIN
        needed_y = abs(raw_dy) * CROP_SAFETY_MARGIN
        required_ratio = min(
            1 - 2 * needed_x / self.frame_w,
            1 - 2 * needed_y / self.frame_h,
        )
        target = np.clip(required_ratio, self._crop_floor, self._crop_ceiling)

        if target < self.crop_ratio_now:
            self.crop_ratio_now = max(target, self.crop_ratio_now - CROP_ATTACK_STEP)
        else:
            self.crop_ratio_now = min(self._crop_ceiling, self.crop_ratio_now + CROP_RELEASE_STEP)

        self._apply_crop_ratio(self.crop_ratio_now)

    def _estimate_motion(self, prev_gray, curr_gray):
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        if prev_pts is None or len(prev_pts) < 10:
            return 0.0, 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        status = status.reshape(-1).astype(bool)
        good_prev, good_curr = prev_pts[status], curr_pts[status]
        if len(good_prev) < 10:
            return 0.0, 0.0

        # Median (not mean) so a handful of points tracked onto a moving
        # subject don't drag the global camera-motion estimate off.
        dx, dy = np.median((good_curr - good_prev).reshape(-1, 2), axis=0)
        return float(dx), float(dy)

    def _intentional_motion_ratio(self):
        """0 = pure back-and-forth vibration, 1 = clean sustained pan/tilt/zoom."""
        if len(self._motion_window) < 3:
            return 0.0
        deltas = np.array(self._motion_window)
        net = np.linalg.norm(deltas.sum(axis=0))
        total_path = np.sum(np.linalg.norm(deltas, axis=1)) + 1e-6
        return float(net / total_path)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        track_gray = cv2.resize(gray, self._track_size)

        if self._prev_track_gray is not None:
            dx, dy = self._estimate_motion(self._prev_track_gray, track_gray)
            dx, dy = dx / self._track_scale, dy / self._track_scale
        else:
            dx, dy = 0.0, 0.0
        self._prev_track_gray = track_gray
        self._motion_window.append((dx, dy))
        self.last_motion = (dx, dy)

        self._traj_x += dx
        self._traj_y += dy

        ratio = self._intentional_motion_ratio()
        r_scale = 1.0 - ratio * (1.0 - self._min_r_scale)
        self._kf.measurementNoiseCov = self._base_measurement_noise_cov * r_scale

        self._kf.predict()
        smoothed = self._kf.correct(np.array([[self._traj_x], [self._traj_y]], np.float32))
        smooth_x, smooth_y = float(smoothed[0, 0]), float(smoothed[1, 0])

        raw_correction_x = smooth_x - self._traj_x
        raw_correction_y = smooth_y - self._traj_y
        self._update_crop(raw_correction_x, raw_correction_y)

        correction_x = np.clip(raw_correction_x, -self._max_shift_x, self._max_shift_x)
        correction_y = np.clip(raw_correction_y, -self._max_shift_y, self._max_shift_y)

        shift_matrix = np.array([[1, 0, correction_x], [0, 1, correction_y]], np.float32)
        shifted = cv2.warpAffine(frame, shift_matrix, (self.frame_w, self.frame_h))

        x0, y0 = self._crop_origin
        cw, ch = self._crop_size
        cropped = shifted[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(cropped, (self.frame_w, self.frame_h))
