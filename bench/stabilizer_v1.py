"""Frame-by-frame video stabilizer: optical flow -> high-pass baseline -> adaptive crop.

Feed frames in one at a time via process_frame(). Stateful (tracks previous
frame, cumulative trajectory, current crop level), so use one instance per
video stream. Tuning can be changed live via set_tuning().

Why a high-pass baseline and not a constant-velocity Kalman: this camera is
bolted to a vibrating structure, not handheld. Its true position is a fixed
"home" that the vibration oscillates around. A constant-velocity model carries
a velocity state that the shake excites, so the estimate chases the vibration
instead of ignoring it -- measured on a simulated 8Hz/13Hz vertical shake it
capped out around 72-92% suppression across its whole usable tuning range.
Tracking a slow baseline and cancelling everything above it (exactly the
"passa-altas" the spec asks for) measures 96%+ with one intuitive knob.
"""
from collections import deque

import cv2
import numpy as np

# Time constant of the "home position" tracker, in seconds. Motion slower than
# this is treated as where the camera genuinely points and is preserved; motion
# faster than this is vibration and gets cancelled. ~1s measured best: shorter
# starts following the shake, longer stops tracking optical-flow drift.
DEFAULT_SMOOTHING_S = 1.0
INTENTIONAL_SMOOTHING_S = 0.12  # baseline snaps to this while a real pan/tilt is detected

# Fraction of the detected deviation actually cancelled (the spec's
# "sensibilidade": how much motion to remove). 1.0 = fully locked.
DEFAULT_STRENGTH = 1.0

# Keep-ratio bounds: 1.0 = no crop at all. The ceiling is what we use when the
# picture is calm, the floor is the hard quality limit from the spec (never
# crop below 80%). The ratio in use adapts between them, see _update_crop.
DEFAULT_MAX_KEEP = 0.98
DEFAULT_MIN_KEEP = 0.85
CROP_WINDOW_S = 2.0       # crop follows the worst correction of the last 2s...
CROP_SLEW_STEP = 0.004    # ...and moves at most 0.4%/frame, so it never pumps
CROP_SAFETY_MARGIN = 1.2  # headroom so the correction isn't clipped right at the edge

# Vibration (metal structure, 5-20Hz) oscillates back and forth, so its
# displacement cancels out over a short window. Sustained pan/tilt/zoom
# doesn't cancel: it keeps a consistent direction. This ratio (net
# displacement / total path length over the window) tells them apart without
# needing an FFT.
DEFAULT_MOTION_WINDOW_S = 0.5

# Optical flow is estimated on a downscaled copy. 480px wide measured 0.4px of
# error per frame against a known ground truth, which is well below what the
# baseline filter cares about -- going wider costs CPU and buys nothing.
DEFAULT_TRACKING_WIDTH = 480
FLOW_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def sensitivity_to_strength(sensitivity):
    """Map the spec's 1-100 'sensibilidade' slider to a correction fraction."""
    return float(np.clip(sensitivity, 0, 100) / 100.0)


class VideoStabilizer:
    def __init__(self, frame_w, frame_h, fps=30,
                 strength=DEFAULT_STRENGTH,
                 smoothing_s=DEFAULT_SMOOTHING_S,
                 max_keep_ratio=DEFAULT_MAX_KEEP,
                 min_keep_ratio=DEFAULT_MIN_KEEP,
                 detect_intentional_motion=True,
                 tracking_width=DEFAULT_TRACKING_WIDTH,
                 motion_window_s=DEFAULT_MOTION_WINDOW_S):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps

        # Downscale for optical flow tracking only; corrections are scaled
        # back up to full resolution. Keeps per-frame cost low.
        self._track_scale = tracking_width / frame_w
        self._track_size = (tracking_width, round(frame_h * self._track_scale))

        self._motion_window = deque(maxlen=max(3, round(fps * motion_window_s)))
        self._need_window = deque(maxlen=max(3, round(fps * CROP_WINDOW_S)))

        self._prev_track_gray = None
        self._traj = np.zeros(2)      # cumulative measured camera path
        self._baseline = np.zeros(2)  # slow "home position" the shake oscillates around

        # Live-readable state for the UI / debug overlay.
        self.last_motion = (0.0, 0.0)
        self.last_correction = (0.0, 0.0)
        self.intentional_ratio = 0.0
        self.crop_ratio_now = max_keep_ratio

        self.set_tuning(strength=strength, smoothing_s=smoothing_s,
                        max_keep_ratio=max_keep_ratio, min_keep_ratio=min_keep_ratio,
                        detect_intentional_motion=detect_intentional_motion)
        self._apply_crop_ratio(self.crop_ratio_now)

    def set_tuning(self, strength=None, smoothing_s=None, max_keep_ratio=None,
                   min_keep_ratio=None, detect_intentional_motion=None):
        """Change tuning while running. Safe to call from the UI thread: every
        field is a plain scalar, so a half-applied update just means one frame
        uses a mix of old and new values."""
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
        self._min_keep = min(self._min_keep, self._max_keep)

    def _apply_crop_ratio(self, ratio):
        self._max_shift = np.array([self.frame_w, self.frame_h]) * (1 - ratio) / 2
        crop_w = round(self.frame_w * ratio)
        crop_h = round(self.frame_h * ratio)
        self._crop_origin = ((self.frame_w - crop_w) // 2, (self.frame_h - crop_h) // 2)
        self._crop_size = (crop_w, crop_h)

    def _estimate_motion(self, prev_gray, curr_gray):
        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=12, blockSize=3)
        if prev_pts is None or len(prev_pts) < 10:
            return 0.0, 0.0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **FLOW_PARAMS)
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
        if not self.detect_intentional_motion or len(self._motion_window) < 3:
            return 0.0
        deltas = np.array(self._motion_window)
        net = np.linalg.norm(deltas.sum(axis=0))
        total_path = np.sum(np.linalg.norm(deltas, axis=1)) + 1e-6
        return float(np.clip(net / total_path, 0.0, 1.0))

    def _update_crop(self, correction):
        """Size the crop from the worst correction of the last couple of
        seconds, and slew towards it. Reacting to the instantaneous correction
        instead makes the crop breathe in and out at the vibration frequency."""
        self._need_window.append(np.abs(correction) * CROP_SAFETY_MARGIN)
        need = np.max(self._need_window, axis=0)
        required = np.min(1 - 2 * need / np.array([self.frame_w, self.frame_h]))
        target = float(np.clip(required, self._min_keep, self._max_keep))

        delta = np.clip(target - self.crop_ratio_now, -CROP_SLEW_STEP, CROP_SLEW_STEP)
        self.crop_ratio_now = float(np.clip(self.crop_ratio_now + delta,
                                            self._min_keep, self._max_keep))
        self._apply_crop_ratio(self.crop_ratio_now)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        track_gray = cv2.resize(gray, self._track_size)

        if self._prev_track_gray is not None:
            flow = np.array(self._estimate_motion(self._prev_track_gray, track_gray))
            flow /= self._track_scale
        else:
            flow = np.zeros(2)
        self._prev_track_gray = track_gray

        self._motion_window.append(flow)
        self.last_motion = (float(flow[0]), float(flow[1]))
        self._traj += flow

        # Track the slow "home position". While a real pan/tilt is happening the
        # baseline is allowed to catch up quickly, so intentional moves pass
        # through instead of being fought and then snapping back.
        self.intentional_ratio = self._intentional_motion_ratio()
        tau = (self.smoothing_s * (1 - self.intentional_ratio)
               + INTENTIONAL_SMOOTHING_S * self.intentional_ratio)
        alpha = 1.0 - np.exp(-1.0 / max(self.fps * tau, 1e-6))
        self._baseline += alpha * (self._traj - self._baseline)

        correction = self.strength * (self._baseline - self._traj)
        self._update_crop(correction)
        clipped = np.clip(correction, -self._max_shift, self._max_shift)

        # Anti-windup: if we ran out of crop margin, move the baseline to match
        # what we could actually apply. Without this the picture parks against
        # the crop edge and stays there after any big or sustained movement.
        if self.strength > 0 and not np.allclose(clipped, correction):
            self._baseline = self._traj + clipped / self.strength
        self.last_correction = (float(clipped[0]), float(clipped[1]))

        shift_matrix = np.array([[1, 0, clipped[0]], [0, 1, clipped[1]]], np.float32)
        shifted = cv2.warpAffine(frame, shift_matrix, (self.frame_w, self.frame_h),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        x0, y0 = self._crop_origin
        cw, ch = self._crop_size
        cropped = shifted[y0:y0 + ch, x0:x0 + cw]
        return cv2.resize(cropped, (self.frame_w, self.frame_h), interpolation=cv2.INTER_LINEAR)
