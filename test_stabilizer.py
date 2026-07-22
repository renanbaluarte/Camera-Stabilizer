"""Self-check: synthetic jitter in, verify the stabilizer damps it out.

Run directly: .venv/bin/python test_stabilizer.py
"""
import time

import cv2
import numpy as np

from src.stabilizer import VideoStabilizer

W, H = 640, 360


def _make_textured_frame():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (H + 60, W + 60, 3), dtype=np.uint8)
    for _ in range(40):
        x, y = rng.integers(0, W), rng.integers(0, H + 60)
        r = rng.integers(10, 40)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(frame, (x, y), r, color, -1)
    return frame


def _shaky_frames(n=60, amplitude=15):
    base = _make_textured_frame()
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(n):
        dx, dy = rng.integers(-amplitude, amplitude + 1, 2)
        ox, oy = 30 + dx, 30 + dy
        frames.append(base[oy:oy + H, ox:ox + W].copy())
    return frames


def _residual_motion(frames):
    stab = VideoStabilizer(W, H)
    out = [stab.process_frame(f) for f in frames]
    shifts = []
    prev_gray = None
    for f in out:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            shift, _ = cv2.phaseCorrelate(prev_gray.astype(np.float32), gray.astype(np.float32))
            shifts.append(shift)
        prev_gray = gray
    return np.array(shifts)


def test_stabilization_reduces_jitter():
    raw_frames = _shaky_frames()
    raw_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in raw_frames]
    raw_shifts = np.array([
        cv2.phaseCorrelate(raw_gray[i - 1], raw_gray[i])[0] for i in range(1, len(raw_gray))
    ])
    stabilized_shifts = _residual_motion(raw_frames)

    raw_jitter = np.std(raw_shifts)
    stabilized_jitter = np.std(stabilized_shifts)
    assert stabilized_jitter < raw_jitter * 0.5, (
        f"expected stabilized jitter well below raw jitter, got raw={raw_jitter:.2f} "
        f"stabilized={stabilized_jitter:.2f}"
    )


def _panning_frames(n=60, speed=2.0):
    """Slow sustained pan (consistent direction), unlike the random jitter above."""
    rng = np.random.default_rng(2)
    pano_w = W + int(speed * n) + 40
    pano = rng.integers(0, 255, (H + 40, pano_w, 3), dtype=np.uint8)
    for _ in range(80):
        x, y = rng.integers(0, pano_w), rng.integers(0, H + 40)
        r = rng.integers(10, 40)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(pano, (x, y), r, color, -1)
    return [pano[20:20 + H, 20 + int(i * speed):20 + int(i * speed) + W].copy() for i in range(n)]


def test_intentional_pan_is_not_fought():
    speed = 2.0  # px/frame, well within PTZ range, well below vibration frequency
    frames = _panning_frames(speed=speed)
    raw_total = speed * (len(frames) - 1)

    stab = VideoStabilizer(W, H)
    out = [stab.process_frame(f) for f in frames]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in out]
    shifts = [cv2.phaseCorrelate(grays[i - 1], grays[i])[0] for i in range(1, len(grays))]
    stabilized_total = abs(sum(s[0] for s in shifts))

    assert stabilized_total > raw_total * 0.6, (
        f"pan intencional sendo suprimido: esperava >= {raw_total * 0.6:.1f}px de deslocamento "
        f"na saida, obteve {stabilized_total:.1f}px"
    )


def test_processing_is_within_frame_budget():
    """Requisito: < 33ms/frame (30fps) para nao acumular atraso nem perder frames."""
    w, h = 1920, 1080
    stab = VideoStabilizer(w, h)
    frame = np.random.default_rng(3).integers(0, 255, (h, w, 3), dtype=np.uint8)

    for _ in range(5):
        stab.process_frame(frame)  # aquecimento (aloca buffers, JIT interno do opencv)

    times_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        stab.process_frame(frame)
        times_ms.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    assert avg_ms < 33, f"processamento medio {avg_ms:.1f}ms excede o orcamento de 33ms/frame (30fps)"


if __name__ == "__main__":
    test_stabilization_reduces_jitter()
    test_intentional_pan_is_not_fought()
    test_processing_is_within_frame_budget()
    print("OK")
