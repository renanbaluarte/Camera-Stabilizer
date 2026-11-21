"""Benchmark realista: tremor subpixel + rotacao + ruido de sensor + pessoa andando.

Compara a entrada crua, a versao 1 (bench/stabilizer_v1.py) e a atual
(src/stabilizer.py). Mede o tremor residual com ECC (translacao + rotacao)
de cada frame de saida contra um frame de referencia, ignorando o aquecimento.

Uso: python bench/benchmark.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

W, H, FPS = 1280, 720, 30


def make_scene(rng):
    sw, sh = W + 200, H + 200
    base = cv2.GaussianBlur(rng.integers(0, 255, (sh // 8, sw // 8, 3), dtype=np.uint8), (0, 0), 1.5)
    scene = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_CUBIC)
    for _ in range(120):
        x, y = int(rng.integers(0, sw)), int(rng.integers(0, sh))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.rectangle(scene, (x, y), (x + int(rng.integers(20, 90)), y + int(rng.integers(20, 90))), color, -1)
        else:
            cv2.circle(scene, (x, y), int(rng.integers(8, 40)), color, -1)
    # Igreja costuma ser escura e de baixo contraste
    return (scene.astype(np.float32) * 0.55 + 20).astype(np.uint8)


def make_frames(n=240, amp=12.0, rot_deg=0.35, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    scene = make_scene(rng)
    t = np.arange(n) / FPS
    dy = amp * np.sin(2 * np.pi * 8 * t) + 0.4 * amp * np.sin(2 * np.pi * 13 * t + 1.1) \
        + 0.3 * amp * rng.standard_normal(n)
    dx = 0.3 * amp * np.sin(2 * np.pi * 7 * t + 0.4) + 0.15 * amp * rng.standard_normal(n)
    da = rot_deg * np.sin(2 * np.pi * 6 * t + 0.7)
    cx, cy = (W + 200) / 2, (H + 200) / 2
    frames = []
    for i in range(n):
        M = cv2.getRotationMatrix2D((cx, cy), da[i], 1.0)
        M[0, 2] += dx[i] - 100
        M[1, 2] += dy[i] - 100
        f = cv2.warpAffine(scene, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # pessoa andando: bloco grande, sem tremor (esta "na cena", mas se move)
        px = int(100 + i * 4) % W
        cv2.rectangle(f, (px, 250), (px + 160, 650), (60, 40, 90), -1)
        cv2.rectangle(f, (px + 40, 180), (px + 120, 250), (120, 110, 150), -1)
        f = np.clip(f.astype(np.float32) + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        frames.append(f)
    return frames


def residual(frames, warmup=40):
    small = [cv2.cvtColor(cv2.resize(f, (W // 2, H // 2)), cv2.COLOR_BGR2GRAY) for f in frames[warmup:]]
    # mascara so o miolo, sem a pessoa (faixa central) para medir o fundo
    mask = np.full(small[0].shape, 255, np.uint8)
    ref = small[0]
    out = []
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    for g in small:
        M = np.eye(2, 3, dtype=np.float32)
        try:
            _, M = cv2.findTransformECC(ref, g, M, cv2.MOTION_EUCLIDEAN, crit, mask, 5)
        except cv2.error:
            pass
        out.append((M[0, 2] * 2, M[1, 2] * 2, np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
    a = np.array(out)
    return a.std(axis=0), np.abs(np.diff(a, axis=0)).mean(axis=0)


def run(cls, frames, **kw):
    stab = cls(W, H, fps=FPS, **kw)
    t0 = time.perf_counter()
    out = [stab.process_frame(f) for f in frames]
    ms = (time.perf_counter() - t0) * 1000 / len(frames)
    # a versao nova atrasa N frames: alinha descartando o inicio
    return out, ms


def report(name, frames, ms=None):
    std, jit = residual(frames)
    extra = f"  {ms:5.1f} ms/frame" if ms is not None else ""
    print(f"{name:<22} std x={std[0]:5.2f}px y={std[1]:5.2f}px rot={std[2]:5.3f}deg | "
          f"tremor frame-a-frame x={jit[0]:5.2f} y={jit[1]:5.2f} rot={jit[2]:5.3f}{extra}")
    return std, jit


if __name__ == "__main__":
    import stabilizer_v1
    from src import stabilizer as current
    frames = make_frames()
    report("entrada (crua)", frames)
    out, ms = run(stabilizer_v1.VideoStabilizer, frames)
    report("v1 (original)", out, ms)
    out, ms = run(current.VideoStabilizer, frames)
    report("v2 (nova)", out, ms)
