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


def _motion(t, amp, rot_deg, rng_phases):
    """Movimento continuo da camera no tempo t (s): dx, dy (px), angulo (graus).
    Senoides fixas (8/13Hz vertical, 7Hz horizontal, 6Hz rotacao) mais
    componentes 'aleatorias' de 9-18Hz com fase sorteada."""
    p = rng_phases
    dy = (amp * np.sin(2 * np.pi * 8 * t) + 0.4 * amp * np.sin(2 * np.pi * 13 * t + 1.1)
          + 0.2 * amp * np.sin(2 * np.pi * 9.7 * t + p[0]) + 0.15 * amp * np.sin(2 * np.pi * 16.3 * t + p[1]))
    dx = (0.3 * amp * np.sin(2 * np.pi * 7 * t + 0.4) + 0.12 * amp * np.sin(2 * np.pi * 11.1 * t + p[2])
          + 0.08 * amp * np.sin(2 * np.pi * 17.9 * t + p[3]))
    da = rot_deg * np.sin(2 * np.pi * 6 * t + 0.7) + 0.3 * rot_deg * np.sin(2 * np.pi * 12.4 * t + p[4])
    # balanco lento da estrutura (1.5-2.5Hz): o mais dificil de separar de
    # um movimento intencional
    dy = dy + 0.35 * amp * np.sin(2 * np.pi * 2.3 * t + p[0] * 0.5)
    dx = dx + 0.25 * amp * np.sin(2 * np.pi * 1.7 * t + p[2] * 0.5)
    return dx, dy, da


class Clip(list):
    """Lista de frames que tambem sabe a verdade: de onde cada pixel veio na cena."""

    def truth(self, i, pts):
        """Pixel (x, y) do frame de entrada i -> coordenada na cena."""
        t = i / FPS + self.readout / FPS * (pts[:, 1] / H)
        dx, dy, da = _motion(t, self.amp, self.rot_deg, self.phases)
        a = np.radians(da)
        u = pts[:, 0] + 100 - dx - self.cx
        v = pts[:, 1] + 100 - dy - self.cy
        return np.stack([np.cos(a) * u - np.sin(a) * v + self.cx,
                         np.sin(a) * u + np.cos(a) * v + self.cy], axis=1)


def make_frames(n=240, amp=12.0, rot_deg=0.35, noise=5.0, seed=0, readout=0.0):
    """readout: tempo de leitura do sensor como fracao do periodo do frame
    (0 = obturador global; 0.5-0.9 = rolling shutter tipico de CMOS, que
    causa o efeito 'geleia' quando a camera vibra)."""
    rng = np.random.default_rng(seed)
    scene = make_scene(rng)
    clip = Clip()
    clip.amp, clip.rot_deg, clip.readout = amp, rot_deg, readout
    clip.phases = rng.uniform(0, 2 * np.pi, 5)
    clip.cx, clip.cy = (W + 200) / 2, (H + 200) / 2
    X, Y = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    for i in range(n):
        m = clip.truth(i, pts).astype(np.float32)
        f = cv2.remap(scene, m[:, 0].reshape(H, W), m[:, 1].reshape(H, W),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # pessoa andando: bloco grande, sem tremor (esta "na cena", mas se move)
        px = int(100 + i * 4) % W
        cv2.rectangle(f, (px, 250), (px + 160, 650), (60, 40, 90), -1)
        cv2.rectangle(f, (px + 40, 180), (px + 120, 250), (120, 110, 150), -1)
        f = np.clip(f.astype(np.float32) + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        clip.append(f)
    return clip


_gx, _gy = np.meshgrid(np.linspace(0.05, 0.95, 12) * W, np.linspace(0.05, 0.95, 8) * H)
GRID = np.stack([_gx.ravel(), _gy.ravel()], axis=1)


def _similarity_residual(a, b):
    """RMS do que sobra depois de explicar a->b por uma similaridade
    (translacao+rotacao+escala): a parte 'gelatinosa' do movimento."""
    A = np.zeros((2 * len(a), 4))
    A[0::2] = np.stack([a[:, 0], -a[:, 1], np.ones(len(a)), np.zeros(len(a))], 1)
    A[1::2] = np.stack([a[:, 1], a[:, 0], np.zeros(len(a)), np.ones(len(a))], 1)
    prm, *_ = np.linalg.lstsq(A, b.reshape(-1), rcond=None)
    return np.sqrt(np.mean((A @ prm - b.reshape(-1)) ** 2) * 2)


def measure(clip, mappings, warmup=40):
    """mappings[j] = (indice do frame de entrada, pontos da grade de saida no
    frame de entrada). Retorna (tremor frame-a-frame, geleia, desvio), em px
    de cena, medidos contra a verdade sintetica."""
    S = [clip.truth(i, p) for i, p in mappings[warmup:]]
    d = [np.sqrt(np.mean(np.sum((S[j] - S[j - 1]) ** 2, axis=1))) for j in range(1, len(S))]
    jel = [_similarity_residual(S[j - 1], S[j]) for j in range(1, len(S))]
    S = np.array(S)
    dev = np.sqrt(np.mean(np.sum((S - S.mean(axis=0)) ** 2, axis=2)))
    return float(np.mean(d)), float(np.mean(jel)), float(dev)


def run_raw(clip):
    return [(i, GRID) for i in range(len(clip))], None


def run_v1(clip, **kw):
    import stabilizer_v1
    stab = stabilizer_v1.VideoStabilizer(W, H, fps=FPS, **kw)
    maps = []
    t0 = time.perf_counter()
    for i, f in enumerate(clip):
        stab.process_frame(f)
        (x0, y0), (cw, ch) = stab._crop_origin, stab._crop_size
        p = np.stack([x0 + (GRID[:, 0] + 0.5) * cw / W - 0.5,
                      y0 + (GRID[:, 1] + 0.5) * ch / H - 0.5], 1) - np.array(stab.last_correction)
        maps.append((i, p))
    return maps, (time.perf_counter() - t0) * 1000 / len(clip)


def run_current(clip, **kw):
    from src.stabilizer import VideoStabilizer
    stab = VideoStabilizer(W, H, fps=FPS, **kw)
    maps = []
    t0 = time.perf_counter()
    for f in clip:
        stab.process_frame(f)
        maps.append((stab.output_index, stab.map_output_to_input(GRID)))
    return maps, (time.perf_counter() - t0) * 1000 / len(clip)


def report(name, clip, result):
    maps, ms = result
    jit, jel, dev = measure(clip, maps)
    extra = f"  {ms:5.1f} ms/frame" if ms is not None else ""
    print(f"{name:<30} tremor frame-a-frame={jit:6.2f}px  geleia={jel:5.2f}px  desvio={dev:5.2f}px{extra}")
    return jit, jel, dev


if __name__ == "__main__":
    for readout, label in ((0.0, "obturador global"), (0.7, "rolling shutter 70% (CMOS)")):
        print(f"--- {label} ---")
        clip = make_frames(readout=readout)
        report("entrada (crua)", clip, run_raw(clip))
        report("v1 (original)", clip, run_v1(clip))
        report("atual", clip, run_current(clip))
        report("atual sem correcao de geleia", clip, run_current(clip, jelly_strength=0))
