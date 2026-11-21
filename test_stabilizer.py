"""Self-checks: estabilizacao contra tremor sintetico, e a logica de
reconexao do pipeline.

Run directly: .venv/bin/python test_stabilizer.py
"""
import threading
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


def _vertical_shake_frames(n=150, fps=30, amp=0.02 * H):
    """Tremor vertical dominante, como a estrutura de metal da igreja:
    fundamental ~8Hz mais um harmonico ~13Hz, quase nada na horizontal.

    A amplitude e uma fracao da altura do frame (2%, ~22px em 1080p) para
    que o tremor fique dentro do orcamento de crop, como no caso real."""
    rng = np.random.default_rng(7)
    margin = 80
    scene = rng.integers(0, 255, (H + 2 * margin, W + 2 * margin, 3), dtype=np.uint8)
    for _ in range(60):
        x = int(rng.integers(0, scene.shape[1] - 120))
        y = int(rng.integers(0, scene.shape[0] - 120))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(scene, (x, y), (x + 100, y + 100), color, -1)

    t = np.arange(n) / fps
    dy = amp * np.sin(2 * np.pi * 8.0 * t) + 0.4 * amp * np.sin(2 * np.pi * 13.0 * t + 1.1)
    dx = 0.15 * amp * np.sin(2 * np.pi * 7.0 * t + 0.4)
    frames = [scene[margin + int(round(y)):margin + int(round(y)) + H,
                    margin + int(round(x)):margin + int(round(x)) + W].copy()
              for x, y in zip(dx, dy)]
    return frames, dy


def _vertical_residual(frames, warmup=30):
    """Desvio vertical de cada frame em relacao ao primeiro, em pixels."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    han = cv2.createHanningWindow((W, H), cv2.CV_32F)
    ref = grays[warmup]
    return np.array([cv2.phaseCorrelate(ref, g, han)[0][1] for g in grays[warmup:]])


def test_vertical_shake_is_strongly_suppressed():
    """O caso real reportado: tremor de cima para baixo. Antes da correcao o
    filtro de velocidade constante parava em ~72% e continuava visivel."""
    frames, dy = _vertical_shake_frames()
    raw = np.std(_vertical_residual(frames))

    stab = VideoStabilizer(W, H)
    out = [stab.process_frame(f) for f in frames]
    residual = np.std(_vertical_residual(out))

    assert residual < raw * 0.15, (
        f"tremor vertical ainda visivel: entrada std={raw:.2f}px, "
        f"saida std={residual:.2f}px (esperado abaixo de {raw * 0.15:.2f}px)"
    )


def test_tuning_changes_output_while_running():
    """Regressao dos sliders inertes: set_tuning tem de valer nos frames
    seguintes, sem reiniciar a captura. Uma sequencia continua so, com a
    mudanca no meio -- reprocessar os mesmos frames criaria um salto de cena
    que mascararia o efeito."""
    frames, _ = _vertical_shake_frames(n=300)
    half = len(frames) // 2

    stab = VideoStabilizer(W, H, strength=0.0)
    out = []
    for i, f in enumerate(frames):
        if i == half:
            stab.set_tuning(strength=1.0)
        out.append(stab.process_frame(f))

    # Cada metade medida contra a propria referencia, descartando a transicao.
    off_residual = np.std(_vertical_residual(out[:half], warmup=30))
    on_residual = np.std(_vertical_residual(out[half + 30:], warmup=30))

    assert on_residual < off_residual * 0.3, (
        f"set_tuning nao teve efeito: sem estabilizar={off_residual:.2f}px, "
        f"depois de ligar={on_residual:.2f}px"
    )


def _bench():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "bench"))
    import benchmark
    return benchmark


def test_rotation_and_noise_are_suppressed():
    """Caso realista com obturador global: tremor subpixel + rotacao + balanco
    lento + ruido de sensor + pessoa andando na frente. Medido contra a
    verdade sintetica (de onde cada pixel de saida veio na cena). A v1 deixava
    ~2px de tremor frame-a-frame aqui; exigimos menos de 3% da entrada."""
    b = _bench()
    clip = b.make_frames(n=150)
    raw, _, _ = b.measure(clip, b.run_raw(clip)[0])
    out, _, _ = b.measure(clip, b.run_current(clip)[0])
    assert out < raw * 0.03, f"tremor {out:.2f}px (entrada {raw:.2f}px)"


def test_rolling_shutter_jelly_is_removed():
    """Sensor CMOS lendo linha a linha (70% do periodo do frame): sob
    vibracao cada linha anda diferente, o efeito 'geleia'. Uma correcao so
    de frame inteiro nao tira isso (e a v2 chegava a piorar o tremor)."""
    b = _bench()
    clip = b.make_frames(n=150, readout=0.7)
    raw_jit, raw_jel, _ = b.measure(clip, b.run_raw(clip)[0])
    jit, jel, _ = b.measure(clip, b.run_current(clip)[0])
    _, jel_off, _ = b.measure(clip, b.run_current(clip, jelly_strength=0)[0])
    assert jel < raw_jel * 0.1, f"geleia {jel:.2f}px (entrada {raw_jel:.2f}px)"
    assert jel < jel_off * 0.2, f"correcao de geleia sem efeito: {jel:.2f} vs {jel_off:.2f}px"
    assert jit < raw_jit * 0.05, f"tremor {jit:.2f}px (entrada {raw_jit:.2f}px)"


def test_audio_delay_matches_lookahead():
    stab = VideoStabilizer(W, H, fps=30, lookahead_frames=8)
    assert abs(stab.audio_delay_ms() - 266.7) < 0.1
    assert abs(stab.audio_delay_ms(processing_ms=20) - 286.7) < 0.1
    frames, _ = _vertical_shake_frames(n=20)
    for f in frames:
        stab.process_frame(f)
    assert stab.output_index == len(frames) - 1 - 8, "frame devolvido nao e o de N frames atras"


def test_profiles_save_load_delete():
    """Perfis: salvar, reabrir (novo processo), lembrar o ultimo, excluir.
    Presets prontos nao podem ser sobrescritos nem excluidos."""
    import tempfile
    from pathlib import Path
    from src.profiles import PRESETS, ProfileStore
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "profiles.json"
        store = ProfileStore(path, seed_path=Path(d) / "nenhum.json")
        assert store.last == next(iter(PRESETS))
        store.save("Culto domingo", {"smoothing_s": 2.2, "crop": 87, "lookahead": 6})
        again = ProfileStore(path, seed_path=Path(d) / "nenhum.json")
        assert again.last == "Culto domingo"
        assert again.get("Culto domingo")["crop"] == 87
        assert again.get("Culto domingo")["rotation"] is True  # default preenchido
        for bad in (lambda: again.save(next(iter(PRESETS)), {}),
                    lambda: again.delete(next(iter(PRESETS)))):
            try:
                bad()
                raise AssertionError("preset foi alterado")
            except ValueError:
                pass
        again.delete("Culto domingo")
        assert "Culto domingo" not in ProfileStore(path).names()
        path.write_text("{lixo", encoding="utf-8")
        assert ProfileStore(path, seed_path=Path(d) / "nenhum.json").names() == list(PRESETS)  # corrompido nao quebra


def test_bundled_profile_is_created_once():
    """O perfil 'Igreja' vem com o programa (config/default_profiles.json),
    e criado na primeira execucao, ja selecionado, e editavel. Se o usuario
    excluir, nao volta sozinho."""
    import tempfile
    from pathlib import Path
    from src.profiles import ProfileStore
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "profiles.json"
        store = ProfileStore(path)
        assert store.last == "Igreja" and not store.is_preset("Igreja")
        assert store.get("Igreja")["lookahead"] == 8 and store.get("Igreja")["precision"] == "Alta"
        store.delete("Igreja")
        assert "Igreja" not in ProfileStore(path).names()


def _run_restart_with(fake, source, retry_delay=0.01, stop_event=None):
    """Roda run_with_restart com o pipeline trocado por um dublê."""
    import camera_to_obs
    original = camera_to_obs.run_pipeline
    camera_to_obs.run_pipeline = fake
    try:
        camera_to_obs.run_with_restart(source, retry_delay=retry_delay,
                                       stop_event=stop_event, log=lambda *_: None)
    finally:
        camera_to_obs.run_pipeline = original


def test_camera_restarts_until_told_to_stop():
    """Placa que da erro ou desconecta tem de voltar sozinha; Parar tem de
    encerrar de verdade em vez de reconectar pra sempre."""
    calls = []
    stop = threading.Event()

    def fake(source, stop_event=None, log=None, **kwargs):
        calls.append(source)
        if len(calls) == 1:
            raise RuntimeError("placa sumiu")   # falha dura
        if len(calls) == 2:
            return                              # desconexao silenciosa
        stop_event.set()                        # usuario apertou Parar

    _run_restart_with(fake, "0", stop_event=stop)
    assert len(calls) == 3, f"esperava 3 tentativas, houve {len(calls)}"
    assert stop.is_set()


def test_video_file_runs_once():
    """Arquivo que chega ao fim e termino normal, nao falha: nao pode entrar
    em loop reproduzindo pra sempre."""
    calls = []

    def fake(source, stop_event=None, log=None, **kwargs):
        calls.append(source)

    _run_restart_with(fake, "clipe.mp4")
    assert len(calls) == 1, f"arquivo reiniciou {len(calls)} vezes"


if __name__ == "__main__":
    test_stabilization_reduces_jitter()
    test_intentional_pan_is_not_fought()
    test_processing_is_within_frame_budget()
    test_vertical_shake_is_strongly_suppressed()
    test_tuning_changes_output_while_running()
    test_rotation_and_noise_are_suppressed()
    test_rolling_shutter_jelly_is_removed()
    test_audio_delay_matches_lookahead()
    test_profiles_save_load_delete()
    test_bundled_profile_is_created_once()
    test_camera_restarts_until_told_to_stop()
    test_video_file_runs_once()
    print("OK")
