"""Le da placa de captura, estabiliza, publica numa camera virtual.

No OBS, adicione a camera virtual resultante como fonte "Dispositivo de
Captura de Video" (no Windows ela aparece via o backend OBS Virtual Camera,
entao precisa do OBS instalado).

Le os parametros de config/default_settings.json.

Uso:
  .venv/bin/python camera_to_obs.py 0            # indice da camera/placa
  .venv/bin/python camera_to_obs.py 0 --fps 30
  .venv/bin/python camera_to_obs.py 0 --width 1280 --height 720   # se a placa nao aguentar 1080p

Ctrl+C para parar. Para a interface grafica, use app.py.
"""
import argparse
import json
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat

from src.stabilizer import VideoStabilizer

CONFIG_PATH = Path(__file__).parent / "config" / "default_settings.json"
RETRY_DELAY_S = 2.0
READ_TIMEOUT_S = 3.0


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def open_capture(source, width, height, fps):
    is_index = str(source).isdigit()
    backend = cv2.CAP_DSHOW if sys.platform == "win32" and is_index else 0
    cap = cv2.VideoCapture(int(source) if is_index else source, backend)
    if not cap.isOpened():
        raise RuntimeError(f"nao consegui abrir: {source}")

    # Muitas placas de captura USB abrem num modo baixo por padrao (ex:
    # 640x480) ate serem forcadas pro modo nativo. MJPG costuma ser
    # necessario pra liberar 1080p30 dentro da banda do USB.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


class FrameReader:
    """Le a placa numa thread propria: a decodificacao MJPG roda em paralelo
    com a estabilizacao, e uma placa que trava sem dar erro deixa de
    congelar o pipeline (get() estoura o tempo e o pipeline reconecta)."""

    def __init__(self, cap, drop_late=True):
        self._cap = cap
        self._drop_late = drop_late  # camera: descarta atrasado; arquivo: espera
        self._q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                frame = None
            if self._drop_late and self._q.full():
                try:
                    self._q.get_nowait()  # atrasado: descarta o mais velho
                except queue.Empty:
                    pass
            while not self._stop.is_set():
                try:
                    self._q.put(frame, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if frame is None:
                return

    def get(self, timeout):
        """Proximo frame; None se a fonte acabou; levanta queue.Empty se travou."""
        return self._q.get(timeout=timeout)

    def close(self):
        self._stop.set()
        # se a thread estiver presa dentro de read(), release() tambem pode
        # travar: solta em segundo plano pra nao segurar a reconexao
        threading.Thread(target=self._cap.release, daemon=True).start()


def run_pipeline(source, fps=30, width=1920, height=1080, config=None,
                  frame_callback=None, stop_event=None, on_stabilizer=None, log=print):
    """Loop bloqueante: captura -> estabiliza -> publica na camera virtual.

    frame_callback(original, stabilized, stats), se fornecido, e chamado a
    cada frame (usado pela GUI pra preview + status). on_stabilizer(stab), se
    fornecido, recebe o VideoStabilizer assim que ele existe, pra quem chama
    poder ajustar a sintonia em tempo real via stab.set_tuning(). stop_event
    (threading.Event), se fornecido, encerra o loop de forma limpa quando
    setado.
    """
    cap = open_capture(source, width, height, fps)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (width, height):
        log(f"aviso: pedi {width}x{height} mas a fonte negociou {w}x{h}")
    is_camera = str(source).isdigit()
    reader = FrameReader(cap, drop_late=is_camera)
    try:
        stab = VideoStabilizer(w, h, fps=fps, **(config or {}))
        if on_stabilizer:
            on_stabilizer(stab)

        with pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=PixelFormat.BGR) as vcam:
            log(f"camera virtual ativa: {vcam.device} ({w}x{h} @ {fps}fps)")
            loop_ms = deque(maxlen=30)
            proc_ms = deque(maxlen=60)
            t_prev = time.perf_counter()
            while stop_event is None or not stop_event.is_set():
                try:
                    frame = reader.get(timeout=READ_TIMEOUT_S)
                except queue.Empty:
                    log(f"a fonte nao entrega frames ha {READ_TIMEOUT_S:g}s (travou?)")
                    break
                if frame is None:
                    break
                t0 = time.perf_counter()
                stabilized = stab.process_frame(frame)
                proc_ms.append((time.perf_counter() - t0) * 1000)
                vcam.send(stabilized)
                if not is_camera:
                    vcam.sleep_until_next_frame()  # arquivo: segura no fps; camera ja vem no ritmo

                now = time.perf_counter()
                loop_ms.append((now - t_prev) * 1000)
                t_prev = now
                if frame_callback:
                    avg_loop = sum(loop_ms) / len(loop_ms)
                    avg_proc = sum(proc_ms) / len(proc_ms)
                    stats = {
                        "fps": 1000.0 / avg_loop if avg_loop else 0.0,
                        "frame_ms": avg_proc,
                        "audio_delay_ms": stab.audio_delay_ms(avg_proc),
                        "lookahead": stab.lookahead,
                        "crop_ratio": stab.crop_ratio_now,
                        "motion": stab.last_motion,
                        "correction": stab.last_correction,
                        "intentional": stab.intentional_ratio,
                        "rotation": stab.last_rotation_deg,
                        "jelly": stab.last_jelly_px,
                        "points": stab.tracked_points,
                    }
                    frame_callback(frame, stabilized, stats)
    finally:
        reader.close()


def run_with_restart(source, retry_delay=RETRY_DELAY_S, stop_event=None, log=print, **kwargs):
    """Mantem o pipeline no ar durante o culto: se a placa desconectar ou o
    processamento estourar, reabre sozinho. So para quando stop_event for
    setado. Arquivos de video rodam uma vez so -- chegar ao fim deles e
    termino normal, nao falha.

    Cobre desconexao, excecao e placa que trava sem dar erro (a leitura roda
    numa thread propria e o pipeline desiste apos READ_TIMEOUT_S).
    """
    if not str(source).isdigit():
        run_pipeline(source, stop_event=stop_event, log=log, **kwargs)
        return

    attempt = 0
    while stop_event is None or not stop_event.is_set():
        try:
            run_pipeline(source, stop_event=stop_event, log=log, **kwargs)
            # Retorno normal de uma camera significa que os frames pararam de
            # chegar -- placa desconectada, driver reiniciado, cabo solto.
            log("a fonte parou de entregar frames (desconectou?)")
        except Exception as exc:
            log(f"erro no processamento: {exc}")

        if stop_event is not None and stop_event.is_set():
            return
        attempt += 1
        log(f"reconectando em {retry_delay:g}s (tentativa {attempt})...")
        if stop_event is not None:
            stop_event.wait(retry_delay)  # acorda na hora se mandarem parar
        else:
            time.sleep(retry_delay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="indice da camera (ex: 0) ou caminho de video")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    config = load_config()
    print(f"config: {config or 'padrao (sem config/default_settings.json)'}")
    try:
        run_with_restart(args.source, fps=args.fps, width=args.width,
                         height=args.height, config=config)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
