"""Preview lado a lado: original vs estabilizado.

Uso:
  .venv/bin/python demo.py 0          # webcam (index 0)
  .venv/bin/python demo.py clipe.mp4  # arquivo de video

Le os parametros de estabilizacao de config/default_settings.json (edite o
arquivo para ajustar intensidade: kalman_q mais baixo = suavizacao mais forte).

Pressione 'q' para sair.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.stabilizer import VideoStabilizer

CONFIG_PATH = Path(__file__).parent / "config" / "default_settings.json"


def load_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def main(source):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        sys.exit(f"nao consegui abrir: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    config = load_config()
    print(f"config: {config or 'padrao (sem config/default_settings.json)'}")
    stab = VideoStabilizer(w, h, **config)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stabilized = stab.process_frame(frame)
        side_by_side = np.hstack([frame, stabilized])
        cv2.imshow("original | estabilizado", side_by_side)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "0")
