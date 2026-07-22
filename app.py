"""Interface simples do estabilizador: ajusta parametros, mostra preview,
controla a camera virtual que o OBS le como fonte.

Uso: python app.py  (ou o executavel gerado por build.bat)
"""
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_to_obs import run_pipeline
from src.stabilizer import sensitivity_to_kalman_q

PREVIEW_WIDTH = 480


class App:
    def __init__(self, root):
        self.root = root
        root.title("Camera Stabilizer")
        root.resizable(False, False)

        self._stop_event = threading.Event()
        self._worker = None
        self._preview_q = queue.Queue(maxsize=1)

        self._build_ui()
        self._poll_preview()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        form = ttk.Frame(self.root, padding=10)
        form.grid(row=0, column=0, sticky="nw")

        ttk.Label(form, text="Fonte (indice da camera ou arquivo)").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(form, textvariable=self.source_var, width=20).grid(row=0, column=1, sticky="w")

        ttk.Label(form, text="Resolucao").grid(row=1, column=0, sticky="w")
        res_frame = ttk.Frame(form)
        res_frame.grid(row=1, column=1, sticky="w")
        self.width_var = tk.IntVar(value=1920)
        self.height_var = tk.IntVar(value=1080)
        self.fps_var = tk.IntVar(value=30)
        ttk.Entry(res_frame, textvariable=self.width_var, width=6).pack(side="left")
        ttk.Label(res_frame, text="x").pack(side="left")
        ttk.Entry(res_frame, textvariable=self.height_var, width=6).pack(side="left")
        ttk.Label(res_frame, text="@").pack(side="left")
        ttk.Entry(res_frame, textvariable=self.fps_var, width=4).pack(side="left")
        ttk.Label(res_frame, text="fps").pack(side="left")

        ttk.Label(form, text="Sensibilidade").grid(row=2, column=0, sticky="w")
        self.sensitivity_var = tk.IntVar(value=70)
        ttk.Scale(form, from_=1, to=100, variable=self.sensitivity_var, orient="horizontal"
                  ).grid(row=2, column=1, sticky="we")

        ttk.Label(form, text="Crop maximo (%)").grid(row=3, column=0, sticky="w")
        self.crop_var = tk.IntVar(value=85)
        ttk.Scale(form, from_=80, to=95, variable=self.crop_var, orient="horizontal"
                  ).grid(row=3, column=1, sticky="we")

        self.intentional_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Detectar movimento intencional (pan/tilt/zoom)",
                         variable=self.intentional_var).grid(row=4, column=0, columnspan=2, sticky="w")

        self.overlay_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Mostrar overlay de debug",
                         variable=self.overlay_var).grid(row=5, column=0, columnspan=2, sticky="w")

        btns = ttk.Frame(form)
        btns.grid(row=6, column=0, columnspan=2, pady=(8, 0), sticky="w")
        self.start_btn = ttk.Button(btns, text="Iniciar", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Parar", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="parado")
        ttk.Label(form, textvariable=self.status_var, wraplength=260
                  ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.preview_label = ttk.Label(self.root)
        self.preview_label.grid(row=0, column=1, padx=10, pady=10)

    def _config(self):
        return {
            "kalman_q": sensitivity_to_kalman_q(self.sensitivity_var.get()),
            "min_crop_ratio": self.crop_var.get() / 100.0,
            "detect_intentional_motion": self.intentional_var.get(),
        }

    def _start(self):
        self._stop_event.clear()
        config = self._config()
        source = self.source_var.get()
        width, height, fps = self.width_var.get(), self.height_var.get(), self.fps_var.get()

        def target():
            try:
                run_pipeline(source, fps=fps, width=width, height=height, config=config,
                             frame_callback=self._on_frame, stop_event=self._stop_event,
                             log=print)
            except Exception as exc:
                self.root.after(0, lambda: self.status_var.set(f"erro: {exc}"))
            finally:
                self.root.after(0, self._on_stopped)

        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("iniciando...")

    def _stop(self):
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self.status_var.set("parando...")

    def _on_stopped(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _on_close(self):
        self._stop_event.set()
        self.root.destroy()

    def _on_frame(self, original, stabilized, stats):
        """Roda na thread de captura. So monta a preview; o frame enviado
        pra camera virtual (em run_pipeline) nunca leva o overlay."""
        preview = stabilized
        if self.overlay_var.get():
            preview = preview.copy()
            h, w = preview.shape[:2]
            cx, cy = w // 2, h // 2
            dx, dy = stats["motion"]
            cv2.arrowedLine(preview, (cx, cy), (int(cx + dx * 5), int(cy + dy * 5)),
                             (0, 0, 255), 2, tipLength=0.3)
            cv2.putText(
                preview,
                f"{stats['fps']:.0f} fps  {stats['frame_ms']:.1f}ms  crop {stats['crop_ratio'] * 100:.0f}%",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        scale = PREVIEW_WIDTH / preview.shape[1]
        thumb = cv2.resize(preview, (PREVIEW_WIDTH, round(preview.shape[0] * scale)))
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

        try:
            self._preview_q.get_nowait()
        except queue.Empty:
            pass
        self._preview_q.put_nowait((rgb, stats))

    def _poll_preview(self):
        try:
            rgb, stats = self._preview_q.get_nowait()
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview_label.configure(image=img)
            self.preview_label.image = img
            self.status_var.set(
                f"rodando - {stats['fps']:.0f} fps / {stats['frame_ms']:.1f}ms por frame / "
                f"crop atual {stats['crop_ratio'] * 100:.0f}%"
            )
        except queue.Empty:
            pass
        self.root.after(33, self._poll_preview)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
