"""Interface simples do estabilizador: ajusta parametros, mostra preview,
controla a camera virtual que o OBS le como fonte.

Os controles agem em tempo real: mexer num slider muda o processamento no
frame seguinte, sem precisar parar e iniciar de novo.

Uso: python app.py  (ou o executavel gerado por build.bat)
"""
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from camera_to_obs import run_with_restart
from src.stabilizer import sensitivity_to_strength

PREVIEW_WIDTH = 480


class App:
    def __init__(self, root):
        self.root = root
        root.title("Camera Stabilizer")
        root.resizable(False, False)

        self._stop_event = threading.Event()
        self._worker = None
        self._stab = None  # VideoStabilizer vivo, quando rodando
        self._preview_q = queue.Queue(maxsize=1)
        self._live_tuning = {}   # espelho dos sliders, lido pela thread de captura
        self._notice = "parado"  # ultima mensagem do pipeline (reconexao, erro)
        self._last_frame_at = 0.0

        self._build_ui()
        self._poll_preview()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _slider(self, parent, row, label, var, lo, hi, fmt):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                  length=170, command=lambda _v: self._apply_tuning()
                  ).grid(row=row, column=1, sticky="we")
        value = ttk.Label(parent, width=6, anchor="e")
        value.grid(row=row, column=2, sticky="e")
        # trace, e nao o command do Scale, pra o numero acompanhar tambem
        # quando o valor for mudado por codigo.
        var.trace_add("write", lambda *_: value.config(text=fmt.format(var.get())))
        value.config(text=fmt.format(var.get()))

    def _build_ui(self):
        form = ttk.Frame(self.root, padding=10)
        form.grid(row=0, column=0, sticky="nw")

        ttk.Label(form, text="Fonte (indice da camera ou arquivo)").grid(row=0, column=0, sticky="w")
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(form, textvariable=self.source_var, width=18).grid(row=0, column=1, sticky="w")

        ttk.Label(form, text="Resolucao").grid(row=1, column=0, sticky="w")
        res = ttk.Frame(form)
        res.grid(row=1, column=1, columnspan=2, sticky="w")
        self.width_var = tk.IntVar(value=1920)
        self.height_var = tk.IntVar(value=1080)
        self.fps_var = tk.IntVar(value=30)
        ttk.Entry(res, textvariable=self.width_var, width=6).pack(side="left")
        ttk.Label(res, text="x").pack(side="left")
        ttk.Entry(res, textvariable=self.height_var, width=6).pack(side="left")
        ttk.Label(res, text="@").pack(side="left")
        ttk.Entry(res, textvariable=self.fps_var, width=4).pack(side="left")
        ttk.Label(res, text="fps").pack(side="left")

        # DoubleVar, nao IntVar: ttk.Scale escreve float na variavel. Todas as
        # variaveis nascem antes dos widgets, porque o command de um Scale pode
        # disparar durante a construcao e _tuning() le todas elas.
        self.sensitivity_var = tk.DoubleVar(value=100)
        self.smoothing_var = tk.DoubleVar(value=1.0)
        self.crop_var = tk.DoubleVar(value=90)
        self.lookahead_var = tk.DoubleVar(value=4)
        self.intentional_var = tk.BooleanVar(value=True)
        self.rotation_var = tk.BooleanVar(value=True)
        self.adaptive_var = tk.BooleanVar(value=False)
        self.overlay_var = tk.BooleanVar(value=False)

        self._slider(form, 2, "Sensibilidade", self.sensitivity_var, 0, 100, "{:.0f}")
        self._slider(form, 3, "Suavizacao (s)", self.smoothing_var, 0.2, 3.0, "{:.2f}")
        self._slider(form, 4, "Crop maximo (%)", self.crop_var, 80, 98, "{:.0f}")
        self._slider(form, 5, "Atraso / look-ahead (frames)", self.lookahead_var, 0, 15, "{:.0f}")

        checks = [
            ("Detectar movimento intencional (pan/tilt)", self.intentional_var, self._apply_tuning),
            ("Corrigir rotacao (recomendado)", self.rotation_var, self._apply_tuning),
            ("Crop adaptativo (zoom so quando precisa)", self.adaptive_var, self._apply_tuning),
            ("Mostrar overlay de debug", self.overlay_var, None),
        ]
        for i, (text, var, cmd) in enumerate(checks):
            ttk.Checkbutton(form, text=text, variable=var, command=cmd
                            ).grid(row=6 + i, column=0, columnspan=3, sticky="w")

        ttk.Label(form, text="Dica: com atraso N, ponha N x 33ms de atraso\n"
                             "de sincronia no audio do OBS (30fps).",
                  foreground="#666").grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))

        btns = ttk.Frame(form)
        btns.grid(row=11, column=0, columnspan=3, pady=(8, 0), sticky="w")
        self.start_btn = ttk.Button(btns, text="Iniciar", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Parar", command=self._stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="parado")
        ttk.Label(form, textvariable=self.status_var, wraplength=300, justify="left"
                  ).grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self.preview_label = ttk.Label(self.root)
        self.preview_label.grid(row=0, column=1, padx=10, pady=10)

    def _tuning(self):
        return {
            "strength": sensitivity_to_strength(self.sensitivity_var.get()),
            "smoothing_s": self.smoothing_var.get(),
            "min_keep_ratio": self.crop_var.get() / 100.0,
            "detect_intentional_motion": self.intentional_var.get(),
            "stabilize_rotation": self.rotation_var.get(),
            "adaptive_crop": self.adaptive_var.get(),
            "lookahead_frames": int(round(self.lookahead_var.get())),
        }

    def _apply_tuning(self):
        """Chamado a cada mexida de slider. Se ja esta rodando, muda na hora.

        Guarda um espelho em dict porque a thread de captura precisa reler a
        sintonia depois de uma reconexao, e ler tk.Variable fora da thread da
        interface nao e seguro.
        """
        self._live_tuning = self._tuning()
        if self._stab is not None:
            self._stab.set_tuning(**self._live_tuning)

    def _start(self):
        if self._worker and self._worker.is_alive():
            # Espera a captura anterior soltar a camera antes de reabrir.
            self._stop_event.set()
            self._worker.join(timeout=3)
        self._stop_event.clear()

        self._live_tuning = self._tuning()
        source = self.source_var.get()
        width, height, fps = self.width_var.get(), self.height_var.get(), self.fps_var.get()

        def target():
            try:
                run_with_restart(source, fps=fps, width=width, height=height,
                                 config=self._live_tuning, frame_callback=self._on_frame,
                                 stop_event=self._stop_event,
                                 on_stabilizer=self._on_stabilizer, log=self._log)
            except Exception as exc:
                self._log(f"erro: {exc}")
            finally:
                self._stab = None
                self.root.after(0, self._on_stopped)

        self._worker = threading.Thread(target=target, daemon=True)
        self._worker.start()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_var.set("iniciando...")

    def _log(self, message):
        """Chamado da thread de captura. So guarda a string; quem mostra e
        _poll_preview, na thread da interface."""
        print(message)
        self._notice = message

    def _on_stabilizer(self, stab):
        # Depois de uma reconexao o estabilizador e novo e nasce com a sintonia
        # da partida, entao reaplica o que os sliders mostram agora.
        stab.set_tuning(**self._live_tuning)
        self._stab = stab

    def _stop(self):
        self._stop_event.set()
        self.stop_btn.config(state="disabled")
        self._notice = "parado"
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
        scale = PREVIEW_WIDTH / stabilized.shape[1]
        preview = cv2.resize(stabilized, (PREVIEW_WIDTH, round(stabilized.shape[0] * scale)))

        if self.overlay_var.get():
            h, w = preview.shape[:2]
            cx, cy = w // 2, h // 2
            mx, my = stats["motion"]
            corr_x, corr_y = stats["correction"]
            # Seta vermelha: tremor detectado. Verde: correcao aplicada.
            cv2.arrowedLine(preview, (cx, cy),
                            (cx + int(mx * scale * 3), cy + int(my * scale * 3)),
                            (0, 0, 255), 2, tipLength=0.3)
            cv2.arrowedLine(preview, (cx, cy),
                            (cx + int(corr_x * scale * 3), cy + int(corr_y * scale * 3)),
                            (0, 200, 0), 2, tipLength=0.3)
            cv2.putText(preview, f"{stats['fps']:.0f}fps {stats['frame_ms']:.0f}ms "
                                 f"crop {stats['crop_ratio'] * 100:.0f}%",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(preview, f"corrigindo {abs(corr_y):.0f}px vert  "
                                 f"rot {stats.get('rotation', 0):+.2f}deg  "
                                 f"{stats.get('points', 0)} pts",
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        self._last_frame_at = time.monotonic()
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        try:
            self._preview_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._preview_q.put_nowait((rgb, stats))
        except queue.Full:
            pass

    def _poll_preview(self):
        try:
            rgb, stats = self._preview_q.get_nowait()
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview_label.configure(image=img)
            self.preview_label.image = img
            corr_y = abs(stats["correction"][1])
            mode = "pan/tilt" if stats["intentional"] > 0.6 else "vibracao"
            self.status_var.set(
                f"rodando - {stats['fps']:.0f} fps / {stats['frame_ms']:.0f} ms por frame\n"
                f"crop {stats['crop_ratio'] * 100:.0f}% - corrigindo {corr_y:.0f}px vert, "
                f"{stats.get('rotation', 0):+.2f} graus - {mode}"
            )
        except queue.Empty:
            pass

        # Sem frame recente = fonte caiu ou esta reconectando: mostra o aviso
        # do pipeline em vez de deixar as estatisticas velhas congeladas.
        if time.monotonic() - self._last_frame_at > 1.0:
            self.status_var.set(self._notice)
        self.root.after(33, self._poll_preview)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
