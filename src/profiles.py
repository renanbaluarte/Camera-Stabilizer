"""Perfis de configuracao: presets prontos + perfis salvos pelo usuario.

Os perfis do usuario ficam em %APPDATA%\\CameraStabilizer\\profiles.json (ou
~/.camera_stabilizer/profiles.json fora do Windows), fora da pasta do
programa, para sobreviver a atualizacoes do executavel.

Formato do arquivo:
  {"last": "<nome do ultimo perfil usado>", "profiles": {"<nome>": {...}}}
"""
import json
import os
import sys
from pathlib import Path

# Valores da interface (nao do estabilizador): e o que os sliders mostram.
DEFAULTS = {
    "source": "0", "width": 1920, "height": 1080, "fps": 30,
    "sensitivity": 100, "smoothing_s": 1.5, "crop": 90, "lookahead": 4,
    "intentional": True, "rotation": True, "adaptive": False, "overlay": False,
}

# Presets somente leitura. Ver README > "Configuracoes recomendadas".
PRESETS = {
    "Recomendado - camera fixa": {},
    "Tremor forte": {"smoothing_s": 2.0, "crop": 85, "lookahead": 8},
    "Sem atraso (nao mexer no audio)": {"smoothing_s": 1.0, "lookahead": 0},
    "PTZ com muito movimento": {"smoothing_s": 0.6, "crop": 92, "lookahead": 4},
}


def bundled_seed_path():
    """config/default_profiles.json, tanto rodando do codigo quanto no .exe
    do PyInstaller (que extrai o --add-data em sys._MEIPASS)."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    return base / "config" / "default_profiles.json"


def default_path():
    base = os.environ.get("APPDATA")
    folder = Path(base) / "CameraStabilizer" if base else Path.home() / ".camera_stabilizer"
    return folder / "profiles.json"


class ProfileStore:
    def __init__(self, path=None, seed_path=None):
        self.path = Path(path) if path else default_path()
        self._data = {"last": None, "profiles": {}, "seeded": []}
        self._load()
        self._seed(Path(seed_path) if seed_path else bundled_seed_path())

    def _load(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data.get("profiles"), dict):
                self._data["profiles"] = data["profiles"]
            self._data["last"] = data.get("last")
            self._data["seeded"] = list(data.get("seeded", []))
        except (OSError, ValueError, AttributeError):
            pass  # arquivo ausente ou corrompido: comeca do zero

    def _seed(self, seed_path):
        """Cria os perfis iniciais que vem com o programa. Cada um e criado
        uma vez so: se o usuario excluir ou editar, a escolha dele vale."""
        try:
            seed = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        added = False
        for name, settings in seed.get("profiles", {}).items():
            if name in self._data["seeded"] or name in PRESETS:
                continue
            self._data["profiles"].setdefault(name, settings)
            self._data["seeded"].append(name)
            if not self._data["last"] and seed.get("last") == name:
                self._data["last"] = name
            added = True
        if added:
            try:
                self._save()
            except OSError:
                pass

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)  # troca atomica: queda de energia nao corrompe

    @staticmethod
    def is_preset(name):
        return name in PRESETS

    def names(self):
        return list(PRESETS) + sorted(n for n in self._data["profiles"] if n not in PRESETS)

    def get(self, name):
        if name in PRESETS:
            return {**DEFAULTS, **PRESETS[name]}
        if name in self._data["profiles"]:
            return {**DEFAULTS, **self._data["profiles"][name]}
        raise KeyError(name)

    def save(self, name, settings):
        name = name.strip()
        if not name:
            raise ValueError("nome vazio")
        if name in PRESETS:
            raise ValueError("presets prontos nao podem ser sobrescritos; use outro nome")
        self._data["profiles"][name] = {k: settings[k] for k in DEFAULTS if k in settings}
        self._data["last"] = name
        self._save()

    def delete(self, name):
        if name in PRESETS:
            raise ValueError("presets prontos nao podem ser excluidos")
        self._data["profiles"].pop(name, None)
        if self._data["last"] == name:
            self._data["last"] = None
        self._save()

    @property
    def last(self):
        last = self._data["last"]
        return last if last in self.names() else next(iter(PRESETS))

    def set_last(self, name):
        if name != self._data["last"]:
            self._data["last"] = name
            try:
                self._save()
            except OSError:
                pass
