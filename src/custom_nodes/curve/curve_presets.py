from __future__ import annotations

from pathlib import Path

from .curve_spec import CurveSpec

_DEFAULT_DIR = Path(__file__).parent / "presets"


def save_preset(spec: CurveSpec, name: str, directory: Path | None = None) -> Path:
    """Serialise spec to <directory>/<name>.json and return the path."""
    d = Path(directory) if directory else _DEFAULT_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(spec.to_json(), encoding="utf-8")
    return path


def load_preset(name_or_path: str, directory: Path | None = None) -> CurveSpec:
    """Load a preset by name (no extension) or by full path."""
    p = Path(name_or_path)
    if not p.suffix:
        d = Path(directory) if directory else _DEFAULT_DIR
        p = d / f"{name_or_path}.json"
    if not p.exists():
        raise FileNotFoundError(f"Preset not found: {p}")
    return CurveSpec.from_json(p.read_text(encoding="utf-8"))


def list_presets(directory: Path | None = None) -> list[str]:
    """Return the names (stems) of all presets in directory."""
    d = Path(directory) if directory else _DEFAULT_DIR
    if not d.exists():
        return []
    return [f.stem for f in sorted(d.glob("*.json"))]
