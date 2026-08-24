import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "configs" / "config.yaml"
STAGES_DIR = ROOT / "configs" / "stages"

# Canonical stage budgets (override path or name via --stage)
STAGES = {
    "S0": "S0_smoke.yaml",
    "S0_smoke": "S0_smoke.yaml",
    "S1": "S1_baseline.yaml",
    "S1_baseline": "S1_baseline.yaml",
    "S2": "S2_research.yaml",
    "S2_research": "S2_research.yaml",
    "S3": "S3_final.yaml",
    "S3_final": "S3_final.yaml",
}


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot access and nested dict conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            if isinstance(v, dict) and not isinstance(v, ConfigDict):
                self[k] = ConfigDict(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_stage_path(stage: str | Path | None) -> Path | None:
    """Resolve stage name (S0/S1/...) or path to a stage YAML. None if unset."""
    if stage is None:
        return None
    p = Path(stage)
    if p.exists() and p.is_file():
        return p.resolve()
    key = str(stage).strip()
    if key in STAGES:
        path = STAGES_DIR / STAGES[key]
        if path.exists():
            return path
    raise FileNotFoundError(f"Unknown stage '{stage}'. Known: {sorted(set(STAGES))}")


def load_config(config_path: str | Path | None = None, stage: str | Path | None = None) -> ConfigDict:
    """Load base YAML, optionally deep-merge a stage override (training budget only)."""
    target_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    stage_path = resolve_stage_path(stage)
    if stage_path is not None:
        with open(stage_path, "r", encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    return ConfigDict(data)


# Global default configuration instance (base only; stages via load_config/CLI)
cfg = load_config()
