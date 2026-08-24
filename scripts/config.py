import os
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"


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


def load_config(config_path: str | Path | None = None) -> ConfigDict:
    """Loads configuration from YAML file and returns an attribute-accessible ConfigDict."""
    target_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return ConfigDict(data)


# Global default configuration instance
cfg = load_config()
