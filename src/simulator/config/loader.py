from pathlib import Path
from typing import Any

import yaml

REQUIRED_TOP_LEVEL_KEYS = ("run", "simulation", "domain", "scenario")


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Configuration file must be a YAML file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping/dictionary.")

    missing_keys = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in config]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"Configuration is missing required top-level keys: {missing}")

    return config
