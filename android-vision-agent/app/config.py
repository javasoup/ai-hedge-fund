"""Load config.yaml with env-var overrides (AVA_<SECTION>_<KEY>)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(os.environ.get("AVA_CONFIG", Path(__file__).parent.parent / "config.yaml"))


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Override cfg[section][key] from AVA_SECTION_KEY env vars when present."""
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            env_key = f"AVA_{section.upper()}_{key.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                # Preserve the original type where we can.
                original = values[key]
                if isinstance(original, bool):
                    values[key] = raw.lower() in ("1", "true", "yes", "on")
                elif isinstance(original, int):
                    values[key] = int(raw)
                elif isinstance(original, float):
                    values[key] = float(raw)
                else:
                    values[key] = raw
    return cfg


def load_config() -> dict[str, Any]:
    with open(_CONFIG_PATH) as fh:
        cfg = yaml.safe_load(fh) or {}
    return _apply_env_overrides(cfg)


CONFIG = load_config()
