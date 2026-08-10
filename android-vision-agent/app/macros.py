"""Record successful action sequences and replay them without the models.

The van will often repeat the same handful of flows ("water heater on"). Once
the vision+reasoning loop solves a goal, we persist the concrete actions so the
next call is a fast, offline, deterministic replay — the models only re-engage
if replay fails (app layout changed).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CONFIG


class MacroStore:
    def __init__(self) -> None:
        mcfg = CONFIG.get("macros", {})
        self.enabled = bool(mcfg.get("enabled", True))
        self.path = Path(mcfg.get("path", "macros.json"))
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self._data = {}

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))

    @staticmethod
    def _key(goal: str) -> str:
        return goal.strip().lower()

    def get(self, goal: str) -> list[dict[str, Any]] | None:
        return self._data.get(self._key(goal))

    def save(self, goal: str, actions: list[dict[str, Any]]) -> None:
        if not self.enabled or not actions:
            return
        self._data[self._key(goal)] = actions
        self._save()

    def all(self) -> dict[str, list[dict[str, Any]]]:
        return dict(self._data)


_STORE: MacroStore | None = None


def get_store() -> MacroStore:
    global _STORE
    if _STORE is None:
        _STORE = MacroStore()
    return _STORE
