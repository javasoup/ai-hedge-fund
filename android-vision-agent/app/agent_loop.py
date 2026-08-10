"""The control loop: capture → detect → reason → act → verify.

This is the orchestrator that ties ADB, YOLO, and Gemma together and turns a
high-level goal into concrete taps. It also records successful runs as macros
and replays them when available.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .adb import Device, get_device
from .config import CONFIG
from .macros import get_store
from .perception.yolo_detector import Element, get_detector
from .reasoning.gemma_agent import Action, get_agent


@dataclass
class Step:
    index: int
    action: str
    detail: str
    reason: str = ""


@dataclass
class RunResult:
    goal: str
    success: bool
    steps: list[Step] = field(default_factory=list)
    used_macro: bool = False
    message: str = ""


def _screen_hash(image: np.ndarray) -> str:
    import cv2

    small = cv2.resize(image, (32, 32))
    return hashlib.md5(small.tobytes()).hexdigest()


def _resolve_action(action: Action, elements: list[Element]) -> tuple[str, tuple | None, str]:
    """Turn a planner Action into (kind, args, human_detail)."""
    if action.type == "tap":
        if action.element_index is not None and 0 <= action.element_index < len(elements):
            x, y = elements[action.element_index].center
        elif action.x is not None and action.y is not None:
            x, y = action.x, action.y
        else:
            return "noop", None, "tap without a valid target"
        return "tap", (x, y), f"tap ({x},{y})"
    if action.type == "swipe" and None not in (action.x, action.y, action.x2, action.y2):
        return "swipe", (action.x, action.y, action.x2, action.y2), "swipe"
    if action.type == "text" and action.value:
        return "text", (action.value,), f"text {action.value!r}"
    if action.type == "key" and action.value:
        return "key", (action.value,), f"key {action.value}"
    if action.type in ("done", "wait"):
        return action.type, None, action.type
    return "noop", None, f"unhandled action {action.type}"


class AgentRunner:
    def __init__(self, device: Device | None = None) -> None:
        self.device = device or get_device()
        self.detector = get_detector()
        self.agent = get_agent()
        self.store = get_store()
        loop = CONFIG.get("loop", {})
        self.max_steps = int(loop.get("max_steps", 12))
        self.stall_limit = int(loop.get("stall_limit", 3))

    # ---- public ----------------------------------------------------------
    def run(self, goal: str, max_steps: int | None = None) -> RunResult:
        if not self.device.is_connected():
            return RunResult(goal, False, message="no ADB device connected")

        macro = self.store.get(goal)
        if macro:
            replayed = self._replay(goal, macro)
            if replayed.success:
                return replayed
            # fall through to live solving if replay failed

        return self._solve(goal, max_steps or self.max_steps)

    # ---- live solve ------------------------------------------------------
    def _solve(self, goal: str, max_steps: int) -> RunResult:
        result = RunResult(goal, False)
        history: list[str] = []
        recorded: list[dict[str, Any]] = []
        last_hash = ""
        stall = 0

        for i in range(max_steps):
            image = self.device.screencap()
            h = _screen_hash(image)
            stall = stall + 1 if h == last_hash else 0
            last_hash = h
            if stall >= self.stall_limit:
                result.message = "screen stopped changing (stuck)"
                break

            elements = self.detector.detect(image)
            action = self.agent.decide(goal, image, elements, history)
            kind, args, detail = _resolve_action(action, elements)
            result.steps.append(Step(i, action.type, detail, action.reason))
            history.append(f"{action.type}: {detail} ({action.reason})")

            if kind == "done":
                result.success = True
                result.message = action.reason or "goal reported complete"
                break
            if kind in ("wait", "noop"):
                continue

            self._apply(kind, args)
            recorded.append({"kind": kind, "args": list(args) if args else []})

        if result.success and recorded:
            self.store.save(goal, recorded)
        return result

    # ---- macro replay ----------------------------------------------------
    def _replay(self, goal: str, macro: list[dict[str, Any]]) -> RunResult:
        result = RunResult(goal, True, used_macro=True)
        try:
            for i, step in enumerate(macro):
                self._apply(step["kind"], tuple(step.get("args", [])))
                result.steps.append(Step(i, step["kind"], f"replay {step['kind']} {step.get('args')}"))
        except Exception as exc:  # noqa: BLE001 - fall back to live solve
            return RunResult(goal, False, used_macro=True, message=f"macro replay failed: {exc}")
        result.message = "replayed macro"
        return result

    # ---- effectors -------------------------------------------------------
    def _apply(self, kind: str, args: tuple | None) -> None:
        if kind == "tap":
            self.device.tap(*args)
        elif kind == "swipe":
            self.device.swipe(*args)
        elif kind == "text":
            self.device.text(*args)
        elif kind == "key":
            self.device.key(*args)


def result_to_dict(result: RunResult) -> dict[str, Any]:
    d = asdict(result)
    return d
