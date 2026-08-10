"""Gemma 3 4B multimodal planner.

Given the goal, the current screenshot, and YOLO's detected elements, decide the
single next action. We hand the model the *screenshot* (it can see) AND the
*numbered element boxes* (precise coordinates it can't produce reliably itself),
then ask it to pick an element index or emit a coordinate-free action. That
grounding step is what keeps taps accurate.

Backends:
  - "ollama": POST to a local Ollama server running e.g. `gemma3:4b`
  - "openai": any OpenAI-compatible /chat/completions endpoint with vision
  - "stub":   deterministic, no model — picks the first element (for testing)
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

import numpy as np

from ..config import CONFIG
from ..perception.yolo_detector import Element

# Valid actions the planner may return.
_ACTIONS = {"tap", "swipe", "text", "key", "done", "wait"}


@dataclass
class Action:
    type: str                      # one of _ACTIONS
    element_index: int | None = None
    x: int | None = None
    y: int | None = None
    x2: int | None = None
    y2: int | None = None
    value: str | None = None       # text to type / keycode / reason
    reason: str = ""

    @classmethod
    def done(cls, reason: str = "") -> "Action":
        return cls(type="done", reason=reason)


_SYSTEM = (
    "You operate an Android phone by choosing ONE next action to reach a goal. "
    "You are shown a screenshot and a numbered list of tappable elements with "
    "pixel boxes. Prefer referring to an element by its index; only use raw "
    "coordinates if the target is not in the list. "
    "Respond with STRICT JSON only, no prose, matching one of:\n"
    '  {"type":"tap","element_index":N,"reason":"..."}\n'
    '  {"type":"tap","x":X,"y":Y,"reason":"..."}\n'
    '  {"type":"swipe","x":X,"y":Y,"x2":X2,"y2":Y2,"reason":"..."}\n'
    '  {"type":"text","value":"...","reason":"..."}\n'
    '  {"type":"key","value":"KEYCODE_BACK","reason":"..."}\n'
    '  {"type":"done","reason":"goal achieved"}\n'
    '  {"type":"wait","reason":"loading"}'
)


def _encode_image(image: np.ndarray, max_side: int) -> str:
    import cv2

    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("failed to encode image for the reasoner")
    return base64.b64encode(buf.tobytes()).decode()


def _elements_prompt(goal: str, elements: list[Element], history: list[str]) -> str:
    lines = [f"GOAL: {goal}", "", "ELEMENTS (index: label conf box=[x1,y1,x2,y2] center=(cx,cy)):"]
    for i, e in enumerate(elements):
        cx, cy = e.center
        txt = f' text="{e.text}"' if e.text else ""
        lines.append(f"  {i}: {e.label} {e.confidence:.2f} box={list(e.box)} center=({cx},{cy}){txt}")
    if history:
        lines += ["", "ACTIONS SO FAR:"] + [f"  - {h}" for h in history[-6:]]
    lines += ["", "Return ONE action as strict JSON."]
    return "\n".join(lines)


def _parse_action(raw: str) -> Action:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return Action(type="wait", reason=f"unparseable: {raw[:80]}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Action(type="wait", reason=f"bad json: {raw[:80]}")
    a_type = str(data.get("type", "wait"))
    if a_type not in _ACTIONS:
        a_type = "wait"
    return Action(
        type=a_type,
        element_index=data.get("element_index"),
        x=data.get("x"),
        y=data.get("y"),
        x2=data.get("x2"),
        y2=data.get("y2"),
        value=data.get("value"),
        reason=str(data.get("reason", "")),
    )


class GemmaAgent:
    def __init__(self) -> None:
        rcfg = CONFIG.get("reasoning", {})
        self.backend = rcfg.get("backend", "ollama")
        self.model = rcfg.get("model", "gemma3:4b")
        self.endpoint = rcfg.get("endpoint", "http://localhost:11434").rstrip("/")
        self.temperature = float(rcfg.get("temperature", 0.1))
        self.image_max_side = int(rcfg.get("image_max_side", 1080))

    def decide(
        self,
        goal: str,
        image: np.ndarray,
        elements: list[Element],
        history: list[str],
    ) -> Action:
        if self.backend == "stub":
            return self._decide_stub(elements)
        prompt = _elements_prompt(goal, elements, history)
        img_b64 = _encode_image(image, self.image_max_side)
        try:
            if self.backend == "ollama":
                raw = self._call_ollama(prompt, img_b64)
            elif self.backend == "openai":
                raw = self._call_openai(prompt, img_b64)
            else:
                return Action(type="wait", reason=f"unknown backend {self.backend}")
        except Exception as exc:  # noqa: BLE001 - never crash the loop on a model hiccup
            return Action(type="wait", reason=f"reasoner error: {exc}")
        return _parse_action(raw)

    # ---- backends --------------------------------------------------------
    def _call_ollama(self, prompt: str, img_b64: str) -> str:
        import requests

        resp = requests.post(
            f"{self.endpoint}/api/generate",
            json={
                "model": self.model,
                "system": _SYSTEM,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _call_openai(self, prompt: str, img_b64: str) -> str:
        import requests

        resp = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        ],
                    },
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _decide_stub(self, elements: list[Element]) -> Action:
        if not elements:
            return Action.done("no elements detected")
        return Action(type="tap", element_index=0, reason="stub: first element")


_AGENT: GemmaAgent | None = None


def get_agent() -> GemmaAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = GemmaAgent()
    return _AGENT
