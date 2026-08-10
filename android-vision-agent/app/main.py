"""FastAPI surface for the Android Vision Agent.

The camper-van controller talks to THIS. It sends a high-level goal; the agent
figures out the taps by looking at the screen.
"""
from __future__ import annotations

import io

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .adb import ADBError, get_device
from .agent_loop import AgentRunner, result_to_dict
from .config import CONFIG
from .macros import get_store

app = FastAPI(title="Android Vision Agent", version="0.1.0")

_runner: AgentRunner | None = None


def runner() -> AgentRunner:
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner


class IntentRequest(BaseModel):
    goal: str = Field(..., description="High-level goal, e.g. 'turn on the water heater'")
    max_steps: int | None = Field(None, description="Override the per-intent action cap")


@app.get("/health")
def health() -> dict:
    device = get_device()
    return {
        "status": "ok",
        "device_connected": device.is_connected(),
        "device_serial": device.serial or "(auto)",
        "perception_backend": CONFIG["perception"]["backend"],
        "reasoning_backend": CONFIG["reasoning"]["backend"],
        "reasoning_model": CONFIG["reasoning"]["model"],
    }


@app.get("/screen")
def screen() -> Response:
    import cv2

    try:
        image = get_device().screencap()
    except ADBError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="failed to encode screenshot")
    return Response(content=io.BytesIO(buf.tobytes()).getvalue(), media_type="image/png")


@app.post("/intent")
def intent(req: IntentRequest) -> dict:
    if not req.goal.strip():
        raise HTTPException(status_code=422, detail="goal must not be empty")
    try:
        result = runner().run(req.goal, req.max_steps)
    except ADBError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result_to_dict(result)


@app.get("/macros")
def macros() -> dict:
    return get_store().all()
