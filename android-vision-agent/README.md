# Android Vision Agent

Drive an Android device by **looking at its screen** — no accessibility tree,
pure vision — and expose the whole thing behind a small HTTP API so external
systems (e.g. a camper-van Bluetooth controller) can request high-level
intents like `water_heater_on` and have the agent figure out the taps.

## Why this exists

A spare Android device stays plugged into an edge box (Jetson). It runs
whatever Bluetooth apps control the van's hardware. This service captures the
device screen, reasons about what to tap to accomplish a goal, and injects the
taps over ADB. Successful flows are recorded as **macros** so repeats are fast
and work offline.

## The model stack (and who does what)

The perception and reasoning jobs are deliberately split across two models,
because neither can do the other's job well:

| Stage | Model | Job |
|-------|-------|-----|
| **Ground** | YOLO 26 | Detect *tappable UI elements* and return **precise pixel boxes**. VLMs are bad at exact coordinates; YOLO is not. |
| **Reason** | Gemma 3 4B (multimodal) | Look at the screenshot + the detected boxes, decide *which* element advances the goal, and plan the next step. |
| **Act** | ADB | Inject the tap / swipe / text and capture the result. |

```
  goal ─▶ ┌─────────────────── agent loop ───────────────────┐
          │ capture ─▶ YOLO detect ─▶ Gemma decide ─▶ adb act │
          │    ▲                                        │     │
          │    └──────────────── verify ◀───────────────┘     │
          └──────────────────────────────────────────────────┘
```

### Two things you MUST know before trusting it

1. **YOLO out of the box does not know UI elements.** Stock YOLO weights are
   trained on COCO (people, cars, dogs). To detect buttons/toggles/fields you
   must **fine-tune YOLO on a UI dataset** (e.g. RICO, or your own captured
   screens). Until you do, `yolo_detector.py` falls back to a generic
   contour/edge element proposer so the pipeline still runs end to end — see
   `app/perception/yolo_detector.py`.

2. **Hardware.** Gemma 3 4B multimodal is heavy. The original **Jetson Nano
   (4GB Maxwell)** will struggle; target a **Jetson Orin Nano (8GB)** and run
   Gemma 4-bit via Ollama or llama.cpp. The reasoning backend is swappable in
   `app/reasoning/gemma_agent.py` (`OLLAMA` by default).

## Layout

```
android-vision-agent/
├── app/
│   ├── main.py                 # FastAPI: /health, /screen, /intent
│   ├── adb.py                  # screencap, tap, swipe, text over ADB
│   ├── agent_loop.py           # capture → detect → reason → act → verify
│   ├── macros.py               # record & replay successful flows
│   ├── perception/
│   │   └── yolo_detector.py    # YOLO 26 UI element detection (+ fallback)
│   └── reasoning/
│       └── gemma_agent.py      # Gemma 3 4B multimodal planner (Ollama)
├── scripts/
│   └── poc_tap.py              # standalone end-to-end proof of concept
├── config.yaml                 # device, models, loop limits
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 0. On the Jetson: install adb + platform tools, plug in / pair the device
adb devices                      # confirm the device shows up

# 1. Python deps
pip install -r requirements.txt

# 2. (optional) pull Gemma for reasoning
ollama pull gemma3:4b

# 3. Prove the pipeline end to end (capture → detect → tap first element)
python scripts/poc_tap.py

# 4. Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Then from the van controller:

```bash
curl -X POST localhost:8080/intent \
  -H 'content-type: application/json' \
  -d '{"goal": "turn on the water heater"}'
```

## API

| Method | Path       | Body                              | Returns |
|--------|------------|-----------------------------------|---------|
| GET    | `/health`  | –                                 | device + model status |
| GET    | `/screen`  | –                                 | current screenshot (PNG) |
| POST   | `/intent`  | `{"goal": str, "max_steps": int}` | step trace + success flag |
| GET    | `/macros`  | –                                 | recorded macros |

## Status

This is a **runnable skeleton**. ADB control is real. YOLO and Gemma are wired
with graceful fallbacks so the loop runs before you've installed the models or
fine-tuned YOLO. Search for `TODO` to find the spots that need your models /
data.
