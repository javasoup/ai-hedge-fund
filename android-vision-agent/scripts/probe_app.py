#!/usr/bin/env python3
"""Reality check: does your ONE app actually need vision?

Even when an app is "the only way in", Android usually still exposes its
buttons through the accessibility / UIAutomator tree — structured data with
exact bounds and text labels, far more reliable (and cheaper) than vision.

This script dumps the current screen's accessibility tree and tells you:
  - how many nodes exist
  - how many are clickable and how many carry text
  - a verdict: is this a true "canvas" (vision required) or not?

Run it while your target app is on screen. If it reports plenty of clickable,
labelled nodes, consider driving the app via the accessibility tree instead of
(or ahead of) the YOLO+Gemma vision loop — it will be dramatically more robust.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adb import ADBError, get_device  # noqa: E402


def main() -> int:
    device = get_device()
    if not device.is_connected():
        print("ERROR: no ADB device connected. Run `adb devices`.", file=sys.stderr)
        return 1

    # `uiautomator dump` writes XML to the device; pull it to stdout.
    try:
        device._run(["shell", "uiautomator", "dump", "/sdcard/window_dump.xml"])
        xml = device._run(["shell", "cat", "/sdcard/window_dump.xml"])
    except ADBError as exc:
        print(f"ADB error: {exc}", file=sys.stderr)
        print("If this fails, the foreground app likely blocks UIAutomator "
              "(secure/canvas surface) — that's a real vision use case.")
        return 1

    if not isinstance(xml, str) or "<hierarchy" not in xml:
        print("No accessibility hierarchy returned — looks like a true canvas. "
              "Vision (YOLO+Gemma) is the right call here.")
        return 0

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        print(f"could not parse UIAutomator XML: {exc}", file=sys.stderr)
        return 1

    nodes = list(root.iter("node"))
    clickable = [n for n in nodes if n.get("clickable") == "true"]
    labelled = [n for n in nodes if (n.get("text") or n.get("content-desc"))]

    print(f"total nodes:      {len(nodes)}")
    print(f"clickable nodes:  {len(clickable)}")
    print(f"labelled nodes:   {len(labelled)}")
    print()
    print("Sample of clickable, labelled elements:")
    shown = 0
    for n in clickable:
        label = n.get("text") or n.get("content-desc") or ""
        if not label:
            continue
        print(f"  - {label!r:40}  bounds={n.get('bounds')}  id={n.get('resource-id','')}")
        shown += 1
        if shown >= 12:
            break

    print()
    if len(clickable) >= 3 and len(labelled) >= 3:
        print("VERDICT: This app IS visible to the accessibility tree.")
        print("  -> You can drive it via UIAutomator (exact, offline, no GPU).")
        print("  -> Vision is optional here; keep it only as a fallback.")
    else:
        print("VERDICT: Sparse/empty tree — likely a canvas surface.")
        print("  -> Vision (YOLO+Gemma) is genuinely warranted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
