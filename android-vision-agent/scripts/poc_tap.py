#!/usr/bin/env python3
"""End-to-end proof of concept: capture → detect → tap the first element.

Run this first, before the API, to confirm the physical pipeline works on your
Jetson + device. It does NOT need Gemma or trained YOLO weights — it exercises
ADB capture, the element detector (fallback is fine), and tap injection.

    python scripts/poc_tap.py            # detect + tap element 0
    python scripts/poc_tap.py --dry-run  # detect only, no tap
    python scripts/poc_tap.py --save out.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a loose script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adb import ADBError, get_device          # noqa: E402
from app.perception.yolo_detector import get_detector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Vision→ADB tap proof of concept")
    ap.add_argument("--dry-run", action="store_true", help="detect only, do not tap")
    ap.add_argument("--index", type=int, default=0, help="element index to tap")
    ap.add_argument("--save", metavar="PATH", help="save an annotated screenshot")
    args = ap.parse_args()

    device = get_device()
    if not device.is_connected():
        print("ERROR: no ADB device connected. Run `adb devices`.", file=sys.stderr)
        return 1

    try:
        w, h = device.screen_size()
        print(f"device screen: {w}x{h}")
        image = device.screencap()
    except ADBError as exc:
        print(f"ADB error: {exc}", file=sys.stderr)
        return 1

    detector = get_detector()
    elements = detector.detect(image)
    print(f"detector backend: {detector.backend} — {len(elements)} elements")
    for i, e in enumerate(elements[:10]):
        print(f"  [{i}] {e.label} conf={e.confidence:.2f} center={e.center} box={e.box}")

    if args.save:
        _annotate_and_save(image, elements, args.save)
        print(f"saved annotated screenshot -> {args.save}")

    if not elements:
        print("no elements detected; nothing to tap.")
        return 0

    idx = args.index
    if not (0 <= idx < len(elements)):
        print(f"index {idx} out of range (0..{len(elements) - 1})", file=sys.stderr)
        return 1

    target = elements[idx]
    if args.dry_run:
        print(f"[dry-run] would tap element {idx} at {target.center}")
        return 0

    print(f"tapping element {idx} at {target.center} ...")
    device.tap(*target.center)
    print("done.")
    return 0


def _annotate_and_save(image, elements, path: str) -> None:
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for i, e in enumerate(elements):
        x1, y1, x2, y2 = e.box
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, str(i), (x1 + 2, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(path, bgr)


if __name__ == "__main__":
    raise SystemExit(main())
