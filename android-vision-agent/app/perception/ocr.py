"""Optional OCR to attach text labels to detected UI elements.

Why: YOLO returns *where* a tappable thing is, not *what it says*. Handing the
reasoner "element 4: box=[...] text='Water heater'" instead of a bare box turns
guessing into reading — the single biggest accuracy win for the vision path.

Backends (auto-detected, all optional):
  - pytesseract  (light; needs the `tesseract` system binary — good on Jetson)
  - none         (no OCR installed — Gemma still reads the full screenshot)

Kept deliberately dependency-light: if nothing is available, callers get "".
"""
from __future__ import annotations

import numpy as np


class _OCR:
    def __init__(self) -> None:
        self._backend = "none"
        self._tess = None
        try:
            import pytesseract  # noqa: F401

            self._tess = pytesseract
            self._backend = "tesseract"
        except Exception:  # noqa: BLE001 - OCR is optional
            self._backend = "none"

    @property
    def available(self) -> bool:
        return self._backend != "none"

    def text_in(self, image: np.ndarray, box: tuple[int, int, int, int]) -> str:
        """Best-effort OCR of a single element crop. Returns '' if unavailable."""
        if self._backend != "tesseract":
            return ""
        x1, y1, x2, y2 = box
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return ""
        crop = image[y1:y2, x1:x2]
        try:
            text = self._tess.image_to_string(crop, config="--psm 7").strip()
        except Exception:  # noqa: BLE001 - never break perception on OCR failure
            return ""
        # collapse whitespace/newlines
        return " ".join(text.split())


_OCR_SINGLETON: _OCR | None = None


def get_ocr() -> _OCR:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = _OCR()
    return _OCR_SINGLETON
