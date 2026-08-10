"""YOLO 26 UI-element detector, with a no-model fallback.

Contract: `detect(image) -> list[Element]` where each Element carries a pixel
bounding box and a tap point (box centre). Gemma later chooses *which* element.

IMPORTANT: stock YOLO weights detect COCO objects (people, cars…), NOT UI
widgets. Point `perception.weights` at weights fine-tuned on a UI dataset
(RICO, or your own captured van-app screens). Until such a file exists, we fall
back to an OpenCV contour proposer so the whole pipeline still runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import CONFIG


@dataclass
class Element:
    box: tuple[int, int, int, int]          # x1, y1, x2, y2 in pixels
    label: str = "element"
    confidence: float = 0.0
    text: str = ""                          # optional OCR text (future)
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector:
    def __init__(self) -> None:
        pcfg = CONFIG.get("perception", {})
        self.backend = pcfg.get("backend", "yolo")
        self.conf = float(pcfg.get("conf", 0.25))
        self.max_elements = int(pcfg.get("max_elements", 60))
        self._weights = pcfg.get("weights", "")
        self._model = None
        if self.backend == "yolo":
            self._try_load_yolo()

    def _try_load_yolo(self) -> None:
        weights = Path(self._weights)
        if not weights.exists():
            print(f"[perception] weights {weights} not found — using contour fallback. "
                  f"Fine-tune YOLO on UI elements and set perception.weights.")
            self.backend = "fallback"
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(str(weights))
            print(f"[perception] loaded YOLO weights: {weights}")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on the edge
            print(f"[perception] YOLO load failed ({exc}); using contour fallback.")
            self.backend = "fallback"

    # ---- public API ------------------------------------------------------
    def detect(self, image: np.ndarray) -> list[Element]:
        if self.backend == "yolo" and self._model is not None:
            elements = self._detect_yolo(image)
        else:
            elements = self._detect_fallback(image)
        elements.sort(key=lambda e: e.confidence, reverse=True)
        return elements[: self.max_elements]

    def _detect_yolo(self, image: np.ndarray) -> list[Element]:
        results = self._model.predict(image, conf=self.conf, verbose=False)
        elements: list[Element] = []
        for res in results:
            names = res.names
            for b in res.boxes:
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                cls = int(b.cls[0]) if b.cls is not None else -1
                elements.append(
                    Element(
                        box=(x1, y1, x2, y2),
                        label=names.get(cls, "element") if names else "element",
                        confidence=float(b.conf[0]) if b.conf is not None else 0.0,
                    )
                )
        return elements

    def _detect_fallback(self, image: np.ndarray) -> list[Element]:
        """Cheap, model-free element proposer: find rectangular blobs.

        Good enough to exercise the loop and to prove tapping works; NOT a
        substitute for a trained detector.
        """
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        min_area = (w * h) * 0.0008    # ignore specks
        max_area = (w * h) * 0.5       # ignore full-screen backgrounds
        elements: list[Element] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < min_area or area > max_area:
                continue
            aspect = cw / max(ch, 1)
            if aspect > 12 or aspect < 0.08:   # drop hairlines
                continue
            elements.append(
                Element(box=(x, y, x + cw, y + ch), label="candidate", confidence=0.1)
            )
        return elements


_DETECTOR: Detector | None = None


def get_detector() -> Detector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Detector()
    return _DETECTOR
