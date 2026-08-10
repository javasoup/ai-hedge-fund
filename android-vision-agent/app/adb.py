"""Thin, real wrapper around the `adb` binary.

This is the part that actually moves: it captures the framebuffer and injects
input events. No Android app install required — just USB (or network) debugging
enabled on the device.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

import numpy as np

from .config import CONFIG


class ADBError(RuntimeError):
    pass


@dataclass
class Device:
    serial: str | None = None
    settle_ms: int = 900

    # ---- low level -------------------------------------------------------
    def _base(self) -> list[str]:
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _run(self, args: list[str], *, binary: bool = False) -> bytes | str:
        try:
            proc = subprocess.run(
                self._base() + args,
                capture_output=True,
                check=True,
                timeout=30,
            )
        except FileNotFoundError as exc:  # adb not installed
            raise ADBError("`adb` binary not found on PATH — install android-platform-tools") from exc
        except subprocess.CalledProcessError as exc:
            raise ADBError(f"adb {' '.join(args)} failed: {exc.stderr.decode(errors='ignore')}") from exc
        except subprocess.TimeoutExpired as exc:
            raise ADBError(f"adb {' '.join(args)} timed out") from exc
        return proc.stdout if binary else proc.stdout.decode(errors="ignore")

    # ---- introspection ---------------------------------------------------
    def is_connected(self) -> bool:
        try:
            out = subprocess.run(["adb", "devices"], capture_output=True, timeout=10).stdout.decode()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        lines = [ln for ln in out.splitlines()[1:] if ln.strip() and "\tdevice" in ln]
        if self.serial:
            return any(ln.startswith(self.serial) for ln in lines)
        return len(lines) >= 1

    def screen_size(self) -> tuple[int, int]:
        """Return (width, height) in pixels."""
        out = self._run(["shell", "wm", "size"])  # e.g. "Physical size: 1080x2340"
        for token in out.replace("\n", " ").split():
            if "x" in token and token.replace("x", "").isdigit():
                w, h = token.split("x")
                return int(w), int(h)
        raise ADBError(f"could not parse screen size from: {out!r}")

    # ---- perception input ------------------------------------------------
    def screencap(self) -> np.ndarray:
        """Capture the current screen as an RGB numpy array (H, W, 3)."""
        png = self._run(["exec-out", "screencap", "-p"], binary=True)
        import cv2  # local import so the module loads even without opencv

        arr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ADBError("failed to decode screencap PNG")
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # ---- actions ---------------------------------------------------------
    def tap(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(int(x)), str(int(y))])
        self._settle()

    def swipe(self, x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> None:
        self._run(["shell", "input", "swipe", *map(lambda v: str(int(v)), (x1, y1, x2, y2)), str(ms)])
        self._settle()

    def text(self, value: str) -> None:
        # adb input text does not handle spaces well; encode them.
        self._run(["shell", "input", "text", value.replace(" ", "%s")])
        self._settle()

    def key(self, keycode: str) -> None:
        """e.g. keycode='KEYCODE_BACK' or 'KEYCODE_HOME'."""
        self._run(["shell", "input", "keyevent", keycode])
        self._settle()

    def _settle(self) -> None:
        time.sleep(self.settle_ms / 1000.0)


def get_device() -> Device:
    dev = CONFIG.get("device", {})
    return Device(serial=dev.get("serial") or None, settle_ms=int(dev.get("settle_ms", 900)))
