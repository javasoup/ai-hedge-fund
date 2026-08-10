"""Autoterm / Planar (Binar) serial framing — decode only.

Community-reverse-engineered structure of the UART between an Autoterm heater
and its control panel:

    +--------+------+--------+------+------+-------- ... --------+---------+
    | 0xAA   | TYPE | LEN    | 0x00 | ID   |   payload (LEN B)   | CRC(2B) |
    +--------+------+--------+------+------+-------- ... --------+---------+
      preamble  dir   payload         msg                         CRC16
                      length          id                          MODBUS

  TYPE (direction):  0x03 = panel -> heater (request)
                     0x04 = heater -> panel (response)
  CRC: CRC16/MODBUS over every byte except the two CRC bytes. Some units send
       it high-byte-first, some low-byte-first, so we accept either and report
       which matched (a useful fingerprint on its own).

This module NEVER transmits. It only parses bytes you have already captured,
so it is safe to run against a live heater.

References:
  - https://github.com/schroeder-robert/autoterm-air-2d-serial-control
  - https://github.com/prclm/AutotermHeaterController
  - https://grimoire314.wordpress.com/2019/03/21/autoterm-planar-diesel-heater-controller-reverse-engineering-part-2/
"""
from __future__ import annotations

from dataclasses import dataclass

PREAMBLE = 0xAA
DIR_TO_HEATER = 0x03      # panel/controller -> heater (request)
DIR_FROM_HEATER = 0x04    # heater -> panel/controller (response)
HEADER_LEN = 5            # AA, TYPE, LEN, 0x00, ID
CRC_LEN = 2
MAX_PAYLOAD = 128         # sanity cap; a "length" above this is treated as noise


def crc16_modbus(data: bytes) -> int:
    """Standard CRC16/MODBUS (poly 0xA001, init 0xFFFF, reflected)."""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return crc & 0xFFFF


@dataclass
class Frame:
    raw: bytes
    direction: int
    msg_id: int
    payload: bytes
    crc_received: int
    crc_ok: bool
    crc_byte_order: str          # "big" | "little" | "none"

    @property
    def direction_name(self) -> str:
        return {
            DIR_TO_HEATER: "panel->heater",
            DIR_FROM_HEATER: "heater->panel",
        }.get(self.direction, f"0x{self.direction:02X}?")

    def summary(self) -> str:
        crc = "CRC ok" if self.crc_ok else "CRC BAD"
        order = f"/{self.crc_byte_order}" if self.crc_ok else ""
        return (
            f"{self.direction_name:14} id=0x{self.msg_id:02X} "
            f"len={len(self.payload):<3} {crc}{order}  "
            f"payload={self.payload.hex(' ')}"
        )


class FrameParser:
    """Incremental, self-resynchronizing parser.

    Feed it whatever bytes arrive; it returns any complete frames found. On any
    inconsistency it advances a single byte and hunts for the next 0xAA, so a
    noisy or non-Autoterm stream simply yields few/no valid frames rather than
    crashing — exactly the signal we want for the compatibility check.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self._buf.extend(data)
        frames: list[Frame] = []
        while True:
            frame, consumed = self._try_one()
            if consumed == 0:
                break
            del self._buf[:consumed]
            if frame is not None:
                frames.append(frame)
        return frames

    def _try_one(self) -> tuple[Frame | None, int]:
        buf = self._buf
        if not buf:
            return None, 0

        # Align to a preamble.
        if buf[0] != PREAMBLE:
            idx = buf.find(PREAMBLE)
            return (None, len(buf)) if idx == -1 else (None, idx)

        if len(buf) < HEADER_LEN:
            return None, 0                       # need the length byte first

        length = buf[2]
        if length > MAX_PAYLOAD:                 # implausible -> false preamble
            return None, 1

        total = HEADER_LEN + length + CRC_LEN
        if len(buf) < total:
            return None, 0                       # wait for the rest of the frame

        raw = bytes(buf[:total])
        crc_calc = crc16_modbus(raw[:-CRC_LEN])
        crc_big = (raw[-2] << 8) | raw[-1]
        crc_little = (raw[-1] << 8) | raw[-2]
        if crc_calc == crc_big:
            ok, order, recv = True, "big", crc_big
        elif crc_calc == crc_little:
            ok, order, recv = True, "little", crc_little
        else:
            # Bad CRC at this alignment: treat the preamble as spurious.
            return None, 1

        frame = Frame(
            raw=raw,
            direction=raw[1],
            msg_id=raw[4],
            payload=raw[HEADER_LEN:HEADER_LEN + length],
            crc_received=recv,
            crc_ok=ok,
            crc_byte_order=order,
        )
        return frame, total
