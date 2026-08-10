#!/usr/bin/env python3
"""No-hardware self-test: build valid Autoterm frames and confirm the parser
decodes them, resynchronizes past garbage, and rejects bad CRCs.

Run: python selftest.py   (needs no serial port, no heater)
"""
from __future__ import annotations

from autoterm_protocol import (
    CRC_LEN,
    DIR_FROM_HEATER,
    DIR_TO_HEATER,
    HEADER_LEN,
    FrameParser,
    crc16_modbus,
)


def build_frame(direction: int, msg_id: int, payload: bytes, *, order: str = "big") -> bytes:
    body = bytes([0xAA, direction, len(payload), 0x00, msg_id]) + payload
    crc = crc16_modbus(body)
    hi, lo = (crc >> 8) & 0xFF, crc & 0xFF
    tail = bytes([hi, lo]) if order == "big" else bytes([lo, hi])
    return body + tail


def main() -> int:
    p = FrameParser()

    # 1. A clean request + response.
    req = build_frame(DIR_TO_HEATER, 0x01, bytes([0x00, 0x2A]))
    resp = build_frame(DIR_FROM_HEATER, 0x01, bytes([0x01, 0x64, 0x00, 0x05]), order="little")
    frames = p.feed(req + resp)
    assert len(frames) == 2, f"expected 2 frames, got {len(frames)}"
    assert frames[0].direction_name == "panel->heater" and frames[0].crc_ok
    assert frames[1].direction_name == "heater->panel" and frames[1].crc_ok
    assert frames[1].crc_byte_order == "little"
    assert frames[0].payload == bytes([0x00, 0x2A])
    print("ok: clean request/response, both CRC orders")

    # 2. Leading garbage before a real frame must be skipped.
    frames = p.feed(b"\x00\xff\x13" + build_frame(DIR_TO_HEATER, 0x0F, b"\x99"))
    assert len(frames) == 1 and frames[0].msg_id == 0x0F, "resync after garbage failed"
    print("ok: resync past leading garbage")

    # 3. A corrupted CRC must NOT be reported as a valid frame.
    bad = bytearray(build_frame(DIR_FROM_HEATER, 0x02, b"\x11\x22"))
    bad[-1] ^= 0xFF
    frames = p.feed(bytes(bad))
    assert all(f.crc_ok is False or f.msg_id != 0x02 for f in frames), "bad CRC accepted"
    print("ok: bad CRC rejected")

    # 4. Fragmented delivery (byte-at-a-time) still assembles one frame.
    p2 = FrameParser()
    whole = build_frame(DIR_TO_HEATER, 0x03, b"\xde\xad\xbe\xef")
    got = []
    for b in whole:
        got += p2.feed(bytes([b]))
    assert len(got) == 1 and got[0].payload == b"\xde\xad\xbe\xef", "stream reassembly failed"
    print("ok: byte-at-a-time reassembly")

    print("\nALL SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
