#!/usr/bin/env python3
"""Read-only Autoterm/Timberline compatibility sniffer.

Listens to the UART between your Timberline (Elwell) control box and the Autoterm
Binar burner and answers ONE question: does that link speak the stock Autoterm
protocol that existing open-source libraries decode, or a custom Elwell framing?

It NEVER transmits. Wire only the adapter's RX and GND (leave TX disconnected)
so it is physically incapable of sending anything to the heater.

Usage:
    python autoterm_sniff.py --port /dev/ttyUSB0                 # one data line
    python autoterm_sniff.py --port /dev/ttyUSB0 --port2 /dev/ttyUSB1
    python autoterm_sniff.py --port /dev/ttyUSB0 --baud 9600 --seconds 30 --raw

Tips:
  * The Autoterm panel<->heater link is commonly 2400 baud, 8N1. If you see
    bytes but zero valid frames, try --baud 9600 / 4800 / 115200.
  * The two directions travel on two separate data wires. Tap one to identify
    the protocol; tap both (--port + --port2, two adapters) to see everything.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from autoterm_protocol import DIR_FROM_HEATER, DIR_TO_HEATER, FrameParser


def _open_serial(port: str, baud: int):
    try:
        import serial  # pyserial
    except ImportError:
        sys.exit("pyserial not installed — run: pip install -r requirements.txt")
    try:
        return serial.Serial(port=port, baudrate=baud, bytesize=8,
                             parity="N", stopbits=1, timeout=0.05)
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        sys.exit(f"could not open {port} @ {baud}: {exc}")


class Stats:
    def __init__(self) -> None:
        self.bytes = 0
        self.frames = 0
        self.crc_ok = 0
        self.directions: Counter[str] = Counter()
        self.msg_ids: Counter[int] = Counter()
        self.crc_orders: Counter[str] = Counter()


def _verdict(s: Stats) -> str:
    if s.bytes == 0:
        return (
            "NO DATA on the line.\n"
            "  -> Check: adapter RX on a real data wire? GND shared? adapter set to 5V?\n"
            "  -> Try a different --baud (9600, 4800, 115200) and confirm the heater is powered."
        )
    if s.frames == 0 or s.crc_ok == 0:
        return (
            "BYTES but NO valid Autoterm frames.\n"
            "  -> Most likely wrong baud (try 9600/4800/115200), OR\n"
            "  -> the Elwell<->Binar link uses a NON-standard framing.\n"
            "     If baud sweeps still fail, expect reverse-engineering rather than a drop-in library."
        )
    ratio = s.crc_ok / max(s.frames, 1)
    known_dir = set(s.directions) <= {"panel->heater", "heater->panel"}
    if ratio >= 0.7 and known_dir:
        return (
            "LOOKS LIKE STOCK AUTOTERM.\n"
            f"  -> {s.crc_ok}/{s.frames} frames passed CRC16/MODBUS with valid 0x03/0x04 directions.\n"
            "  -> Existing libraries (csreades/AutothermDieselRepeater, prclm/AutotermHeaterController)\n"
            "     should adapt. Safe next step: build read-only monitoring, THEN consider control."
        )
    return (
        "PARTIAL match — some valid frames, but irregular.\n"
        f"  -> {s.crc_ok}/{s.frames} passed CRC; directions seen: {dict(s.directions)}.\n"
        "  -> Possibly Autoterm-derived with Elwell changes. Worth a deeper decode before trusting a library."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only Autoterm protocol compatibility sniffer")
    ap.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyUSB0")
    ap.add_argument("--port2", help="optional second data line (other direction)")
    ap.add_argument("--baud", type=int, default=2400, help="baud rate (default 2400)")
    ap.add_argument("--seconds", type=float, default=0, help="capture duration; 0 = until Ctrl-C")
    ap.add_argument("--raw", action="store_true", help="also print raw frame hex")
    args = ap.parse_args()

    ports = [(args.port, _open_serial(args.port, args.baud), FrameParser(), "A")]
    if args.port2:
        ports.append((args.port2, _open_serial(args.port2, args.baud), FrameParser(), "B"))

    stats = Stats()
    print(f"listening @ {args.baud} 8N1 on {', '.join(p[0] for p in ports)} "
          f"({'until Ctrl-C' if args.seconds == 0 else f'{args.seconds:g}s'}) — read-only\n")
    start = time.monotonic()
    try:
        while True:
            if args.seconds and (time.monotonic() - start) >= args.seconds:
                break
            idle = True
            for name, ser, parser, tag in ports:
                chunk = ser.read(4096)
                if not chunk:
                    continue
                idle = False
                stats.bytes += len(chunk)
                for f in parser.feed(chunk):
                    stats.frames += 1
                    stats.crc_ok += int(f.crc_ok)
                    stats.directions[f.direction_name] += 1
                    stats.msg_ids[f.msg_id] += 1
                    if f.crc_ok:
                        stats.crc_orders[f.crc_byte_order] += 1
                    label = f"[{tag}] " if args.port2 else ""
                    print(label + f.summary())
                    if args.raw:
                        print(f"      raw: {f.raw.hex(' ')}")
            if idle:
                time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        for _, ser, _, _ in ports:
            ser.close()

    print("\n" + "=" * 60)
    print(f"bytes captured : {stats.bytes}")
    print(f"frames parsed  : {stats.frames}  (CRC ok: {stats.crc_ok})")
    if stats.directions:
        print(f"directions     : {dict(stats.directions)}")
    if stats.crc_orders:
        print(f"crc byte order : {dict(stats.crc_orders)}")
    if stats.msg_ids:
        ids = ", ".join(f"0x{i:02X}({n})" for i, n in stats.msg_ids.most_common())
        print(f"message ids    : {ids}")
    print("=" * 60)
    print("VERDICT:")
    print(_verdict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
