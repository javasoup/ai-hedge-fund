# Autoterm / Timberline read-only sniffer

A **listen-only** tool to answer one question before you invest in any heater
integration:

> Does the link between my Timberline (Elwell) control box and the Autoterm
> Binar burner speak the **stock Autoterm protocol** — the one existing
> open-source libraries decode — or a **custom Elwell framing**?

If it's stock Autoterm, projects like
[csreades/AutothermDieselRepeater](https://github.com/csreades/AutothermDieselRepeater)
and [prclm/AutotermHeaterController](https://github.com/prclm/AutotermHeaterController)
should adapt to your van. If it isn't, you'll know you're facing
reverse-engineering rather than a drop-in — before you cut a single wire for
control.

## ⚠️ Safety — read this first

This is a **diesel combustion + engine-coolant** heater. The Elwell control box
manages the burn, the coolant pump, and the safety cutoffs.

- This tool **only reads**. Wire **only the adapter's RX and GND**. **Leave the
  adapter's TX pin disconnected** so it is physically incapable of transmitting
  to the heater.
- Do **not** send commands to the burner while the Elwell box is controlling it,
  and do **not** bypass the safety controller. Monitoring is safe; control is a
  separate, careful project.

## Hardware

- A **5V-capable USB–UART adapter** (e.g. FT232). Set it to **5V logic**, not
  3.3V — the Autoterm link is 5V.
- Jumper wires. Connect:
  - adapter **GND** → system/heater **GND** (shared ground is essential)
  - adapter **RX** → the **data wire** you want to listen to
  - adapter **TX** → **nothing** (deliberately)

The panel↔heater link uses **two separate data wires**, one per direction:

```
  [Elwell control box] ==(wire 1: panel->heater)==> [Autoterm Binar]
        ^                                                  |
        ============(wire 2: heater->panel)================
```

Tap **one** wire to identify the protocol. Tap **both** (two adapters →
`--port` and `--port2`) to capture the full conversation. Find the wires from
the Binar's control connector; the
[Elwell Timberline technical manual](https://www.manualslib.com/manual/3961970/Elwell-Timberline-1-0.html)
has the control-box pinout.

## Run

```bash
pip install -r requirements.txt

# verify the decoder with no hardware:
python selftest.py

# listen on one data line (default 2400 baud, 8N1):
python autoterm_sniff.py --port /dev/ttyUSB0

# both directions, print raw hex, stop after 30s:
python autoterm_sniff.py --port /dev/ttyUSB0 --port2 /dev/ttyUSB1 --raw --seconds 30
```

Operate the heater from its touchscreen (turn on, change setpoint) while it
captures — that's what generates the traffic worth reading.

## Reading the verdict

At exit the tool prints a summary and one of:

| Verdict | Meaning | Next step |
|---------|---------|-----------|
| **NO DATA** | Nothing on the line | Recheck RX wire / shared GND / 5V setting; sweep `--baud` (9600, 4800, 115200) |
| **BYTES but NO valid frames** | Wrong baud, or non-standard framing | Sweep baud first; if still nothing decodes, it's custom Elwell → reverse-engineering |
| **PARTIAL match** | Some frames pass CRC, but irregular | Likely Autoterm-derived with Elwell tweaks; deeper decode before trusting a library |
| **LOOKS LIKE STOCK AUTOTERM** | Frames pass CRC16/MODBUS with valid 0x03/0x04 directions | The existing libraries should adapt — build **read-only monitoring** next |

## What it decodes

The Autoterm frame (community-reverse-engineered):

```
0xAA | TYPE | LEN | 0x00 | ID | payload(LEN bytes) | CRC16-MODBUS (2 bytes)
       0x03=panel->heater
       0x04=heater->panel
```

`autoterm_protocol.py` validates the CRC (accepting either byte order and
reporting which matched — itself a useful fingerprint) and never transmits.
`selftest.py` proves the decoder on synthetic frames with no serial port.

## Files

| File | Role |
|------|------|
| `autoterm_protocol.py` | Frame parser + CRC16/MODBUS (decode only) |
| `autoterm_sniff.py` | Read-only serial capture CLI + verdict |
| `selftest.py` | No-hardware verification of the decoder |
| `requirements.txt` | `pyserial` |
