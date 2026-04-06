#!/usr/bin/env python3
# (c) Copyright 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
AIE New Trace Parser

Parses AIE trace data captured via the new declarative trace API (aie.trace ops).
Handles two trace modes not covered by the standard parse.py infrastructure:

  - Event-PC mode  (mode 01): captures PC address when each configured event fires
  - Execution mode (mode 10): captures branch/loop control flow (E_atom, N_atom, New_PC, LC)

For Event-Time mode (mode 00) use the standard:
  python/utils/trace/parse.py --input trace.txt --mlir design.mlir --output trace.json

Usage:
    parse_trace.py --input trace.txt --output trace.json
    parse_trace.py --input trace.txt --output trace.json --mode eventpc
    parse_trace.py --input trace.txt --output trace.json --mode execution
    parse_trace.py --input trace.txt --output trace.json -v 2
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Frame type enumeration (all modes)
# ---------------------------------------------------------------------------

class FrameType(Enum):
    """AIE trace frame types across all modes."""
    PACKET_HEADER = "PacketHeader"
    START         = "Start"         # 64-bit (Event-Time/Event-PC) or 32-bit (Execution)
    STOP          = "Stop"          # 32-bit
    EVENT_PC      = "Event_PC"      # 32-bit — Event-PC mode only
    E_ATOM        = "E_atom"        # 4-bit  — Execution mode: direct branch taken
    N_ATOM        = "N_atom"        # 4-bit  — Execution mode: direct branch not taken
    NEW_PC        = "New_PC"        # 16-bit — Execution mode: indirect branch with new PC
    LC            = "LC"            # 32-bit — Execution mode: loop counter update
    SINGLE0       = "Single0"       # 8-bit  — Event-Time: single event, 4-bit cycle count
    SINGLE1       = "Single1"       # 16-bit — Event-Time: single event, 10-bit cycle count
    SINGLE2       = "Single2"       # 24-bit — Event-Time: single event, 18-bit cycle count
    MULTIPLE0     = "Multiple0"     # 16-bit — Event-Time: multi-event, 4-bit cycle count
    MULTIPLE1     = "Multiple1"     # 24-bit — Event-Time: multi-event, 10-bit cycle count
    MULTIPLE2     = "Multiple2"     # 32-bit — Event-Time: multi-event, 18-bit cycle count
    FILLER        = "Filler"        # 4-bit or 8-bit padding
    REPEAT0       = "Repeat0"       # 8-bit  — repeat count < 16
    REPEAT1       = "Repeat1"       # 16-bit — repeat count 0–1023
    SYNC          = "Sync"          # 8-bit  — heartbeat / flush
    UNKNOWN       = "Unknown"


# ---------------------------------------------------------------------------
# Trace event dataclass
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    """One decoded trace frame."""
    index:         int
    frame_type:    FrameType
    raw_hex:       str
    bit_length:    int = 32

    # Shared fields
    packet_id:    Optional[int]  = None
    pc_address:   Optional[int]  = None   # 14-bit PC (Event-PC Start or Execution Stop)
    timer_value:  Optional[int]  = None   # 56-bit free-running clock (Start frame)
    trace_mode:   Optional[int]  = None   # 0=Event-Time, 1=Event-PC, 2=Execution
    overrun:      Optional[bool] = None   # Start frame overrun flag

    # Event-PC mode
    events_fired: Optional[List[int]] = None  # positional indices of events that fired

    # Execution mode
    branch_taken: Optional[bool] = None
    new_pc:       Optional[int]  = None
    loop_count:   Optional[int]  = None
    lc_overflow:  Optional[bool] = None

    # Event-Time mode
    event_id:     Optional[int]  = None   # 3-bit event index (Single frames)
    event_mask:   Optional[int]  = None   # 8-bit event bitmask (Multiple frames)
    cycle_count:  Optional[int]  = None   # delta cycles since previous event

    # Repeat compression
    repeat_count: Optional[int]  = None

    # Packet header fields
    pkt_col:  Optional[int] = None
    pkt_row:  Optional[int] = None
    pkt_type: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "index":      self.index,
            "type":       self.frame_type.value,
            "raw_value":  self.raw_hex,
            "bit_length": self.bit_length,
        }
        if self.packet_id   is not None: d["packet_id"]   = self.packet_id
        if self.pkt_col     is not None: d["col"]         = self.pkt_col
        if self.pkt_row     is not None: d["row"]         = self.pkt_row
        if self.pkt_type    is not None: d["pkt_type"]    = self.pkt_type
        if self.pc_address  is not None: d["pc_address"]  = f"0x{self.pc_address:04x}"
        if self.timer_value is not None: d["timer_value"] = f"0x{self.timer_value:014x}"
        if self.trace_mode  is not None:
            d["trace_mode"] = {0: "Event-Time", 1: "Event-PC", 2: "Execution"}.get(
                self.trace_mode, f"Mode-{self.trace_mode}"
            )
        if self.overrun      is not None: d["overrun"]      = self.overrun
        if self.events_fired is not None: d["events_fired"] = self.events_fired
        if self.branch_taken is not None: d["branch_taken"] = self.branch_taken
        if self.new_pc       is not None: d["new_pc"]       = f"0x{self.new_pc:04x}"
        if self.loop_count   is not None: d["loop_count"]   = self.loop_count
        if self.lc_overflow  is not None: d["lc_overflow"]  = self.lc_overflow
        if self.event_id     is not None: d["event_id"]     = self.event_id
        if self.event_mask   is not None: d["event_mask"]   = f"0b{self.event_mask:08b}"
        if self.cycle_count  is not None: d["cycle_count"]  = self.cycle_count
        if self.repeat_count is not None: d["repeat_count"] = self.repeat_count
        return d


# ---------------------------------------------------------------------------
# Trace mode constants
# ---------------------------------------------------------------------------

MODE_EVENT_TIME = 0
MODE_EVENT_PC   = 1
MODE_EXECUTION  = 2


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class AIENewTraceParser:
    """
    Parser for AIE trace data produced by the new declarative trace API.

    Supports Event-PC mode and Execution mode.  Mode is auto-detected from
    each packet's Start frame; an explicit override can be supplied via the
    ``default_mode`` constructor argument.

    Frame layout references
    -----------------------
    All sub-32-bit frames are MSB-aligned within the 32-bit word they occupy.

    Common (all modes):
      PacketHdr (32b): verified with odd parity; carries col/row/type/id
      Start (64b):     bits[31:27]=11110, bit[26]=overrun, bits[25:24]=mode,
                       bits[23:0]=timer[55:32] || bits[31:0]=timer[31:0]
      Start (32b, Execution only):
                       bits[31:27]=11110, bit[26]=overrun, bits[25:24]=10,
                       bits[13:0]=PC (14-bit)
      Stop (32b):      bits[31:26]=110111
      Sync (8b):       bits[31:24]=11111111  (MSB-aligned)
      Filler (8b):     bits[31:24]=11111110  (MSB-aligned; Execution: 11101000)
      Repeat0 (8b):    bits[31:28]=1110, bits[27:24]=count (MSB-aligned)
      Repeat1 (16b):   bits[31:26]=110110, bits[25:16]=count (MSB-aligned)

    Event-PC mode (mode 01):
      Event_PC (32b):  bits[31:26]=110001, bits[25:18]=event_mask (8 positional bits),
                       bits[17:14]=reserved, bits[13:0]=PC (14-bit)

    Execution mode (mode 10):
      LC (32b):        bits[31:29]=010, bit[28]=overflow, bits[27:0]=loop_count
      New_PC (16b):    bits[31:30]=10, bits[29:16]=new_pc (14-bit)  (MSB-aligned)
      E_atom (4b):     bits[31:28]=0001  (branch taken)             (MSB-aligned)
      N_atom (4b):     bits[31:28]=0000  (branch not taken)         (MSB-aligned)
      Filler0 (4b):    bits[31:28]=0010                              (MSB-aligned)
    """

    def __init__(
        self,
        trace_file: str,
        default_mode: Optional[int] = None,
        verbosity: int = 1,
        event_names: Optional[List[str]] = None,
    ):
        self.trace_file   = trace_file
        self.default_mode = default_mode  # None = auto-detect per stream
        self.verbosity    = verbosity
        # Optional names for the 8 configurable trace events (index 0-7).
        # Defaults to "Event_0" … "Event_7" when not supplied.
        self.event_names: List[str] = event_names or [f"Event_{i}" for i in range(8)]
        self.raw_words:   List[int]        = []
        self.frames:      List[TraceEvent] = []
        # Per-stream mode state: key = (col, row, pkt_type, pkt_id) -> mode int
        self._stream_mode: Dict[tuple, int] = {}

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load hex-encoded trace words (one per line) from ``trace_file``."""
        with open(self.trace_file) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        self.raw_words.append(int(line, 16))
                    except ValueError:
                        if self.verbosity >= 2:
                            print(f"Warning: skipping non-hex line: {line!r}", file=sys.stderr)
        if self.verbosity >= 1:
            nz = sum(1 for w in self.raw_words if w)
            print(f"Loaded {len(self.raw_words)} words ({nz} non-zero)")

    # ------------------------------------------------------------------
    # Packet header parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _odd_parity(word: int) -> bool:
        return bin(word).count("1") % 2 == 1

    @classmethod
    def _parse_packet_header(cls, word: int) -> Optional[Dict[str, int]]:
        """
        Attempt to parse *word* as a packet header.
        Returns a dict with col/row/pkt_type/pkt_id, or None if invalid.
        """
        if not cls._odd_parity(word):
            return None
        # Unused bits must be zero
        if ((word >> 5) & 0x7F) or ((word >> 19) & 0x1) or ((word >> 28) & 0x7):
            return None
        return {
            "col":      (word >> 21) & 0x7F,
            "row":      (word >> 16) & 0x1F,
            "pkt_type": (word >> 12) & 0x3,
            "pkt_id":   word & 0x1F,
        }

    # ------------------------------------------------------------------
    # Single-word classifier
    # ------------------------------------------------------------------

    def _current_mode(self, stream_key: Optional[tuple]) -> Optional[int]:
        # Per-stream mode (set by a Start frame) always takes priority.  The
        # default_mode only applies to streams where no Start frame has been seen yet.
        if stream_key is not None:
            known = self._stream_mode.get(stream_key)
            if known is not None:
                return known
        return self.default_mode  # may be None

    def _classify(
        self,
        word: int,
        idx: int,
        next_word: Optional[int],
        stream_key: Optional[tuple],
    ) -> TraceEvent:
        hex_str  = f"{word:08x}"
        mode     = self._current_mode(stream_key)

        b31_26 = (word >> 26) & 0x3F
        b31_30 = (word >> 30) & 0x3
        b31_29 = (word >> 29) & 0x7
        b31_28 = (word >> 28) & 0xF
        b31_27 = (word >> 27) & 0x1F

        # ---- Packet header ----
        # Always check packet headers first.  The structural constraints (odd
        # parity + multiple zero-bit fields) make false positives extremely
        # unlikely against real trace frames (E_atom, N_atom, etc.), and packet
        # headers genuinely appear at every 8-word packet boundary regardless
        # of stream mode.
        hdr = self._parse_packet_header(word)
        if hdr is not None:
            return TraceEvent(
                index=idx, frame_type=FrameType.PACKET_HEADER, raw_hex=hex_str,
                bit_length=32,
                packet_id=hdr["pkt_id"], pkt_col=hdr["col"],
                pkt_row=hdr["row"], pkt_type=hdr["pkt_type"],
            )

        # ---- Start frame ----
        if b31_27 == 0b11110:
            overrun    = bool((word >> 26) & 0x1)
            trace_mode = (word >> 24) & 0x3
            if stream_key is not None:
                old = self._stream_mode.get(stream_key)
                if old is not None and old != trace_mode and self.verbosity >= 1:
                    print(
                        f"Warning: mode change on stream {stream_key}: "
                        f"{old} -> {trace_mode}",
                        file=sys.stderr,
                    )
                self._stream_mode[stream_key] = trace_mode

            if trace_mode == MODE_EXECUTION:
                # 32-bit Start with PC in bits[13:0]
                return TraceEvent(
                    index=idx, frame_type=FrameType.START, raw_hex=hex_str,
                    bit_length=32, trace_mode=trace_mode, overrun=overrun,
                    pc_address=word & 0x3FFF,
                )
            else:
                # 64-bit Start with 56-bit timer
                timer_high = word & 0x00FFFFFF
                timer_low  = next_word if next_word is not None else 0
                return TraceEvent(
                    index=idx, frame_type=FrameType.START, raw_hex=hex_str,
                    bit_length=64, trace_mode=trace_mode, overrun=overrun,
                    timer_value=(timer_high << 32) | timer_low,
                )

        # ---- Stop frame: bits[31:26] = 110111 ----
        if b31_26 == 0b110111:
            # Execution: Stop carries current PC in bits[13:0]
            pc = (word & 0x3FFF) if mode == MODE_EXECUTION else None
            return TraceEvent(
                index=idx, frame_type=FrameType.STOP, raw_hex=hex_str,
                bit_length=32, pc_address=pc,
            )

        # ---- Sync: bits[31:24] = 0xFF (MSB-aligned) ----
        if ((word >> 24) & 0xFF) == 0xFF:
            return TraceEvent(
                index=idx, frame_type=FrameType.SYNC, raw_hex=hex_str, bit_length=8,
            )

        # ---- Filler (8-bit): bits[31:24] = 0xFE (MSB-aligned) ----
        if ((word >> 24) & 0xFF) == 0xFE:
            return TraceEvent(
                index=idx, frame_type=FrameType.FILLER, raw_hex=hex_str, bit_length=8,
            )

        # ---- Repeat1 (16-bit): bits[31:26] = 110110 ----
        if b31_26 == 0b110110:
            return TraceEvent(
                index=idx, frame_type=FrameType.REPEAT1, raw_hex=hex_str,
                bit_length=16, repeat_count=(word >> 16) & 0x3FF,
            )

        # ---- Repeat0 (8-bit): bits[31:28] = 1110 ----
        if b31_28 == 0b1110:
            return TraceEvent(
                index=idx, frame_type=FrameType.REPEAT0, raw_hex=hex_str,
                bit_length=8, repeat_count=(word >> 24) & 0x0F,
            )

        # ---- Event-PC mode frames ----
        if mode == MODE_EVENT_PC or mode is None:
            # Event_PC (32b): bits[31:26] = 110001
            if b31_26 == 0b110001:
                mask = (word >> 18) & 0xFF
                return TraceEvent(
                    index=idx, frame_type=FrameType.EVENT_PC, raw_hex=hex_str,
                    bit_length=32,
                    events_fired=[i for i in range(8) if (mask >> i) & 1],
                    pc_address=word & 0x3FFF,  # 14-bit PC
                )

        # ---- Event-Time mode frames ----
        if mode == MODE_EVENT_TIME or mode is None:
            # Order matters: check longer/more-constrained patterns first.
            # Multiple2 (32b): bits[31:26] = 110101
            if b31_26 == 0b110101:
                return TraceEvent(
                    index=idx, frame_type=FrameType.MULTIPLE2, raw_hex=hex_str,
                    bit_length=32,
                    event_mask=(word >> 18) & 0xFF,
                    cycle_count=word & 0x3FFFF,
                )
            # Multiple1 (24b): bits[31:26] = 110100
            if b31_26 == 0b110100:
                return TraceEvent(
                    index=idx, frame_type=FrameType.MULTIPLE1, raw_hex=hex_str,
                    bit_length=24,
                    event_mask=(word >> 18) & 0xFF,
                    cycle_count=(word >> 8) & 0x3FF,
                )
            # Single2 (24b): bit[31]=1, bits[30:29]=01
            if ((word >> 31) & 1) == 1 and ((word >> 29) & 3) == 0b01:
                return TraceEvent(
                    index=idx, frame_type=FrameType.SINGLE2, raw_hex=hex_str,
                    bit_length=24,
                    event_id=(word >> 26) & 0x7,
                    cycle_count=(word >> 8) & 0x3FFFF,
                )
            # Single1 (16b): bit[31]=1, bits[30:29]=00
            if ((word >> 31) & 1) == 1 and ((word >> 29) & 3) == 0b00:
                return TraceEvent(
                    index=idx, frame_type=FrameType.SINGLE1, raw_hex=hex_str,
                    bit_length=16,
                    event_id=(word >> 26) & 0x7,
                    cycle_count=(word >> 16) & 0x3FF,
                )
            # Multiple0 (16b): bits[31:28] = 1100
            if b31_28 == 0b1100:
                return TraceEvent(
                    index=idx, frame_type=FrameType.MULTIPLE0, raw_hex=hex_str,
                    bit_length=16,
                    event_mask=(word >> 20) & 0xFF,
                    cycle_count=(word >> 16) & 0xF,
                )
            # Single0 (8b): bit[31]=0
            if ((word >> 31) & 1) == 0:
                return TraceEvent(
                    index=idx, frame_type=FrameType.SINGLE0, raw_hex=hex_str,
                    bit_length=8,
                    event_id=(word >> 28) & 0x7,
                    cycle_count=(word >> 24) & 0xF,
                )

        # ---- Execution mode frames ----
        if mode == MODE_EXECUTION or mode is None:
            # LC (32b): bits[31:29] = 010
            if b31_29 == 0b010:
                return TraceEvent(
                    index=idx, frame_type=FrameType.LC, raw_hex=hex_str,
                    bit_length=32,
                    loop_count=word & 0x0FFFFFFF,
                    lc_overflow=bool((word >> 28) & 0x1),
                )
            # New_PC (16b): bits[31:30] = 10
            if b31_30 == 0b10:
                return TraceEvent(
                    index=idx, frame_type=FrameType.NEW_PC, raw_hex=hex_str,
                    bit_length=16, new_pc=(word >> 16) & 0x3FFF,
                )
            # E_atom / N_atom / Filler0 (4-bit): bits[31:28]
            if b31_28 == 0b0001:
                return TraceEvent(
                    index=idx, frame_type=FrameType.E_ATOM, raw_hex=hex_str,
                    bit_length=4, branch_taken=True,
                )
            if b31_28 == 0b0000:
                return TraceEvent(
                    index=idx, frame_type=FrameType.N_ATOM, raw_hex=hex_str,
                    bit_length=4, branch_taken=False,
                )
            if b31_28 == 0b0010:
                return TraceEvent(
                    index=idx, frame_type=FrameType.FILLER, raw_hex=hex_str, bit_length=4,
                )

        return TraceEvent(
            index=idx, frame_type=FrameType.UNKNOWN, raw_hex=hex_str, bit_length=32,
        )

    # ------------------------------------------------------------------
    # Main parse loop
    # ------------------------------------------------------------------

    def parse(self) -> List[TraceEvent]:
        """Parse all loaded words into TraceEvent objects."""
        self.frames = []
        current_stream: Optional[tuple] = None
        idx = 0

        while idx < len(self.raw_words):
            word = self.raw_words[idx]
            if word == 0:
                idx += 1
                continue

            next_word = self.raw_words[idx + 1] if idx + 1 < len(self.raw_words) else None
            frame = self._classify(word, idx, next_word, current_stream)

            if frame.frame_type == FrameType.PACKET_HEADER:
                current_stream = (frame.pkt_col, frame.pkt_row, frame.pkt_type, frame.packet_id)

            self.frames.append(frame)

            # 64-bit Start frames consume two words
            if frame.frame_type == FrameType.START and frame.bit_length == 64:
                idx += 2
            else:
                idx += 1

        if self.verbosity >= 1:
            print(f"Parsed {len(self.frames)} frames")
        return self.frames

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def _count_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.frames:
            counts[f.frame_type.value] = counts.get(f.frame_type.value, 0) + 1
        return counts

    def summary(self) -> None:
        """Print a human-readable summary to stdout."""
        SEP = "=" * 80
        counts = self._count_types()
        total  = len(self.frames)

        # Determine dominant mode from Start frames
        modes_seen = {f.trace_mode for f in self.frames if f.trace_mode is not None}
        mode_names = {0: "Event-Time", 1: "Event-PC", 2: "Execution"}
        mode_label = "/".join(mode_names.get(m, str(m)) for m in sorted(modes_seen)) if modes_seen else "Unknown"

        print("\n" + SEP)
        print(f"{mode_label.upper()} TRACE SUMMARY")
        print(SEP)
        print(f"\nTotal words processed: {len(self.raw_words)}")
        print(f"Total events parsed: {total}")
        print(f"Total packets: {counts.get('PacketHeader', 0)}")

        print(f"\nFrame Type Distribution:")
        for name, count in sorted(counts.items()):
            pct = count / total * 100 if total else 0
            print(f"  {name:<20s}: {count:5d} ({pct:5.1f}%)")

        # ---- Event-PC mode analysis ----
        pc_frames = [f for f in self.frames if f.frame_type == FrameType.EVENT_PC]
        if pc_frames:
            event_hits = [0] * 8
            for f in pc_frames:
                for e in (f.events_fired or []):
                    if 0 <= e < 8:
                        event_hits[e] += 1
            total_hits = sum(event_hits)
            if total_hits:
                print(f"\nEvent Trigger Statistics:")
                for i, c in enumerate(event_hits):
                    if c:
                        pct = c / total_hits * 100
                        name = self.event_names[i] if i < len(self.event_names) else f"Event_{i}"
                        print(f"  {name:<20s}: {c:5d} ({pct:5.1f}%)")

            # PC address frequency
            pc_counts: Dict[int, int] = {}
            for f in pc_frames:
                if f.pc_address is not None:
                    pc_counts[f.pc_address] = pc_counts.get(f.pc_address, 0) + 1
            if pc_counts:
                top_pcs = sorted(pc_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                print(f"\nTop 10 Most Frequent PC Addresses (14-bit offset):")
                for pc, cnt in top_pcs:
                    print(f"  0x{pc:04x}: {cnt} times")

            # Repeating 3-PC patterns
            if len(pc_frames) >= 3:
                pattern_counts: Dict[tuple, int] = {}
                pcs = [f.pc_address for f in pc_frames if f.pc_address is not None]
                for i in range(len(pcs) - 2):
                    pat = (pcs[i], pcs[i + 1], pcs[i + 2])
                    pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
                if pattern_counts:
                    top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"\nTop 5 Repeating 3-PC Patterns:")
                    for pat, cnt in top_patterns:
                        print(f"  {'0x{:04x} -> 0x{:04x} -> 0x{:04x}'.format(*pat)}: {cnt} times")

            # Execution pattern analysis
            unique_pcs = len(pc_counts)
            print(f"\nExecution Pattern Analysis:")
            print(f"  Total PC captures: {len(pc_frames)}")
            print(f"  Unique PC addresses: {unique_pcs}")
            if unique_pcs:
                rep = len(pc_frames) / unique_pcs
                print(f"  Average repetition: {rep:.1f}x")
                if rep > 10:
                    print(f"  -> Indicates tight loop execution")

        # ---- Execution mode analysis ----
        exec_frames = [
            f for f in self.frames
            if f.frame_type in (FrameType.E_ATOM, FrameType.N_ATOM, FrameType.NEW_PC, FrameType.LC)
        ]
        if exec_frames:
            taken     = sum(1 for f in self.frames if f.frame_type == FrameType.E_ATOM)
            not_taken = sum(1 for f in self.frames if f.frame_type == FrameType.N_ATOM)
            indirect  = sum(1 for f in self.frames if f.frame_type == FrameType.NEW_PC)
            lc_count  = sum(1 for f in self.frames if f.frame_type == FrameType.LC)
            total_br  = taken + not_taken
            print(f"\nExecution-Trace Branch Statistics:")
            print(f"  Branches taken    : {taken}")
            if total_br:
                print(f"  Branches not taken: {not_taken} ({not_taken / total_br * 100:.0f}%)")
            print(f"  Indirect branches : {indirect}")
            print(f"  Loop counter frames: {lc_count}")

        print("\n" + SEP)

    # ------------------------------------------------------------------
    # JSON serialisation
    # ------------------------------------------------------------------

    def to_json_dict(self) -> Dict[str, Any]:
        counts = self._count_types()
        return {
            "trace_file":    self.trace_file,
            "total_words":   len(self.raw_words),
            "total_frames":  len(self.frames),
            "frame_counts":  counts,
            "frames":        [f.to_dict() for f in self.frames],
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  "-i", required=True, help="Trace file (hex words, one per line)")
    p.add_argument("--output", "-o", required=True, help="Output JSON file")
    p.add_argument(
        "--mode", choices=["auto", "eventpc", "execution", "eventtime"],
        default="auto",
        help="Force trace mode (default: auto-detect from Start frames). "
             "'eventtime' is not fully decoded here; use parse.py instead.",
    )
    p.add_argument(
        "--event-names", nargs="+", metavar="NAME",
        help="Names for trace events 0-7 (space-separated, e.g. INSTR_EVENT_0 LOCK_STALL ...)",
    )
    p.add_argument(
        "--verbosity", "-v", type=int, default=1, choices=[0, 1, 2],
        help="Verbosity level (0=quiet, 1=normal, 2=debug)",
    )
    p.add_argument(
        "--no-summary", dest="summary", action="store_false", default=True,
        help="Suppress the printed summary",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    mode_map = {"auto": None, "eventpc": MODE_EVENT_PC,
                "execution": MODE_EXECUTION, "eventtime": MODE_EVENT_TIME}
    forced_mode = mode_map[args.mode]

    parser = AIENewTraceParser(
        args.input,
        default_mode=forced_mode,
        verbosity=args.verbosity,
        event_names=args.event_names,
    )
    parser.load()
    parser.parse()

    with open(args.output, "w") as fh:
        json.dump(parser.to_json_dict(), fh, indent=2)

    if args.verbosity >= 1:
        print(f"Wrote {args.output}")

    if args.summary:
        parser.summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
