<!---//===- README.md ---------------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Out-of-order S2MM on NPU3

`N` packet senders merge into one S2MM channel that is in out-of-order (OoO)
mode. Each packet lands in the receive buffer descriptor (BD) named by the BD
lookup index in its packet header, whatever order the packets arrive in.
AIE4 S2MM channels in core, memory and shim tiles support this mode, which
works as it does on AIE2.

## How it works

* **Senders.** Each sender BD sets `out_of_order_id`, the 4-bit BD lookup
  index (0..15) that the MM2S channel writes into bits [8:5] of the packet
  header. Each sender also has its own route `pkt_id`. The two fields are
  separate. Every packet flow sets `keep_pkt_header = true` because the
  receiving S2MM channel reads the header.
* **Receiver.** An `aiex.dma_configure_task(..., S2MM, ...) {out_of_order}`
  lists one receive BD per lookup index. Lowering sets `Enable_Out_of_Order`
  in the channel's `DMA_S2MM_<n>_Ctrl` register. The task's `repeat_count`
  is the number of packets minus one, and the task can't issue a
  task-complete token because the lookup index uses the same header bits.
  Hardware ignores `Next_BD` in this mode, so the BDs are written unchained.
* **Completion.** Each receive BD releases a counting lock. A consumer
  acquires the lock with a count of `N` and then reports completion with a
  token that the host can wait on.

Sender `s` sends input slice `s` with lookup index `(s + 1) % 4`. So slot
`(s + 1) % 4` of the output holds slice `s`. The output is rotated relative
to the input, which shows that the header index places each packet.

## Designs

| Design | Receiver | Receive BDs | Completion |
| --- | --- | --- | --- |
| `memtile_recv.mlir` | MEM (0,1) S2MM 0 | private BDs 0-3 | MEM MM2S 0 acquires `full` >= 4 and drains the merged buffer to shim (0,0) S2MM 0, which issues the token |
| `shim_upper_recv.mlir` | shim (0,0) S2MM 1 | BDs 16-19 (upper bank) | shim S2MM 0 acquires `cons` >= 4, then receives a looped-back sentinel and issues the token |

The senders are four shim MM2S channels in two columns. Neither design has
an `aie.core`.

### BD lookup index is local to the receiving channel's BD bank

On AIE4, each S2MM channel can see 16 BDs:

* A core tile has 16 BDs, shared by all of its channels.
* Each memory tile channel has 16 private BDs, with local ids 0-15.
* A shim tile splits its 32 BDs into two banks. S2MM 0 uses BDs 0-15, and
  S2MM 1 uses BDs 16-31.

The 4-bit index names a BD inside the receiving channel's bank, so index `k`
selects BD `bank_start + k`. On shim S2MM 1, that is BD `16 + k`. The shim's
`Next_BD` field and the task's start-BD field use the same bank-local
numbering. `shim_upper_recv.mlir` shows this on hardware: indices 0-3 place
the packets into BDs 16-19.

## Running

```
make          # builds memtile_recv.elf, shim_upper_recv.elf, test.exe
make run      # runs both on an NPU3 device
```

`test.exe <elf> <words>` prints `PASS: <elf>: ...` when every slot holds the
expected slice. If two senders swap lookup indices, exactly their two slots
(128 words) mismatch. That result shows the header index chooses where each
packet lands.
