<!--
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# MoE control packets — runtime DMA-BD reprogramming

A small, self-contained reproduction of the **clever bit** of the GPT-OSS-20B MoE
decode layer (`mirror-flm-xclbin/GPTOSS/gpt_npu_bin`, documented in the repo-root
`GPTOSS_LAYER_ARCH.md`, §6 "MoE control-packet mechanism"): a compute core decides
at runtime *which* data to fetch, then **rewrites a DMA buffer descriptor on the
fly** by emitting AIE *control packets* over a statically-routed packet flow to
another tile's `TileControl` port — no host round-trip.

In GPT-OSS the `router_slave` core picks a token's top-4 experts and control-packet-
reprograms the four expert-weight **shim** tiles so each streams the selected
expert's MXFP4 weights from DDR into the projection tiles. The control-packet path
is identical whether the target is a shim tile (DDR) or a memtile (on-chip); this
example drives a **memtile** (the proven path from `../microbench/memtile_program_cost`)
to keep it standalone and runnable, and adds the MoE-defining ingredient: the BD
address that gets programmed is **computed from runtime data**.

## What it does

```
  shim(0,0) ──scores──► core(0,2) S2MM0          host supplies per-expert scores
  core(0,2) ── top-k argmax  (the "router")
  core(0,2) MM2S0 ──control packets──► memtile(0,1) TileControl
                                        └─ reprograms memtile MM2S0 BD:
                                           src = experts[idx], len = CHUNK
  memtile(0,1) MM2S0 ──experts[idx]──► core(0,2) S2MM1
  core(0,2) MM2S1 ──results──► shim(0,0) S2MM0   selected idxs + fetched chunks
```

The memtile holds `NUM_EXPERT` "expert weight" chunks (`CHUNK` int32 each, baked
with a known pattern `experts[e][i] == 1000*e + i`). The host sets `scores`; the
core argmax-selects the `TOPK` winners at runtime and, for each, builds a 23-word
control program whose BD source-address word is `base + idx*CHUNK`, sends it to the
memtile's `TileControl` port, and receives exactly that expert's chunk. The host
verifies the selected indices **and** the fetched data, proving the BD address was
patched from runtime data.

## GPT-OSS correspondence

| this example | GPT-OSS (`gpt_npu_bin`) |
|---|---|
| `scores` (host int32) | 32 bf16 router logits + bias |
| top-k argmax | `select_top_k_with_index` (bubble sort + stable softmax) |
| memtile MM2S BD patch | `configure_BD_x_MM2S_y_dma_bd_len` on the expert shim BD |
| 23-word control program | `control_packet_gen` / `packet_flow_gen` (`packet_control_gen.hpp`) |
| `packetflow(... TileControl)` | `packetflow(pkt, router_slave → IT[col] TileControl)` |
| memtile (on-chip) source | expert-weight **shim** tile (DDR) |

Deliberate simplifications: target tile (memtile vs shim/DDR), and selection
(argmax vs sort+softmax). The control-packet emission and BD-rewrite are the same.

## Running

```bash
source env.sh        # from repo root
cd mlir-aie/programming_examples/pythoc/moe_control_packets

python moe_control_packets.py                              # random scores
python moe_control_packets.py --scores 10,90,30,40,99,20,5,70   # experts 4,1 win
```

Expected tail:

```
  selected (hw) : [4, 1]
  expert[4] first/last fetched: 4000 / 4015
  expert[1] first/last fetched: 1000 / 1015
  top-k select : OK
  fetched data : OK
PASS
```

Requires XRT + a Ryzen AI NPU2 (AIE2P) device and a built `mlir-aie` (see the
repo `CLAUDE.md`). The lit `RUN:` line builds and runs it on hardware.

## Two non-obvious requirements (learned the hard way)

Both bit during development and are worth knowing if you adapt this pattern:

1. **`ctrl_buf` must be `volatile`** (`ptr[volatile[i32], True]`). The control
   program is consumed by the MM2S DMA hardware, never read back by the core. LLVM
   can't see the DMA reading the buffer, marks the pointer `writeonly`, and
   **dead-store-eliminates** the constant program words — the memtile then receives
   garbage and never streams (its completion lock stays 0). `volatile` keeps every
   store and its ordering. (Inspect `build/pythoc_objects/moe_router.opt.ll`: without
   volatile only the one runtime store to `ctrl_buf[8]` survives.)

2. **The fetch loop must be `for k in range(TOPK)` (trace-time unrolled), not a
   runtime `while`.** A real loop around the control-DMA launch does not deliver on
   this path; unrolling to straight-line code does. (GPT-OSS likewise issues its
   per-expert fetches as unrolled/explicit calls.)

Neither is needed in `memtile_program_cost`, which sends a single, all-constant
control program once — exactly the case that hides both issues.

## Retargeting at a shim tile (GPT-OSS's real configuration)

Only the addressed registers change; the `shim` register module matches
`packet_control_gen.hpp`:

| register | offset | hpp constant |
|---|---|---|
| `DMA_BD0_0` | `0x1D000` | `get_SHM_bd_x_0_address(bd) = 0x1D000 + bd*0x20` |
| `DMA_MM2S_0_Ctrl` | `0x1D210` | — |
| `DMA_MM2S_0_Task_Queue` | `0x1D214` | `Shimtile_MM2S_X_TASK_QUEUE_addr` |

Point the `packetflow` dest at `{shim_tile, WireBundle.TileControl, 0}`. The BD
source becomes the DDR buffer-object virtual base + expert offset; GPT-OSS first
reads that virtual base back over a *response* packet flow
(`get_SHM_bd_virtual_address` → `control_packet_gen` with `operation=read`) before
patching, which this example omits.

## Files

- `moe_control_packets.py` — the example (host driver + `@aie_kernel` router + IRON placement)
- `README.md` — this file
