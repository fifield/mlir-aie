# Memtile DMA programming cost — core (single-transfer) vs CDO vs host (NPU2 / aie2p)

**What this measures:** when a compute core programs an *idle memtile's* MM2S DMA
at runtime via control packets, how much does that cost vs. the static CDO path
or host MMIO — and how does it scale with transfer size, BD-chain depth, and
channel count?

Measured with `memtile_program_cost.py` (core `Timer_Low` brackets). One memtile
MM2S → compute-core S2MM transfer of N int32 words; the core times the
program phase and the transfer phase. mlir-aie `652a02018`, Peano `f4933ef736b0`.

## Run it

```bash
source env.sh        # repo root: venv + XRT + PYTHONPATH for aiecc
cd programming_examples/pythoc/microbench/memtile_program_cost

# size sweep: core (single transfer) vs cdo vs host
python memtile_program_cost.py --modes core_static,cdo,host --sizes 8,32,128,512,2048,8192

# chain-depth + multi-channel
python memtile_program_cost.py --modes multi --depths 1,2,4,8 --chain-n 256

python slides/make_charts.py     # regenerate the deck figures (writes slides/img/*.png)
```

Modes: `core` (build the program on-core), `core_static` (host-baked buffer,
recommended), `cdo` (PDI-configured), `host` (`npu_write32` MMIO), `multi`
(2 channels), and the chain-depth sweep via `--depths`. Requires XRT + a built
mlir-aie; runs on NPU2 (Strix Halo) hardware.

The core programs the idle memtile by **streaming the whole control-packet
program as ONE transfer**: the 6 write-commands (reset, unreset, 2× BD halves,
enable, start-queue) are concatenated into one buffer and sent in a single
`MM2S` packet. On aie2p the core DMA inserts the NoC stream header only for the
*first* command, so commands 2..N carry an explicit NoC header word
(`NOC_CTRL_HDR = 0x80000005` for id=5/type=0; cf.
`AIETranslateControlPacketsToUI32Vec` in `lib/Targets/AIETargetNPU.cpp`). One
transfer ⇒ one core-MM2S0 completion ⇒ one lock wait, vs one packet + lock per
command.

The `core_static` path bakes that buffer at PDI load (no runtime stores → no
exposure to the Peano store-scheduling bug, works for any N); the `core` path
builds it on the core (folds to a constant store). Both measure identically.

**What the two intervals bracket** (core reads `Timer_Low` at t0, t1, t2; see the
`measurement_timeline` figure in `slides/`):

- `program = t1 − t0` — the core *issuing the config*. For `core`, the
  control-packet send (one transfer, lock1≥1). For `cdo`/`host` the core programs
  nothing — it only arms its own receive BD (~8 cyc), so their `program` ≈ 0.
- `transfer = t2 − t1` — from config-issued until **all N words have landed at the
  core** (its S2MM lock0). It overlaps the data DMA; ~identical for `core` and
  `cdo`. The data time is **not** subtracted from `program` — the two are measured
  separately. The **host**'s MMIO programming latency falls *inside* `transfer`
  (the core idles for it), hence `host-prog = transfer_host − transfer_cdo`.

## Cost to make the memtile deliver the data (core cycles)

| N (i32) | core-prog (1 xfer) | host-prog* | cdo-prog | transfer | core-prog % of total |
|--------:|-------------------:|-----------:|---------:|---------:|---------------------:|
|       8 |        58 |       ~861 |      ~0  |       73 |  44% |
|      32 |        58 |       ~861 |      ~0  |       94 |  38% |
|     128 |        58 |       ~861 |      ~0  |      136 |  30% |
|     512 |        58 |       ~861 |      ~0  |      325 |  15% |
|    2048 |        58 |       ~861 |      ~0  |     1102 |   5% |
|    8192 |        58 |       ~861 |      ~0  |     4168 |   1% |

`*host-prog` is the host MMIO programming cost, observed as the extra cycles the
core idles waiting for data vs the CDO baseline (`host_transfer − cdo_transfer`);
the core's own `program` counter reads ~0 in host/cdo modes.

## Headline: send the program as ONE transfer

| programming method | core cyc | note |
|--------------------|---------:|------|
| 6 control packets (one per command, lock-synced each) | 270 | prior approach |
| **1 control transfer (6 commands concatenated)** | **58** | **4.6× cheaper** |

Each lock-synced packet round-trip costs ~35 cyc of drain/wait; collapsing 6
packets into a single transfer removes 5 of those round-trips. The result is a
**fixed ~58 core cycles**, independent of N.

## Takeaways

- **Core programming the memtile in one control transfer is a fixed ~58 core
  cycles, independent of N** (vs 270 for the packet-per-command path).
- **CDO is ~free at runtime (~0 cycles)** — the channel is configured at PDI load.
- **Host (`npu_write32`) costs ~861 cycles** — *much more* than the core path. The
  10 shim MMIO writes routed through the column control are far slower than one
  core stream transfer.
- **Transfer is ~linear in N** (~0.5 cyc/word + ~40): 73 cyc @ N=8 →
  4168 cyc @ N=8192. Identical for all modes (same DMA).
- **Amortization:** core programming is 44% of total at N=8 but only 1% at
  N=8192; below ~128 words the ~58-cycle setup matters, above ~512 it is noise.

## Multi-channel (single transfer, N=256)

Both channels' programs (12 commands) are concatenated into one 47-word transfer.
**MM2S1 (an odd memtile DMA channel) must use BD ID [24-47]** — even channels
(MM2S0/S2MM0) use [0-23], odd channels (MM2S1/S2MM1) use [24-47]
(`AIETargetModel::isBdChannelAccessible`). Putting MM2S1 on BD24 makes the second
channel deliver.

| channels | commands / words | program (cyc) | both deliver? |
|---------:|-----------------:|--------------:|:-------------:|
| 1 |  6 / 23 | 58 | yes |
| 2 | 12 / 47 | 79 | **yes** |

Programming a second channel adds only **+21 cyc** (its 6 extra commands in the
same transfer). Both channels deliver in parallel — the earlier "ch1 never fires"
caveat was the illegal BD1-for-MM2S1 assignment, now fixed.

## BD-chain depth (use_next, N=256) — needs the paced path

A D-deep `use_next`-linked memtile MM2S chain does **not** deliver when the whole
program is sent as one concatenated transfer (the head BD with `use_next` set
yields no output). Chained-BD writes require the lock-synced separate-packet
path; with it the chain works:

| D | program (cyc) | transfer (cyc) | Δprogram/BD |
|--:|--------------:|---------------:|------------:|
| 1 |  269 |  178 |  —   |
| 2 |  405 |  303 | +136 |
| 4 |  645 |  576 | +120 |
| 8 | 1137 | 1080 | +123 |

Linear: `program ≈ 269 + 123·(D−1)`. Each extra `use_next` BD ≈ **+123 cyc** = 2
lock-synced control packets. (Independent channels don't need this — only the
`use_next` chain head does.) Compare `dma_overhead`: a core programming its *own*
BD costs ~16 cyc/BD.

## Cost ranking (programming a memtile MM2S, runtime)

    cdo (~0, paid at load)  <  core 1-transfer (~58)  <  core 6-packet (~270)  <  host (~861)

## Notes / caveats

- The single-transfer program is `CTRL_WORDS = 23` words for one channel: 6
  commands + 5 embedded NoC headers (the DMA supplies the first). Multi = 47.
- `program` includes the per-transfer stream-drain + single lock wait; it varies
  a few cycles run-to-run but the structure (flat in N) is stable.
- core/cdo transfer numbers track within ~1 cycle — the only difference is who
  programs the channel, not the DMA.
- Chained (`use_next`) BD setup is the one case that still needs the paced
  packet-per-command path; everything else benefits from the single transfer.

## Files

- `memtile_program_cost.py` — kernels (all modes, parameterized), MLIR design,
  single-transfer control-packet builder, and the sweep runner.
- `slides/memtile_program_cost.md` — Marp deck (incl. the measurement-timeline
  diagram explaining exactly what `program`/`transfer` bracket).
- `slides/make_charts.py` — regenerates `slides/img/*.png` from the measured data.
- This README — methodology, results, and register/BD-channel model.

Related: `../dma_overhead/` measures the cost of a core programming its *own* tile
DMA (~16 cyc/BD); `core-programs-memtile-dma-ctrlpkt` PoC in `../ctrl_packet_dma/`.
