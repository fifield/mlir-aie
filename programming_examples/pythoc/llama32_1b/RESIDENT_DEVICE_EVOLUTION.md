# Evolving pythoc llama32_1b decode toward a single resident `aie.device`

Design analysis: how far the current per-stage decode pipeline is from a
mirror-flm-style persistent spatial layer, what overlaps today, what can be
fused/repacked now, and a staged path.

> Status: **Path A floor reached (2026-06-08).** The RMS folds and the
> d1d3d4_rms flip described below as "what can be fused today" have landed for
> both BF16 and AWQ — see the "Status update" section immediately below for
> what shipped and what's next. Companion to `DEVICE_PACKING_ANALYSIS.md`
> (which covers the 232→104 packing already landed) and the mirror-flm
> `generic_decoding_layer` spatial design used as the convergence target.

## Status update (2026-06-08): Path A floor reached

The "what can be fused TODAY" items below are now done. Current decode device
layout (per layer, each `aie.device` = one full-device LoadPDI):

| Call | Kernel | Pack mode (default) | Devices |
|---|---|---|---|
| 1 | `rms_gemv_rope` / `_awq` | `rgr1_ddr` | **1** (RMS folded into QKV+RoPE pack) |
| — | decode attention | CPU | (host hop — still the structural gate) |
| 2 | `o_gemv_ffn` / `_awq` | `d1d3d4_rms` | **3** (O+add1 / gate+up+swiglu+RMS / down+add2) |
| 3 | `lm_head_gemv` / `_awq` | Lever A 8→1 | **1** |

Per token ≈ (1 + 3)×16 + 1 = **~65 LoadPDIs** (was ~104 at the start of this
doc, ~232 pre-packing).

What landed:
- **`rgr1_ddr`** — folds the standalone `r_rms_seg` RMSNorm device into the
  `rgr2_qkv_rope_pack` as a row-3 tile on spare shim channels, keeping the DDR
  handoff (arg2). Call 1: 2→1 device. BF16 + AWQ (`rms_gemv_rope_awq`).
- **`d1d3d4_rms`** defaulted on for BF16 `o_gemv_ffn` (was AWQ-only): folds
  `rm_rms_seg` into the gate/up tiles via the fused matvec_rms kernel.
  Call 2: 4→3 devices. (BF16 throughput-wash, AWQ slight win — structural step.)
- Both bit-exact vs unpacked (`make hf-gate` 4/4 BF16, AWQ paris 2/2);
  BF16 11.86 tok/s, no regression.

These are perf-neutral on purpose — the value is structural (more stages
device-resident, fewer reconfigs) on the path to the resident layer. The two
gates and refined roadmap are in **"Path to a single resident device"** and
**"Suggested ordering"** at the end of this doc (both updated for this status).

## TL;DR

Decode is not three architectures — it is **one matvec engine invoked ~33×
per token**, with a CPU attention hop wedged into the middle of every layer.
The CPU-attention break and the per-op-dispatch design are mutually
reinforcing: attention is on CPU "because LQ=1 dispatch overhead beats NPU
GEMV", but that only holds *because the engine isn't resident*. Make the
engine persistent and the argument inverts — which is exactly why mirror-flm
runs decode attention on-NPU and can hold the whole layer resident.

## The workload, quantified

Llama-3.2-1B: `dim=2048, n_heads=32, head_dim=64, kv_heads=8 (kv_dim=512),
intermediate=8192, layers=16, vocab=128256`.

Per-token bf16 weight stream:

| Group | Params | Bytes (bf16) |
|---|---|---|
| attn (Wq 2048², Wk/Wv 2048×512, Wo 2048²) | ~10.5M | ~21 MB |
| ffn (gate/up 8192×2048, down 2048×8192) | ~50.3M | ~101 MB |
| **per layer** | ~60.8M | ~122 MB |
| × 16 layers | ~973M | ~1.95 GB |
| lm_head (128256×2048) | ~263M | ~0.5 GB |
| **per token total** | | **~2.5 GB** |

At ~88 ms/token (bf16) that is ~28 GB/s effective — well under Strix LPDDR5
peak. **Decode is dominated by reconfig/DMA overhead, not raw bandwidth** —
which is why the 232→104 packing bought ~6.7% with zero added compute.

> **AWQ note (corrected):** AWQ uint4 decode is now **faster** than bf16,
> roughly **+2–3 tok/s** on the current design. (The README table showing
> AWQ slower than bf16, and the "dequant chain is the limiter" framing, are
> stale — update them.) This strengthens the resident-engine case: 4-bit
> weights cut the ~2 GB/token stream ~4×, and a persistent pipeline that
> overlaps dequant with DMA+MAC removes the dequant-serialization penalty
> that the old dispatched design paid.

## How much the three dispatches overlap

All three decode devices are the **same fabric**: 8 columns, rows
shim(0)/mem(1)/compute(2), running the **same kernel** `matvec_vectorized_bf16_bf16`
(`mv_pythoc.o`) with the same shim-col-0 broadcast and the same per-column
weight gather. They differ only in loop bounds and which DDR BOs they touch.

| Op | Device | Kernel | Herd | Differs by |
|---|---|---|---|---|
| V/Q/K GEMV | rms_gemv_rope | `matvec_…bf16` | 8c × row2 | M/K bounds, weight BO |
| O GEMV | o_gemv_ffn | same | 8c × row2 | M=2048/K=2048 |
| gate/up GEMV | o_gemv_ffn | same | 8c × row2 | M=8192 |
| down GEMV | o_gemv_ffn | `dg_…` K=8192 | 8c × row2 | K=8192 |
| LM head | lm_head_gemv | same | 8c × row2 | 8 part × N_OUTER=16 |

Satellites reuse col-0 cores: RMSNorm (appears in **both** rms_gemv_rope and
o_gemv_ffn), RoPE, silu_and_mul, two residual adds. ~8 of ~10 distinct ops
are literally the same engine. mirror-flm builds this **once** as the
persistent `q4_proj_engine` (16 cores serving QKV+O+gate+up+down, selected by
packet ID); pythoc rebuilds + re-dispatches it ~33×/token.

### Redundant every dispatch

- Each `aiex.run` reapplies a fresh per-segment PDI: switchbox config +
  memtile/shim BDs + **core ELF reload** — even though all 16 layers are
  structurally identical devices.
- Weights flagged `static_indices` (no host copy-back) but still DMA'd from
  DDR every call.
- Activations round-trip DDR at every phase boundary (proj→rope, og→a1,
  gg/ug→sw→dg) and every layer boundary.

## The structural obstacle

Residual stream: `x → rms → qkv → ATTN → o_proj(+x) → rms → ffn(+) → x'`.
CPU attention sits dead-center. **While decode attention is on the host, no
single device can span a layer** — forced break + host sync + re-enter each
layer.

## What can be fused / repacked TODAY (no attention rework)

> Progress (2026-06-08): the RMS-fold portions landed (see "Status update").
> Each item below is annotated [DONE] / [PARTIAL] / [TODO].

1. [TODO] **Persistent PDI across the 16 identical layers** — biggest cheap win.
   rms_gemv_rope layer 0 ≡ layer 15 (same ELF/tiles); only weight base ptr +
   RoPE position + KV ptr change. Load PDI once, loop 16× with new runtime
   args → ~3 reconfigs/token instead of ~104. Doesn't touch attention. *(see
   deep-dive below)* — this is "step 5" in the updated ordering; the right form
   is an internal 16-layer loop inside one layer device's runtime_sequence.
2. [TODO] **RoPE in-place on the GEMV cores** — each core holds 4 Q-heads
   (256/col, 64-dim) post-GEMV; rotate there before scatter. Kills the
   proj→rope DDR round-trip. (rgr1_ddr folds RoPE into the *device* but it is
   still a separate row-3 tile with a DDR handoff, not in-core.)
3. [PARTIAL→TODO] **Collapse all matvecs into one persistent proj engine** —
   d1d3d4 taken to its limit: one configured-once device per side of the
   attention break, fed {O, gate, up, down, lm_head} by packet ID / runtime
   arg. The RMS folds cut device count (call1=1, call2=3) but the matvec cores
   are NOT yet reused-by-packet-ID across projections. This is the **packet-fed
   proj-engine — the recommended next step** (step 1 in the updated ordering).
4. [PARTIAL] **Keep residual `x` resident across og→ffn** — res1 / second add
   stay in L2 instead of DDR scatter/gather. d1d3d4(_rms) folds the adds into
   the matvec devices (packet-routed), but res1 still crosses DDR between the
   d1/d3/d4 devices; full L2 residency comes with the single-device proj-engine.

## What spatial repacking does NOT buy

Running Q/K/V on disjoint columns concurrently (vs all-8-then-all-8) does not
help: single-token GEMV is bandwidth-bound and total shim bandwidth (8c × 2
MM2S) is fixed. Concurrency only removes a *reconfigure* — the packing win in
a spatial costume, not a bandwidth win. The genuine spatial win is
mirror-flm's: persistent cores that **overlap weight-DMA with compute**
(prefetch tile N+1 while MAC-ing N), impossible across dispatched phase
boundaries.

## Path to a single resident device

**Path A — keep CPU attention:** two resident devices/layer + resident
lm_head, via items 1–4. Cores stay loaded; per layer re-run rgr → CPU hop →
re-run ogf. q/k/v/attn_out/x still DDR round-trip (~18 KB/layer, negligible).

**Path B — move decode attention on-NPU → one resident layer device:** the
real mirror-flm convergence. Tile budget fits (mirror packs a layer in ~27 of
32 cores). Cost = packet-routed core-to-core dataflow (pythoc doc flags it as
"syntactically present but unexercised" — exactly what mirror-flm already
paid for). Once the layer is one persistent device, on-NPU GEMV attention
wins and 16 layers pipeline as a weight stream. This is also where 4-bit
weights unlock the bandwidth wall.

## Deep dive (1): persistent PDI — what's already done vs the real tax

Traced through `kernel_builder/cache.py`, the builders, and the mlir-aie
compiler. The "reload the PDI every layer" hypothesis is **half wrong** — the
expensive part lives one level down than expected.

### What's already persistent (host/XRT level)

`cache.py:_XRTRunner.load()` builds `xrt.device → xrt.elf → xrt.hw_context →
xrt.ext.kernel` **once per kernel-name** (3 total: rgr, ogf, lm_head) and
caches it in `self._loaded[name]`. Every one of the 16 layers calls
`load_and_run`, which on a hit just does `run = xrt.run(runner.kernel);
set_arg(...) per BO; start(); wait2()` (`cache.py:511-516`). Weights live in 16
pre-allocated BO sets (`_L0`..`_L15`), written once at preload; only
activations + RoPE LUT are re-copied per call.

**So "load the xclbin/ELF once, loop 16× changing only runtime args" is
already the design at the host level.** Any README/doc text implying a
per-call host PDI reload is misleading — fix it.

### Where the reconfig tax actually is (on-device LoadPDI)

Each compiled artifact is **one dispatcher `aie.device` + N sub-device
`aie.device` blocks** (cached rgr MLIR = 7 devices, ogf = 9, lm_head = 9). The
dispatcher's `aiex.runtime_sequence` issues, per sub-device:

```
aiex.configure @q_matvec_bf16_0 { aiex.run @..._sequence(args) }
```

`aiex.configure` lowers to `aiex.npu.load_pdi`
(`AIEMaterializeRuntimeSequences.cpp:139`), and the compiler states plainly:

> `// LoadPDI resets the whole device, hence cannot do partial reconfiguration`
> — `AIEMaterializeRuntimeSequences.cpp:119`

So **every `aiex.configure` is a full-device reset + PDI reload**, executed
*on-device inside a single host `xrt.run`*. Per token: rgr 2×16 + ogf 4×16 +
lm_head 8 = **104 full-device LoadPDIs** (this is the "232→104" axis; packing
already removed 128). Calibration: 128 LoadPDIs ≈ the 6 ms that packing saved
(94→88 ms) ⇒ **~47 µs per LoadPDI**, and the residual 104 ≈ ~4.9 ms ≈ ~5.6%
of decode. **That ~5.6% is the entire ceiling left for any PDI-level work.**

### Lever A — implement the compiler's own TODO (low-risk)

Right above the lowering, unfixed:

> `// TODO: add code to remove repeated @configure ops`
> `// TODO: add check that liveness of two aie.configures do not overlap`
> — `AIEMaterializeRuntimeSequences.cpp:102-105`

Consecutive identical configures are **not** deduped today. The clean target
is **lm_head**, the one multi-segment device packing never touched: 8
partitions = 8 *byte-identical* PDIs (same 8-col herd, same ELFs), differing
only in weight/out BOs. Re-emit as **one `aiex.configure @p_matvec` + 8
`aiex.run`** with per-partition args ⇒ 8→1 LoadPDI, −7/token. Pure overhead
removal, no compute change, isolated from the attention path. Magnitude is
small (~0.3–0.5%) but it's free and it proves the dedup mechanism.

### Lever B — unify the matvec herd to one PDI across rgr/ogf/lm_head

Q/K/V, O, gate/up/down and lm_head are the *same* `matvec_…bf16` on the *same*
8-col herd — only bounds + BOs differ. Make them **one device symbol**,
parameterized by runtime args (BO addrs + M/K bounds), so all GEMVs share one
PDI and Lever-A dedup collapses them to a single LoadPDI per host dispatch.
Bounded by the whole-device-reset + non-overlapping-liveness rules, this works
*within* a dispatch; across dispatches each `xrt.run` still re-LoadPDIs once.
Net could approach ~1 LoadPDI per host dispatch (≈33/token vs 104) — up to
~3–4% on top of Lever A.

### The ceiling, and the door it points to

True cross-layer PDI persistence is blocked: LoadPDI resets the **whole**
device, the framework forbids two live configs, and `xrt.run` has no
"PDI already resident, skip" path. With CPU attention forcing a host return
mid-layer, no single config survives the 32-dispatch layer loop. **PDI tricks
top out near ~5%.**

But the compiler reveals the *right* mechanism:

> `// Skip cores without elf_file (e.g., lightweight reset devices that only`
> `//  need DMA/lock reconfiguration).` — `AIERT.cpp:993`

The hardware supports a **DMA/lock-only reconfigure** that leaves core
programs loaded — exactly mirror-flm's model (cores never reload; only weight
DMA streams change). `aiex.configure` doesn't expose it (it always emits
LoadPDI). Wiring a "keep cores resident, swap only DMA BDs/locks" reconfigure
path is the real bridge from "persist PDI" to the resident design — and the
single most valuable compiler-level direction here.

### Prototype: Lever A landed on lm_head (`builders/lm_head_gemv.py`)

Implemented the 8→1 collapse. Change is isolated to the builder; the host
(`cache.py`) is untouched.

**What changed:** instead of emitting 8 separate partition devices
(`@p0..@p7_matvec_bf16_0`) and 8 `aiex.configure`s, the builder now emits **one
shared partition device** `@p_matvec_bf16_0` with a 3-arg runtime_sequence
`(x, w, y)`, and the dispatcher emits a **single `aiex.configure` containing 8
`aiex.run`s**, each passing that partition's `(weight, output)` BO pair. The
outer `@lm_head_gemv` host signature stays 17 args, so the runtime BO layout is
unchanged.

**IR-level validation (all passing in this env):**

| Metric | Baseline | Lever A |
|---|---|---|
| `aie.device` blocks | 9 | **2** |
| `aiex.configure` | 8 | **1** |
| `npu.load_pdi` after `--aie-materialize-runtime-sequences` | 8 | **1** |
| `load_pdi` after full host lowering (materialize → substitute-shim-dma → assign-bd-ids → dma-tasks-to-npu) | — | **1**, 8497 npu ops, clean |

So the **8→1 whole-device-reset goal is achieved and the host-program lowering
that Lever A touches compiles cleanly end-to-end.**

**End-to-end validation (PASSING on real HF weights):**

`make compile` builds `lm_head_gemv.elf` through aiecc in 1.1s (both packed and
unpacked decode configs). `make hf-gate` (gold-standard answer-level gate on
`unsloth/Llama-3.2-1B-Instruct`) passes **both** cases with bit-identical
decode tokens `[271, 791, 6864, 315, 9822, 374, 12366, 13, 128009]` →
`'\n\nThe capital of France is Paris.'`:

- `test_hf_answer_gate_paris` (packed default rgr2_ddr/d1d3d4) — **PASSED**
- `test_hf_answer_gate_unpacked_baseline` (pack_mode=none) — **PASSED**

`make profile N_TOKENS=64` → **11.90 tok/s** (packed), no regression vs the
~11.4 baseline. The 8→1 LoadPDI saving is ~7 of ~104 segments/token (sub-1%),
so it sits within run-to-run noise — as predicted. The value is not the
isolated speedup; it's that this **proves the configure-dedup mechanism**
(one `aiex.configure` + N `aiex.run`) end-to-end on hardware, which Lever B
reuses across the much larger rgr/ogf matvec set.

> Env note: the gate needs `transformers` + `jinja2` in the venv (installed
> during validation); without them it *silently skips* with a misleading
> "HF cache does not contain the model" reason even when the weights are
> cached. Worth fixing the skip message to distinguish missing-weights from
> missing-tokenizer-deps.

### Net recommendation for item (1)

1. Land Lever A on lm_head (8→1 configure) — small, safe, validates dedup.
2. Generalize dedup + unify the matvec herd PDI (Lever B) — ~few %.
3. Recognize the ~5% PDI ceiling: the big win is the DMA-only reconfigure path
   (`AIERT.cpp:993`) and/or moving attention on-device (Path B). Don't expect
   PDI persistence alone to move decode much past packing's gains.

## Suggested ordering

**Done (2026-06-08) — Path A floor:** RMS folds (`rgr1_ddr`) + `d1d3d4_rms`
default + lm_head Lever A. Decode = call1(1) / CPU attn / call2(3) / lm_head(1),
~65 LoadPDIs/token, bit-exact, BF16 + AWQ.

**Remaining roadmap to a single persistent kernel.** Two gates:
*Gate 1 = on-NPU decode attention* (structural — while attention is on the host
no single device can span a layer). *Gate 2 = the packet-fed proj-engine + fit*
(only 4 compute rows/col, so the ~7-stage residual chain + attention can't be
row-stacked — the cores must be reused-by-packet-ID, FLM `q4_proj_engine` style,
with intermediates flowing core-to-core / via L2 instead of DDR).

| # | Step | Attn? | Effect |
|---|---|---|---|
| 1 | **Packet-fed proj-engine on call 2** — one core set reused across O/gate/up/down by packet ID; res1/normed2/swiglu resident on-chip. **Recommended next.** | no | call2 3→1; builds the core-reuse + core-to-core primitive everything else needs |
| 2 | Same engine on call 1 (RoPE/RMS fold into the engine, in-core) | no | call1 stays 1, removes the in-pack DDR handoffs |
| 3 | On-NPU decode attention (GQA, KV cache from DDR) — **Gate 1** | yes | removes the host hop; lets a device span a layer |
| 4 | Fuse call1 + attn + call2 → one persistent layer device; residual resident across the layer | — | layer = 1 device |
| 5 | Loop the 16 layers inside the layer device's `runtime_sequence` (weights = per-layer DDR base/BOs) | — | ~65 → ~1 LoadPDI/token, no compiler change |
| 6 | DMA/lock-only reconfigure (`AIERT.cpp:993`) for cross-token residency — compiler change | — | true persistence; the last ~5% |
| — | Re-evaluate AWQ on the resident engine (4-bit cuts the ~2 GB/token stream ~4×) | — | bandwidth-wall unlock |

The first earned place for a *new* design doc is step 1 — a packet-fed
proj-engine spec (core map, packet-ID role table, L2/cascade intermediate
layout, runtime-arg ABI). Until then this doc is the single roadmap.

## Implementation methodology — converge, then collapse

Every step in the table above is landed as a sequence of **bit-exact
increments**, never as a single rewrite. The rules:

1. **Converge shape before merging function.** To fuse two `aie.device`s,
   first make them *structurally identical* in the dimension that matters —
   same core-body kernel, same tile/lock layout, same `flow`→`packetflow`
   form — one small change at a time. Each change is validated bit-exact with
   the devices **still separate and still host-dispatched**. The two devices
   "look the same" long before they become one. (Precedent: the d1d4 packs
   converged matvec+add into a stacked herd before anything was removed.)

2. **Keep the safety net until correctness is proven.** Leave the existing DDR
   handoffs, separate devices, and host orchestration in place as the working
   reference. They cost reconfig/dispatch overhead — that is *fine*; we are not
   chasing the perf win mid-flight. Correctness first, every step.

3. **Remove the win last, in isolated reversible steps.** Only after the
   compute/dataflow is proven do we take the irreversible/optimization moves —
   collapse N devices into one, drop the redundant `aiex.configure`, move
   orchestration host→device, loop the 16 layers internally. Each is its own
   small commit, gated behind a pack-mode flag so the prior path stays
   selectable for A/B (as `rgr2_ddr` / `d1d3d4` / `none` already are).

4. **No increment may change tokens.** `make hf-gate` bit-exact is the gate on
   every step. Perf may move (often a wash, sometimes a transient regression
   mid-sequence) — acceptable; only the final collapse is expected to pay off.

Worked example — how step 1 (packet-fed proj-engine on call 2) decomposes:

| sub | change | devices? | dispatch? | gate |
|---|---|---|---|---|
| 1a | ✅ **DONE** (2026-06-09) result path `flow`→`packetflow` (single ID, same dest) in d3 — pack mode `d1d3d4_rms_pkt` | unchanged (3) | unchanged | bit-exact ✓ |
| 1b | make O/gate/up/down core bodies byte-identical (differ only by runtime args) | unchanged (3) | unchanged | bit-exact |
| 1c | add packet-ID role table; one core set *can* do O→gate→up→down, but devices still run separately | unchanged (3) | unchanged | bit-exact |
| 1d | route res1/normed2/swiglu core-to-core / L2 (DDR handoff still present, just unused) | unchanged (3) | unchanged | bit-exact |
| 1e | **collapse** to one device fed by packet ID; drop the now-dead DDR handoffs + extra configures | **3→1** | reduced | bit-exact |

Steps 1a–1d are pure convergence (zero reconfig/dispatch change); 1e is the
single irreversible collapse, landing on already-proven pieces.

### Proj-engine probe: mode-RTP-switched matvec (measured 2026-06-09)

Before committing to how O/gate/up/down share one core (1b–1c), measured the
two viable ways to make one matvec body cover K=2048 and K=8192:

- **Key structural fact.** `kernels/matvec.py` and `matvec_k8192.py` are
  algorithmically identical; K is *already* a runtime arg (`while j < k`). The
  ONLY binding compile-time constant is the `loop_range(32)` vs `loop_range(128)`
  trip-count hint that unlocks Peano's hardware loop + software pipelining.
  So "feed K as an RTP into one loop" is wrong (loses the hint); the right shape
  is **two hinted bodies + a `mode` switch** — exactly the switch-statement idea.

- **Program-memory budget.** AIE2P core = **16 KB** program / 64 KB data
  (`xaie2pgbl_reginit.c:177`). Current `.text`: plain matvec ~850 B, RMS-fused
  matvec 4384 B, swiglu 1952 B, add 384 B. Program memory is *not* the
  constraint; the 4-row limit is about distinct stacked cores, not code size.

- **Measured (pack mode `d1d3d4_rms_fmv`).** `kernels/matvec_fused.py` carries
  both bodies behind a per-tile `mode` RTP. Fusion **fully proven**: the SAME
  `matvec_fused_pythoc.o` (1744 B) is linked into BOTH the O matvec (D1, mode 0,
  K=2048) and the down matvec (D4, mode 1, K=8192) -- symbol `matvec_fused_bf16`
  present in both core ELFs, 16 mode RTPs (8 + 8). Result: **bit-exact** (Paris);
  per-token **~84 ms vs ~86 ms baseline (perf-neutral)**; each fused matvec core
  `.text` **1280 B** (O: 864→1280 inline→fused; down: 848→1280), **7.8 % of the
  16 KB budget**.

- **Verdict.** One ELF serves two K-roles selected at runtime by a mode RTP, with
  zero perf cost and ample program-memory headroom. Concatenating core bodies
  behind one mode RTP is the confirmed proj-engine primitive. A full proj-engine
  core (4 matvec roles + RMS branch + swiglu, ~5–7 KB) fits 16 KB with margin.
  This green-lights 1b–1c: one reused matvec core serving all four projections by
  mode RTP, each arm keeping its compile-time `loop_range`. Next: extend the same
  fused kernel to gate/up (mode 0, K=2048) and add the RMS arm, then collapse the
  separate matvec cores onto one reused core (the actual 3→1 device fold).

  Cache note: the decode KernelCache keys on the pack-mode *signature*, not the
  IR text, so editing a builder without changing its `pack_mode` string is a
  false cache hit -- delete `decode_kernel_cache/<name>.elf` (+ `.npu.air.mlir`,
  `.<name>.work/`) to force a rebuild when iterating on a fixed pack mode.

  Done (2026-06-09): all four matvecs (O/gate/up mode 0, down mode 1) run on the
  one `matvec_fused_pythoc.o` under `d1d3d4_rms_fmv` (commits 82a04e4, 8f68c5e).

### Collapse plan: 3 devices -> 1 (approved 2026-06-09)

Goal: one `aie.device` / one `aiex.configure` for call-2 (3 LoadPDIs -> 1),
intermediates via DDR initially (safety net), removed later.

**Why the earlier "cram down into the shared core" fork was a false constraint.**
The partition is ~99.94 % matvec / ~0.06 % post-op (add+rms+swiglu). The post-op
rows are 8-wide for *dataflow locality* (consume the matvec's per-column output
slice in place), not compute -- so we are not tile-starved and down does NOT
need to share the O/gate/up core. Balanced single-device tile map:

| row | core-set | reused for | notes |
|---|---|---|---|
| 2 | matvec-A (8 col) | O -> gate -> up (3 waves) | K=2048, exact buffers, one kernel |
| 3 | matvec-B (8 col) | down (1 wave) | K=8192, exact buffers |
| 4 | add (8 col) | add1 -> add2 (2 waves) | same op, per-column |
| 5 | swiglu (8 col) + rms (col 0) | swiglu; rms (cross-column reduce) | rms stays 1 tile |

Core reuse is achieved by the **single runtime_sequence dispatching waves
sequentially** to a stream-processor core (the core already loops forever; we
just feed it O-weights, then gate-weights, then up-weights). No per-wave mode
change needed on row 2 (all K=2048). The row-1 mem tile time-shares across waves.

Staged, each bit-exact behind its own pack flag, default `d1d3d4_rms` untouched:

| step | change | devices | gate |
|---|---|---|---|
| C1 | merge D1+D3 -> one device: O/gate/up on reused matvec-A (row2), add1 (row3), swiglu (row4), rms (row5 col0); DDR intermediates; keep D4 separate | 3 -> 2 | bit-exact |
| C2 | fold D4 (down + add2) into the same device: down on matvec-B (row3 -> needs relayout; add1/add2 reuse one add core) | 2 -> 1 | bit-exact |
| C3 | drop DDR handoffs for the on-chip intermediates (proj/res1/normed2/gate/up/swiglu) -> core-to-core / L2 | 1 | bit-exact |

### C1 implementation spec (ready to build in a fresh session)

Status: **DONE (2026-06-09).** Landed as pack mode `c1_merged` in
`builders/o_gemv_ffn.py` (`_emit_call2_merged`), exactly per this spec: one
device, rows 2..5, sequential 6-stage runtime, DDR intermediates, D4 separate
(call 2 = 2 configures). Bit-exact (`test_hf_answer_gate_o_gemv_ffn_c1_merged`
+ full gate 7/7); ~84 ms/token vs ~88 baseline (slight win). Per token: 65→33
LoadPDIs. Three hard-won implementation notes (apply to anything packet-fed):

1. **Packet IDs are mask-constrained, never reuse id 0.** All ids passing
   the same slave port get merged into ONE masked `aie.rule`; chained
   pass-throughs (rows 3/4/5) must mask exactly at every hop, and any id the
   mask falsely captures fails routing. Working scheme: matvec/W/x = 1,
   add = 8, swiglu = 12, rms = 13 ({8,12,13} → mask-clean at rows 1/3/4/5),
   and ALL results share id 1 (the shim S2MM port needs no demux).
2. **Exact delivery counts on shared channels.** The standalone matvec
   shim broadcasts x with repeat 32/outer while cores consume 16 — harmless
   standalone, but leftover stream data jams the shared MM2S1 queue and
   deadlocks the next stage. C1 sets x repeat = 16/outer (exact).
3. **A deadlocked run wedges the NPU partition for the NEXT process** (its
   first command ERT-times-out; the timeout reset eventually clears it, but
   the half-wedged state can persist a few runs and mimics flaky tests).
   `tools/test_c1_wedge.py` = run+probe bisect harness; sequential c1 and
   c1+D4 dispatch is wedge-free (verified 16x).

Original spec follows. Default `d1d3d4_rms` and the committed
`d1d3d4_rms_fmv` are untouched; C1 is selectable via
`PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c1_merged`. Next: C2 (fold D4 in).

### C2 status (2026-06-10): built, blocked on stale-PDI switch state

Pack modes `c2_rms` (C2a: RMS folded into gate/up waves on the reused row-2
core, rms tile/stage gone, D4 kept) and `c2_merged` (C2b: + row-5 K=8192 down
herd, add herd runs add1+add2, ONE configure for call 2) are implemented and
IR-clean (c2_merged = 1 LoadPDI). Two findings:

1. **Inline rms depth bug:** `rms_norm_packed_bf16` (matvec_rms .ll inline)
   deadlocks the core when called inside an `scf.for` wave loop at depth 3;
   unrolling gate/up straight-line (depth 2, as d3) fixes it.
2. **BLOCKER — c2 add1 wave starves on the X-broadcast fan's extreme
   columns (ISOLATED 2026-06-10).** The deadlock is c2-internal, not a
   cross-PDI stale-master effect (the earlier stale-master hypothesis is
   withdrawn — see retraction below). Reproducer `tools/test_c2_add_starve.py`
   + per-col res1 readback: the O matvec wave (proj) completes on all 8
   columns, but the following add1 eltwise wave (proj + x_resid -> res1)
   never produces output on specific columns, stalling everything downstream.
   The starved columns track the **X-broadcast source column**
   (`PYTHOC_C2_XCOL`), deterministically (3/3 each):

   | XCOL | broadcast fan | starved add1 cols |
   |---|---|---|
   | 0 | east-going | **0** |
   | 3 | both ways | **0 and 7** |
   | 7 | west-going | none (add1 ok; a later stage still stalls) |

   So the matvec X-broadcast transits the shim row on the **same MM2S1 lane**
   that each column's add1 `in1` enters on; the fan's terminal columns lose
   the arbitration and never receive `in1`. This is a structural shim-channel
   conflict in the c2 row map (add reuses MM2S1, shared with the broadcast),
   independent of packet ids. (Renumbering ids to distinct single bits —
   matvec=1/add=2/swiglu=4/down=8 — landed and removes a *separate* rule-mask
   aliasing hazard, `rule(27,8)` dropping bit 2 to merge add=8 & swiglu=12,
   but does NOT fix the starvation.) **Fix direction:** give add1 `in1` a
   physical path disjoint from the X-broadcast (e.g. deliver the broadcast via
   a mem-tile, or move add inputs onto MM2S0-side channels), or stage the
   broadcast so it fully retires before add1 claims MM2S1. Until then C2 stays
   off-default behind its flags.

   **FIXED (2026-06-10): the cause was OUTPUT convergence, not the broadcast.**
   Two changes, both landed; c2_rms and c2_merged now run bit-exact (Paris,
   gate 9/9) and complete the rgr->ogf swap 3/3:

   - **Root cause — distinct output packet ids.** Every producer drained to
     its column's shim `S2MM0` with the *same* pkt id 1: matvec-y (mem),
     add, swiglu (and down for c2_merged). Multiple producers wired to one
     shim S2MM0 with one id is a static multi-producer routing conflict that
     starved exactly one column's add drain (col 0). Giving each producer a
     distinct id — **matvec-y=1, add=5, swiglu=6, down=7** — removes it and
     `res1` fills all 8 columns. (This is the same shared-slave-port hazard as
     the input-side `rule(27,8)` aliasing, but on the S2MM convergence side.)
     The earlier `PYTHOC_C2_XCOL` sweep only shifted *which* column lost the
     arbitration, which is why it looked like a broadcast-fan effect.
   - **Mem-tile X delivery (the requested change, kept).** The matvec
     activation ships per-column via each column's mem-tile (`shim[c] ->
     mem[c] S2MM5 -> mat[c] DMA0`, pkt 16, 3-slot ring for the O/gate/up
     lengths) instead of the shim-row broadcast fan — `_memx`, default on for
     c2_rms. This removes the fan (the resident-engine-correct X path) but was
     *not* what unblocked the deadlock; the output-id fix was. (c2_merged
     keeps the old broadcast: its mem channels 2/3 hold the down chains.)

   **C2 is now functionally complete and the BF16 default (2026-06-10):**
   `c2_merged` = decode call 2 in ONE device / ONE `aiex.configure` =
   **1 LoadPDI** (was 3). Per-token BF16 decode ≈ call1(1) + CPU attn +
   call2(1) + lm_head(1) ≈ **~49 LoadPDIs/token** (was ~65 at the Path A
   floor, ~104 pre-C1). ~83 ms/token, bit-exact (hf-gate 9/9). Default set in
   `aie_ir_gen.py` (`_O_GEMV_FFN_PACK_DEFAULT = "c2_merged"`); step back with
   `PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=d1d3d4_rms|d1d3d4|none`.

   **AWQ collapsed too (2026-06-10).** `builders/o_gemv_ffn_awq.py` now has
   `_emit_awq_call2_c2` (`c2_rms`/`c2_merged`), ported from the BF16 C2 by
   substituting the uint4-dequant matvec kernels (`awq_matvec_vectorized_u4_bf16`
   / `dg_…`, ui8 weights with `row_bytes` strides) — everything else (RMS fold,
   per-column mem-tile activation, distinct shim-S2MM0 output ids
   matvec-y=1/add=5/swiglu=6/down=7, the add/swiglu cores) is identical because
   the AWQ matvec call shape matches the BF16 one. `c2_merged` is the AWQ
   default: decode call 2 = ONE device / ONE configure / **1 LoadPDI** (was 3),
   bit-exact (`make hf-gate QUANT=awq` 2/2), **~14.5 tok/s** (AWQ stays faster
   than BF16). The distinct-output-id fix was needed identically, as predicted.

### Retraction (2026-06-10): minimal stale-master reproducers were invalid

Two earlier commits (`6eba6697e`, `1962fb23b`) added
`tools/test_stale_master_repro.py` claiming to isolate the blocker with tiny
kernel-less passthrough "victim" devices that deadlock after a parking PDI.
On re-examination those victims **never complete as standalone PDIs on this
board at all** — they time out 4/4 even following themselves, and the
absolute-simplest 1-tile shim→core→shim passthrough times out 3/3 standalone,
while full decode runs fine. So "victim deadlocks after rgr" was confounded:
the device shape simply never completes here, independent of any predecessor.
The reproducer file is removed. Why a hand-built minimal IRON passthrough
fails to dispatch standalone while the production builders' devices succeed is
an open, separate question (candidate: some init/quiesce step the production
path emits that the toy omits). The genuine, still-valid signal is the C2
builder itself: a real-traffic device that completes standalone yet deadlocks
after rgr in-pipeline (`test_c2_rgr_swap.py`). Isolating the blocker needs a
*real* device as the victim, or a true blank-device baseline (driver reload /
`xrt-smi reset`, neither available in this shared session).

**New builder** `_emit_call2_merged(sym="c1_merged")` — ONE `@device` containing
four core-sets, composed by copying the proven core/mem bodies verbatim from the
standalone emitters, only changing their tile row:

| role | tile row | copy core/mem from | kernel | reused |
|---|---|---|---|---|
| matvec-A | row 2 (8 col) + mem row 1 | `_emit_matvec_seg_k2048` | `mv_pythoc.ll` (plain K=2048) | O, gate, up (3 waves) |
| add1 | row 3 (8 col) | `_emit_eltwise_add_seg` | inline addf | once |
| swiglu | row 4 (8 col) | `_emit_sw_silu_mul_seg` | `silu_and_mul_bf16.o` | once |
| rms | row 5, col 0 only | `_emit_rm_rms_seg` | `rms_norm_2048_bf16.o` | once |

matvec-A is a stream processor reused for O/gate/up by the runtime_sequence
feeding it three weight/x/out groups in order (NO mode RTP — all K=2048 plain;
`matvec_fused` is NOT needed for C1). D4 (`d4_dg_a2_pack`, down+add2) stays
separate; dispatch becomes `(c1_merged, d4_dg_a2_pack)` = 2 configures.

**Host args** (`o_gemv_ffn_host_arg_types`, 15): 0=wo 1=attn_out 2=proj 3=x_resid
4=res1 5=ffn_norm_w 6=normed2 7=wgate 8=gate 9=wup 10=up 11=swiglu 12=wdown
13=down 14=output. C1 produces res1(4) and swiglu(11) to DDR; D4 consumes them.

**runtime_sequence stages** (sequential; each awaits its predecessor's output
token before the dependent next stage issues; DDR in/out every stage):

| # | stage | core | w/in0 | x/in1 | out | rows |
|---|---|---|---|---|---|---|
| 1 | O      | matvec | arg0 wo   | arg1 attn_out | arg2 proj    | 2048 (n_outer 2) |
| 2 | add1   | add    | arg2 proj | arg3 x_resid  | arg4 res1    | 2048 |
| 3 | rms    | rms    | arg5 normw| arg4 res1     | arg6 normed2 | 2048 |
| 4 | gate   | matvec | arg7 wgate| arg6 normed2  | arg8 gate    | 8192 (n_outer 8) |
| 5 | up     | matvec | arg9 wup  | arg6 normed2  | arg10 up     | 8192 |
| 6 | swiglu | swiglu | arg8 gate | arg10 up      | arg11 swiglu | 8192 |

**Shim channel / packet scheme** (budget = 2 MM2S + 2 S2MM per col; stages are
sequential so the SAME physical channels are reused, demuxed by packet ID):
per col `c` use MM2S0, MM2S1, S2MM0 (S2MM1 spare). One air_channel per physical
channel, multiple `packetflow`s sharing it:

- `MM2S0[c]`: pkt0 -> mem[c] (matvec W); pkt1 -> add[c] (in0); pkt2 -> sw[c] (in0); col0 also pkt3 -> rms (w)
- `MM2S1[c]`: pkt0 -> mat[c] (matvec X — broadcast: 8 packetflows all sourced from shim[0] MM2S1); pkt1 -> add[c] (in1); pkt2 -> sw[c] (in1); col0 also pkt3 -> rms (x)
- `S2MM0[c]`: pkt0 <- mem[c] (matvec Y); pkt1 <- add[c] (out); pkt2 <- sw[c] (out); col0 also pkt3 <- rms (out)

Packet IDs: INPUT pkt id is set on the shim MM2S BD in the runtime_sequence per
stage (`dma_bd(..., packet=(0, id))`). OUTPUT pkt id is baked into the producing
core's `@mem` out BD (mat/mem=0, add=1, sw=2, rms=3) since for S2MM the id rides
the source (compute/mem) side. matvec internal mem<->compute hops stay circuit
`flow` (no packets). Template for shim-channel packet sharing: `_emit_matvec_add_pack_k2048`
(D1) already does pkt0=weights / pkt1=residual on one shim MM2S0.

Wrinkle: matvec X is a col-0 broadcast (shim[0] MM2S1 -> all 8 mat tiles) while
add/sw in1 are per-col; so col-0 MM2S1 carries pkt0(broadcast)+pkt1+pkt2(+pkt3),
cols 1-7 MM2S1 carry only pkt1/pkt2. The mem tile (row1) is matvec-only here.

**Wiring**: add `c1_merged` to the valid-set in `build_o_gemv_ffn_module`, emit
`_emit_call2_merged()` + the existing `_emit_matvec_add_pack_k8192("d4_dg_a2_pack", ...)`,
and set `dispatch_sequence = ("c1_merged", "d4_dg_a2_pack")`. Selectable via
`PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c1_merged`.

**Validate**: delete `decode_kernel_cache/o_gemv_ffn.{elf,npu.air.mlir}` +
`.o_gemv_ffn.work/`, then run from `build_peano/`:
`PYTHOC_LLAMA_O_GEMV_FFN_PACK_MODE=c1_merged HF_MODEL_ID=unsloth/Llama-3.2-1B-Instruct
python3 ../llama32_1b_inference.py --run-only --n-tokens 10 --profile --quant bf16
--model instruct --hf-model-id unsloth/Llama-3.2-1B-Instruct`. Gold tokens
`[271,791,6864,315,9822,374,12366,13,128009]` -> "...Paris.". Add a gate test
mirroring `test_hf_answer_gate_o_gemv_ffn_fused_matvec`.

**Expected first-run failure mode**: deadlock (packet-route or lock-choreography
bug). Debug with the aie-xray / trace tooling; the 6-stage await ordering and the
per-core output packet ids are the usual suspects. Budget several build/run iters.

Open implementation detail: row-1 mem-tile DMA channel budget when one column's
mem tile stages weights for multiple sequential matvec waves (time-shared, but
the @memtile_dma BD chains are compile-fixed -- may need a wave-agnostic chain).
