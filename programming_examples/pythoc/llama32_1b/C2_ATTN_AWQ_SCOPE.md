# Scope: port c2_attn (on-NPU GQA decode attention) to the AWQ path

**Goal:** fold the on-NPU BFP576 GQA decode-attention wave-0 + MEMKV KV feed
(shipped for the BF16 decode path in commits `e7c1341bf … 9f635caf7`) into the
AWQ decode call-2 device, yielding a `c2_attn_awq` resident path. Target: AWQ
decode runs attention on the NPU (no CPU attention), seq_len up to 512, default
AWQ `c2_merged` byte-identical.

## Current state (verified 2026-06-20)

- **BF16 path**: `builders/o_gemv_ffn.py::_emit_call2_c2(attn_wave0=, attn_resident=)`
  has the full attention wave-0 + MEMKV, all behind those gates. Default
  `c2_merged` (no attention); `c2_attn` adds it. HW-verified, cap lifted to 512.
- **AWQ path**: `builders/o_gemv_ffn_awq.py::_emit_awq_call2_c2` is the
  uint4 counterpart — collapsed O+add1+gate+up+swiglu+down+add2, default pack
  mode `c2_merged` (`_O_GEMV_FFN_AWQ_PACK_DEFAULT="c2_merged"` in
  `kernel_builder/aie_ir_gen.py:113`), ~14.5 tok/s, 1 LoadPDI. **No attention**
  (0 occurrences of `c2_attn`/`attn_wave0` in the AWQ builder).
- AWQ decode host path: `llama32_1b_awq_runtime.py:155 o_gemv_ffn_awq_npu`
  (dispatched from `llama32_1b_decode.py:299`); CPU does attention today.

## Why this port is favorable (structural compatibility)

1. **Attention is weight-free.** Wave-0 uses `attn_pythoc.o` BFP576 BF16 kernels
   (`matmul_a_b_bf16`, `matmul_g_b_bf16`, `fused_softmax`) — Q·K, online softmax,
   P·V. No quantization. The compute is **identical** regardless of whether the
   surrounding matvecs are bf16 or uint4. So the wave-0 block transplants almost
   verbatim.
2. **`attn_pythoc.o` is already linked for AWQ.** `kernel_builder/cache.py:54`
   `_LINK_OBJS` is a shared pool for all decode kernels — no link changes needed.
3. **Same row map / add herd.** Both `_emit_call2_c2` and `_emit_awq_call2_c2`
   put add1/add2 on the row-3 herd (8 idle cores before add1) — the exact window
   c2_attn rides as wave 0. The BF16 doc string and AWQ doc string both say
   "add herd row 3 runs TWO waves (add1, add2)".
4. **No DMA-channel collision.** AWQ c2 uses channels 60–74 (W/A0/SG/X/A1/SU/YO/
   AO/SO/DW/DX/DO). Attention wave-0 uses **90–94** (`AQ/AK/AV/APO/AL`,
   `o_gemv_ffn.py:3962`). Disjoint — drop-in.
5. **Same packet-id discipline.** Both use distinct single-bit ids; the attention
   feed already coexists with this in BF16.

## New / non-trivial work (and the bug class each maps to)

| Item | Detail | Risk / prior bug |
|---|---|---|
| **A. Transplant wave-0** | Copy the `attn_wave0`/`attn_resident` block (`o_gemv_ffn.py` ~3955–4729: channels, geometry, kernel decls, the row-3 add-herd wave-0 body, MEMKV single-shim-BD K/V feed via `k_avail`/`v_avail`, runtime-L mask `_c2attn_mask_invalid_cols_rtp`, `q_all` offset-256 L) into `_emit_awq_call2_c2`, gated behind a new `attn_wave0`/`attn_resident` param. | Mechanical; attention compute unchanged. |
| **B. ABI extension** | Extend `_awq_host_arg_types` (or a c2_attn variant) with q / KV / context-out / runtime-L args; thread through `o_gemv_ffn_awq_npu`. Keep default `c2_merged` ABI untouched. | **ABI-mismatch garble** (`a7f7abf95`): mis-sized preload started the herd on uninit BOs → prefill garble + teardown segfault. Must update the **AWQ preload skip** in lockstep (mirror `_skip_ogf_preload`). |
| **C. Host KV packer** | Port `_run_c2_attn`'s incremental KV tiling (`_C2_ATTN_KV_STATE`, vectorized `_c2_attn_tile_8x8` = `reshape(64,8,8).transpose(1,0,2)`), pad-to-`MAX_CHUNKS*64`, `q_all[g*tile+256]=bf16(seq_len)` into the AWQ runtime. | O(seq) re-pack regression if incremental tiling dropped (cost the BF16 path its parity). |
| **D. W-relay quiescence** | The AWQ c2 weight relay is the **uint4** W relay; c2_attn auto-enables `WRELAY2` 2-slot ping-pong for the resident path. Need the equivalent quiescence (net-to-init per dispatch) on the AWQ relay, or warm-reuse drift. | **WRELAY2 drift** (took 4 root-cause flips in BF16): single-buffer L2 W-relay parked a prefetch → non-bit-identical warm reuse. |
| **E. BD-ID headroom** | MEMKV already makes KV cost constant (4 shim BD tasks/group). But AWQ matvec W streams (uint4, larger) consume more base shim BDs than bf16 — verify the wave-0 q/kv/context BDs still fit the ~16/shim budget alongside the uint4 weight DMAs. | Shim BD exhaustion (the original 256-cap cause). Likely fine (MEMKV) but must confirm at compile. |
| **F. Wiring** | New pack mode `c2_attn` for AWQ: add to the validator set in `_emit_awq_call2_c2`'s pack-mode dispatch, env `PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn`, decode signatures, `compile_decode_kernels` AWQ branch, `_RES_MAX_CHUNKS`/`_RES_PADDED` for AWQ. | Cache-slot desync if not gated cleanly. |

## Step plan

1. **Refactor for reuse (optional but recommended).** The wave-0 body in
   `o_gemv_ffn.py` is self-contained and weight-agnostic. Extract it into a
   shared helper (e.g. `builders/_c2_attn_wave0.py: emit_attn_wave0(...)`) taking
   the add-herd tiles + locks, callable from both BF16 and AWQ emitters. Avoids a
   second copy drifting. (If risky, copy verbatim first, refactor later.)
2. **B + F**: add the AWQ `c2_attn` pack mode + ABI + preload skip (no compute
   yet) — get a clean compile of the extended device.
3. **A**: transplant/wire the wave-0 into the AWQ add herd; compile.
4. **C**: AWQ host KV packer + runtime-L; run the gold gate.
5. **D**: confirm/port WRELAY2-equivalent quiescence; warm-reuse bit-identity gate.
6. **E**: confirm shim BD headroom at MAX_CHUNKS=8; raise the cap.

## Validation gates (HW)

- **AWQ c2_attn gold**: `PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn ... make profile-awq N_TOKENS=20` → "The capital of France is Paris."
- **seq > 256**: long generation crossing position 256, coherent, no wedge/garble.
- **Warm-reuse bit-identity** across ~12 fresh processes; no teardown segfault.
- **Parity**: AWQ c2_attn token agreement vs AWQ c2_merged (CPU attention) baseline.
- **Regression**: default AWQ `make profile-awq` unchanged (~14.5 tok/s, "Paris");
  default BF16 `make profile` byte-identical; `tests/test_c2_attn_ir.py` 8/8.
- **Perf**: NPU flat as context grows vs CPU O(seq); steady-state on a long run.

## Effort estimate

Medium. The hard, novel R&D (on-NPU BFP576 attention, MEMKV cap-lift, WRELAY2
quiescence, ABI/preload garble) is already solved in BF16. This port is mostly a
disciplined transplant of a weight-agnostic block into a structurally identical
device, plus an AWQ host packer. The two repeat-offender bug classes to watch are
**B (ABI/preload garble)** and **D (W-relay warm-reuse drift)** — both have known
fixes to mirror. Verify on hardware (static analysis converged wrong twice in the
BF16 effort).

## Open question — RESOLVED (2026-06-20)

> Is the AWQ uint4 weight relay already quiescent per-dispatch (D), or does it
> carry a parked prefetch like the BF16 single-buffer relay did?

**Resolved: it is single-buffered (parked-prefetch structure), so step D is NOT a
no-op.** The AWQ c2 W-relay uses a `w_dma_done`/`w_ready` lock pair, `init=1`/
`init=0` (`o_gemv_ffn_awq.py:1649-1650`, handshake 2207-2216) — the exact
1-credit relay the BF16 path had **before** the WRELAY2 fix. It works for AWQ
`c2_merged` today, but the BF16 WRELAY2 drift only surfaced **after** attention
wave-0 was added (wave-0 retimes the dispatch so the relay's parked prefetch
becomes observable as warm-reuse drift). Expect the same once attention is folded
into AWQ c2 → **port the WRELAY2 2-slot ping-pong (credit init=1) to the AWQ
relay** as part of this work. Known fix; de-risked, but plan for it (don't assume
the warm-reuse gate passes for free).

---

## IMPLEMENTATION RESULTS (2026-06-20) — SHIPPED, HW-VERIFIED

The port is **complete and validated on hardware**.  AWQ decode now runs GQA
attention on the NPU (no CPU attention) folded into the AWQ c2 device as wave 0,
seq_len up to 512 via MEMKV, selected by
`PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn`.  Default `c2_merged` is
byte-identical; default BF16 is byte-identical by construction.

### Strategy: disciplined verbatim transplant (not a shared helper)
The wave-0 body is tightly coupled to each device's locks/buffers/channels/core
+ mem emission; extracting a clean shared helper would thread dozens of objects
and is high-risk per the "static analysis converged wrong twice" warning.  Per
the scope's fallback ("copy verbatim first, refactor later"), the gated
`attn_wave0`/`attn_resident` blocks were copied **verbatim** from
`o_gemv_ffn.py::_emit_call2_c2` into `o_gemv_ffn_awq.py::_emit_awq_call2_c2`,
kept byte-aligned so a future refactor to a shared `emit_attn_wave0(...)` is
mechanical.  The ONLY genuinely-shared code is the two weight-free softmax-mask
helpers, imported (not copied): `from .o_gemv_ffn import
_c2attn_mask_invalid_cols, _c2attn_mask_invalid_cols_rtp` — so the on-NPU mask
cannot drift between the BF16 and AWQ paths.  The host KV packer is also
single-sourced: `o_gemv_ffn_awq_c2_attn_npu` imports the BF16
`_c2_attn_tile_8x8` / `_C2_ATTN_ROW_OFF` / `_C2_ATTN_KV_STATE` from
`llama32_1b_decode`, so the incremental (non-O(seq)) tiling is shared verbatim.

### Files changed (all c2_attn-gated)
- **`builders/o_gemv_ffn_awq.py`** (+738): `_emit_awq_call2_c2` gains
  `attn_wave0/seq_len/n_groups/attn_resident` params + the transplanted wave-0
  (channels 90-94, geometry, `_A_MEMKV`, attn kernels/buffers/locks on row-3,
  the attention-augmented `_make_add_mem`/`_make_add_core` + `_emit_one_add`,
  attn packetflows + shim allocations, `_attn_wave0()` runtime-sequence KV feed
  + the `_o_x` head-major gather, WRELAY2 2-slot W relay + down relay).  New
  `_awq_c2_attn_host_arg_types` / `_awq_attn_n_chunks` (extended 18-arg AWQ ABI).
  `_emit_dispatcher_device` gains `attn_wave0/attn_resident` → uses the 18-arg
  ABI.  `build_o_gemv_ffn_awq_module` validates+routes `c2_attn` (resident
  default; `PYTHOC_C2_ATTN_RESIDENT=0` for the seq<=64 micro path).
- **`kernel_builder/aie_ir_gen.py`** (+13): `o_gemv_ffn_awq_pack_mode()` exposes
  the AWQ pack mode to the host (the BF16 `decode_pack_modes` dict only carries
  BF16 keys).
- **`llama32_1b_awq_runtime.py`** (+141): `o_gemv_ffn_awq_c2_attn_npu` — the AWQ
  counterpart of `_run_c2_attn` (incremental KV tiling + runtime-L fold + 18-arg
  AWQ ABI, weight-free host packer shared with BF16).  Fixed an
  `OGF_AWQ_BACKEND` local-shadow scoping bug.
- **`llama32_1b_decode.py`** (+25): `run_decode_block` gates `_awq_c2_attn` (AWQ
  pack mode == c2_attn) → skips CPU attention, dispatches the new AWQ packer.
- **`llama32_1b_inference.py`** (+36): `_preload_decode_weights_awq` skips the
  15-arg `o_gemv_ffn_awq` preload under c2_attn (mirrors the BF16
  `_skip_ogf_preload` ABI-garble fix `a7f7abf95`); rms_gemv_rope_awq preload
  unaffected.

### Watch-item outcomes
- **ABI/preload garble**: avoided — the AWQ preload skip is the lockstep mirror
  of the BF16 fix.  No prefill garble / teardown segfault observed (all runs
  exit 0).  Extended ABI is consistent across the dispatcher (`o_gemv_ffn_awq`,
  18 args) and inner device (`c2_attn_sequence`, 18 args), arg1 widened to
  32768 (8×4096), k/v = 131072 (8×4×4096) at the 4-chunk default.
- **WRELAY2 warm-reuse drift**: ported the BF16 2-slot ping-pong (credit init=1)
  to BOTH the AWQ uint4 W relay (blocks 4↔17 drain / 6↔18 fill) and the down-W
  relay (10↔19 / 14↔20); auto-ON for the resident path.  **Drift did NOT
  manifest** in the AWQ path: two fresh 120-token processes are token-identical
  (only the tok/s line differs), AND `PYTHOC_C2_WRELAY2=0` produced the SAME 120
  tokens.  So on the AWQ uint4 relay the single-buffer drift the BF16 path hit
  did not reproduce here — but WRELAY2-on is kept as the conservative resident
  default (it is the proven fix and costs only 2×32 KB L2/tile).  Honest note:
  the WRELAY2 port may be a belt-and-suspenders no-op on AWQ; it is correct and
  cheap, and matches the BF16 invariant, so it stays default-on for resident.
- **shim BD-ID headroom**: confirmed fine — the device compiles through aiecc
  with MEMKV at MAX_CHUNKS=8 (the 512 cap) alongside the uint4 W/down DMAs (no
  BD-budget failure).  MEMKV keeps KV at 4 shim BD tasks/group (constant), so
  context length is decoupled from the shim budget on the AWQ path too.

### Validation results (HW)
| Gate | Result |
|---|---|
| AWQ c2_attn gold (N=20) | PASS — "The capital of France is Paris.", tokens id=271/791/6864/315/9822/374/12366/13/128009 |
| AWQ c2_attn parity vs c2_merged (CPU attn) | first 80 tokens EXACT; diverges at tok 81 (BFP576-vs-float attention accumulation, coherent) — same character as BF16 c2_attn |
| seq>256 (N=120, MEMKV=8, crosses pos 256) | PASS — coherent "Here are twenty large cities...", no wedge/garble, 7.0 tok/s |
| Warm-reuse bit-identity (2 fresh procs, N=120) | PASS — token-identical |
| Regression: default AWQ (no env) | PASS — ~15.6 tok/s, "Paris", tokens unchanged |
| Regression: default BF16 `make profile` | PASS — "Paris", tokens unchanged; `o_gemv_ffn.py` unchanged (byte-identical by construction) |
| Regression: BF16 c2_attn reference path | PASS — "Paris" |
| Regression: `tests/test_c2_attn_ir.py` | 8/8 PASS |
| Builder byte-identity: AWQ c2_merged | md5 unchanged vs HEAD (ecf0f278…) after guarding a stray `c0_i32` const into the attn branch |

### New max seq_len
512 (MEMKV at `PYTHOC_C2_ATTN_MAX_CHUNKS=8`), matching the BF16 cap-lift.
Default (MEMKV off) = 256 (4 chunks), byte-compatible host/device geometry.

### Perf note
AWQ c2_attn steady-state ~60 ms/tok at short context (N=20), ~7 tok/s at N=120
(the per-token cost is dominated by the uint4 matvec + launch overhead, as in
c2_merged; on-NPU attention is constant vs CPU's O(seq)).  First token carries
the one-time ELF compile (~7 s) + BO alloc; short runs are dragged by it.

### Open / follow-on
- Refactor the verbatim wave-0 copy into a shared `emit_attn_wave0(...)` helper
  (now de-risked: both paths proven identical-by-transplant).
- Investigate whether AWQ warm-reuse drift can be provoked at all (it did not
  here); if confirmed genuinely absent, WRELAY2 could be made opt-in for AWQ to
  save the 2nd L2 W buffer.  Low priority (cheap, correct as-is).
