# O-wave cross-token state hazard in c2_attn — root cause + quiescence-by-design

Read-only design investigation, 2026-06-19. Strix Halo / aie2p, mlir-aie PythoC,
`llama32_1b`. **No code was changed.** Everything below is from IR/source
inspection of `builders/o_gemv_ffn.py`, `builders/c2_attn.py`, the attention add
core/mem, the `@runtime_sequence`, and the scope docs
(`ATTN_DECODE_GQA_SCOPE.md`, `RESIDENT_DEVICE_EVOLUTION.md`). Hardware was
available but the root cause is determinable from the BD-chain / lock topology
without a run; §1 states confidence and what a confirming run would add.

The framing the user asked for is the spine of this doc: the bug is a symptom of
a missing **quiescence invariant** — *every piece of device state must return to
its exact init value at task end* so dispatch N+1 starts byte-identical to
dispatch 0. The c2_attn O-wave violates that invariant in (at least) one
concrete place and is structurally fragile in several others. §2 enumerates the
full state taxonomy, §3 gives the per-class remedy, §4 the on-device self-re-arm
migration, §5 the residency connection, §6 prioritized recommendations.

---

## 0. The two delivery paths — which one c2_attn actually uses (load-bearing)

The O-wave's activation X reaches the row-2 matvec herd by ONE of two mechanisms,
selected by the `_memx` knob at `o_gemv_ffn.py:4040`:

```python
_memx = (not with_down) and os.environ.get("PYTHOC_C2_MEMX", "1") == "1"
```

For the c2 dispatcher, `with_down` is set from `_c2_down` (`o_gemv_ffn.py:5412`):

```python
_c2_down = pack_mode in {"c2_merged", "c2_attn"}      # 5412
... _emit_call2_c2(pack_mode, with_down=_c2_down, attn_wave0=_c2_attn, ...)  # 5456
```

So **for `c2_attn`, `with_down=True` ⇒ `_memx=False`.** c2_attn does **not** use
the per-column mem-relay X ring (the `_mxb`/`x_empty`/`x_full` machinery at
`4948–4976`). That relay is dead in this path. Instead c2_attn uses the
**single-source shim broadcast** (`_mat_wave._x_once` branch, `o_gemv_ffn.py:5121`):

```python
else:
    x_tasks = [_x_once(f"air_channel_{X_CH}", x_emit, 1)]   # 5121, pkt id 1
```

and `_x_once` configures that broadcast task with **`repeat_count=0`**
(`o_gemv_ffn.py:5104–5109`):

```python
def _x_once(chan_name, bd_emit, pid):
    t = dma_configure_task_for(chan_name, repeat_count=0)   # 5105
    with bds(t) as bd: bd_emit(bd, pid)
    dma_start_task(t)
    return t
```

The X BD itself (`_o_x`, `o_gemv_ffn.py:5206–5209`) reads from `args[1]` (the wide
attn_out scratch) and gathers the 4 real GQA head-rows head-major:

```python
dma_bd(args[1], offset=0, len=EMB_DIM,
       dimensions=[(8, 4096), (256, 1)], packet=(0, pid))   # 5207-5208
```

That single shim MM2S read, packet id 1, fans out to all 8 row-2 matvec mem
tiles, each landing it in the tile-local `normed` buffer (mat-mem block[4],
`o_gemv_ffn.py:4318–4323`). The matvec core consumes it in place for 32 O-wave
chunks (`o_gemv_ffn.py:4363–4372`).

**This is the central difference vs c2_merged:** in c2_merged, `args[1]` is the
host-written `attn_out` DDR BO — freshly written + BO-synced from the host every
call, so the shim X-broadcast reads a known-good, host-coherent buffer each
dispatch. In c2_attn, `args[1]` is **produced on-device by the wave-0 attention**
(`_attn_wave0`'s `APO_CH` S2MM task lands each group's tile into
`args[1][g*4096..]`, `o_gemv_ffn.py:5089–5098`), and the same shim then reads it
back for the O-wave X. The producer and consumer of `args[1]` are now both
on-device, coupled only by the host runtime_sequence's `dma_await_task` ordering
— **not** by a host BO sync. That coupling is where the hazard lives.

---

## 1. Root cause

**Confidence: HIGH that the hazard is a producer→consumer coherence/ordering gap
on the device-written `args[1]` scratch between the attention APO write and the
O-wave shim X read; MEDIUM on exactly which of the two coherence sub-mechanisms
dominates (DDR write-not-globally-visible vs await-token semantics).** It is NOT
the per-column X-relay over-delivery (that path is OFF in c2_attn, §0), and it is
NOT a lock-init imbalance in the matvec O-ring (the lock accounting balances per
token — see below). The state that carries across the dispatch boundary is the
**contents of the `args[1]` DDR scratch as seen by the shim MM2S X read**, plus
the fact that nothing forces that scratch (or the shim read's view of it) back to
a known state at task end.

### 1a. The lock/BD accounting *does* balance per token (rules out candidates b, c, d)

I traced the per-token acquire/release balance for every lock the O-wave touches.
All return to init at token end, so the documented candidates (b) lock-count
drift, (c) ring-slot pointer drift, and (d) BD-chain mid-cycle position are
**not** the cause for the O-wave's own rings:

* **mat-tile `x_avail`/`x_ready`** (`_six_locks`, init `x_avail=1`, `x_ready=0`,
  `o_gemv_ffn.py:4075–4076`). Per token the mat-mem X ring runs exactly 3 BDs
  (O block[4], gate block[5], up block[6], `4318–4342`): each acquires `x_avail`
  GE1 and releases `x_ready`; the core acquires `x_ready` 3× and releases
  `x_avail` 3× (`4364/4372`, `4383/4396` ×2). 3 in / 3 out ⇒ both locks return to
  (1,0). Balanced.
* **mat-tile `y_done`/`y_full`** init (1,0) (`4077–4078`): O wave does
  N_CHUNKS_O=32 iters, gate+up do N_CHUNKS_GU=128 each; the mem MM2S0 y-drain ring
  (block[1], self-cycling `4313`) drains exactly as many. Balanced per token.
* **mat-tile `w_avail`/`w_ready`** init (1,0): W ring self-cycles (block
  `_w_blk+1`, `4349`), one W per chunk, released/acquired symmetrically. Balanced.
* **The X BD ring `next_bd` topology is a closed 3-cycle** O→gate→up→O
  (`4323/4332/4342`). After exactly 3 deliveries it is back at block[4] — i.e.
  the ring pointer returns to its start position every token. No mid-cycle drift.

So the matvec O-ring is *internally* quiescent: its locks and BD pointer return
to init. **The leak is not inside the matvec tile's own state machine.** This is
consistent with the measured symptom: `attn_out` is correct every token, the
O-ring runs the same number of iters every token, yet `proj` drifts run0→run1.

### 1b. The actual carried state: the device-written `args[1]` scratch + shim read coherence

The hazard is the **handoff buffer `args[1]` itself and the shim DMA that reads
it**, which are NOT covered by any lock that resets per dispatch:

1. **Producer/consumer are decoupled across the dispatch boundary.** Within one
   dispatch, `_attn_wave0()` runs first (`o_gemv_ffn.py:5237`), its APO out-tasks
   are `issue_token=True` and the sequence `dma_await_task`s them
   (`5099–5100`) before `_mat_wave` issues the O-wave X read (`5239`). So
   *intra*-dispatch ordering (attn write → O read) is enforced by the await. But
   the await only guarantees the S2MM **shim-side** token fired; for a
   DDR-resident scratch read straight back by another shim DMA in the same
   dispatch, whether the just-written bytes are *globally visible* to the X read
   depends on DDR write-completion / cache-coherence semantics that the host BO
   sync normally provides and that are **absent here** (c2_merged gets coherence
   for free from the host `attn_out` BO write+sync; c2_attn never syncs `args[1]`
   — it is pure device round-trip). This is exactly the trap in
   [[resident-runner-output-bo-sync]]: *output BOs never sync to device; bounce
   designs read stale/uninitialized scratch.* On token 0 (cold PDI) the scratch
   region happens to be in a benign state; from token 1 on, the residue of the
   previous token's `args[1]` (or a partially-visible new write) is what the X
   read sees → deterministic-after-warmup ~17% drift, which is the exact measured
   signature (`ATTN_DECODE_GQA_SCOPE.md:476–484`).

2. **Nothing returns `args[1]` to a known init state at task end.** There is no
   zero-fill, no host re-sync, no producer re-arm of `args[1]` between dispatches.
   The buffer is whatever attention last wrote. A quiescence invariant would
   require `args[1]` (and the shim read's notion of "fresh data") to be
   re-established every dispatch. Today that re-establishment is implicit and only
   correct on the cold (token-0) path.

3. **The shim broadcast task is freed and re-created per dispatch
   (`_x_once`/`dma_free_task`), so the *task* quiesces — but the DATA it reads
   does not.** `repeat_count=0` is correct (exactly one delivery, fanned to 8
   consumers — this is NOT an over-delivery; the prior "X repeat 32 vs consume 16"
   jam from the standalone matvec is not present here). So candidate (a),
   over-delivery residue in the shared MM2S queue, is **ruled out for c2_attn**:
   the count is exact. The residue that drifts is in DDR (`args[1]`), not in the
   MM2S queue.

### 1c. Why token 0 is correct and token 1+ wrong — the "warmup" tell

The cold PDI load (`aiex.configure`) resets the whole device
(`AIEMaterializeRuntimeSequences.cpp:119`, cited `RESIDENT_DEVICE_EVOLUTION.md:203`)
and `args[1]` is in its post-LoadPDI state. The first token's attention writes
`args[1]`, the await fires, and the X read either sees the fresh write (because
the cold path's timing/coherence happens to line up) or sees post-reset zeros
overlaid by the write — either way token 0 is correct. On token 1+, the device is
NOT reloaded (resident PDI reuse — the whole point of residency), so `args[1]`
retains token-0's contents AND the device DMA/cache state is mid-cycle; the X read
now races a not-yet-globally-visible new write against stale prior contents. The
result is deterministic once the steady-state DMA phase relationship settles
(hence "deterministic-after-warmup"). This is the classic *device-written
activation handoff drifts on PDI reuse vs host-synced BO* failure, called out
verbatim as a reusable hazard in `ATTN_DECODE_GQA_SCOPE.md:566–568`.

### 1d. What a read-only run would add (not required for the diagnosis)

The IR topology is sufficient to localize the carried state to `args[1]`
coherence. A confirming read-only run (via `tools/test_c2_attn.py stepR` /
`c2_attn_resident_npu`, which the scope doc already exercised to *measure* the
drift) could pin sub-mechanism 1 vs 2 by: (i) inserting a host read-back +
re-write of `args[1]` between tokens (host-side, no device edit) and checking the
drift vanishes — confirms it's `args[1]` data coherence, not a device lock; or
(ii) zeroing `args[1]` from the host between tokens. Both are host-harness-only
and would not edit builders/kernels. I did not run them because the existing
measured evidence in the scope doc (attn_out bit-exact every token; proj drifts
0.001→0.018 run0→run1; legacy single-chunk c2_attn shows the identical drift;
c2_merged with host-synced `attn_out` does NOT drift) already isolates the
variable to "device-written vs host-synced `args[1]`," which is precisely the
coherence gap above.

**Bottom line:** the carried state is the `args[1]` DDR scratch and the shim X
read's coherence with the on-device attention write. It leaks because the c2_attn
path removed the host BO write+sync that gave c2_merged a fresh, globally-visible
`attn_out` every dispatch, and replaced it with a device round-trip that has no
quiescence/coherence guarantee at the dispatch boundary.

---

## 2. State taxonomy — everything that persists across a dispatch

For each decode device (rgr / c2_merged / c2_attn / lm_head), the state that
survives from dispatch N to N+1 unless explicitly reset:

| # | State class | Where (c2_attn) | Quiesces today? | How it leaks |
|---|---|---|---|---|
| 1 | **Core loop position** (parked forever-loop) | every compute core: `for _ in range_(sys.maxsize)` — mat `4362`, add/attn `4569`, sw `4680`, down `4732` | YES (in practice) | Core parks at the top-of-loop `use_lock(...ready..., GE1)` between dispatches; it is mid-`scf.for` but at a deterministic acquire point. Safe ONLY because the lock it blocks on is itself at init. If any feeding lock is off by one, the core resumes at the wrong wave. Fragile-by-construction, not leaking today. |
| 2 | **Lock counts** | mat `_six_locks` (`4071`), add/attn `a_lock` ids 6–13 (`4275`), io `_io_locks`, mem `mem_locks` (`4051`) | YES for the O-ring (proven balanced §1a); attention locks balanced per token (`q/k/v_avail`/`o_full` acquired+released symmetrically, `4571–4634`) | A lock leaks only if a wave's acquire/release count differs across dispatches. Today they balance, but the discipline is implicit — there is no assertion that ∑acquire==∑release per dispatch. The `_memx` x_empty/x_full relay (`4067`, OFF in c2_attn) would be a leak source if enabled with a count mismatch. |
| 3 | **BD-chain position** (`next_bd` cyclic rings) | ALL rings are forever cycles: mat X 3-cycle (`4323/4332/4342`), mat y self (`4313`), mat W self (`4349`), add MM2S0/S2MM0/S2MM1 rings (`4426–4503`), memtile rings (`4922–5005`) | YES *iff* deliveries-per-dispatch == ring length (or a divisor) | A cyclic ring quiesces only when the per-dispatch delivery count is an exact multiple of the cycle length so the pointer lands back at slot 0. The mat X 3-cycle gets exactly 3 deliveries/token (O,gate,up) ⇒ returns to slot 0. The attn rings carry "attn-once + add-twice" sized cycles (`4418`) tuned to exactly that count. **This is the "EXACT delivery counts" discipline as a BD-position invariant** — it holds today but is hand-tuned and brittle (the C1/C2 wedge was a delivery-count violation, [[packet-id-mask-rules]]). |
| 4 | **In-flight / over-delivered DMA queue data** | shim MM2S X broadcast (pkt 1), shim S2MM convergence (ids 1/5/6/7) | YES for c2_attn X (count exact, `repeat_count=0`, §1b·3) | The documented jam (standalone matvec broadcast x repeat 32 vs consume 16) is the canonical leak of this class. c2_attn's X is exact so no MM2S residue. Distinct output pkt ids (1/5/6/7) prevent S2MM convergence mis-merge ([[stale-pdi-switch-masters]]). Not the c2_attn bug, but the highest-risk class in general. |
| 5 | **DDR handoff scratch `args[1]`** (the device-written attn_out) | `args[1]`, written by APO (`5089–5098`), read by O-wave X (`5206–5209`) | **NO** | **This is the c2_attn root cause (§1).** No zero/re-sync/re-arm at task end; coherence between the device write and the device read-back is not guaranteed at the dispatch boundary the way a host BO write+sync guarantees it for c2_merged. |
| 6 | **RTP buffers** | resident-L folded into q padding (`a_rtp[col]="q_padding"`, `4294`; read via `fptosi` `4582`) | YES (host re-writes q every token; L rides in q) | The runtime-L mask is exact every position (`ATTN_DECODE_GQA_SCOPE.md:471–474`). RTP is re-delivered each dispatch via the q DMA, so it quiesces. (A `write32`-baked RTP would NOT quiesce — it's build-time-constant; this is why L is DMA'd, see comment `4291–4293`.) |
| 7 | **L1 (core-tile SRAM)** — `gp`, `sprun`, `up`, `g`, `normed`, `y`, `w` | per-tile buffers | Mixed: explicitly re-init where it matters, implicitly carried elsewhere | Attention zero-fills `gp/sprun/g` and neg-inf-fills `up` at the TOP of every token (`4586–4588`, `4599`) ⇒ those quiesce by construction. `normed` is overwritten by the X read each token (block[4]) ⇒ quiesces as long as the X read delivers correct data (which is exactly the §1 failure — if X is stale, `normed` carries stale). So L1 `normed` is a *secondary* carrier downstream of the `args[1]` leak. |
| 8 | **L2 (memtile SRAM)** — `mem_buf_w/y/x` | row-1 memtiles | Overwritten per wave; **survives PDI swap** | L2 RAM persists across a PDI swap ([[memtile-survives-pdi-swap]]) — so any L2-resident handoff must be treated as carried state. In c2_attn the L2 W/y buffers are rewritten each wave (gated by w/y locks) so they quiesce; the `_memx` X relay L2 buffer is unused (OFF). But the PDI-swap-survival fact means a future L2-resident attn_out handoff (the natural fix to avoid DDR) MUST add an explicit reset, exactly like #5. |

**Summary of who leaks:** only **#5 (DDR handoff `args[1]`)** actively leaks today
and causes the c2_attn bug; **#7 `normed`** is its downstream victim. #1/#3/#4 are
*quiescent-but-brittle* — they hold only because delivery counts are hand-tuned;
they are the latent next-bug class. #2/#6/#8 quiesce by construction in the
current wiring.

---

## 3. Quiescence-by-design — the principle and per-class remedy

**Principle.** A dispatch is *quiescent-by-construction* iff at task end every
state class in §2 is provably back at its dispatch-0 init value, with the proof
being structural (counts/topology), not empirical (it happened to work). Then
dispatch N+1 == dispatch 0 byte-for-byte and cross-token carryover is impossible.
The invariant to enforce, per class:

* **#5 DDR handoff (the actual bug) — REMEDY: make the handoff coherent or
  reset it.** Three options, cheapest first:
  1. **Keep the host BO write+sync** for `attn_out` (i.e. don't fold attention
     into the device; the c2_merged contract). This is the *zero-risk* fix and is
     the existing recommendation (`ATTN_DECODE_GQA_SCOPE.md:570` — keep attention
     on CPU). It trivially satisfies #5 because the host re-establishes `args[1]`
     every dispatch.
  2. **On-device explicit reset/barrier:** after the O-wave consumes `args[1]`,
     have a producer leg zero `args[1]` (or a lock that forces the X read to wait
     for a *fresh* attention write with a global-visibility barrier). Cost: one
     extra zero-fill DMA per token (negligible bytes). This makes #5 quiesce.
  3. **Move the handoff to L1/L2 with a lock** instead of DDR round-trip, so the
     coherence is lock-mediated (L1/L2 are not subject to the DDR
     write-visibility gap). But note #8: L2 survives PDI swap, so the lock must
     also re-init — which it does if it's a normal `aie.lock` reset by LoadPDI on
     token 0 and balanced thereafter.
* **#3/#4 BD-position & DMA-queue — REMEDY: exact-delivery-count discipline as a
  *checked* invariant, OR one-shot finite BD chains.**
  - *Forever cyclic rings* (today) quiesce only if deliveries/dispatch is a
    multiple of the cycle length. This is the cheapest at steady state (no re-arm)
    but is the fragile path — a single count change wedges the next dispatch
    ([[packet-id-mask-rules]] "a wedge persists into the next process").
  - *One-shot finite BD chains* (the user's preferred direction): each dispatch
    arms a finite chain that runs exactly the deliveries it needs and ends (EndOp,
    no `next_bd` back-edge). The chain is re-armed per dispatch (by host today, by
    a core via control packets in the §4 design). This makes #3 quiesce *by
    construction* — there is no persistent ring pointer to land wrong; the chain
    is gone at task end. **Trade-off:** per-dispatch re-arm cost (host: a few
    `dma_configure_task_for` calls — already paid in `_mat_wave`/`_eltwise_wave`;
    on-device: ~58 cyc/transfer of control-packet programming, [[core-programs-memtile-dma-ctrlpkt]]). For decode (overhead-bound, not BW-bound) the re-arm cost is in the noise; the robustness is worth it.
* **#1 core loop position — REMEDY: park at a single, lock-gated quiescent
  point.** A `sys.maxsize` forever core is acceptable IFF it always parks at the
  same top-of-loop acquire on a lock that is at init between dispatches (true
  today). The *stronger* form is a finite per-dispatch core invocation, but that
  costs a re-launch. The middle ground: keep the forever loop but make the FIRST
  acquire each iteration the canonical quiescent barrier and assert all feeding
  locks are at init there.
* **#2 lock counts — REMEDY: per-wave acquire==release as a build-time
  assertion.** The emitter already balances them; making it an explicit invariant
  (count acquires/releases per lock per wave at emit time, assert equal) converts
  "happens to balance" into "provably quiesces."
* **#6 RTP — REMEDY: DMA-delivered (not write32-baked) RTPs**, already done (L in
  q padding). A baked RTP is build-time-constant and cannot vary per token without
  a new PDI; the DMA path re-delivers every dispatch ⇒ quiesces.
* **#7/#8 L1/L2 — REMEDY: explicit re-init at wave entry.** Attention already
  zero/neg-inf fills its L1 accumulators at token top (`4586–4588`). The general
  rule: every accumulator/handoff buffer gets an explicit fill at the start of the
  wave that owns it, so its prior-token contents never matter. #8 additionally
  needs this because L2 survives PDI swap.

**The unifying trade-off:** forever cyclic rings minimize per-dispatch work but
require hand-proven exact-delivery counts and are the empirical wedge source;
one-shot finite chains (+ dynamic re-enqueue) cost a small re-arm per dispatch but
make #1/#3/#4 quiesce structurally. For an *overhead-bound* decode workload the
re-arm cost is negligible and the robustness directly buys residency (§5).

---

## 4. On-device migration — the self-quiescing, self-re-arming resident core

Today the `@runtime_sequence` (host) does all dispatch-boundary work:
configure/start/await/free DMA tasks, deliver RTPs (L in q), and implicitly
re-establish `args[1]` (only correctly on c2_merged via BO sync). To make a core
*resident and self-quiescing*, each of those host responsibilities migrates onto a
core tile via **control packets** — proven in this tree (see below). "It's just
engineering," and most of the pieces exist.

What the runtime_sequence does, and what moving it on-device takes:

| runtime_sequence responsibility | c2_attn site | On-device mechanism | Cost | Proven here? | Risk |
|---|---|---|---|---|---|
| **DMA task configure + start** (`dma_configure_task_for`/`dma_start_task`) | `_mat_wave`, `_eltwise_wave`, `_attn_wave0`, `_x_once` (`5104–5298`) | Core builds a 6-command control program (reset, unreset, 2× BD halves, enable, start-queue) and streams it as ONE packet to the target tile's `TileControl` port | ~58 cyc/transfer; ~16 cyc/extra BD ([[dynamic-dma-overhead]]) | **YES** — `moe_control_packets/moe_control_packets.py` (core reprograms a memtile MM2S BD from runtime data); `microbench/memtile_program_cost/` (cost study, the base path) | Peano store-scheduling bug: BD words must be compile-time constants or use the own-tile 0x8xxxx alias (`mdv6/kernels/rn3_chain_pythoc.py:192–231`). Raw `0x1D000` sext-lowers to a wedging pointer. |
| **Channel reset** | implicit in LoadPDI today | The 6-command program's cmd0/cmd1 ARE the channel reset (write MM2S/S2MM `_CTRL`) — no CDO needed | folded into the configure cost | **YES** — included in `memtile_program_cost` / moe path ([[core-programs-memtile-dma-ctrlpkt]]) | Must include the reset; odd DMA channels use BD ids 24–47 (matches c2's `bd_id=24+_i` pinning, `4965/4974`) |
| **DMA await** (`dma_await_task`, the attn→O ordering, `5099–5100`) | host | Replace with a core `use_lock(...AcquireGreaterEqual...)` on a completion lock the producer leg releases — pure on-device sync | 0 extra DMA; one lock | **YES** — `dynamic_dma_use_lock.py`, `ctrl_packet_codex/core_to_core_ctrl_packet.py` (sender sets receiver lock via ctrl packet) | localized lock index rule (`localized_lock_index()`); self-tile base = 48+k ([[pythoc-use-lock]]) |
| **DMA free / re-arm** (`dma_free_task`) | host | One-shot finite chain re-armed by the next control-packet program; OR `aie.objectfifo` native `repeat_count`/`set_repeat_count` for a fixed multi-delivery | re-arm = one more control transfer | **PARTIAL** — `set_repeat_count` exists & lowers ([[wtreplay-objectfifo-native-repeat]]); the explicit re-enqueue-per-token is `rn3_chain_pythoc.py` wt-replay (ACTIVE) | `use_next` BD chains need *paced* packets ([[core-programs-memtile-dma-ctrlpkt]]) |
| **RTP / lock writes** (L delivery; any `write32`) | L rides in q DMA | A core writes the target tile's lock/RTP via a control packet (`write_tm` to the lock/RTP register, or a TileControl ctrl packet) | a few words/packet | **YES** — `ctrl_packet_opus.py` (PASS), `dynamic_dma_add_one.py` (PASS) | host must first enable the processor bus: `npu_maskwrite32(Core_Processor_Bus,1,1,col,row)` (the documented gotcha) |
| **Re-establish `args[1]` coherence** (the #5 fix) | host BO sync (c2_merged) / nothing (c2_attn) | Producer leg (attention) writes the handoff to L1/L2 with a lock; consumer (O-wave) acquires that lock — coherence is lock-mediated, no DDR round-trip, no host sync needed | one lock + on-chip buffer | partially — L2 handoff pattern exists in the codebase but the attn_out→O L2 handoff is **UNBUILT** | L2 survives PDI swap (#8) → the lock must re-init (it does via LoadPDI on token 0) |

**The composite design — a self-quiescing resident matvec core:** at the top of
its forever loop the row-2 core (a) acquires a completion lock the attention leg
releases (replacing the host await), (b) consumes the on-chip handoff (replacing
the DDR `args[1]` round-trip — fixes #5), (c) runs its 3 waves, (d) emits a
control-packet program that re-arms (resets+reprograms) its own and its memtile's
next-token BD chains as **one-shot finite chains** (fixes #3/#4 structurally), and
(e) releases the handoff buffer back to init (fixes #7). No host runtime_sequence
step is needed between dispatches; the device is byte-identical at each loop top.
This is exactly the "self-re-arming resident core" the user describes, and every
sub-mechanism except the on-chip attn_out→O handoff (b) is already proven on HW in
this tree.

**Proven vs unbuilt (honest inventory):**
- PROVEN (lit/HW PASS): control-packet BD programming (`moe_control_packets`,
  `ctrl_packet_opus`, `memtile_program_cost`), core self-programs own DMA
  (`dynamic_dma_add_one`, `dynamic_dma_use_lock`), core→core lock-set via ctrl
  packet (`ctrl_packet_codex`), per-token dynamic re-enqueue
  (`rn3_chain_pythoc.py` wt-replay, ACTIVE in production chain), objectfifo native
  repeat (`set_repeat_count`).
- UNBUILT for c2_attn specifically: the on-chip (L1/L2) attn_out→O-wave handoff
  with a coherence lock (the #5 fix), and a row-2 core that emits its own
  re-arm control program inside its forever loop.

---

## 5. Connection to the endgame (RESIDENT_DEVICE steps 5–6)

Designing for quiescence **gets residency substantially for free**, and the
inverse is also true (residency is what *exposed* this hazard — per-position PDIs
ran each c2_attn exactly once/process and masked #5, `ATTN_DECODE_GQA_SCOPE.md:483–484`).

* **Step 5 (16-layer loop in one runtime_sequence):** a layer device looped 16×
  per token reuses ONE PDI across 16 dispatches. That is precisely "dispatch N+1
  must == dispatch 0." Every leak in §2 that is masked by per-dispatch PDI reload
  becomes a real bug under a 16-iteration loop — #5 first, then #3/#4 if any
  count is off. **Quiescence-by-construction is the *enabling precondition* for
  the 16-layer loop**, not a separate feature. The resident c2_attn build already
  proved the mechanism (1 PDI, no wedge, runtime-L RTP) and proved that the ONLY
  thing blocking the loop is the #5 handoff coherence
  (`ATTN_DECODE_GQA_SCOPE.md:505–512`).
* **Step 6 (DMA/lock-only reconfigure, `AIERT.cpp:993`):** the HW can re-arm
  DMA/locks WITHOUT reloading the core ELF — "Skip cores without elf_file … only
  need DMA/lock reconfiguration" (`RESIDENT_DEVICE_EVOLUTION.md:250–256`). This is
  the hardware blessing for the §4 design: the cores stay resident (loop position
  #1 never resets — which is fine BECAUSE we made it quiescent), and only the
  DMA BDs/locks are re-armed each token — exactly what a core emitting one-shot
  control-packet re-arm programs does on-device. So the §4 self-re-arming core IS
  the on-device realization of the `AIERT.cpp:993` reconfigure path, and L2
  surviving the swap (#8) is the same fact that makes a persistent on-chip handoff
  buffer viable.
* **Autonomous self-driving loop:** once (a) the handoff is on-chip + lock-coherent
  (#5 fixed) and (b) re-arm is on-device control packets (#3/#4 one-shot), the
  device drives its own 16-layer/N-token loop with the host only supplying
  weights/activations — the autonomous resident decode. Quiescence is the property
  that makes that loop *correct*; without it the loop drifts after dispatch 0,
  which is exactly what we observe.

So: **yes, designing for full quiescence gets residency for free** — the same
invariant (state returns to init at task end) is simultaneously the correctness
condition for the 16-layer loop and the structural requirement that lets a core
re-arm itself on-device. The decode lever remains residency + AWQ
(`ATTN_DECODE_GQA_SCOPE.md:562–564`), and quiescence is the gate to residency.

---

## 6. Prioritized recommendations (no code written)

Ranked by effort/risk vs payoff. All are design directions; nothing here was
implemented.

1. **Smallest fix that eliminates the c2_attn hazard — keep `attn_out` host-synced
   (don't device-produce it), OR add an on-device `args[1]` reset/barrier.**
   *Effort: trivial (the host-synced form is literally c2_merged's contract).
   Risk: none.* Per `ATTN_DECODE_GQA_SCOPE.md:570`, the project already concluded
   on-NPU decode attention doesn't pay off, so the *operational* recommendation is
   keep attention on CPU → `args[1]` stays a host-synced BO → #5 cannot leak. This
   is the smallest correct fix and is already the standing recommendation. If
   on-device attention is revived later, the minimal device fix is option 3a/3b
   below (zero/re-sync `args[1]` per dispatch, or lock-mediate it).

2. **Principled redesign — on-chip lock-coherent attn_out→O handoff + checked
   exact-delivery invariants.** Replace the DDR `args[1]` round-trip with an
   L1/L2 buffer released/acquired by a lock (fixes #5 structurally, no DDR
   coherence gap), and add emit-time assertions that every wave's per-lock
   acquire==release and every cyclic ring's deliveries-per-dispatch is a multiple
   of its cycle length (turns #2/#3/#4 from "happens to balance" into "provably
   quiesces"). *Effort: moderate — touches the ~5k-line matvec core's `normed`
   handoff (the surgery the scope doc flags as out-of-scope,
   `ATTN_DECODE_GQA_SCOPE.md:509–511`); the assertions are cheap. Risk: medium
   (core surgery), but it is the fix that makes the 16-layer loop correct.*

3. **On-device re-arm north star — self-quiescing resident core via control
   packets.** Implement the §4 composite: core emits its own one-shot finite BD
   re-arm program (reset+reprogram+start-queue as one control transfer), replaces
   host awaits with completion locks, and holds the handoff on-chip. *Effort:
   high (new on-device control-packet glue in the matvec core), but every
   sub-mechanism except the attn_out→O on-chip handoff is already proven on HW
   here (moe_control_packets / rn3 wt-replay / dynamic_dma_use_lock). Risk:
   high — Peano store-scheduling bug (constant-only BD words / own-tile 0x8xxxx
   alias), pacing for `use_next` chains, BD-id 24–47 ranges, processor-bus enable.
   Payoff: this IS the `AIERT.cpp:993` DMA/lock-only reconfigure realized
   on-device = the autonomous resident decode loop (steps 5–6).* This is the
   north star but only worth it once residency (not attention) is the active
   thread — which the scope doc says it should be.

### What I could and could not determine read-only

- **Determined (HIGH confidence):** the carried state is the device-written
  `args[1]` DDR scratch + the shim X read's coherence with it; the matvec O-ring's
  own locks/BD-pointer balance and quiesce per token (ruling out candidates b/c/d
  and the over-delivery candidate a — `_memx` is OFF in c2_attn and the broadcast
  is `repeat_count=0` exact); the full §2 taxonomy and which classes leak; that
  every on-device-re-arm sub-mechanism except the on-chip handoff is already
  proven on HW in this tree.
- **Not pinned without a run (MEDIUM):** which sub-mechanism of #5 dominates —
  DDR-write-not-globally-visible vs await-token-vs-DDR-completion semantics. Both
  are the same class (device-round-trip coherence absent host BO sync) and the
  same remedies fix both; a host-only read-back/zero of `args[1]` between tokens
  (no builder/kernel edit) would discriminate them, and the existing
  `tools/test_c2_attn.py` harness already measured the drift that the diagnosis
  rests on.
