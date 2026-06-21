# bf16-MAC (full bf16, fp32-accumulate) attention matmul prototype

Goal: recover the ~1% BFP16-EBS8 operand-quantization error in the two on-NPU
decode attention matmuls (`Q·Kᵀ` and `P·V`) by doing them in full bf16 with
fp32 accumulate, **NO bfp16-ebs8 down-convert**. Behind `PYTHOC_ATTN_HP=1`;
default (flag off) stays byte-identical (BFP576 untouched).

Status: **DONE — success gate met.** Paris decodes correctly with the flag on,
the new kernels are tile-for-tile HW-verified, and the precision improvement is
measured. AMD Strix Halo aie2p, mlir-aie/PythoC.

## What the bf16 matmul replaced (the key finding)

There is **no native bf16 matrix-multiply instruction on AIE2P.** AMD's own
`aie_api` `mmul<...,bfloat16,...>` is always emulated, by either (a) converting
operands to `v64bfp16ebs8` and calling the BFP576 8×8×8 op — exactly what the
*existing* `matmul_a_b_bf16` / `matmul_g_b_bf16` do, and the source of the ~1%
error — or (b) the `mac_4x8_8x8_bf16` recipe: 8 element-wise 32-lane bf16 MACs
(`I512_I512_ACC1024_bf_mac_conf`, conf=60) with operand broadcasts/transposes,
keeping full bf16 mantissa and fp32 accumulate. This prototype implements path
(b) in PythoC, replacing the per-(output-tile, k-block) BFP576 op.

The 8×8×8 tile body (verified empirically on HW, not by static reasoning):
- A sub-tile → `T16_4x8` (mode 29) column-transpose; per column k, replicate the
  4-row column to 32 lanes via `concat`×4 then `T16_8x4` (mode 28) → `x_k`
  (lane `r*8+c` = `A[r,k]`, independent of c).
- B sub-tile → `y_k` = row k broadcast across the 4 rows via
  `vextract_broadcast128_I512` (8-bf16 group broadcast). For `A@Bᵀ` (the `a_b`
  kernel) B is first 8×8-transposed with a 2-op `T16_8x8_lo/hi` (modes 52/53).
- `acc += x_k * y_k` over k = the 8×8 output tile, fp32 accumulator, truncated to
  bf16 only at store. Output stays cm8x8 (`(col//8)*512 + row*8 + (col%8)`), so
  fused_softmax and the rest of the pipeline are unaffected.

Empirically pinned (HW) primitives — `vbroadcast64_bf512` / `vbcst_shuffle*`
**hang on HW** and are avoided; only `vshuffle`, `vextract_broadcast128_I512`,
`vector_extract`, `concat`, `vector_cast` are used.

A subtle correctness point that bit once: the flash-attention pipeline calls
`matmul_g_b` as a **read-modify-write** (`gp` is the running output accumulator,
pre-scaled by `mul_r_gp` before the matmul adds into it). So both `_hp` kernels
initialize their f32 accumulator from the **existing** output tile (`load_v` +
`v32bf16_to_v32accfloat`), matching the BFP576 kernels' `C += A·op(B)` — not
`C = A·op(B)`. Starting from zero gave garbage logits (cos≈0.16) until fixed.

## Files changed (all flag-gated; default path untouched)

- `kernels/attn.py`
  - import: added `vextract_broadcast128_I512`.
  - new kernels `matmul_a_b_bf16_hp` (C=A@Bᵀ) and `matmul_g_b_bf16_hp` (C=A@B),
    inserted **before** `fused_softmax` (which must stay last — it's the
    compile_pythoc_source entry point; everything before it is a TU helper).
- `kernels/build.py` (compile_attn): added `vextract_broadcast128_I512` to the
  intrinsic `extras` so the new helpers compile into `attn_pythoc.o`.
- `builders/o_gemv_ffn_awq.py` (`attn_kernels` dict, the AWQ c2_attn device):
  `PYTHOC_ATTN_HP=1` → selects `matmul_a_b_bf16_hp` / `matmul_g_b_bf16_hp`.
- `builders/o_gemv_ffn.py` (`_emit_call2_c2` attn_kernels): same flag gate
  (covers the BF16 c2_attn path + stepR's resident device).
- `builders/attn_decode.py`: `_attn_hp_sym()` helper gates the two standalone
  decode-attn external_func declarations.

Default (flag off) emits the BFP576 symbol names unchanged — verified the
flag-off `o_gemv_ffn_awq.aie.mlir` references `matmul_a_b_bf16` (non-hp), and
`tests/test_c2_attn_ir.py` passes 8/8.

## Program memory (fits)

The attention-core ELF on the c2_attn resident herd (the core that links the
attn matmuls) with `_hp`: **10144 bytes** of program memory, vs the 16 KB/core
limit → ~62% used, comfortable headroom. `attn_pythoc.o` whole-TU text grew to
~10.9 KB (built with `--function-sections`, so each core gc-sections to only the
symbols it calls). The device compiled with no `Overflow of program memory`.

## Numerical results

### Tile-for-tile vs CPU fp32 (the clean isolated proof; HW, random 64×64)
| matmul | BFP576 default (rel) | bf16-MAC `_hp` (rel) |
|---|---|---|
| `matmul_a_b` (A@Bᵀ) | 9.06e-03 | **2.71e-03** |
| `matmul_g_b` (A@B)  | 1.08e-02 | **2.99e-03** |

The `_hp` variants are ~3.3–3.6× more accurate — the ~1% → ~0.3% recovery,
exactly the bf16 operand-rounding floor (not zero, as expected).

### Success gate — Paris (AWQ c2_attn, flag on)
`PYTHOC_ATTN_HP=1 PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn make profile-awq
N_TOKENS=20` →
`A: The capital of France is Paris.`  ✅
(device MLIR confirmed linking `matmul_a_b_bf16_hp` / `matmul_g_b_bf16_hp`.)

### Correlation harness (teacher-forced, REF=BFP576, CAND=`_hp`, 48 tokens,
prompt "who's a good dog?")
`tools/compare_logit_dumps.py /tmp/bfp.npz /tmp/hp.npz`:
- cosine: mean 0.99919, min 0.99351
- pearson: mean 0.99951
- top-5 overlap: 4.90/5
- argmax agreement: **47/48 (97.9%)** — one near-tie greedy flip, as anticipated
- symKL: mean 3.19e-3, max 1.59e-2

(Before the RMW fix this was cos≈0.16 / argmax 2-of-48 = the diagnostic that
caught the accumulator bug.)

### stepR oracle (device-vs-CPU end-to-end c2 output rel error)
Default BFP576: rel ≈ 0.9–2.1% (pos=0 FAILs at 2.1%), det=True.
`_hp` (flag): all positions PASS, det=True, pos=0 improves 2.1%→1.06%.
The end-to-end stepR metric is dominated by the *downstream* bf16 GEMVs +
bf16 output quantization, so it dilutes the attention-only delta — the
tile-level table above is the clean attention measurement. stepR's value here is
the **no-regression + determinism** confirmation (warm-reuse bit-identity holds).

### tok/s cost
Decode is launch/overhead-bound (effective M≈4), so the extra bf16 shuffle work
is ~free at wall-clock: Paris run ~1321 ms/tok (`_hp`) vs ~1261 ms/tok (BFP576),
within run-to-run noise. No meaningful throughput cost, as predicted.

## Determinism / constraints honored
- Default flag-off path byte-identical (BFP576 symbols unchanged); IR test 8/8.
- `attn_pythoc.o` shared by prefill + BF16 decode + AWQ decode — adding the two
  `_hp` helpers did not break those (they're unused symbols, gc-sectioned out)
  and did not blow program memory.
- Warm-reuse bit-identity (stepR det=True) preserved.

## How to reproduce
```
# success gate
PYTHOC_ATTN_HP=1 PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn \
  make profile-awq N_TOKENS=20 AWQ_WEIGHTS=<awq_repacked>
# correlation: REF then CAND (teacher-forced), then compare
PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn PYTHOC_LOGIT_DUMP=/tmp/bfp.npz \
  make profile-awq N_TOKENS=48 PROMPT="who's a good dog?" AWQ_WEIGHTS=<...>
PYTHOC_ATTN_HP=1 PYTHOC_LLAMA_O_GEMV_FFN_AWQ_PACK_MODE=c2_attn \
  PYTHOC_FORCE_TOKENS=/tmp/bfp.npz PYTHOC_LOGIT_DUMP=/tmp/hp.npz \
  make profile-awq N_TOKENS=48 PROMPT="who's a good dog?" AWQ_WEIGHTS=<...>
python3 tools/compare_logit_dumps.py /tmp/bfp.npz /tmp/hp.npz
# NOTE: toggling PYTHOC_ATTN_HP requires clearing the AWQ device cache:
#   rm -f build_peano/decode_kernel_cache/o_gemv_ffn_awq.* ; \
#   rm -rf build_peano/decode_kernel_cache/.o_gemv_ffn_awq.work ; \
#   then drop "o_gemv_ffn_awq" from decode_kernel_cache/manifest.json ; \
#   and rebuild attn_pythoc.o (compile_attn) if attn.py changed.
```
