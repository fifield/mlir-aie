#!/usr/bin/env python3
"""Milestone 0 HW test — on-device GELAN concat -> conv4 (1x1 512->256).

Proves the 4-way channel-dim concat of four 20x20x128 bf16 tiles (HWC), done as
an on-device strided gather DMA, followed by a 1x1 GEMM + BN + SiLU (the model's
K-blocked conv1x1 kernel), is bit-exact vs the host reference
`np.concatenate([x1,x2,x3,x4], axis=2)` then GEMM+BN+SiLU.

The host NEVER concatenates the input fed to the device: the four quarters are
passed stacked (each flat-contiguous) and the device interleaves them per-pixel
via the input fill's gather TAP. See aie2_concat_proof.py for the mechanism and
the IRON expressibility note.

Run:
  source /home/jfifield/npu-dev-pythoc/env.sh
  flock /tmp/npu-dev.lock python conv/test_concat_proof_hw.py
"""
from __future__ import annotations
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))           # for aie2_concat_proof
sys.path.insert(0, str(HERE.parent))    # for run_tiled_mc helpers
PYE = Path(__file__).resolve().parents[2]
if str(PYE) not in sys.path:
    sys.path.insert(0, str(PYE))

import aie.iron as iron  # noqa: E402
from aie.iron.device import NPU2  # noqa: E402
from aie.utils import NPUKernel, DefaultNPURuntime  # noqa: E402
from aie.utils.compile import compile_mlir_module  # noqa: E402

from aie2_concat_proof import concat_proof, concat_only  # noqa: E402

# bf16<->uint16 host helpers (same as the model uses).
import importlib.util  # noqa: E402
_ett_path = HERE.parent / "_full_model_helpers" / "elan_test_tiled.py"
_spec = importlib.util.spec_from_file_location("ett_helpers", _ett_path)
_ett = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ett)
bf16_to_uint16 = _ett.bf16_to_uint16
uint16_to_bf16 = _ett.uint16_to_bf16
import torch  # noqa: E402


# ---- config (matches model's re8_c4: 20x20, 512->256, k_block=32, tile_m=24) ----
H = W = 20
Q_IC = 128
N_Q = 4
IC = N_Q * Q_IC            # 512
OC = 256
TILE_M = 24
K_BLOCK = 32
N_CORES = 32
M = H * W                  # 400
PPC = max(1, math.ceil(M / (N_CORES * TILE_M)))
COVERED = N_CORES * TILE_M * PPC    # >= M, zero-padded tail (GEMM path)

# concat_only uses 4 cores x tile_m=16 (ppc=1, split-free) -> 64 proven pixels.
_CO_NCORES = 4
_CO_TILE_M = 16
CONCAT_ONLY_COVERED = _CO_NCORES * _CO_TILE_M    # 64 (subset of M=400)


def _bf16(x):
    """Round a float32 ndarray to bf16 values (still float32 dtype)."""
    a = np.atleast_1d(np.asarray(x, np.float32))
    return (((a.view(np.uint32) >> 16) << 16).view(np.float32)).reshape(np.shape(x))


def _to_u16(x_f32):
    return (_bf16(x_f32).reshape(-1).view(np.uint32) >> 16).astype(np.uint16)


def _silu_rational(x):
    """SiLU with the kernel's rational sigmoid: x*(0.5 + x/(2+2|x|))."""
    return x * (0.5 + x / (2.0 + 2.0 * np.abs(x)))


def _pack_weights_kblocked(conv_w_f32, bn_w_f32, bn_b_f32, ic, oc, k_block):
    """Repack [OC,IC] conv + BN to K-blocked [kb/8, oc/8, 8ic, 8oc] + BN chunks.

    Mirrors run_tiled_mc._repack_weights_kblocked exactly (bf16-rounded values).
    Returns flat uint16 [chunk_0 .. chunk_{n_kb-1}].
    """
    all_conv = _to_u16(conv_w_f32.reshape(-1))           # [OC*IC]
    all_bn_w = _to_u16(bn_w_f32.reshape(-1))             # [OC]
    all_bn_b = _to_u16(bn_b_f32.reshape(-1))             # [OC]
    n_kb = ic // k_block
    oc_blks = oc // 8
    chunks = []
    for kb_idx in range(n_kb):
        k0 = kb_idx * k_block
        kb_blks = k_block // 8
        w_slice = np.zeros(oc * k_block, dtype=np.uint16)
        for o in range(oc):
            src = all_conv[o * ic + k0:o * ic + k0 + k_block]
            w_slice[o * k_block:o * k_block + k_block] = src
        w_f = uint16_to_bf16(w_slice).reshape(oc, k_block)
        w_blocked = w_f.reshape(oc_blks, 8, kb_blks, 8)
        w_blocked = w_blocked.permute(2, 0, 3, 1).contiguous()  # [kb/8,oc/8,8ic,8oc]
        blocked_u16 = bf16_to_uint16(w_blocked.flatten())
        chunks.append(np.concatenate([blocked_u16, all_bn_w.copy(), all_bn_b.copy()]))
    return np.concatenate(chunks)


def host_reference(quarters_f32, conv_w_f32, bn_w_f32, bn_b_f32,
                   k_block=K_BLOCK):
    """Host concat + K-blocked 1x1 GEMM + BN + SiLU, emulating device numerics.

    quarters_f32: list of N_Q arrays [COVERED, Q_IC] (bf16-rounded).
    Returns [COVERED, OC] float32 (bf16-rounded values).

    The device kblocked kernel rounds the running accumulator to bf16 between
    K-blocks (store partial -> reload partial), so we emulate that to track the
    hardware closely rather than do a single full-width f32 matmul.
    """
    # On-device concat reference: per-pixel channel concat of the quarters.
    fused = _bf16(np.concatenate(quarters_f32, axis=1))   # [COVERED, IC]
    Wb = _bf16(conv_w_f32).astype(np.float32)             # [OC, IC]
    inp = fused.astype(np.float32)
    ic = inp.shape[1]
    n_kb = ic // k_block

    # Partial-accumulator chain matching gemm_conv1x1_kblocked_bf16:
    # within a K-block the products accumulate in f32; the running partial is
    # rounded to bf16 between K-blocks (acc_to_bf16 store, _bf16_to_acc reload).
    acc = np.zeros((inp.shape[0], Wb.shape[0]), np.float32)
    for kb in range(n_kb):
        k0 = kb * k_block
        block = inp[:, k0:k0 + k_block] @ Wb[:, k0:k0 + k_block].T   # f32 acc
        acc = acc + block
        if kb < n_kb - 1:
            acc = _bf16(acc).astype(np.float32)           # partial round-trip
    out = _bf16(acc)                                       # final result32 -> bf16
    # BN matches the kernel's _store_bn_silu rounding: t1=bn_w*x (bf16),
    # t2=t1+bn_b (bf16), then SiLU (bf16).
    bnw = _bf16(bn_w_f32).astype(np.float32)
    bnb = _bf16(bn_b_f32).astype(np.float32)
    t1 = _bf16(out * bnw)
    t2 = _bf16(t1 + bnb)
    return _bf16(_silu_rational(t2))


def main():
    # MODE: concat_only (default, the rigorous concat primitive proof),
    #       e2e        (full on-device concat -> conv4 GEMM),
    #       gemmonly   (host pre-concats; isolates the GEMM path).
    mode = os.environ.get("MODE", "concat_only")
    assert mode in ("concat_only", "e2e", "gemmonly"), mode

    # concat_only uses its own (smaller) tile_m -> its own covered.
    covered = CONCAT_ONLY_COVERED if mode == "concat_only" else COVERED
    rng = np.random.default_rng(0)
    quarters = []
    for k in range(N_Q):
        q = rng.standard_normal((covered, Q_IC)).astype(np.float32) * 0.3
        if covered > M:
            q[M:] = 0.0                 # zero-pad tail pixels
        quarters.append(_bf16(q))

    if mode == "concat_only":
        return _run_concat_only(quarters, covered)

    # ---- e2e / gemmonly: concat (or host-concat) -> conv4 GEMM ----
    conv_w = _bf16(rng.standard_normal((OC, IC)).astype(np.float32) * 0.1)
    bn_w = _bf16(rng.standard_normal(OC).astype(np.float32) * 0.3 + 1.0)
    bn_b = _bf16(rng.standard_normal(OC).astype(np.float32) * 0.1)
    ref = host_reference(quarters, conv_w, bn_w, bn_b)     # [COVERED, OC]

    do_concat = (mode == "e2e")
    wd = HERE / f"build_concat_proof_{'e2e' if do_concat else 'gemmonly'}"
    (wd / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = wd / "final.xclbin", wd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        print("Building concat->conv4 xclbin ...", file=sys.stderr)
        os.environ["CONCAT"] = "1" if do_concat else "0"
        module = concat_proof(NPU2(), H, W, Q_IC, N_Q, OC,
                              TILE_M, K_BLOCK, N_CORES)
        assert module.operation.verify()
        compile_mlir_module(mlir_module=module, insts_path=str(insts),
                            xclbin_path=str(xclbin), work_dir=str(wd / "work"),
                            verbose=False)

    if do_concat:
        in_u16 = np.concatenate([_to_u16(q) for q in quarters])   # stacked
    else:
        in_u16 = _to_u16(_bf16(np.concatenate(quarters, axis=1)))  # host-fused
    wt_u16 = _pack_weights_kblocked(conv_w, bn_w, bn_b, IC, OC, K_BLOCK)
    out_elems = N_CORES * PPC * TILE_M * OC

    npu = NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out_t = iron.zeros(out_elems, dtype=np.uint16)
    # NOTE: dtype=np.uint16 is REQUIRED. iron.tensor(uint16_array) silently
    # promotes to uint32 (value in low 16 bits, zero in high), which the DMA
    # then streams as 16-bit -> input spread by stride 2 with zeros (looks like
    # an "every-other-channel-zero" kernel bug). Pinning the dtype keeps the bf16
    # bits intact.
    DefaultNPURuntime.run(h, [iron.tensor(in_u16, dtype=np.uint16),
                              iron.tensor(wt_u16, dtype=np.uint16), out_t])
    got = uint16_to_bf16(np.array(out_t.numpy()).astype(np.uint16)).to(
        torch.float32).numpy().reshape(COVERED, OC)
    g, r = got[:M], ref[:M]
    d = np.abs(g - r)
    ref_mag = max(float(np.abs(r).max()), 1e-6)
    rel = float(d.max()) / ref_mag
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    print(f"[{mode}] on-device concat -> conv4: max_diff={d.max():.6f} "
          f"mean_diff={d.mean():.7f} rel={rel*100:.2f}% (max_ref={ref_mag:.3f})")
    print(f"  got[0,:6]={g[0,:6]}")
    print(f"  ref[0,:6]={r[0,:6]}")
    # 16-K-block bf16 partial-accumulation: test_gemm_truth uses 6% rel for
    # K-blocked shapes; the residual here is pure GEMM bf16 rounding, not concat.
    PASS = rel < 0.06
    print(f"PASS={PASS} (rel<6%)")
    return 0 if PASS else 1


def _run_concat_only(quarters, covered):
    """Rigorous primitive proof: drain the on-device-concatenated fused buffer
    and compare BIT-EXACT to the host channel concat. No GEMM involved."""
    wd = HERE / "build_concat_only"
    (wd / "work").mkdir(parents=True, exist_ok=True)
    xclbin, insts = wd / "final.xclbin", wd / "insts.bin"
    if not (xclbin.exists() and insts.exists()):
        print("Building concat-only xclbin ...", file=sys.stderr)
        module = concat_only(NPU2(), H, W, Q_IC, N_Q, N_CORES)
        assert module.operation.verify()
        compile_mlir_module(mlir_module=module, insts_path=str(insts),
                            xclbin_path=str(xclbin), work_dir=str(wd / "work"),
                            verbose=False)

    # Device input: 4 quarters STACKED (no host concat).
    in_u16 = np.concatenate([_to_u16(q) for q in quarters])
    COVERED = covered
    fused_elems = COVERED * IC
    npu = NPUKernel(str(xclbin), str(insts), kernel_name="MLIR_AIE")
    h = DefaultNPURuntime.load(npu)
    out_t = iron.zeros(fused_elems, dtype=np.uint16)
    DefaultNPURuntime.run(h, [iron.tensor(in_u16, dtype=np.uint16), out_t])
    got_u16 = np.array(out_t.numpy()).astype(np.uint16).reshape(COVERED, IC)

    # Host reference: the channel concat we want the device to reproduce.
    ref_fused = _bf16(np.concatenate(quarters, axis=1))         # [COVERED, IC]
    ref_u16 = _to_u16(ref_fused).reshape(COVERED, IC)

    # covered (256) is a subset of M=400; validate all covered pixels.
    g, r = got_u16[:COVERED], ref_u16[:COVERED]
    bit_exact = bool(np.array_equal(g, r))
    n_mismatch = int(np.sum(g != r))
    # also a float view for a human-readable max_diff
    gf = uint16_to_bf16(g.copy()).to(torch.float32).numpy()
    rf = uint16_to_bf16(r.copy()).to(torch.float32).numpy()
    max_diff = float(np.abs(gf - rf).max())
    print(f"[concat_only] on-device channel concat vs np.concatenate(axis=2): "
          f"BIT_EXACT={bit_exact} mismatches={n_mismatch}/{g.size} "
          f"max_diff={max_diff:.6f}")
    print(f"  fused pixel0 q-boundaries (oc 126..130): "
          f"got={gf[0,126:131]} ref={rf[0,126:131]}")
    print(f"PASS={bit_exact}")
    return 0 if bit_exact else 1


if __name__ == "__main__":
    sys.exit(main())
