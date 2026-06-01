"""Per-layer GEMM (1x1 conv) shape registry + sizing helpers.

Lists every (label, in_h, in_w, IC, OC) 1x1-conv shape the model dispatches,
and provides the L1/L2 budget calculations needed to pick (tile_m, k_block,
ppc) for the merged-ELF builder.

(Previously lived in gemm_conv1x1/build_gemm_conv1x1.py, which was an
xclbin builder. The xclbin path was retired in the Phase G+ cleanup; only
the shape table + sizing helpers are still needed.)
"""
import math

N_CORES = 32

# L1 budget for GEMM conv1x1 (must match aie2_gemm_conv1x1.py).
# depth=1, stack=8KB, RTP=32B reserved.
_GEMM_L1 = 65536 - 8192 - 32

# L2 (memtile) budget for accumulated ppc-worth of in+out per col.
_L2_BUDGET = 400 * 1024
_XRT_BUF_MAX = 16 * 1024 * 1024
_MAX_K_BLOCKS = 16

# (label, in_h, in_w, IC, OC). Derived from the rt() call sites in
# test_full_model_mc.py that route through run_gemm_conv1x1_mc /
# run_gemm_pair_mc.
MODEL_LAYERS_1x1 = [
    ("elan_c1",   160, 160,  64,  64),
    ("elan_c4",   160, 160, 128,  64),
    ("re4_c1",     80,  80, 128, 128),
    ("re4_rn1",    80,  80,  64,  32),
    ("re4_c4",     80,  80, 256, 128),
    ("re6_c1",     40,  40, 192, 192),
    ("re6_rn1",    40,  40,  96,  48),
    ("re6_rnm",    40,  40,  96,  96),
    ("re6_c4",     40,  40, 384, 192),
    ("re8_c1",     20,  20, 256, 256),
    ("re8_rn1",    20,  20, 128,  64),
    ("re8_rnm",    20,  20, 128, 128),
    ("re8_c4",     20,  20, 512, 256),
    ("spp_c1",     20,  20, 256, 128),
    ("re12_c1",    40,  40, 448, 192),
    ("re15_c1",    80,  80, 320, 128),
    ("re15_c4",    80,  80, 256, 128),
    ("re15_rnm",   80,  80,  64,  64),
    ("re18_c1",    40,  40, 288, 192),
    ("re21_c1",    20,  20, 384, 256),
]


def _gemm_tile_m(ic, oc_block):
    """Max tile_m (mult of 4) that fits in L1 for non-K-blocked path."""
    wt_bytes = (ic * oc_block + 2 * oc_block) * 2
    remaining = _GEMM_L1 - wt_bytes
    if remaining <= 0:
        return 0
    return (remaining // ((ic + oc_block) * 2) // 4) * 4


def _gemm_tile_m_kblocked(ic, oc, k_block):
    """Max tile_m (mult of 4) for K-blocked config."""
    wt_chunk_bytes = (k_block * oc + 2 * oc) * 2
    remaining = _GEMM_L1 - wt_chunk_bytes
    if remaining <= 0:
        return 0
    return (remaining // ((ic + oc) * 2) // 4) * 4


def choose_k_block(ic, oc, M):
    """Return (k_block, tile_m). k_block=0 means non-K-blocked path."""
    tm_full = _gemm_tile_m(ic, oc)
    if tm_full >= 16:
        return 0, min(tm_full, 256)
    best_kb, best_tm, best_calls = 0, 0, float("inf")
    for n_kb in range(2, _MAX_K_BLOCKS + 1):
        kb = ic // n_kb
        if kb < 8 or kb % 8 != 0 or ic % kb != 0:
            continue
        tm = _gemm_tile_m_kblocked(ic, oc, kb)
        tm = min(tm, 256)
        if tm < 4:
            continue
        calls = math.ceil(M / (tm * N_CORES))
        if calls < best_calls or (calls == best_calls and n_kb < ic // best_kb):
            best_kb, best_tm, best_calls = kb, tm, calls
    return best_kb, best_tm


def compute_ppc(M, tile_m, ic, oc_block):
    """Optimal patches_per_core for non-K-blocked GEMM."""
    ideal = math.ceil(M / (N_CORES * tile_m))
    in_bytes = N_CORES * tile_m * ic * 2
    out_bytes = N_CORES * tile_m * oc_block * 2
    max_xrt_in = _XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = _XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999
    col_in = 4 * tile_m * ic * 2
    col_out = 4 * tile_m * oc_block * 2
    wt = (ic * oc_block + 2 * oc_block) * 2
    per_ppc = col_in + col_out
    max_l2 = (_L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999
    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))


def compute_ppc_kblocked(M, tile_m, ic, oc, k_block):
    """Optimal patches_per_core for K-blocked GEMM."""
    ideal = math.ceil(M / (N_CORES * tile_m))
    in_bytes = N_CORES * tile_m * ic * 2
    out_bytes = N_CORES * tile_m * oc * 2
    max_xrt_in = _XRT_BUF_MAX // in_bytes if in_bytes > 0 else 999
    max_xrt_out = _XRT_BUF_MAX // out_bytes if out_bytes > 0 else 999
    col_in = 4 * tile_m * ic * 2
    col_out = 4 * tile_m * oc * 2
    wt = (k_block * oc + 2 * oc) * 2
    per_ppc = col_in + col_out
    max_l2 = (_L2_BUDGET - wt) // per_ppc if per_ppc > 0 else 999
    return max(1, min(ideal, max_xrt_in, max_xrt_out, max_l2, 32))
