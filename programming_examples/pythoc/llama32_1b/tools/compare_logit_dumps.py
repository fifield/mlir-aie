#!/usr/bin/env python3
"""Correlate two per-token logit dumps (PYTHOC_LOGIT_DUMP .npz files).

Intended use: compare AWQ c2_attn (on-NPU BFP576 attention) against AWQ
c2_merged (CPU fp32 attention) on a TEACHER-FORCED token stream, so every
position has identical context and the logit vectors are comparable. Run the
reference (c2_merged) first with PYTHOC_LOGIT_DUMP, then the candidate
(c2_attn) with PYTHOC_FORCE_TOKENS=<ref.npz> PYTHOC_LOGIT_DUMP=<cand.npz>.

Per token it reports cosine similarity, Pearson r, top-5 overlap, argmax
agreement, and the symmetric KL of the softmax distributions -- the
distributional checks that L-infinity error and argmax-only comparison miss.

Usage: python tools/compare_logit_dumps.py REF.npz CAND.npz
"""
import sys
import numpy as np


def _softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def main():
    ref_p, cand_p = sys.argv[1], sys.argv[2]
    ref = np.load(ref_p)
    cand = np.load(cand_p)
    a = ref["logits"].astype(np.float64)
    b = cand["logits"].astype(np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"comparing {n} tokens  ref={ref_p}  cand={cand_p}\n")
    print(f"{'tok':>3} {'cos':>9} {'pearson':>9} {'top5':>5} "
          f"{'argmax':>7} {'symKL':>9}")
    cos_all, r_all, kl_all, top5_all, argmax_hits = [], [], [], [], 0
    for i in range(n):
        x, y = a[i], b[i]
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        r = float(np.corrcoef(x, y)[0, 1])
        tx = set(np.argsort(x)[-5:])
        ty = set(np.argsort(y)[-5:])
        top5 = len(tx & ty)
        ax, ay = int(x.argmax()), int(y.argmax())
        agree = ax == ay
        argmax_hits += int(agree)
        px, py = _softmax(x), _softmax(y)
        kl = float(np.sum(px * np.log((px + 1e-12) / (py + 1e-12)))
                   + np.sum(py * np.log((py + 1e-12) / (px + 1e-12))))
        cos_all.append(cos); r_all.append(r); kl_all.append(kl)
        top5_all.append(top5)
        print(f"{i:>3} {cos:>9.6f} {r:>9.6f} {top5:>4}/5 "
              f"{('=' if agree else f'{ax}!={ay}'):>7} {kl:>9.3e}")
    print(f"\nsummary over {n} tokens:")
    print(f"  cosine     min={min(cos_all):.6f}  mean={np.mean(cos_all):.6f}")
    print(f"  pearson    min={min(r_all):.6f}  mean={np.mean(r_all):.6f}")
    print(f"  top5 overlap mean={np.mean(top5_all):.2f}/5")
    print(f"  argmax agreement {argmax_hits}/{n} "
          f"({100*argmax_hits/n:.1f}%)")
    print(f"  symKL      max={max(kl_all):.3e}  mean={np.mean(kl_all):.3e}")


if __name__ == "__main__":
    main()
