# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""C3.0 — per-call decode breakdown for the c2_merged pipeline.

Wraps KernelCache.load_and_run and decode_attention_cpu with timers (no
production changes) and runs the standard instruct prompt. Reports, per decode
token: rms_gemv_rope (call 1), CPU attention, o_gemv_ffn (call 2), lm_head,
and the unattributed host remainder.

Run from build_peano:
    python3 ../tools/profile_c30.py --n-tokens 9
"""

import argparse
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

EVENTS = []  # (name, seconds), chronological


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=9)
    ap.add_argument("--quant", default="bf16", choices=["bf16", "awq"])
    ap.add_argument("--awq-weights", default=None)
    args_cli = ap.parse_args()

    import llama32_1b_inference as inf
    import llama32_1b_decode as dec
    from kernel_builder.cache import KernelCache

    orig_run = KernelCache.load_and_run

    def timed_run(self, name, *a, **kw):
        t0 = time.perf_counter()
        r = orig_run(self, name, *a, **kw)
        EVENTS.append((name, time.perf_counter() - t0))
        return r

    KernelCache.load_and_run = timed_run

    orig_attn = dec.decode_attention_cpu

    def timed_attn(*a, **kw):
        t0 = time.perf_counter()
        r = orig_attn(*a, **kw)
        EVENTS.append(("cpu_attn", time.perf_counter() - t0))
        return r

    dec.decode_attention_cpu = timed_attn

    args = SimpleNamespace(
        run_only=True, compile_only=False, verbose=False, trace=None,
        synthetic_weights=False, quant=args_cli.quant,
        awq_weights=args_cli.awq_weights,
        hf_model_id="unsloth/Llama-3.2-1B-Instruct", model="instruct",
        n_tokens=args_cli.n_tokens, awq_decode_experimental=False,
        cpu_attn=True,
    )
    session = inf.build_session(args)

    EVENTS.clear()
    t0 = time.perf_counter()
    generated, _ = inf.run_once(
        session, "What is the capital of France?", n_tokens=args_cli.n_tokens)
    wall = time.perf_counter() - t0
    print(f"\ntokens: {generated}")

    # Split decode tokens on lm_head occurrences; first lm_head is prefill's.
    lm = [i for i, (n, _) in enumerate(EVENTS) if n.startswith("lm_head")]
    tok_slices = [EVENTS[lm[i] + 1:lm[i + 1] + 1] for i in range(len(lm) - 1)]
    n = len(tok_slices)
    if n == 0:
        print("no decode tokens recorded"); return

    agg, tok_totals = {}, []
    for sl in tok_slices:
        tot = 0.0
        for name, dt in sl:
            key = ("o_gemv_ffn" if name.startswith("o_gemv_ffn")
                   else "rms_gemv_rope" if name.startswith("rms_gemv_rope")
                   else "lm_head" if name.startswith("lm_head") else name)
            agg[key] = agg.get(key, 0.0) + dt
            tot += dt
        tok_totals.append(tot)

    tok_ms = sum(tok_totals) / n * 1e3
    print(f"\n=== C3.0 per-token breakdown ({n} decode tokens) ===")
    for k in sorted(agg, key=agg.get, reverse=True):
        ms = agg[k] / n * 1e3
        print(f"  {k:16s} {ms:7.2f} ms  ({ms / tok_ms * 100:4.1f}% of attributed)")
    print(f"  {'attributed':16s} {tok_ms:7.2f} ms")
    print(f"  decode wall/token ≈ {wall / n * 1e3:7.2f} ms (incl prefill amortized — ignore)")


if __name__ == "__main__":
    main()
