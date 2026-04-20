"""Repeatable performance profiler for the MDV6 NPU pipeline.

Categorizes per-frame time into:
  - npu_run        : time inside DefaultNPURuntime.run (NPU active)
  - iron_alloc     : iron.tensor + iron.zeros (XRT buffer creation)
  - pack           : weight repacking (_pack_3x3_weights, _repack_*)
  - fuse           : fuse_bn variants
  - numpy          : np.concatenate / np.zeros from host-side assembly
  - cpu_<Layer>    : CPU-resident model layers (RepConv, Detection, etc.)
  - launch_gap     : inter-launch gap not attributable to the categories above
                     (per-launch Python/pyxrt plumbing — the real overhead floor)

Use as:
    from profile_harness import Profiler
    with Profiler(n_warmup=1, n_measure=2) as prof:
        for i in range(prof.n_total_frames):
            run_one_forward_pass()
            prof.next_frame()
    prof.report()

Or via the CLI of test_full_model_mc.py:
    python test_full_model_mc.py --profile 3       # 1 cold + 2 warm
    python test_full_model_mc.py --profile 3 --baseline baseline.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# Categories shown in the report, in display order.
CATEGORIES = [
    "npu_run",
    "iron_alloc",
    "pack",
    "fuse",
    "numpy",
    "cpu_layers",
    "launch_gap",
    "pre_post",
]


@dataclass
class FrameStats:
    wall_ms: float = 0.0
    bucket_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    bucket_n: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sub_bucket_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    sub_bucket_n: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_launches: int = 0
    npu_ms: float = 0.0
    inter_launch_gaps_ms: float = 0.0


class Profiler:
    """Context-managed profiler that patches the runtime + helpers on enter,
    restores on exit. Stats accumulate per frame; call next_frame() between
    forward passes."""

    def __init__(self, n_warmup: int = 1, n_measure: int = 2,
                 cpu_layers: bool = True, sub_buckets: bool = True):
        self.n_warmup = n_warmup
        self.n_measure = n_measure
        self.n_total_frames = n_warmup + n_measure
        self.cpu_layers = cpu_layers
        self.sub_buckets = sub_buckets
        self.frames: list[FrameStats] = []
        self._cur = FrameStats()
        self._frame_start = 0.0
        self._last_launch_end: float | None = None
        self._patched = []  # list of (obj, attr, original) for restore

    # ---- patching helpers ----

    def _patch(self, obj, attr, replacement):
        self._patched.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, replacement)

    def _restore(self):
        for obj, attr, orig in reversed(self._patched):
            setattr(obj, attr, orig)
        self._patched.clear()

    def _bump(self, bucket: str, t_s: float, sub: str | None = None):
        ms = t_s * 1000
        self._cur.bucket_ms[bucket] += ms
        self._cur.bucket_n[bucket] += 1
        if sub and self.sub_buckets:
            self._cur.sub_bucket_ms[sub] += ms
            self._cur.sub_bucket_n[sub] += 1

    # ---- lifecycle ----

    def __enter__(self):
        self._install_hooks()
        self._frame_open = False  # opened lazily by next_frame() / first launch
        self._frame_start = time.perf_counter()
        self._last_launch_end = None
        self._frame_open = True
        return self

    def __exit__(self, *exc):
        # Close the frame opened by __enter__ (or by the last next_frame).
        if self._frame_open:
            self._finalize_frame()
            self._frame_open = False
        self._restore()
        return False

    def next_frame(self):
        """Call after each forward pass to close one frame and start the next."""
        if self._frame_open:
            self._finalize_frame()
        self._cur = FrameStats()
        self._frame_start = time.perf_counter()
        self._last_launch_end = None
        self._frame_open = True

    def _finalize_frame(self):
        wall = (time.perf_counter() - self._frame_start) * 1000
        self._cur.wall_ms = wall
        # pre/post = wall - npu - inter_launch_gaps - host_inside_gaps
        # The "tracked" buckets all happen during gaps; we compute pre_post as
        # the residual after subtracting npu + inter-launch-gaps.
        self._cur.npu_ms = self._cur.bucket_ms.get("npu_run", 0.0)
        self.frames.append(self._cur)

    # ---- hooks ----

    def _install_hooks(self):
        from aie.utils import DefaultNPURuntime
        import aie.iron as iron
        import importlib.util
        import numpy as np

        prof = self
        _base = os.path.dirname(os.path.abspath(__file__))

        # Wrap DefaultNPURuntime.run — also tracks inter-launch gaps.
        _orig_run = DefaultNPURuntime.run

        def _wrap_run(kh, args, *a, **kw):
            now = time.perf_counter()
            if prof._last_launch_end is not None:
                gap = now - prof._last_launch_end
                prof._cur.inter_launch_gaps_ms += gap * 1000
            t0 = time.perf_counter()
            r = _orig_run(kh, args, *a, **kw)
            t1 = time.perf_counter()
            prof._bump("npu_run", t1 - t0)
            prof._cur.n_launches += 1
            prof._last_launch_end = t1
            return r

        self._patch(DefaultNPURuntime, "run", _wrap_run)

        # iron.tensor / iron.zeros (buffer alloc).
        for name in ("tensor", "zeros"):
            orig = getattr(iron, name)

            def make(o, n):
                def w(*a, **kw):
                    t0 = time.perf_counter()
                    r = o(*a, **kw)
                    prof._bump("iron_alloc", time.perf_counter() - t0, sub=f"iron.{n}")
                    return r
                return w

            self._patch(iron, name, make(orig, name))

        # Pack/repack functions in run_tiled_mc.py.
        mcr_spec = importlib.util.spec_from_file_location(
            "mcr_prof", os.path.join(_base, "run_tiled_mc.py"))
        mcr = importlib.util.module_from_spec(mcr_spec)
        mcr_spec.loader.exec_module(mcr)
        self._mcr = mcr  # keep reference

        for name in ("_pack_3x3_weights", "_repack_weights_for_gemm",
                     "_repack_weights_kblocked"):
            if hasattr(mcr, name):
                orig = getattr(mcr, name)

                def make(o, n):
                    def w(*a, **kw):
                        t0 = time.perf_counter()
                        r = o(*a, **kw)
                        prof._bump("pack", time.perf_counter() - t0, sub=n)
                        return r
                    return w

                self._patch(mcr, name, make(orig, name))

        # fuse_bn variants in elan/test_tiled.py.
        ett_spec = importlib.util.spec_from_file_location(
            "ett_prof", os.path.join(_base, "elan", "test_tiled.py"))
        ett = importlib.util.module_from_spec(ett_spec)
        ett_spec.loader.exec_module(ett)
        self._ett = ett

        for name in ("fuse_bn", "fuse_bn_transposed", "fuse_bn_transposed_3x3"):
            if hasattr(ett, name):
                orig = getattr(ett, name)

                def make(o, n):
                    def w(*a, **kw):
                        t0 = time.perf_counter()
                        r = o(*a, **kw)
                        prof._bump("fuse", time.perf_counter() - t0, sub=n)
                        return r
                    return w

                self._patch(ett, name, make(orig, name))

        # Make test_full_model_mc.py see the wrapped versions.
        try:
            import test_full_model_mc as tfmm
            tfmm.run_tiled_mc = mcr.run_tiled_fused_conv_mc
            tfmm.run_gemm_conv1x1 = mcr.run_gemm_conv1x1_mc
            tfmm.fuse_bn = ett.fuse_bn
            tfmm.fuse_bn_transposed = ett.fuse_bn_transposed
            tfmm.fuse_bn_transposed_3x3 = ett.fuse_bn_transposed_3x3
        except ImportError:
            pass

        # numpy: np.concatenate is the dominant numpy cost (per repack).
        _orig_concat = np.concatenate

        def _wcat(*a, **kw):
            t0 = time.perf_counter()
            r = _orig_concat(*a, **kw)
            prof._bump("numpy", time.perf_counter() - t0, sub="np.concatenate")
            return r

        self._patch(np, "concatenate", _wcat)

        # CPU layers (optional).
        if self.cpu_layers:
            try:
                sys.path.insert(0, os.path.join(_base, "../../../python"))
                from mdv6.layers import (
                    Bottleneck, RepNCSP, RepConv, AConv, Detection,
                )
                import torch.nn as nn

                def hook_cls(cls, name):
                    orig = cls.__call__

                    def w(self_, *a, **kw):
                        t0 = time.perf_counter()
                        r = orig(self_, *a, **kw)
                        prof._bump("cpu_layers",
                                   time.perf_counter() - t0,
                                   sub=f"cpu.{name}")
                        return r

                    self._patch(cls, "__call__", w)

                hook_cls(RepConv, "RepConv")
                hook_cls(Detection, "Detection")
                hook_cls(nn.Upsample, "Upsample")
                hook_cls(nn.AvgPool2d, "AvgPool2d")
                # NB: we do NOT hook Bottleneck/RepNCSP/AConv to avoid
                # double-counting their nested children.
            except ImportError:
                pass

    # ---- reporting ----

    def warm_frames(self) -> list[FrameStats]:
        return self.frames[self.n_warmup:]

    def cold_frame(self) -> FrameStats | None:
        return self.frames[0] if self.frames else None

    def warm_avg(self) -> dict[str, float]:
        warm = self.warm_frames()
        if not warm:
            return {}
        n = len(warm)
        out = {"wall_ms": sum(f.wall_ms for f in warm) / n,
               "n_launches": sum(f.n_launches for f in warm) / n,
               "inter_launch_gaps_ms": sum(f.inter_launch_gaps_ms for f in warm) / n}
        cats: dict[str, float] = defaultdict(float)
        for f in warm:
            for k, v in f.bucket_ms.items():
                cats[k] += v
        for k, v in cats.items():
            out[k] = v / n
        return out

    def report(self, baseline_path: str | None = None,
               out_json: str | None = None) -> int:
        """Print a structured report. Returns 0 if all checks pass else 1."""
        if not self.frames:
            print("Profiler: no frames recorded.")
            return 1

        print()
        print("=" * 78)
        print(f"PROFILE — {len(self.frames)} frames "
              f"({self.n_warmup} warmup, {len(self.warm_frames())} measured)")
        print("=" * 78)

        # Per-frame summary.
        print(f"{'Frame':>5}  {'Wall(ms)':>9}  {'NPU(ms)':>8}  "
              f"{'Host(ms)':>9}  {'Launches':>9}  Notes")
        for i, f in enumerate(self.frames):
            tag = "cold" if i < self.n_warmup else "warm"
            print(f"{i:>5}  {f.wall_ms:>9.0f}  {f.npu_ms:>8.0f}  "
                  f"{f.wall_ms - f.npu_ms:>9.0f}  {f.n_launches:>9}  ({tag})")

        # Warm-frame category breakdown.
        warm = self.warm_avg()
        if not warm:
            return 1

        wall = warm["wall_ms"]
        npu = warm.get("npu_run", 0)
        gaps = warm["inter_launch_gaps_ms"]
        n_launches = int(warm["n_launches"])

        # Compute "tracked-in-gap" so we can derive the launch_gap residual.
        tracked_in_gap = sum(
            warm.get(c, 0.0) for c in
            ("iron_alloc", "pack", "fuse", "numpy", "cpu_layers")
        )
        # launch_gap = inter_launch_gaps - tracked_in_gap
        launch_gap = max(0.0, gaps - tracked_in_gap)
        # pre_post = wall - npu - inter_launch_gaps
        pre_post = max(0.0, wall - npu - gaps)

        print()
        print(f"WARM AVERAGE — wall {wall:.0f} ms ({n_launches} launches, "
              f"{1000/wall:.2f} fps)")
        print("-" * 78)
        rows = [
            ("npu_run", npu, "NPU active (in DefaultNPURuntime.run)"),
            ("iron_alloc", warm.get("iron_alloc", 0),
             "XRT buffer creation (iron.tensor/zeros)"),
            ("pack", warm.get("pack", 0),
             "Weight repacking (mostly cache hits after warmup)"),
            ("fuse", warm.get("fuse", 0), "fuse_bn cache lookups"),
            ("numpy", warm.get("numpy", 0), "np.concatenate (host weight assembly)"),
            ("cpu_layers", warm.get("cpu_layers", 0),
             "CPU-resident layers (RepConv, Detection, AvgPool, Upsample)"),
            ("launch_gap", launch_gap,
             f"Per-launch Python/pyxrt plumbing ({launch_gap*1000/n_launches:.0f} µs/call)"
             if n_launches else "Per-launch plumbing"),
            ("pre_post", pre_post,
             "Pre-first-launch + post-last-launch (model setup, last layer)"),
        ]
        for cat, ms, desc in rows:
            pct = 100 * ms / wall if wall else 0
            print(f"  {cat:<14}  {ms:>7.1f} ms  {pct:>5.1f}%   {desc}")
        accounted = sum(r[1] for r in rows)
        print(f"  {'-'*78}")
        print(f"  {'TOTAL':<14}  {accounted:>7.1f} ms  "
              f"{100*accounted/wall:>5.1f}%   "
              f"(unaccounted: {wall - accounted:+.1f} ms)")

        # Sub-bucket detail.
        if self.sub_buckets:
            print()
            print("SUB-BUCKETS (warm-frame averages):")
            warm_subs: dict[str, list[float]] = defaultdict(list)
            warm_n: dict[str, list[int]] = defaultdict(list)
            for f in self.warm_frames():
                for k, v in f.sub_bucket_ms.items():
                    warm_subs[k].append(v)
                for k, v in f.sub_bucket_n.items():
                    warm_n[k].append(v)
            for k in sorted(warm_subs.keys(), key=lambda x: -sum(warm_subs[x])):
                avg_ms = sum(warm_subs[k]) / len(warm_subs[k])
                avg_n = sum(warm_n[k]) / len(warm_n[k])
                if avg_ms < 0.05:
                    continue
                print(f"  {k:<32}  {avg_ms:>7.2f} ms  ({avg_n:>5.0f} calls/frame, "
                      f"{1000*avg_ms/avg_n:>4.1f} µs/call)")

        # Optional baseline comparison.
        rc = 0
        if baseline_path and os.path.exists(baseline_path):
            with open(baseline_path) as fh:
                baseline = json.load(fh)
            print()
            print("BASELINE COMPARISON (regression > 10% on any category):")
            for cat, ms, _ in rows:
                base_ms = baseline.get(cat, 0)
                if base_ms < 5:
                    continue  # ignore noise on tiny categories
                delta = (ms - base_ms) / base_ms * 100
                marker = ""
                if delta > 10:
                    marker = "  ← REGRESSION"
                    rc = 1
                elif delta < -10:
                    marker = "  ← improvement"
                print(f"  {cat:<14}  base {base_ms:>6.1f} ms  "
                      f"now {ms:>6.1f} ms  ({delta:+5.1f}%){marker}")

        # Optional JSON dump for future baselines.
        if out_json:
            payload = {cat: ms for cat, ms, _ in rows}
            payload["wall_ms"] = wall
            payload["n_launches"] = n_launches
            with open(out_json, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"\nWrote {out_json}")

        return rc
