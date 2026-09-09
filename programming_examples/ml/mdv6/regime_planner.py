#!/usr/bin/env python3
"""Offline, sequential MDV6 envelope cost report; no torch, XRT or device imports.

Run ``python3 regime_planner.py --output-dir /tmp/mdv6-planner``. Calibration
uses only profile_baseline.json; historical regressions are validation data.
This is a screening model, not a compiler fit check or a latency predictor
validated outside the baseline. See report['limitations'] before using scores.
"""
import argparse
import ast
import csv
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path

from regime_config import regime_artifacts

ROOT = Path(__file__).resolve().parent


def literal_table(path, name):
    """Read a static assignment without executing the runtime/build script."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing static assignment {name} in {path}")


def gemm_helpers():
    # Builder imports only the standard library and regime_config; unlike the
    # runtime module this needs neither the installed AIE Python nor torch.
    spec = importlib.util.spec_from_file_location(
        "planner_gemm_build", ROOT / "gemm_conv1x1/build_gemm_conv1x1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Layer:
    name: str
    runtime_name: str
    h: int
    w: int
    ic: int
    oc: int
    kind: str
    tile_h: int = 0
    tile_w: int = 1
    oc_block: int = 0
    stride: int = 1
    count: int = 1


@dataclass(frozen=True)
class Envelope:
    name: str
    tile_h: int
    tile_w: int
    ic: int
    oc: int
    ppc: int
    k_block: int = 0
    active_h: int = 0
    active_w: int = 0
    active_oc: int = 0
    input_depth: int = 1


def model_layers():
    """Expand static forward dispatch; RE repeat depth is three in this model.

    Read call arguments so graph aliases (e.g. re21 -> mc_re6_c4) are retained.
    Stem IC uses the build contract, which describes the deployed artifacts.
    """
    configs = {c[0]: c for c in literal_table(ROOT / "conv/build_multicore.py", "CONFIGS")}
    tree = ast.parse((ROOT / "test_full_model_mc.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    layers = []

    def add(name, runtime, h, w, ic, oc, kind, th=0, tw=1, ob=0, stride=1, count=1):
        layers.append(Layer(name, runtime.replace("mc_", "gemm_") if kind == "gemm" else runtime,
                            h, w, ic, oc, kind, th, tw, ob, stride, count))

    # Only direct assignments in the forward body, not diagnostic branches.
    for statement in main.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Name):
            continue
        fn = call.func.id
        label = statement.targets[0].id if isinstance(statement.targets[0], ast.Name) else "unknown"
        a = call.args
        val = lambda i: ast.literal_eval(a[i])
        if fn == "run_re_mc":
            h, w, ic, oc, part, proc = [val(i) for i in range(2, 8)]
            c1, c3, c4, rn1, rn3, rnm = [val(i) for i in range(8, 20, 2)]
            tc1, oc1, tc3, oc3, tc4, oc4, trn1, orn1, trn3, orn3, trnm, ornm = [val(i) for i in range(20, 32)]
            add(label + ".c1", c1, h, w, ic, part, "gemm")
            add(label + ".rn1", rn1, h, w, proc, proc // 2, "gemm", count=4)
            add(label + ".rn3", rn3, h, w, proc // 2, proc // 2, "conv", trn3, trn3, orn3, count=6)
            add(label + ".rnm", rnm, h, w, proc, proc, "gemm", count=2)
            add(label + ".c3", c3, h, w, proc, proc, "conv", tc3, tc3, oc3, count=2)
            add(label + ".c4", c4, h, w, part + 2 * proc, oc, "gemm")
        elif fn == "run_elan_mc":
            h, w, ic, oc = [val(i) for i in range(2, 6)]
            part, proc = val(18), val(19)
            add(label + ".c1", val(6), h, w, ic, part, "gemm")
            add(label + ".c3", val(8), h, w, proc, proc, "conv", val(14), val(14), val(15), count=2)
            add(label + ".c4", val(10), h, w, 4 * proc, oc, "gemm")
        elif fn == "run_aconv_mc":
            runtime = val(0)
            add(label, runtime, val(4), val(5), configs[runtime][4], val(6), "conv", val(7), val(7), val(8), 2)
        elif fn == "rt":
            runtime, h, w, oc = val(0), val(4), val(5), val(6)
            ks, stride = val(11), val(10)
            # Direct rt calls are conv0, conv1, and the two SPP convolutions.
            ic = configs[runtime][4]
            add(label, runtime, h, w, ic, oc, "conv" if ks == 3 else "gemm", val(7), val(8), val(9), stride)
    if len(layers) != 54:
        raise ValueError(f"Forward graph changed: expected 54 layer groups, got {len(layers)}; review extraction")
    return layers


def candidates(layer, helpers, route="legacy"):
    conv_artifacts, gemm_artifacts = regime_artifacts(route)
    ppcs = literal_table(ROOT / "run_tiled_mc.py", "_MC_PPC")
    if layer.kind == "conv":
        base = Envelope(layer.runtime_name, layer.tile_h, layer.tile_w, layer.ic,
                        layer.oc_block, ppcs.get(layer.runtime_name, 1))
        regimes = []
        for artifact in conv_artifacts:
            if layer.runtime_name not in artifact.members:
                continue
            ah, aw, ic, oc, stride, padding, ppc = artifact.members[layer.runtime_name]
            if ic != layer.ic or stride != layer.stride:
                continue
            regimes.append(Envelope(artifact.xclbin_name, artifact.tile_h, artifact.tile_w,
                                    artifact.ic, artifact.oc_block, artifact.patches_per_core,
                                    active_h=ah, active_w=aw, active_oc=oc,
                                    input_depth=artifact.input_depth))
    else:
        kb, tm = helpers.choose_k_block(layer.ic, layer.oc, layer.h * layer.w)
        ppc = (helpers.compute_ppc_kblocked(layer.h * layer.w, tm, layer.ic, layer.oc, kb)
               if kb else helpers.compute_ppc(layer.h * layer.w, tm, layer.ic, layer.oc))
        base = Envelope(f"gemm_t{tm}_ic{layer.ic}_oc{layer.oc}_kb{kb}_p{ppc}", tm, 1, layer.ic, layer.oc, ppc, kb)
        regimes = []
        for artifact in gemm_artifacts:
            for member in artifact.members:
                if (member.runtime_name, member.ic, member.oc, member.k_block) != (layer.runtime_name, layer.ic, layer.oc, kb):
                    continue
                regimes.append(Envelope(artifact.xclbin_name, artifact.tile_m, 1,
                                        artifact.ic, artifact.oc, artifact.patches_per_core,
                                        artifact.k_block,
                                        active_h=member.tile_m if not kb else artifact.tile_m,
                                        active_oc=member.oc if not kb else artifact.oc))
    return [base] + regimes


def measure(layer, env, cores=32):
    if min(layer.h, layer.w, layer.ic, layer.oc, layer.count, env.tile_h,
           env.tile_w, env.ic, env.oc, env.ppc, cores) <= 0:
        raise ValueError("Layer dimensions, envelope dimensions, PPC and cores must be positive")
    ah, aw, aoc = env.active_h or env.tile_h, env.active_w or env.tile_w, env.active_oc or env.oc
    ks = 3 if layer.kind == "conv" else 1
    tiles = (math.ceil(layer.h / ah) * math.ceil(layer.w / aw) if ks == 3
             else math.ceil(layer.h * layer.w / ah))
    oc_groups = math.ceil(layer.oc / aoc)
    batches = math.ceil(tiles / (cores * env.ppc))
    calls = batches * oc_groups * layer.count
    slots = batches * cores * env.ppc * oc_groups * layer.count
    # Conv and non-K GEMM kernels use logical RTP sizes, while K-blocked
    # regime kernels execute the full padded channel envelope.
    active_ic = env.ic if env.k_block else layer.ic
    useful_ic = 3 if layer.runtime_name == "mc_ftconv0" else layer.ic
    useful = layer.h * layer.w * useful_ic * layer.oc * ks * ks * layer.count
    padded = slots * ah * aw * active_ic * aoc * ks * ks
    inp = ((env.tile_h - 1) * layer.stride + ks) * ((env.tile_w - 1) * layer.stride + ks) * env.ic * 2
    out = env.tile_h * env.tile_w * env.oc * 2
    weight_chunk = ((env.k_block or env.ic) * env.oc * ks * ks + 2 * env.oc) * 2
    chunks = env.ic // env.k_block if env.k_block else 1
    stack = 4096 if ks == 3 else 8192
    l1 = inp * env.input_depth + out + weight_chunk + stack + 32
    # Conv super-FIFOs hold every PPC slot at once. GEMM super-FIFOs
    # stream one patch/core per cycle, irrespective of PPC (see generators).
    l2_slots = env.ppc if ks == 3 else 1
    l2 = min(cores, 4) * l2_slots * (inp + out) + weight_chunk
    # Weight buffers are broadcast to each column on each invocation.
    input_bytes, output_bytes = calls * cores * env.ppc * inp, calls * cores * env.ppc * out
    # K-blocked GEMM's weight TAP repeats the complete IC stream for every
    # PPC slot; non-K GEMM and conv retain one weight object across slots.
    weight_replays = env.ppc if env.k_block else 1
    weight_bytes = calls * math.ceil(cores / 4) * weight_chunk * chunks * weight_replays
    reasons = []
    if env.ic < layer.ic or aoc > env.oc or ah > env.tile_h or aw > env.tile_w:
        reasons.append("logical kernel exceeds envelope capacity")
    if useful > padded:
        reasons.append("executed work cannot cover logical MACs")
    if l1 > 65536:
        reasons.append("L1 exceeds 64 KiB estimate")
    if l2 > 400 * 1024:
        reasons.append("L2 exceeds 400 KiB per-column estimate")
    if max(cores * env.ppc * inp, cores * env.ppc * out, weight_chunk * chunks) > 16 * 1024 * 1024:
        reasons.append("XRT argument exceeds 16 MiB")
    if active_ic % 8 or aoc % 8 or (ks == 1 and ah % 4):
        reasons.append("kernel vector alignment violated")
    if env.k_block and (env.ic % env.k_block or env.k_block % 8 or chunks > 16):
        reasons.append("K-block alignment or instruction limit violated")
    occupancy = tiles / (batches * cores * env.ppc)
    return dict(layer_name=layer.name, runtime_name=layer.runtime_name, occurrences=layer.count,
                op_kind=layer.kind, spatial_shape=[layer.h, layer.w], logical_ic=useful_ic,
                logical_oc=layer.oc, candidate_regime=env.name, candidate_core_count=cores,
                candidate_tile=[env.tile_h, env.tile_w], active_tile=[ah, aw],
                candidate_ic_envelope=env.ic, candidate_oc_envelope=env.oc,
                candidate_k_block=env.k_block, candidate_ppc=env.ppc,
                useful_macs=useful, padded_macs=padded, compute_efficiency=useful / padded,
                input_bytes=input_bytes, output_bytes=output_bytes, weight_bytes=weight_bytes,
                transfer_bytes=input_bytes + output_bytes + weight_bytes,
                active_tiles=tiles * oc_groups * layer.count, padded_tiles=slots,
                active_cores_peak=min(cores, math.ceil(tiles / env.ppc)), core_occupancy=occupancy,
                calls_today=calls, calls_after_fusion_lower_bound=layer.count,
                l1_bytes_est=l1, l2_bytes_per_column_est=l2, fits_l1=l1 <= 65536,
                fits_l2=l2 <= 400 * 1024, feasible_estimate=not reasons,
                reject_reasons=reasons)


def best_existing_candidate(layer, envs, score):
    evaluated = [(env, measure(layer, env)) for env in envs]
    feasible = [(env, row) for env, row in evaluated if row["feasible_estimate"]]
    if not feasible:
        reasons = {env.name: row["reject_reasons"] for env, row in evaluated}
        raise ValueError(f"No feasible candidate for layer {layer.name}: {reasons}")
    return min(feasible, key=lambda item: score(item[1]))[0]


def build_report(baseline_path=ROOT / "profile_baseline.json", core_counts=(32,)):
    baseline = json.loads(Path(baseline_path).read_text())
    helpers = gemm_helpers()
    layers = model_layers()
    options = {layer.name: candidates(layer, helpers) for layer in layers}
    base_rows = [measure(layer, options[layer.name][0]) for layer in layers]
    base_calls = sum(row["calls_today"] for row in base_rows)
    if base_calls != baseline["n_launches"]:
        raise ValueError(f"Graph/build choices predict {base_calls} calls, baseline has {baseline['n_launches']}; recalibrate explicitly")
    # One baseline cannot identify compute/bandwidth/submit costs separately.
    # Use npu_run as effective executed-work cost; launch_gap and numpy are
    # separately measured proxies. Do not describe this as pure device compute.
    mac_ms = baseline["npu_run"] / sum(r["padded_macs"] for r in base_rows)
    launch_ms = baseline["launch_gap"] / base_calls
    byte_ms = baseline["numpy"] / sum(r["transfer_bytes"] for r in base_rows)
    fixed_ms = baseline["wall_ms"] - baseline["npu_run"] - baseline["launch_gap"] - baseline["numpy"]

    def score(row):
        return row["padded_macs"] * mac_ms * 32 / row["candidate_core_count"] + row["calls_today"] * launch_ms + row["transfer_bytes"] * byte_ms

    rows = []
    for layer, base in zip(layers, base_rows):
        for cores in core_counts:
            for env in options[layer.name]:
                row = measure(layer, env, cores)
                row["estimated_variable_ms"] = score(row)
                row["calls_ratio_to_standalone"] = row["calls_today"] / base["calls_today"]
                row["work_ratio_to_standalone"] = row["padded_macs"] / base["padded_macs"]
                row["transfer_ratio_to_standalone"] = row["transfer_bytes"] / base["transfer_bytes"]
                row["performance_flags"] = [f"{label} grows >25%" for label, key in
                                            [("launch count", "calls_ratio_to_standalone"), ("executed MACs", "work_ratio_to_standalone"), ("transfer volume", "transfer_ratio_to_standalone")]
                                            if row[key] > 1.25]
                rows.append(row)

    plans = []
    per_regime_launch_deltas = []
    for plan in ("standalone", "per_regime_r1_r3", "current_non_k", "plus_r5", "plus_shared_k", "all_shared", "best_existing_per_layer"):
        selected = []
        for layer in layers:
            envs = options[layer.name]
            eligible = [e for e in envs[1:] if "shared" not in e.name and "r5" not in e.name]
            selected_env = eligible[0] if eligible and plan != "standalone" else envs[0]
            if plan == "per_regime_r1_r3":
                historical = candidates(layer, helpers, "per-regime-r1-r3")
                selected_env = historical[1] if len(historical) > 1 else historical[0]
                before, after = measure(layer, envs[0]), measure(layer, selected_env)
                delta = after["calls_today"] - before["calls_today"]
                if delta:
                    per_regime_launch_deltas.append(dict(layer_name=layer.name,
                        occurrences=layer.count, standalone_calls=before["calls_today"],
                        regime_calls=after["calls_today"], delta=delta,
                        standalone_tile=before["active_tile"], regime_tile=after["active_tile"],
                        standalone_ppc=before["candidate_ppc"], regime_ppc=after["candidate_ppc"]))
            if plan in ("plus_r5", "plus_shared_k", "all_shared"):
                selected_env = next((e for e in envs if "r5" in e.name), selected_env)
            if plan in ("plus_shared_k", "all_shared"):
                selected_env = next((e for e in envs if "shared_gemm" in e.name), selected_env)
            if plan == "all_shared":
                selected_env = next((e for e in envs if "shared_conv" in e.name), selected_env)
            if plan == "best_existing_per_layer":
                selected_env = best_existing_candidate(layer, envs, score)
            selected.append(measure(layer, selected_env))
        plans.append(dict(name=plan, resident_xclbins_est=len({r["candidate_regime"] for r in selected}),
                          total_host_launches=sum(r["calls_today"] for r in selected),
                          total_padded_macs=sum(r["padded_macs"] for r in selected),
                          total_transfer_bytes=sum(r["transfer_bytes"] for r in selected),
                          estimated_wall_ms=fixed_ms + sum(score(r) for r in selected),
                          selections={r["layer_name"]: r["candidate_regime"] for r in selected},
                          rejected_layers=[r["layer_name"] for r in selected if not r["feasible_estimate"]]))
    return dict(schema_version=1, per_regime_launch_check=dict(
                standalone_calls=base_calls,
                predicted_calls=base_calls + sum(r["delta"] for r in per_regime_launch_deltas),
                explanation="R1 non-K 44-row retile adds 15 calls across ELAN and 80x80 RE blocks; R1 K-blocked 52-row/PPC2 removes two. Net +13 = 466, unlike the historical 453 summary. This is contract arithmetic, not a hardware measurement.",
                sources=["test_full_model_mc.py: forward dispatch and grouped occurrences",
                         "regime_config.py: regime_artifacts('per-regime-r1-r3')",
                         "gemm_conv1x1/build_gemm_conv1x1.py: standalone tile/PPC selection"],
                layer_deltas=per_regime_launch_deltas),
                calibration=dict(source=str(baseline_path), launch_ms=launch_ms,
                effective_ms_per_mac=mac_ms, host_assembly_ms_per_byte=byte_ms, fixed_ms=fixed_ms),
                limitations=[
                    "Sequential grouped forward inventory, repeat_num=3; no placement, concurrency, residency eviction or instruction fusion implementation.",
                    "Stem IC comes from deployed build contracts; source model definitions can differ from installed model. Baseline launch-count check detects only scheduling drift.",
                    "L1/L2 are sizing screens, not bank allocation/compiler proof; L1 includes stack and a 32-byte RTP reserve, but barrier buffers, allocator alignment and bank fragmentation are not explicitly accounted. L2 uses the conservative 400 KiB builder budget.",
                    "GEMM L2 holds one patch/core per cycle; conv L2 holds PPC patches/core. Standalone PPC still follows the builder's older conservative PPC-dependent L2 heuristic to reproduce runtime choices.",
                    "npu_run includes submit/wait and scheduling. One aggregate baseline cannot identify kernel-family throughput or bandwidth; wall predictions are heuristic, not validated timings.",
                    "Sub-32-core rows are hypothetical rebuilds with unchanged PPC; no overlap benefit is credited.",
                    "per_regime_r1_r3 uses the restored route. current_non_k and subsequent plans use current extended R1 non-K membership; they are explicit configurations, not complete historical commit reconstructions.",
                    "Context count is distinct artifact names, not runtime handles or measured resident contexts. Host cache effects and CPU operator changes are unmodeled.",
                    "best_existing_per_layer ranks existing envelopes without a resident-context budget; it is a screening suggestion, not a deployable optimized schedule.",
                    "Fusion lower bound assumes one invocation per logical operator; feasibility has not been demonstrated. CPU islands remain in fixed baseline cost.",
                ], plans=plans, candidates=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "profile_baseline.json")
    parser.add_argument("--cores", type=int, nargs="+", choices=(4, 8, 16, 24, 32), default=[32])
    args = parser.parse_args()
    report = build_report(args.baseline, args.cores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regime_planner_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "regime_planner_report.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["candidates"][0]))
        writer.writeheader()
        writer.writerows(report["candidates"])
    for plan in report["plans"]:
        print(f"{plan['name']:16s} {plan['total_host_launches']:4d} calls  {plan['resident_xclbins_est']:2d} artifacts  {plan['estimated_wall_ms']:8.1f} ms heuristic  {len(plan['rejected_layers'])} fit warnings")
    print("Latency estimates are screening heuristics; read limitations in the JSON report.")


if __name__ == "__main__":
    main()
