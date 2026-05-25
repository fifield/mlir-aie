# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# HF-weights answer-level correctness gate for the llama32_1b PythoC port.
#
# This is the GOLD-STANDARD correctness check for each phase: it loads real
# Meta-LLaMA-3.2-1B-Instruct weights from HuggingFace and verifies that the
# end-to-end NPU pipeline still produces a sensible answer to a fixed prompt.
#
# Phase exits should run this when HF weights are available. The
# `make snapshot` harness uses synthetic weights and so cannot detect
# answer-level regressions caused by, e.g., a kernel that subtly mis-rounds
# real activations -- this gate can.
#
# Behavior:
#   * Prompt:           "What is the capital of France?"
#   * n_tokens:         8 decode tokens
#   * Acceptance:       the detokenized first-5 decode tokens must contain
#                       the substring "Paris" (case-insensitive).
#   * Skip condition:   if the HF cache isn't populated locally AND
#                       network/auth would be required, mark xfail with a
#                       clear message rather than failing CI.

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
BUILD_DIR = PROJECT_DIR / "build_peano"
INFERENCE_SCRIPT = PROJECT_DIR / "llama32_1b_inference.py"

HF_MODEL = os.environ.get("HF_MODEL_ID") or "unsloth/Llama-3.2-1B-Instruct"
PROMPT = "What is the capital of France?"
N_TOKENS = 10
EXPECTED_FIRST_N_DECODE = 10
EXPECTED_SUBSTRING = "paris"  # case-insensitive

# Phase 6 / Stage 3 Subtask D: parameterize the gate on QUANT so that
# `make hf-gate QUANT=awq` exercises the AWQ NPU decode path.
# Defaults to bf16 (the original Phase 4 gate). For QUANT=awq, the caller
# must supply HF_GATE_AWQ_WEIGHTS pointing at a repacked AWQ model dir.
HF_GATE_QUANT = os.environ.get("HF_GATE_QUANT", "bf16")
HF_GATE_AWQ_WEIGHTS = os.environ.get("HF_GATE_AWQ_WEIGHTS", "")

# Regex for the per-token log line emitted by --profile mode.
RE_TOKEN = re.compile(r"^\s*Token\s+(\d+):\s*id=(\d+),\s*time=(\d+)ms")


def _hf_cache_has_model(model_id: str) -> bool:
    """Return True iff the HF cache appears to have the requested model
    downloaded locally. We look in the standard locations rather than try
    to import huggingface_hub -- this is a heuristic.
    """
    # Standard cache locations.
    candidates = [
        Path(os.environ.get("HF_HOME", "")) / "hub" if os.environ.get("HF_HOME") else None,
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "")) if os.environ.get("HUGGINGFACE_HUB_CACHE") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    safe_id = "models--" + model_id.replace("/", "--")
    for c in candidates:
        if c is None:
            continue
        target = c / safe_id
        if target.exists():
            # snapshots/<rev>/config.json present == model is materialized.
            for snap in (target / "snapshots").glob("*"):
                if (snap / "config.json").exists():
                    return True
    return False


def _has_tokenizer() -> bool:
    """We need a local AutoTokenizer to detokenize NPU's output. If
    transformers + a local tokenizer aren't present, we can't run this gate.
    """
    try:
        from transformers import AutoTokenizer  # noqa: F401
    except Exception:
        return False
    return _hf_cache_has_model(HF_MODEL)


def _parse_decode_token_ids(stdout: str) -> list[int]:
    ids: list[int] = []
    for line in stdout.splitlines():
        m = RE_TOKEN.match(line)
        if m:
            ids.append(int(m.group(2)))
    return ids


@pytest.mark.skipif(
    not _has_tokenizer(),
    reason=(
        "HuggingFace cache does not contain {model}; this gate requires "
        "real weights. Populate the cache via `huggingface-cli download "
        "{model}` (gated; needs HF auth) and re-run. Synthetic-weight "
        "phase-snapshot tests cover the rest of the pipeline."
    ).format(model=HF_MODEL),
)
def test_hf_answer_gate_paris():
    """Gold-standard answer-level correctness gate.

    With real HF weights, ask "What is the capital of France?" and verify the
    NPU produces a string containing "Paris" within the first 10 decode tokens.
    """
    from transformers import AutoTokenizer

    if not BUILD_DIR.exists():
        pytest.fail(
            f"build_peano/ does not exist at {BUILD_DIR}. "
            f"Run `make compile` first."
        )

    cmd = [
        sys.executable, str(INFERENCE_SCRIPT),
        "--run-only",
        "--n-tokens", str(N_TOKENS),
        "--profile",
        "--prompt", PROMPT,
        "--quant", HF_GATE_QUANT,
        "--model", "instruct",
    ]
    if HF_GATE_QUANT == "awq":
        if not HF_GATE_AWQ_WEIGHTS:
            pytest.skip(
                "HF_GATE_QUANT=awq requires HF_GATE_AWQ_WEIGHTS=<path> "
                "(set via `make hf-gate QUANT=awq AWQ_WEIGHTS=...`)"
            )
        # inference.py rejects `--hf-model-id` with `--quant awq`
        # (tokenizer is loaded from the AWQ-weights dir instead).
        cmd += ["--awq-weights", HF_GATE_AWQ_WEIGHTS,
                "--awq-decode-experimental"]
    else:
        cmd += ["--hf-model-id", HF_MODEL]

    env = os.environ.copy()
    pip_peano = "/home/jfifield/npu-dev-air/venv/lib/python3.12/site-packages/llvm-aie"
    if Path(pip_peano, "bin", "clang++").exists():
        env.setdefault("PEANO_INSTALL_DIR", pip_peano)

    proc = subprocess.run(
        cmd, cwd=BUILD_DIR, capture_output=True, text=True,
        env=env, timeout=900,
    )
    assert proc.returncode == 0, (
        f"inference exited with code {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-2000:]}\n"
    )

    token_ids = _parse_decode_token_ids(proc.stdout)
    assert len(token_ids) >= 1, (
        f"could not parse any 'Token N: id=...' lines from stdout. "
        f"Last 2KB:\n{proc.stdout[-2000:]}"
    )

    # For AWQ mode the AWQ-weights dir ships its own tokenizer; otherwise use HF.
    tokenizer_src = HF_GATE_AWQ_WEIGHTS if HF_GATE_QUANT == "awq" else HF_MODEL
    tok = AutoTokenizer.from_pretrained(tokenizer_src)
    head = token_ids[:EXPECTED_FIRST_N_DECODE]
    text = tok.decode(head, skip_special_tokens=True)
    print(f"\n[hf-gate] first {len(head)} decode tokens -> {head}")
    print(f"[hf-gate] detokenized: {text!r}")

    assert EXPECTED_SUBSTRING in text.lower(), (
        f"expected first {EXPECTED_FIRST_N_DECODE} decode tokens to "
        f"detokenize to a string containing {EXPECTED_SUBSTRING!r}; got "
        f"{text!r} (ids={head}). This is the gold-standard correctness gate "
        f"-- if this fails, a recent kernel swap likely broke real-weight "
        f"behavior even though synthetic-weight snapshots still pass."
    )
