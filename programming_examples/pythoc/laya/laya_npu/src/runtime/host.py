"""Pinned input preparation and FP32 deployment heads (M17 semantics)."""

import math
import numpy as np
import torch
import torch.nn.functional as F
from .vendor.laya.agent import Agent
from .common import DATA

INPUT_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
BUCKETS = (64, 128, 256, 512)
BATCH_SIZES = (1, 2, 4, 8)


def load_agent():
    from importlib.metadata import version

    for package, expected in {"torch": "2.12.0+cpu", "transformers": "5.17.0"}.items():
        if version(package) != expected:
            raise RuntimeError(
                f"{package}: expected {expected}, got {version(package)}"
            )
    return Agent(str(DATA / "model"), device="cpu")


def make_batch(agent, state, questions, pad_to=None):
    from .vendor.laya.common import (
        QTYPES,
        build_sequence,
        collate_items,
        render_options,
    )

    items = []
    for qid, definition in questions.items():
        q = agent._to_internal(definition)
        ids, markers = build_sequence(
            agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"]
        )
        if len(markers) != len(render_options(q)):
            raise ValueError(
                f'question {qid!r} options exceed head_max_len={agent.cfg["head_max_len"]}'
            )
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
    batch = collate_items([items], agent.tok.pad_token_id)
    result = {k: batch[k] for k in INPUT_KEYS}
    if pad_to is not None:
        extra = pad_to - result["input_ids"].shape[1]
        if extra < 0 or pad_to > agent.cfg["max_len"]:
            raise ValueError("Padding length must fit inputs and sequence limit")
        result["input_ids"] = F.pad(
            result["input_ids"], (0, extra), value=agent.tok.pad_token_id
        )
        result["attention_mask"] = F.pad(result["attention_mask"], (0, extra))
    return result


def probabilities(agent, batch, logits, action):
    """Independent temperature selection; return unrounded answers and action paths."""
    rows, temps = [], []
    for i, qt in enumerate(batch["qtype"].tolist()):
        k = int(batch["marker_mask"][i].sum())
        bucket = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        key = f'{("choice", "score", "noul")[qt]}:{bucket}'
        temp = max(
            1e-3, float(agent.temperature_by_options.get(key, agent.temperature[qt]))
        )
        # Match upstream's NumPy FP32 calibration and retain all unrounded values.
        z = logits[i, :k].float().numpy() / temp
        p = np.exp(z - z.max())
        p /= p.sum()
        rows.append(torch.from_numpy(p.copy()))
        temps.append(temp)
    return rows, torch.softmax(action.float(), -1), temps


def format_result(agent, questions, batch, logits, action):
    rows, acts, _ = probabilities(agent, batch, logits, action)
    answers = {}
    for i, (qid, definition) in enumerate(questions.items()):
        q = agent._to_internal(definition)
        p = rows[i].numpy()
        k = len(p)
        confidence = (
            1.0
            if k < 2
            else float(
                np.clip(
                    1 + (p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(k), 0, 1
                )
            )
        )
        answer = {
            "type": q["t"],
            "confidence": round(confidence, 4),
            "action": {"act_probability": round(float(acts[i, 0]), 4)},
        }
        if q["t"] == "choice":
            labels = list(q["crit"])
            answer.update(
                choice=labels[int(p.argmax())],
                probabilities={
                    label: round(float(v), 4) for label, v in zip(labels, p)
                },
            )
        elif q["t"] == "score":
            answer.update(
                score=round(float((np.arange(k) * p).sum()), 4),
                legend={str(j): c for j, c in enumerate(q["crit"])},
                probabilities={str(j): round(float(v), 4) for j, v in enumerate(p)},
            )
        else:
            answer.update(
                noul=round(float(p[1]), 4),
                confidence=round(max(float(p[1]), 1 - float(p[1])), 4),
            )
        answers[qid] = answer
    return {
        "model": "laya-rl-agent",
        "answers": answers,
        "usage": {
            "input_tokens": int(batch["attention_mask"].sum()),
            "output_tokens": 0,
        },
    }


def choose_bucket(length, bucket=None):
    if not isinstance(length, int) or isinstance(length, bool) or not 0 < length <= 512:
        raise ValueError("Input length must be between 1 and 512")
    if bucket is not None:
        if isinstance(bucket, bool) or bucket not in BUCKETS or length > bucket:
            raise ValueError("Explicit bucket must be supported and fit every row")
        return bucket
    return next(value for value in BUCKETS if length <= value)


def prepare_rows(agent, rows, bucket=None):
    """Rows are ordered {id, state, question} mappings; return batch plus metadata."""
    rows = list(rows)
    if len(rows) not in BATCH_SIZES:
        raise ValueError("Supported batch sizes are 1, 2, 4 and 8")
    if agent.cfg.get("max_len") != 512 or agent.cfg.get("head_max_len") != 192:
        raise ValueError(
            "Input preparation requires pinned max_len=512/head_max_len=192"
        )
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Row identifiers must be unique")
    singletons = []
    for row in rows:
        if row["question"].get("type") not in ("choice", "score", "noul"):
            raise ValueError("Unsupported question type")
        item = make_batch(agent, row["state"], {row["id"]: row["question"]})
        positions = item["marker_pos"][0, item["marker_mask"][0]]
        if not positions.numel() or not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError("Missing or unordered candidate markers")
        if not bool((item["input_ids"][0, positions] == agent.tok.mask_token_id).all()):
            raise ValueError("Candidate marker token mismatch")
        singletons.append(item)
    lengths = [int(item["attention_mask"].sum()) for item in singletons]
    physical = choose_bucket(max(lengths), bucket)
    # The action feature always reads top two probabilities. A masked dummy
    # makes the promised positive-option domain defined for one-option rows.
    markers = max(2, max(item["marker_pos"].shape[1] for item in singletons))
    batch = {}
    for key in INPUT_KEYS:
        values = []
        for item in singletons:
            value = item[key]
            if key in ("input_ids", "attention_mask"):
                value = F.pad(
                    value,
                    (0, physical - value.shape[1]),
                    value=agent.tok.pad_token_id if key == "input_ids" else 0,
                )
            elif key in ("marker_pos", "marker_mask"):
                value = F.pad(value, (0, markers - value.shape[1]), value=0)
            values.append(value)
        batch[key] = torch.cat(values, dim=0)
    return {
        "batch": batch,
        "ids": ids,
        "bucket": physical,
        "valid_lengths": lengths,
        "rows": rows,
    }


def prepare(agent, state, questions, bucket=None):
    return prepare_rows(
        agent,
        (
            {"id": key, "state": state, "question": value}
            for key, value in questions.items()
        ),
        bucket,
    )


def cpu_heads(model, h, batch):
    h = h.float()
    idx = batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, h.shape[-1])
    markers = torch.gather(h, 1, idx)
    scorer = model.scorer
    x = F.layer_norm(
        markers,
        scorer[0].normalized_shape,
        scorer[0].weight,
        scorer[0].bias,
        scorer[0].eps,
    )
    x = F.gelu(F.linear(x, scorer[1].weight, scorer[1].bias), approximate="none")
    logits = (
        F.linear(x, scorer[3].weight, scorer[3].bias)
        .squeeze(-1)
        .masked_fill(~batch["marker_mask"], -1e4)
    )
    p = logits.softmax(-1)
    count = batch["marker_mask"].sum(-1).clamp(min=2).float()
    top = p.topk(2, -1).values
    entropy = -(p * p.clamp_min(1e-9).log()).sum(-1) / count.log()
    features = torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255), -1)
    action_input = torch.cat((h[:, 0], features), -1)
    head = model.act_head
    action = F.linear(
        F.gelu(
            F.linear(action_input, head[0].weight, head[0].bias), approximate="none"
        ),
        head[2].weight,
        head[2].bias,
    )
    return logits, action, action_input
