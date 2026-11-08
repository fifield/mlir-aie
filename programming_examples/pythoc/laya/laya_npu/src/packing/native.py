"""Raw checkpoint to native parameter and initialization layouts; CPU only."""

import numpy as np
import torch
from safetensors.torch import load_file


def bits(t):
    return t.to(torch.bfloat16).contiguous().view(torch.uint16).numpy()


def generate(model):
    weights = load_file(str(model))
    torch.set_num_threads(2)
    for key, t in weights.items():
        if t.ndim == 2 and (
            key.startswith("encoder.layers.") or key.startswith("head.layers.")
        ):
            raw = bits(t.T)
            yield {"kind": "matrix_kn", "tensor": key}, raw
            if key.endswith("mlp.Wi.weight"):
                permutation = [
                    i for c in range(8) for p in range(c, 41, 8) for i in (p, 41 + p)
                ]
                yield {
                    "kind": "matrix_gateup_tile64",
                    "tensor": key,
                }, np.ascontiguousarray(raw.reshape(1024, 82, 64)[:, permutation, :])
    for length in (64, 128, 256, 512):
        for key, t in weights.items():
            if (
                t.ndim == 1
                and t.numel() == 1024
                and key.endswith(".weight")
                and "norm" in key
            ):
                value = np.zeros((length, 3072), np.uint16)
                value[:, 1024:2048] = bits(t)
                bias = weights.get(key[:-6] + "bias")
                if bias is not None:
                    value[:, 2048:] = bits(bias)
                yield {
                    "kind": "norm_x_gamma_beta",
                    "tensor": key,
                    "length": length,
                }, value
            if (
                t.ndim == 1
                and key.startswith("head.layers.")
                and ("_bias" in key or ".bias" in key)
                and "norm" not in key
            ):
                width = t.numel()
                value = np.zeros((length * width // 256, 512), np.float32)
                value[:, 256:] = (
                    t.to(torch.bfloat16)
                    .float()
                    .numpy()
                    .reshape(-1, 256)[np.arange(length * width // 256) % (width // 256)]
                )
                yield {"kind": "decision_bias", "tensor": key, "length": length}, value
                if key.endswith("in_proj_bias"):
                    i = np.arange(length * width)
                    src = (i // 256) * 512 + i % 256
                    row, col = np.divmod(i, width)
                    tile = (col // 64) * length * 64 + row * 64 + col % 64
                    dst = (tile // 256) * 512 + tile % 256
                    perm = np.empty_like(value.reshape(-1))
                    perm[dst] = value.reshape(-1)[src]
                    perm[dst + 256] = value.reshape(-1)[src + 256]
                    yield {
                        "kind": "decision_bias_tile64",
                        "tensor": key,
                        "length": length,
                    }, perm
        for row in range(3):
            value = np.tile(bits(weights["type_emb.weight"][row]), length)
            yield {"kind": "type", "row": row, "length": length}, value
            yield {"kind": "type_pair", "row": row, "length": length}, np.concatenate(
                [np.full(value.size, 0x7FC1, np.uint16), value]
            )
        for theta in (10000, 160000):
            inv = 1.0 / theta ** (torch.arange(0, 64, 2).float() / 64)
            angle = torch.arange(length).float()[:, None] * inv[None, :]
            value = np.zeros((16, length, 128), np.uint16)
            value[:, :, 64:96] = bits(angle.cos())
            value[:, :, 96:] = bits(angle.sin())
            yield {"kind": "rope", "theta": theta, "length": length}, value
        q, groups = {64: (16, 1), 128: (32, 1), 256: (32, 2), 512: (64, 2)}[length]
        for local in (0, 1):
            value = np.zeros(32 * groups * groups * 16 + 4 * q, np.uint16)
            for group in range(groups):
                for chunk in range(groups):
                    for qc in range(4):
                        for stage in range(4):
                            i = (((group * groups + chunk) * 4 + qc) * 4 + stage) * 32
                            value[i : i + 3] = [
                                group * 4 * q + qc * q,
                                (stage * groups + chunk) * q,
                                local,
                            ]
            yield {
                "kind": "attention_metadata",
                "local": local,
                "length": length,
            }, value


def engine_cb():
    """src/packing/engine_cb.py (device_cb's engine layout and header)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("laya_npu_engine_cb", Path(__file__).with_name("engine_cb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def engine_header(fields=None):
    value = np.zeros((8, 1024), np.uint16)
    if fields is None:
        fields = [
            (1, 1, 0, 2, 1, 32, 0),
            (1, 1, 0, 2, 1, 82, 0),
            (2, 1, 0, 6, 2, 32, 0),
            (3, 1, 2, 6, 1, 32, 0),
        ]
        fields += [(5, length // 32, 16, 1, 0, 0, 2) for length in (64, 128, 256, 512)]
    for row, (op, units, ntab, ntile, npass, kt, npre) in enumerate(fields):
        words = [0x9A05, op, units, ntab, ntile, npass, kt]
        check = npre
        for w in words:
            check ^= w
        words += [check, npre]
        indices = (np.arange(18) // 8) * 32 + np.arange(18) % 8
        value[row, indices] = np.asarray(words, dtype="<u4").view("<u2")
    return value


def fixture(recipe, weights):
    kind = recipe["kind"]
    if kind == "fill":
        return np.full(recipe["count"], recipe["value"], dtype=recipe["dtype"])
    if kind == "engine_header":
        return engine_header()
    if kind == "engine_header_cb":
        return engine_header(engine_cb().header_fields())
    if kind == "engine_header_s8":
        return engine_header(engine_cb().header_fields(8))
    if kind == "checkpoint_rows":
        embedding = bits(weights["encoder.embeddings.tok_embeddings.weight"])
        types = bits(weights["type_emb.weight"])
        chunks = []
        for source, index, count in recipe["runs"]:
            row = (
                np.full(1024, 0x7FC1, np.uint16)
                if source == "poison"
                else types[index] if source == "type" else embedding[index]
            )
            chunks.append(np.tile(row, count))
        return np.concatenate(chunks)
    raise ValueError(kind)
