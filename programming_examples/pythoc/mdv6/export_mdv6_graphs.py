#!/usr/bin/env python3
"""Export MDV6 network graphs for Netron and browser viewing.

Outputs:
  graphs/mdv6_mit_yolov9c.onnx  - Netron-compatible ONNX graph
  graphs/mdv6_network.html       - self-contained browser architecture diagram
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
PYTHON_DIR = ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

from mdv6.model import MDV6MITYOLOv9c  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "graphs"
OUT_DIR.mkdir(exist_ok=True)


class NetronExportWrapper(nn.Module):
    """Flatten MDV6's list-of-tuples output into named ONNX outputs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        p3, p4, p5 = self.model(x)
        p3_cls, p3_anchor, p3_vec = p3
        p4_cls, p4_anchor, p4_vec = p4
        p5_cls, p5_anchor, p5_vec = p5
        return (
            p3_cls, p3_anchor, p3_vec,
            p4_cls, p4_anchor, p4_vec,
            p5_cls, p5_anchor, p5_vec,
        )


def shape_of(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (tuple, list)):
        return [shape_of(x) for x in obj]
    return str(type(obj).__name__)


def collect_top_level_shapes(model: MDV6MITYOLOv9c, x: torch.Tensor):
    hooks = []
    shapes = {}
    top_names = [
        "conv0", "conv1", "elan2", "aconv3", "rep_elan4", "aconv5",
        "rep_elan6", "aconv7", "rep_elan8", "spp9", "rep_elan12",
        "rep_elan15", "aconv16", "rep_elan18", "aconv19", "rep_elan21", "detect",
    ]

    for name in top_names:
        mod = getattr(model, name)

        def hook(_module, inputs, output, name=name):
            shapes[name] = {
                "input": shape_of(inputs[0]) if inputs else None,
                "output": shape_of(output),
                "params": sum(p.numel() for p in _module.parameters()),
            }

        hooks.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        outputs = model(x)

    for h in hooks:
        h.remove()
    shapes["outputs"] = shape_of(outputs)
    return shapes


def export_onnx(model: MDV6MITYOLOv9c):
    onnx_path = OUT_DIR / "mdv6_mit_yolov9c.onnx"
    wrapper = NetronExportWrapper(model).eval()
    dummy = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    output_names = [
        "p3_class_logits", "p3_anchor_distribution", "p3_bbox_vector",
        "p4_class_logits", "p4_anchor_distribution", "p4_bbox_vector",
        "p5_class_logits", "p5_anchor_distribution", "p5_bbox_vector",
    ]
    # Use the legacy tracer (dynamo=False) with fixed 640x640 shapes. The new
    # torch.export-based ONNX path attempts to solve symbolic shape constraints
    # introduced by channel chunking and currently fails on this model.
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        input_names=["images"],
        output_names=output_names,
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
        dynamo=False,
    )
    return onnx_path


def write_html(shapes: dict):
    html_path = OUT_DIR / "mdv6_network.html"
    nodes = [
        ("input", "Input", "1×3×640×640", 60, 260, "external"),
        ("conv0", "Conv0", "3×3 s2 · 3→32 · P1 320²", 210, 260, "backbone"),
        ("conv1", "Conv1", "3×3 s2 · 32→64 · P2 160²", 380, 260, "backbone"),
        ("elan2", "ELAN2", "64→64 · P2 160²", 560, 260, "backbone"),
        ("aconv3", "AConv3", "avg+3×3 s2 · 64→128", 740, 260, "backbone"),
        ("b3", "RepNCSPELAN4 / B3", "128→128 · P3 80²", 930, 260, "backbone"),
        ("aconv5", "AConv5", "3×3 s2 · 128→192", 210, 410, "backbone"),
        ("b4", "RepNCSPELAN6 / B4", "192→192 · P4 40²", 410, 410, "backbone"),
        ("aconv7", "AConv7", "3×3 s2 · 192→256", 620, 410, "backbone"),
        ("b5", "RepNCSPELAN8 / B5", "256→256 · P5 20²", 820, 410, "backbone"),
        ("spp", "SPPELAN9 / N3", "SPP pools + ELAN · 256→256", 1030, 410, "neck"),
        ("up1", "Upsample ×2 + concat B4", "256+192=448 · 40²", 260, 570, "neck"),
        ("n4", "RepNCSPELAN12 / N4", "448→192 · 40²", 500, 570, "neck"),
        ("up2", "Upsample ×2 + concat B3", "192+128=320 · 80²", 740, 570, "neck"),
        ("p3", "RepNCSPELAN15 / P3", "320→128 · 80²", 980, 570, "head"),
        ("down16", "AConv16 + concat N4", "128→96; 96+192=288", 300, 730, "head"),
        ("p4", "RepNCSPELAN18 / P4", "288→192 · 40²", 560, 730, "head"),
        ("down19", "AConv19 + concat N3", "192→128; 128+256=384", 820, 730, "head"),
        ("p5", "RepNCSPELAN21 / P5", "384→256 · 20²", 1080, 730, "head"),
        ("det", "MultiheadDetection", "3 heads · classes=3 · reg_max=16", 660, 900, "detect"),
    ]
    edges = [
        ("input", "conv0"), ("conv0", "conv1"), ("conv1", "elan2"), ("elan2", "aconv3"), ("aconv3", "b3"),
        ("b3", "aconv5"), ("aconv5", "b4"), ("b4", "aconv7"), ("aconv7", "b5"), ("b5", "spp"),
        ("spp", "up1"), ("b4", "up1"), ("up1", "n4"), ("n4", "up2"), ("b3", "up2"), ("up2", "p3"),
        ("p3", "down16"), ("n4", "down16"), ("down16", "p4"), ("p4", "down19"), ("spp", "down19"), ("down19", "p5"),
        ("p3", "det"), ("p4", "det"), ("p5", "det"),
    ]
    pos = {n[0]: (n[3], n[4]) for n in nodes}
    styles = {
        "external": ("rgba(30,41,59,.75)", "#94a3b8"),
        "backbone": ("rgba(8,51,68,.45)", "#22d3ee"),
        "neck": ("rgba(120,53,15,.38)", "#fbbf24"),
        "head": ("rgba(6,78,59,.45)", "#34d399"),
        "detect": ("rgba(76,29,149,.45)", "#a78bfa"),
    }

    def rect_node(node):
        node_id, title, subtitle, x, y, kind = node
        fill, stroke = styles[kind]
        w, h = (190, 68)
        if node_id in {"det"}:
            w = 260
        return f'''
        <g class="node" id="node-{node_id}">
          <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#0f172a"/>
          <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>
          <text x="{x+14}" y="{y+25}" class="title">{html.escape(title)}</text>
          <text x="{x+14}" y="{y+47}" class="sub">{html.escape(subtitle)}</text>
        </g>'''

    def edge_path(a, b):
        ax, ay = pos[a]
        bx, by = pos[b]
        aw = 260 if a == "det" else 190
        # route from center/right-ish to center/left-ish, simple cubic
        x1, y1 = ax + aw, ay + 34
        x2, y2 = bx, by + 34
        if abs(y2 - y1) > 90:
            # vertical skip/concat routes
            x1, y1 = ax + aw/2, ay + 68
            x2, y2 = bx + 95, by
        c1x = x1 + max(40, abs(x2-x1) * 0.35)
        c2x = x2 - max(40, abs(x2-x1) * 0.35)
        return f'<path d="M {x1:.1f},{y1:.1f} C {c1x:.1f},{y1:.1f} {c2x:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" class="edge" marker-end="url(#arrow)"/>'

    total_params = sum(v.get("params", 0) for k, v in shapes.items() if isinstance(v, dict))
    rows = []
    for name, meta in shapes.items():
        if not isinstance(meta, dict) or name == "outputs":
            continue
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(str(meta.get('input')))}</code></td>"
            f"<td><code>{html.escape(str(meta.get('output')))}</code></td><td>{meta.get('params',0):,}</td></tr>"
        )

    details_html = '  <div class="card" style="margin-top:18px">\n    <h2>Multi-op layer internals</h2>\n    <p class="muted">Expanded views of the composite modules used repeatedly in the MDV6 YOLOv9-c graph. These are logical PyTorch/module views, not the lower-level ONNX operator graph.</p>\n    <div class="detail-grid">\n      <div class="detail">\n        <h3>Conv block</h3>\n        <p>Used throughout. The PythoC kernels generally fuse Conv2d + BatchNorm + SiLU into one device kernel.</p>\n        <svg viewBox="0 0 520 120">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <path d="M120 60 L170 60" class="mini-edge"/><path d="M270 60 L320 60" class="mini-edge"/><path d="M420 60 L470 60" class="mini-edge"/>\n          <rect x="20" y="32" width="100" height="56" class="mini-box mini-op"/><text x="38" y="55" class="mini-title">Conv2d</text><text x="34" y="73" class="mini-sub">k/s/groups</text>\n          <rect x="170" y="32" width="100" height="56" class="mini-box mini-op"/><text x="194" y="55" class="mini-title">BatchNorm</text><text x="198" y="73" class="mini-sub">eps=1e-3</text>\n          <rect x="320" y="32" width="100" height="56" class="mini-box mini-op"/><text x="352" y="55" class="mini-title">SiLU</text><text x="340" y="73" class="mini-sub">or Identity</text>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>AConv downsample</h3>\n        <p>Average-pool prefilter followed by stride-2 3×3 Conv block. Used at stage transitions and head downsample paths.</p>\n        <svg viewBox="0 0 520 120">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <path d="M120 60 L170 60" class="mini-edge"/><path d="M270 60 L320 60" class="mini-edge"/><path d="M420 60 L470 60" class="mini-edge"/>\n          <rect x="20" y="32" width="100" height="56" class="mini-box mini-split"/><text x="36" y="55" class="mini-title">AvgPool2d</text><text x="35" y="73" class="mini-sub">k2 s1</text>\n          <rect x="170" y="32" width="100" height="56" class="mini-box mini-op"/><text x="196" y="55" class="mini-title">Conv</text><text x="188" y="73" class="mini-sub">3×3 s2</text>\n          <rect x="320" y="32" width="100" height="56" class="mini-box mini-merge"/><text x="346" y="55" class="mini-title">Output</text><text x="345" y="73" class="mini-sub">H/2 W/2</text>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>ELAN</h3>\n        <p>Split-channel path: a 1×1 Conv creates two halves, one bypasses while the other goes through two 3×3 Conv blocks; all four tensors concatenate into a final 1×1 Conv.</p>\n        <svg viewBox="0 0 620 190">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <rect x="20" y="70" width="90" height="50" class="mini-box mini-op"/><text x="42" y="92" class="mini-title">Conv1</text><text x="39" y="109" class="mini-sub">1×1</text>\n          <rect x="150" y="70" width="90" height="50" class="mini-box mini-split"/><text x="174" y="92" class="mini-title">Chunk</text><text x="171" y="109" class="mini-sub">x1,x2</text>\n          <rect x="290" y="25" width="90" height="50" class="mini-box mini-op"/><text x="312" y="47" class="mini-title">x1</text><text x="300" y="64" class="mini-sub">bypass</text>\n          <rect x="290" y="115" width="90" height="50" class="mini-box mini-op"/><text x="312" y="137" class="mini-title">Conv2</text><text x="306" y="154" class="mini-sub">3×3</text>\n          <rect x="420" y="115" width="90" height="50" class="mini-box mini-op"/><text x="442" y="137" class="mini-title">Conv3</text><text x="436" y="154" class="mini-sub">3×3</text>\n          <rect x="520" y="70" width="80" height="50" class="mini-box mini-merge"/><text x="545" y="92" class="mini-title">Cat</text><text x="531" y="109" class="mini-sub">x1,x2,x3,x4</text>\n          <path d="M110 95 L150 95" class="mini-edge"/><path d="M240 95 C255 50 265 50 290 50" class="mini-edge"/><path d="M240 95 C255 140 265 140 290 140" class="mini-edge"/><path d="M380 140 L420 140" class="mini-edge"/><path d="M380 50 C450 50 470 74 520 85" class="mini-edge"/><path d="M240 95 C390 95 430 95 520 95" class="mini-edge"/><path d="M380 140 C450 140 470 116 520 105" class="mini-edge"/><path d="M510 140 C535 132 545 125 560 120" class="mini-edge"/>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>RepNCSPELAN</h3>\n        <p>The dominant multi-op block. It wraps ELAN-style split/concat around two sequential RepNCSP branches, each followed by a 3×3 Conv, then a final 1×1 Conv.</p>\n        <svg viewBox="0 0 720 230">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <rect x="20" y="92" width="90" height="50" class="mini-box mini-op"/><text x="42" y="114" class="mini-title">Conv1</text><text x="40" y="131" class="mini-sub">1×1</text>\n          <rect x="145" y="92" width="85" height="50" class="mini-box mini-split"/><text x="166" y="114" class="mini-title">Chunk</text><text x="163" y="131" class="mini-sub">x1,x2</text>\n          <rect x="275" y="38" width="95" height="50" class="mini-box mini-op"/><text x="307" y="60" class="mini-title">x1</text><text x="288" y="77" class="mini-sub">bypass</text>\n          <rect x="275" y="145" width="110" height="50" class="mini-box mini-op"/><text x="303" y="167" class="mini-title">RepNCSP</text><text x="294" y="184" class="mini-sub">conv2 branch</text>\n          <rect x="420" y="145" width="90" height="50" class="mini-box mini-op"/><text x="442" y="167" class="mini-title">Conv</text><text x="436" y="184" class="mini-sub">3×3</text>\n          <rect x="545" y="145" width="110" height="50" class="mini-box mini-op"/><text x="573" y="167" class="mini-title">RepNCSP</text><text x="562" y="184" class="mini-sub">conv3 branch</text>\n          <rect x="545" y="38" width="90" height="50" class="mini-box mini-merge"/><text x="574" y="60" class="mini-title">Cat</text><text x="554" y="77" class="mini-sub">x1,x2,x3,x4</text>\n          <path d="M110 117 L145 117" class="mini-edge"/><path d="M230 117 C250 63 255 63 275 63" class="mini-edge"/><path d="M230 117 C250 170 255 170 275 170" class="mini-edge"/><path d="M385 170 L420 170" class="mini-edge"/><path d="M510 170 L545 170" class="mini-edge"/><path d="M370 63 C445 63 480 63 545 63" class="mini-edge"/><path d="M230 117 C360 108 440 83 545 73" class="mini-edge"/><path d="M510 170 C540 140 555 110 570 88" class="mini-edge"/><path d="M655 170 C680 140 675 90 635 70" class="mini-edge"/>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>RepNCSP + Bottleneck core</h3>\n        <p>Inside RepNCSPELAN branches. Two 1×1 Conv paths; one passes through repeated Bottleneck blocks using RepConv(3×3 + 1×1) then Conv3×3 with optional residual add.</p>\n        <svg viewBox="0 0 720 210">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <rect x="20" y="80" width="80" height="50" class="mini-box mini-split"/><text x="45" y="102" class="mini-title">In</text>\n          <rect x="145" y="35" width="90" height="50" class="mini-box mini-op"/><text x="167" y="57" class="mini-title">Conv1</text><text x="160" y="74" class="mini-sub">1×1</text>\n          <rect x="145" y="125" width="90" height="50" class="mini-box mini-op"/><text x="167" y="147" class="mini-title">Conv2</text><text x="160" y="164" class="mini-sub">1×1</text>\n          <rect x="280" y="35" width="130" height="50" class="mini-box mini-op"/><text x="300" y="57" class="mini-title">Bottleneck×N</text><text x="296" y="74" class="mini-sub">RepConv→Conv(+add)</text>\n          <rect x="455" y="80" width="90" height="50" class="mini-box mini-merge"/><text x="483" y="102" class="mini-title">Cat</text>\n          <rect x="590" y="80" width="90" height="50" class="mini-box mini-op"/><text x="612" y="102" class="mini-title">Conv3</text><text x="605" y="119" class="mini-sub">1×1</text>\n          <path d="M100 105 C120 60 125 60 145 60" class="mini-edge"/><path d="M100 105 C120 150 125 150 145 150" class="mini-edge"/><path d="M235 60 L280 60" class="mini-edge"/><path d="M410 60 C430 75 440 90 455 100" class="mini-edge"/><path d="M235 150 C340 150 400 130 455 112" class="mini-edge"/><path d="M545 105 L590 105" class="mini-edge"/>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>SPPELAN</h3>\n        <p>Compress with 1×1 Conv, run three serial 5×5 max-pools, concatenate the original + pooled features, then restore channels with 1×1 Conv.</p>\n        <svg viewBox="0 0 720 150">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <rect x="20" y="50" width="90" height="50" class="mini-box mini-op"/><text x="42" y="72" class="mini-title">Conv1</text><text x="36" y="89" class="mini-sub">1×1</text>\n          <rect x="155" y="50" width="90" height="50" class="mini-box mini-split"/><text x="176" y="72" class="mini-title">Pool1</text><text x="171" y="89" class="mini-sub">max 5×5</text>\n          <rect x="290" y="50" width="90" height="50" class="mini-box mini-split"/><text x="311" y="72" class="mini-title">Pool2</text><text x="306" y="89" class="mini-sub">max 5×5</text>\n          <rect x="425" y="50" width="90" height="50" class="mini-box mini-split"/><text x="446" y="72" class="mini-title">Pool3</text><text x="441" y="89" class="mini-sub">max 5×5</text>\n          <rect x="560" y="50" width="70" height="50" class="mini-box mini-merge"/><text x="584" y="72" class="mini-title">Cat</text><text x="574" y="89" class="mini-sub">4-way</text>\n          <rect x="650" y="50" width="60" height="50" class="mini-box mini-op"/><text x="660" y="72" class="mini-title">Conv5</text><text x="661" y="89" class="mini-sub">1×1</text>\n          <path d="M110 75 L155 75" class="mini-edge"/><path d="M245 75 L290 75" class="mini-edge"/><path d="M380 75 L425 75" class="mini-edge"/><path d="M515 75 L560 75" class="mini-edge"/><path d="M630 75 L650 75" class="mini-edge"/><path d="M110 75 C240 15 450 15 560 65" class="mini-edge"/><path d="M245 75 C330 30 470 35 560 72" class="mini-edge"/><path d="M380 75 C450 45 505 52 560 80" class="mini-edge"/>\n        </svg>\n      </div>\n      <div class="detail">\n        <h3>Detection head per scale</h3>\n        <p>Each of P3/P4/P5 gets independent class and box-distribution branches. Anchor distributions are converted to vectors by softmax + fixed 1×1×1 Conv3d.</p>\n        <svg viewBox="0 0 720 190">\n          <defs><marker id="miniArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#64748b"/></marker></defs>\n          <rect x="20" y="70" width="90" height="50" class="mini-box mini-detect"/><text x="48" y="92" class="mini-title">P{3,4,5}</text><text x="39" y="109" class="mini-sub">feature map</text>\n          <rect x="160" y="25" width="125" height="50" class="mini-box mini-op"/><text x="178" y="47" class="mini-title">Anchor branch</text><text x="174" y="64" class="mini-sub">Conv, Conv(g), 1×1</text>\n          <rect x="330" y="25" width="120" height="50" class="mini-box mini-detect"/><text x="356" y="47" class="mini-title">Anchor2Vec</text><text x="348" y="64" class="mini-sub">softmax + Conv3d</text>\n          <rect x="500" y="25" width="120" height="50" class="mini-box mini-merge"/><text x="529" y="47" class="mini-title">Box vec</text><text x="532" y="64" class="mini-sub">4×H×W</text>\n          <rect x="160" y="115" width="125" height="50" class="mini-box mini-op"/><text x="181" y="137" class="mini-title">Class branch</text><text x="178" y="154" class="mini-sub">Conv, Conv, 1×1</text>\n          <rect x="330" y="115" width="120" height="50" class="mini-box mini-merge"/><text x="354" y="137" class="mini-title">Class logits</text><text x="365" y="154" class="mini-sub">3×H×W</text>\n          <path d="M110 95 C130 50 140 50 160 50" class="mini-edge"/><path d="M285 50 L330 50" class="mini-edge"/><path d="M450 50 L500 50" class="mini-edge"/><path d="M110 95 C130 140 140 140 160 140" class="mini-edge"/><path d="M285 140 L330 140" class="mini-edge"/>\n        </svg>\n      </div>\n    </div>\n  </div>\n\n'

    content = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>MDV6 MIT YOLOv9-c Network Graph</title>
<style>
  body {{ margin:0; background:#020617; color:#e2e8f0; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  .wrap {{ max-width: 1360px; margin: 0 auto; padding: 28px; }}
  h1 {{ margin:0 0 6px; font-size:26px; }}
  .muted {{ color:#94a3b8; margin-bottom:20px; }}
  .card {{ border:1px solid #1e293b; background:rgba(15,23,42,.72); border-radius:16px; padding:18px; box-shadow:0 20px 80px rgba(0,0,0,.35); }}
  svg {{ width:100%; height:auto; display:block; }}
  .title {{ fill:#f8fafc; font-weight:700; font-size:13px; }}
  .sub {{ fill:#cbd5e1; font-size:10px; }}
  .edge {{ fill:none; stroke:#64748b; stroke-width:1.4; opacity:.88; }}
  .section {{ fill:none; stroke-dasharray:8 5; opacity:.6; }}
  table {{ width:100%; border-collapse:collapse; margin-top:18px; font-size:12px; }}
  th, td {{ border-bottom:1px solid #1e293b; text-align:left; padding:8px 10px; vertical-align:top; }}
  th {{ color:#f8fafc; }} code {{ color:#bae6fd; white-space:normal; }}
  .legend span {{ display:inline-block; margin-right:18px; }} .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
  .detail-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap:16px; margin-top:18px; }}
  .detail {{ border:1px solid #1e293b; border-radius:14px; background:rgba(2,6,23,.45); padding:14px; }}
  .detail h3 {{ margin:0 0 4px; font-size:15px; color:#f8fafc; }}
  .detail p {{ margin:0 0 10px; color:#94a3b8; font-size:12px; line-height:1.45; }}
  .detail svg {{ border-radius:10px; background:rgba(15,23,42,.6); border:1px solid #1e293b; }}
  .mini-title {{ fill:#f8fafc; font-size:11px; font-weight:700; }}
  .mini-sub {{ fill:#cbd5e1; font-size:9px; }}
  .mini-edge {{ fill:none; stroke:#64748b; stroke-width:1.2; marker-end:url(#miniArrow); }}
  .mini-box {{ fill:#0f172a; stroke:#38bdf8; stroke-width:1.2; rx:6; }}
  .mini-op {{ fill:rgba(8,51,68,.7); stroke:#22d3ee; }}
  .mini-split {{ fill:rgba(120,53,15,.45); stroke:#fbbf24; }}
  .mini-merge {{ fill:rgba(6,78,59,.55); stroke:#34d399; }}
  .mini-detect {{ fill:rgba(76,29,149,.55); stroke:#a78bfa; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>MDV6 MIT YOLOv9-c Network Graph</h1>
  <div class="muted">Backbone → GELAN neck → P3/P4/P5 detection heads. ONNX companion is Netron-ready.</div>
  <div class="card">
  <svg viewBox="0 0 1280 1030" role="img" aria-label="MDV6 network graph">
    <defs>
      <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="0.5"/></pattern>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#64748b"/></marker>
    </defs>
    <rect width="1280" height="1030" fill="url(#grid)"/>
    <rect x="40" y="220" width="1140" height="300" rx="18" class="section" stroke="#22d3ee"/>
    <text x="55" y="245" class="sub">BACKBONE</text>
    <rect x="220" y="540" width="980" height="250" rx="18" class="section" stroke="#fbbf24"/>
    <text x="235" y="565" class="sub">NECK / FEATURE FUSION</text>
    <rect x="260" y="700" width="940" height="240" rx="18" class="section" stroke="#34d399"/>
    <text x="275" y="725" class="sub">HEAD</text>
    {''.join(edge_path(a,b) for a,b in edges)}
    {''.join(rect_node(n) for n in nodes)}
  </svg>
  <div class="legend">
    <span><i class="dot" style="background:#22d3ee"></i>Backbone</span>
    <span><i class="dot" style="background:#fbbf24"></i>Neck</span>
    <span><i class="dot" style="background:#34d399"></i>Head</span>
    <span><i class="dot" style="background:#a78bfa"></i>Detection</span>
  </div>
  </div>
  {details_html}
  <div class="card" style="margin-top:18px">
    <h2>Top-level module shape summary</h2>
    <p class="muted">Captured from a 1×3×640×640 dummy forward pass. Total top-level parameter count: {total_params:,}</p>
    <table><thead><tr><th>Module</th><th>Input</th><th>Output</th><th>Params</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
  </div>
  <script type="application/json" id="mdv6-shapes">{html.escape(json.dumps(shapes, indent=2))}</script>
</div>
</body>
</html>'''
    html_path.write_text(content)
    return html_path


def main():
    torch.manual_seed(42)
    model = MDV6MITYOLOv9c(num_classes=3).eval()
    dummy = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    print("Collecting top-level shapes...")
    shapes = collect_top_level_shapes(model, dummy)
    (OUT_DIR / "mdv6_shapes.json").write_text(json.dumps(shapes, indent=2))
    print("Exporting ONNX...")
    onnx_path = export_onnx(model)
    print("Writing HTML...")
    html_path = write_html(shapes)
    print(f"Wrote {onnx_path}")
    print(f"Wrote {html_path}")
    print(f"Wrote {OUT_DIR / 'mdv6_shapes.json'}")


if __name__ == "__main__":
    main()
