# MDV6 per-layer PythoC prototypes

Standalone single-file PythoC + IRON examples, one per MDV6 layer class.
Each prototype reimplements a layer end-to-end (host driver + kernel)
in a single Python file using inline PythoC kernels — no external C++
`.cc` build step.

**These are reference material, not part of the production path.** The
production full-model dispatch (`test_full_model_mc.py` →
`run_tiled_mc.py` → `conv/` + `gemm_conv1x1/` + `kernels/`) does not
import or depend on anything here.

For the conversion methodology that produced these, see
[`../CONVERSION_PATTERN.md`](../CONVERSION_PATTERN.md). The canonical
worked example is `elementwise/elementwise_pythoc.py`.

## What's here

Each subdir is a port of the older
`programming_examples/ml/mdv6/<layer>/{aie2.py, <layer>_bf16.cc}` two-file
pair into one PythoC file.

| Subdir | Layer |
|---|---|
| `aconv/` | AvgPool2d + Conv3x3 (stride 2) + BN + SiLU |
| `batchnorm_silu/` | Standalone BN + SiLU |
| `bottleneck/` | RepConv + Conv3x3 + residual |
| `elan/` | ELAN block (c1 + c3 split/merge + c4) |
| `elementwise/` | add / mul / max (the canonical conversion example) |
| `repconv/` | RepConv (parallel 3x3 + 1x1 branches + BN + SiLU) |
| `repncsp/` | RepNCSP block (rn1 + bottleneck × N + rnm) |
| `repncsp_elan/` | RepNCSPELAN (c1 + 2× RepNCSP + c3 + c4) |
| `sppelan/` | Spatial Pyramid Pooling + ELAN |
