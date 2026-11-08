# Laya on the AMD NPU

Laya, a small question-answering model, running on the AMD NPU (XDNA 2) with kernels written in PythoC and MLIR-AIE.

- [`laya_npu/`](laya_npu/README.md): the NPU implementation. It builds every core, packed weight blob and full-ELF graph from source and the fetched model checkpoint, stages a runtime package, and has unit, build and NPU tests.
- [`laya_local/`](laya_local/README.md): a local `Client` API and HTTP server on top of the staged runtime, with a support-desk demo page and a Snake demo.

## Quick start

From this directory, with the repository environment active:

```bash
source <npu-dev>/env.sh
python laya_npu/fetch_model.py
python laya_npu/rebuild.py --output /tmp/laya_npu-build          # ~30 min
python laya_npu/stage_runtime.py --build /tmp/laya_npu-build --output laya_npu/staged/current
python -m laya_local.server --port 8000                          # open http://127.0.0.1:8000/
```
