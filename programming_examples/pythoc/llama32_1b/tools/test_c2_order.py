import os, sys, time
import numpy as np
from ml_dtypes import bfloat16
L = "/home/jfifield/npu-dev-pythoc/mlir-aie/programming_examples/pythoc/llama32_1b"
sys.path.insert(0, L); os.chdir(L + "/build_peano")
from kernel_builder.cache import KernelCache
EMB, HID, KV = 2048, 8192, 512
z = lambda n: np.zeros(n, dtype=bfloat16)
ogf = [z((EMB,EMB)), z(EMB), z(EMB), z(EMB), z(EMB), z(EMB), z(EMB),
       z((HID,EMB)), z(HID), z((HID,EMB)), z(HID), z(HID), z((EMB,HID)), z(EMB), z(EMB)]
rgr = [z(EMB), z(EMB), z(EMB), z((EMB,EMB)), z(EMB), z((KV,EMB)), z(KV),
       z((KV,EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]
cache = KernelCache(cache_dir="decode_kernel_cache", verbose=False)
cache.load_manifest()
try:
    cache.load_and_run("rms_gemv_rope", None, *rgr, output_indices=[11,12], bo_key="s")
    print("sacrifice ok", flush=True)
except RuntimeError:
    print("sacrifice consumed wedge", flush=True)
for i in range(3):
    cache.load_and_run("rms_gemv_rope", None, *rgr, output_indices=[11,12], bo_key="r")
    print(f"rgr{i} ok", flush=True)
    cache.load_and_run("o_gemv_ffn", None, *ogf, output_indices=[14], bo_key="o")
    print(f"ogf{i} ok", flush=True)
print("PASS")
