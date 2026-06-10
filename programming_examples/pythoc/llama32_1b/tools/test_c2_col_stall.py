import os, sys
import numpy as np
from ml_dtypes import bfloat16
L = "/home/jfifield/npu-dev-pythoc/mlir-aie/programming_examples/pythoc/llama32_1b"
sys.path.insert(0, L); os.chdir(L + "/build_peano")
from kernel_builder.cache import KernelCache
import pyxrt as xrt
EMB, HID, KV = 2048, 8192, 512
rng = np.random.default_rng(7)
rand = lambda s: (rng.standard_normal(s)*0.05+0.05).astype(bfloat16)
z = lambda n: np.zeros(n, dtype=bfloat16)
ogf = [rand((EMB,EMB)), rand(EMB), z(EMB), rand(EMB), z(EMB), rand(EMB), z(EMB),
       rand((HID,EMB)), z(HID), rand((HID,EMB)), z(HID), z(HID), rand((EMB,HID)), z(EMB), z(EMB)]
rgr = [z(EMB), z(EMB), z(EMB), z((EMB,EMB)), z(EMB), z((KV,EMB)), z(KV),
       z((KV,EMB)), z(KV), z(64), z(64), z(EMB), z(KV)]
cache = KernelCache(cache_dir="decode_kernel_cache", verbose=False)
cache.load_manifest()
try:
    cache.load_and_run("rms_gemv_rope", None, *rgr, output_indices=[11,12], bo_key="s")
    print("sacrifice ok", flush=True)
except RuntimeError:
    print("sacrifice consumed wedge", flush=True)
cache.load_and_run("rms_gemv_rope", None, *rgr, output_indices=[11,12], bo_key="r")
print("rgr ok", flush=True)
try:
    cache.load_and_run("o_gemv_ffn", None, *ogf, output_indices=[14], bo_key="o")
    print("ogf COMPLETED", flush=True)
except RuntimeError:
    print("ogf TIMEOUT", flush=True)
bos = cache._cached_bos["o"]
for i, n in {2:"proj", 4:"res1", 8:"gate", 10:"up", 11:"swiglu", 13:"down"}.items():
    bos[i].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    a = np.frombuffer(bos[i].map(), dtype=bfloat16, count=ogf[i].size)
    cols = [int(np.count_nonzero(a[c*(a.size//8):(c+1)*(a.size//8)])) for c in range(8)]
    print(f"{n}: {cols}")
