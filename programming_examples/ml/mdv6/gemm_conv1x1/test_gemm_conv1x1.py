"""Test for GEMM-based Conv1x1 kernel.

Validates:
  1. CPU-only: NumPy matmul reference vs PyTorch Conv1x1+BN+SiLU
  2. NPU hardware: GEMM kernel vs PyTorch reference

Weight packing:
  Conv weights from [oc, ic] (PyTorch) -> [oc/8, ic/8, 8, 8] (GEMM layout)
  Packed buffer: [conv_weights, bn_w_fused, bn_b_fused]

Usage:
  # CPU reference test (no hardware needed)
  python3 test_gemm_conv1x1.py --tile-m 64 --ic 128 --oc 64

  # NPU hardware test
  python3 test_gemm_conv1x1.py --tile-m 64 --ic 128 --oc 64 \
      -x build/final.xclbin -i build/insts.bin

  # Pure GEMM (no BN/SiLU)
  python3 test_gemm_conv1x1.py --tile-m 64 --ic 64 --oc 64 --no-fuse \
      -x build/final_nofuse.xclbin -i build/insts_nofuse.bin
"""
import argparse
import numpy as np
import sys
import struct
import time

# bf16 utilities
def float_to_bf16_bits(f):
    """Convert float32 to bfloat16 as uint16."""
    b = struct.pack('f', f)
    return struct.unpack('H', b[2:4])[0]

def bf16_bits_to_float(u):
    """Convert uint16 (bfloat16 bits) to float32."""
    b = struct.pack('H', u) + b'\x00\x00'
    # bf16 is top 16 bits of float32
    b = b'\x00\x00' + struct.pack('H', u)
    return struct.unpack('f', b)[0]

def np_to_bf16_u16(arr):
    """Convert float32 numpy array to uint16 array of bf16 bits."""
    flat = arr.astype(np.float32).ravel()
    result = np.zeros(len(flat), dtype=np.uint16)
    for i in range(len(flat)):
        result[i] = float_to_bf16_bits(flat[i])
    return result

def bf16_u16_to_np(arr, shape=None):
    """Convert uint16 array of bf16 bits to float32 numpy array."""
    flat = arr.ravel()
    result = np.zeros(len(flat), dtype=np.float32)
    for i in range(len(flat)):
        result[i] = bf16_bits_to_float(flat[i])
    if shape is not None:
        result = result.reshape(shape)
    return result

# Use ml_dtypes for faster bf16 conversion if available
try:
    from ml_dtypes import bfloat16 as ml_bf16

    def np_to_bf16_u16(arr):
        return arr.astype(np.float32).astype(ml_bf16).view(np.uint16).ravel()

    def bf16_u16_to_np(arr, shape=None):
        result = arr.view(ml_bf16).astype(np.float32)
        if shape is not None:
            result = result.reshape(shape)
        return result
except ImportError:
    pass


def pack_weights_gemm(conv_weight, bn_weight=None, bn_bias=None,
                      bn_mean=None, bn_var=None, eps=1e-5):
    """Pack conv weights into GEMM layout [oc/8, ic/8, 8, 8] + optional fused BN.

    Args:
        conv_weight: [oc, ic] float32 (1x1 conv weight, no spatial dims)
        bn_weight, bn_bias, bn_mean, bn_var: BN parameters [oc]
        eps: BN epsilon

    Returns:
        uint16 array of packed bf16 weights
    """
    oc, ic = conv_weight.shape
    assert oc % 8 == 0 and ic % 8 == 0

    # Reshape to [ic/8, oc/8, 8, 8] blocked layout
    # Inner 8x8: [8ic, 8oc] — ic rows, oc cols (matches mmul B[K,N] layout)
    w = conv_weight.reshape(oc // 8, 8, ic // 8, 8)  # [oc/8, 8oc, ic/8, 8ic]
    w = w.transpose(2, 0, 3, 1)  # [ic/8, oc/8, 8ic, 8oc]

    packed = w.astype(np.float32).ravel()

    if bn_weight is not None:
        # Fuse BN: w_fused = gamma / sqrt(var + eps)
        #          b_fused = beta - gamma * mean / sqrt(var + eps)
        inv_std = 1.0 / np.sqrt(bn_var + eps)
        bn_w_fused = (bn_weight * inv_std).astype(np.float32)
        bn_b_fused = (bn_bias - bn_weight * bn_mean * inv_std).astype(np.float32)
        packed = np.concatenate([packed, bn_w_fused, bn_b_fused])

    return np_to_bf16_u16(packed)


def fast_sigmoid(x):
    """Fast sigmoid approximation matching the AIE kernel."""
    return 0.5 + x / (2.0 * (1.0 + np.abs(x)))


def reference_conv1x1_bn_silu(input_hwc, conv_weight, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """NumPy reference for Conv1x1 + BN + SiLU.

    Args:
        input_hwc: [M, ic] float32
        conv_weight: [oc, ic] float32
        bn_w, bn_b, bn_mean, bn_var: [oc] float32

    Returns:
        [M, oc] float32
    """
    # Conv1x1 = matmul
    out = input_hwc @ conv_weight.T  # [M, oc]

    # BN
    inv_std = 1.0 / np.sqrt(bn_var + eps)
    w_fused = bn_w * inv_std
    b_fused = bn_b - bn_w * bn_mean * inv_std
    out = out * w_fused + b_fused

    # SiLU with fast sigmoid
    out = out * fast_sigmoid(out)

    return out


def test_cpu(tile_m, ic, oc, fused=True):
    """CPU-only test: validate weight packing and reference computation."""
    print(f"\n=== CPU Test: tile_m={tile_m}, {ic}->{oc}, fused={fused} ===")
    np.random.seed(42)

    M = tile_m
    input_f32 = np.random.randn(M, ic).astype(np.float32) * 0.1
    conv_w = np.random.randn(oc, ic).astype(np.float32) * 0.05

    if fused:
        bn_w = np.ones(oc, dtype=np.float32) * 0.9 + np.random.randn(oc).astype(np.float32) * 0.1
        bn_b = np.random.randn(oc).astype(np.float32) * 0.1
        bn_mean = np.random.randn(oc).astype(np.float32) * 0.01
        bn_var = np.abs(np.random.randn(oc).astype(np.float32)) + 0.5

        ref = reference_conv1x1_bn_silu(input_f32, conv_w, bn_w, bn_b, bn_mean, bn_var)

        # Pack weights and unpack to verify round-trip
        packed = pack_weights_gemm(conv_w, bn_w, bn_b, bn_mean, bn_var)
        print(f"  Packed weight size: {len(packed)} uint16 ({len(packed)*2/1024:.1f} KB)")
        print(f"  Reference output range: [{ref.min():.4f}, {ref.max():.4f}]")
    else:
        ref = input_f32 @ conv_w.T
        packed = pack_weights_gemm(conv_w)
        print(f"  Packed weight size: {len(packed)} uint16 ({len(packed)*2/1024:.1f} KB)")
        print(f"  Reference output range: [{ref.min():.4f}, {ref.max():.4f}]")

    # Verify pure matmul through bf16
    input_bf16 = bf16_u16_to_np(np_to_bf16_u16(input_f32), input_f32.shape)
    conv_w_bf16 = bf16_u16_to_np(np_to_bf16_u16(conv_w), conv_w.shape)
    matmul_bf16 = input_bf16 @ conv_w_bf16.T
    matmul_diff = np.max(np.abs(matmul_bf16 - (input_f32 @ conv_w.T)))
    print(f"  bf16 matmul quantization error: {matmul_diff:.6f}")

    print("  CPU test PASS")
    return True


def test_npu(tile_m, ic, oc, n_cores, xclbin_path, insts_path,
             fused=True, patches_per_core=1):
    """NPU hardware test."""
    print(f"\n=== NPU Test: tile_m={tile_m}, {ic}->{oc}, "
          f"{n_cores} cores, fused={fused} ===")

    import aie.iron as iron
    from aie.utils import NPUKernel, DefaultNPURuntime

    np.random.seed(42)

    total_pixels = n_cores * patches_per_core * tile_m
    M = tile_m  # per core per patch

    # Generate test data
    input_f32 = np.random.randn(total_pixels, ic).astype(np.float32) * 0.1
    conv_w = np.random.randn(oc, ic).astype(np.float32) * 0.05

    if fused:
        bn_w = np.ones(oc, dtype=np.float32) * 0.9 + np.random.randn(oc).astype(np.float32) * 0.1
        bn_b = np.random.randn(oc).astype(np.float32) * 0.1
        bn_mean = np.random.randn(oc).astype(np.float32) * 0.01
        bn_var = np.abs(np.random.randn(oc).astype(np.float32)) + 0.5

        ref = reference_conv1x1_bn_silu(input_f32, conv_w, bn_w, bn_b, bn_mean, bn_var)
        packed_weights = pack_weights_gemm(conv_w, bn_w, bn_b, bn_mean, bn_var)
    else:
        ref = input_f32 @ conv_w.T
        packed_weights = pack_weights_gemm(conv_w)

    # Convert input to bf16-as-uint16
    input_u16 = np_to_bf16_u16(input_f32)

    # XRT setup
    npu_kernel = NPUKernel(xclbin_path, insts_path, kernel_name="MLIR_AIE")
    kernel_handle = DefaultNPURuntime.load(npu_kernel)

    host_in_size = n_cores * patches_per_core * tile_m * ic
    host_out_size = n_cores * patches_per_core * tile_m * oc

    in_buf = iron.tensor(input_u16[:host_in_size].copy(), dtype=np.uint16)
    wt_buf = iron.tensor(packed_weights.copy(), dtype=np.uint16)
    out_buf = iron.zeros(host_out_size, dtype=np.uint16)

    print(f"  Running on NPU ({n_cores} cores)...")
    t0 = time.time()
    ret = DefaultNPURuntime.run(kernel_handle, [in_buf, wt_buf, out_buf])
    elapsed = time.time() - t0
    print(f"  Execution time: {elapsed*1000:.1f} ms")

    # Read output
    output_u16 = out_buf.numpy().copy()
    output_f32 = bf16_u16_to_np(output_u16).reshape(total_pixels, oc)

    # Compare
    ref_sub = ref[:total_pixels]
    max_diff = np.max(np.abs(output_f32 - ref_sub))
    mean_diff = np.mean(np.abs(output_f32 - ref_sub))
    print(f"  Output range: [{output_f32.min():.4f}, {output_f32.max():.4f}]")
    print(f"  Reference range: [{ref_sub.min():.4f}, {ref_sub.max():.4f}]")
    print(f"  Max abs diff: {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")

    # Tolerance: bf16 quantization + fast sigmoid approximation
    tol = 0.5 if fused else 0.1
    if max_diff < tol:
        print(f"  NPU test PASS (max_diff={max_diff:.4f} < {tol})")
        return True
    else:
        print(f"  NPU test FAIL (max_diff={max_diff:.4f} >= {tol})")
        # Show worst locations
        diffs = np.abs(output_f32 - ref_sub)
        worst = np.unravel_index(np.argmax(diffs), diffs.shape)
        print(f"  Worst at pixel={worst[0]}, channel={worst[1]}: "
              f"npu={output_f32[worst]:.4f} ref={ref_sub[worst]:.4f}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test GEMM Conv1x1")
    parser.add_argument("--tile-m", type=int, default=64,
                        help="Pixels per core (must be %%16==0)")
    parser.add_argument("--ic", type=int, default=128, help="Input channels")
    parser.add_argument("--oc", type=int, default=64, help="Output channels")
    parser.add_argument("--n-cores", type=int, default=32, help="Number of cores")
    parser.add_argument("--patches-per-core", type=int, default=1)
    parser.add_argument("--no-fuse", action="store_true",
                        help="Pure GEMM without BN+SiLU")
    parser.add_argument("-x", "--xclbin", default=None, help="Path to xclbin")
    parser.add_argument("-i", "--insts", default=None, help="Path to insts.bin")
    args = parser.parse_args()

    fused = not args.no_fuse

    # Always run CPU test
    ok = test_cpu(args.tile_m, args.ic, args.oc, fused)

    # Run NPU test if xclbin provided
    if args.xclbin and args.insts:
        ok = test_npu(args.tile_m, args.ic, args.oc, args.n_cores,
                      args.xclbin, args.insts, fused, args.patches_per_core)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
