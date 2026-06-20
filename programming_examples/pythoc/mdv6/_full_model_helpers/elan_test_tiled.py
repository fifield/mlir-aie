#!/usr/bin/env python3
"""Test ELAN(64→64) at 160×160 using host-composed tiled fused conv sub-layers."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../python"))
import torch
from mdv6.layers import ELAN
import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime


def bf16_to_uint16(t):
    return t.view(torch.uint16).cpu().numpy()

def uint16_to_bf16(a):
    return torch.from_numpy(a.copy()).view(torch.bfloat16)

# Per-Module cache of packed uint16 buffers, one entry per (module, variant).
# WeakKeyDictionary so entries die with their Module — prevents stale hits when
# Python recycles object ids across forward passes (mlir-aie-woi: video stream
# inference re-instantiates the model each frame; id-keyed dicts return wrong
# weights from prior frame's freed modules).
import weakref
_FUSE_CACHE = weakref.WeakKeyDictionary()  # module -> {variant: uint16 array}

def fuse_bn(conv_module):
    """Return [conv_weights, fused_bn_w, fused_bn_b] as packed uint16."""
    variants = _FUSE_CACHE.setdefault(conv_module, {})
    if "plain" in variants:
        return variants["plain"]
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)
    out = bf16_to_uint16(torch.cat([
        conv_module.conv.weight.data.flatten(),
        gamma * inv_std,
        beta - gamma * mean * inv_std,
    ]))
    variants["plain"] = out
    return out


def _bn_affine(conv_module):
    """Return (W, B) such that BN(Conv(x)) == W[oc,ic,h,w]·x_conv + B[oc] in
    inference mode. Used by fuse_repconv to collapse the two parallel BN+Conv
    branches of a RepConv into a single 3x3 conv-with-bias."""
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)
    scale = gamma * inv_std                  # [oc]
    bias = beta - gamma * mean * inv_std     # [oc]
    W = conv_module.conv.weight.data * scale.view(-1, 1, 1, 1)
    return W, bias


def fuse_repconv(rep_module):
    """Reparameterize a RepConv module — SiLU(BN(3x3(x)) + BN(1x1(x))) — into
    a single 3x3 conv with bias, packed for the mc_*_rn3 kernel.

    At inference RepConv's two parallel BN+conv branches fold into one 3x3
    conv: ``W_fused = W3_eff + W1_eff_padded_to_3x3``, ``B_fused = B3_eff +
    B1_eff``. The kernel applies its own SiLU at the end, matching RepConv's
    outer ``self.act``. The bn_w slot is set to 1.0 since we already folded
    both BN scales into W_fused.

    Returns uint16 buffer in the same layout as ``fuse_bn``:
    [conv_OIHW_flat, bn_w=1, bn_b=B_fused].
    """
    variants = _FUSE_CACHE.setdefault(rep_module, {})
    if "rep" in variants:
        return variants["rep"]

    # conv1 is the 3x3 branch (Conv with bn + activation=None inside)
    W3, B3 = _bn_affine(rep_module.conv1)
    # conv2 is the 1x1 branch
    W1, B1 = _bn_affine(rep_module.conv2)

    oc, ic, kh, kw = W3.shape
    assert kh == 3 and kw == 3, f"RepConv.conv1 must be 3x3, got {kh}x{kw}"
    assert W1.shape == (oc, ic, 1, 1), \
        f"RepConv.conv2 must be 1x1, got {W1.shape}"

    W_fused = W3.clone()
    # 1x1 lifts to the center of the 3x3 kernel; pad-1 then makes the two
    # branches arithmetically equivalent (the boundary neighbors contribute 0).
    W_fused[:, :, 1, 1] += W1[:, :, 0, 0]
    B_fused = B3 + B1

    bn_w = torch.ones(oc, dtype=W_fused.dtype)
    out = bf16_to_uint16(torch.cat([W_fused.flatten(), bn_w, B_fused]))
    variants["rep"] = out
    return out


def fuse_bn_transposed(conv_module):
    """Return [transposed_conv_weights, fused_bn_w, fused_bn_b] as packed uint16.

    Conv weights are transposed from OC-major [oc][ic] to block layout
    [ic/8][oc/8][8ic][8oc] for contiguous vector loads in the AIE kernel.
    Only supports 1x1 convolutions.
    """
    variants = _FUSE_CACHE.setdefault(conv_module, {})
    if "transposed" in variants:
        return variants["transposed"]
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)

    # Original weight: (oc, ic, 1, 1) for 1x1 conv
    w = conv_module.conv.weight.data.squeeze(-1).squeeze(-1)  # (oc, ic)
    oc, ic = w.shape

    # Transpose to block layout: [ic/8][oc/8][8ic][8oc]
    w_blocks = w.reshape(oc // 8, 8, ic // 8, 8)  # [oc_blk, 8oc, ic_blk, 8ic]
    w_blocks = w_blocks.permute(2, 0, 3, 1)        # [ic_blk, oc_blk, 8ic, 8oc]
    w_transposed = w_blocks.contiguous().flatten()

    out = bf16_to_uint16(torch.cat([
        w_transposed,
        gamma * inv_std,
        beta - gamma * mean * inv_std,
    ]))
    variants["transposed"] = out
    return out

def fuse_bn_transposed_3x3(conv_module):
    """Return [packed_conv_weights, fused_bn_w, fused_bn_b] as packed uint16.

    Conv weights packed from [oc, ic, 3, 3] to [oc/8, ic/8, 9, 8ic, 8oc]
    for contiguous vector loads in the AIE 3x3 conv kernel.
    Each (oc_blk, ic_blk, kpos) has a contiguous 64-element block.
    """
    variants = _FUSE_CACHE.setdefault(conv_module, {})
    if "transposed_3x3" in variants:
        return variants["transposed_3x3"]
    eps = conv_module.bn.eps
    gamma = conv_module.bn.weight.data
    beta = conv_module.bn.bias.data
    mean = conv_module.bn.running_mean.data
    var = conv_module.bn.running_var.data
    inv_std = 1.0 / torch.sqrt(var + eps)

    # Original weight: (oc, ic, 3, 3)
    w = conv_module.conv.weight.data  # [oc, ic, 3, 3]
    oc, ic = w.shape[0], w.shape[1]

    # Reshape to [oc/8, 8oc, ic/8, 8ic, 9]
    w = w.reshape(oc // 8, 8, ic // 8, 8, 9)  # [oc_blk, 8oc, ic_blk, 8ic, 9]
    # Permute to [oc/8, ic/8, 9, 8ic, 8oc]
    w = w.permute(0, 2, 4, 3, 1).contiguous()  # [oc_blk, ic_blk, 9, 8ic, 8oc]

    out = bf16_to_uint16(torch.cat([
        w.flatten(),
        gamma * inv_std,
        beta - gamma * mean * inv_std,
    ]))
    variants["transposed_3x3"] = out
    return out


def extract_patch(image_hwc, tile_row, tile_col, tile_h, tile_w, stride=1, ks=3, pad=1):
    """Extract input patch for tiled conv."""
    H, W, C = image_hwc.shape
    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    in_start_h = tile_row * tile_h * stride - pad
    in_start_w = tile_col * tile_w * stride - pad
    patch = torch.zeros(patch_h, patch_w, C, dtype=image_hwc.dtype)
    vs_h = max(0, in_start_h); vs_w = max(0, in_start_w)
    ve_h = min(H, in_start_h + patch_h); ve_w = min(W, in_start_w + patch_w)
    po_h = vs_h - in_start_h; po_w = vs_w - in_start_w
    patch[po_h:po_h+(ve_h-vs_h), po_w:po_w+(ve_w-vs_w), :] = image_hwc[vs_h:ve_h, vs_w:ve_w, :]
    return patch


def extract_all_patches_u16(image_hwc, tiles_h, tiles_w, tile_h, tile_w,
                            stride=1, ks=3, pad=1):
    """Vectorized equivalent of looping extract_patch over the (tr, tc) grid.

    Returns a uint16 array of shape [tiles_h*tiles_w, patch_size] in row-major
    (tr, tc) order, where patch_size = patch_h*patch_w*C padded up to even.
    Bit-exact with:
        for tr in range(tiles_h):
          for tc in range(tiles_w):
            p = extract_patch(image_hwc, tr, tc, tile_h, tile_w, stride, ks, pad)
            u = bf16_to_uint16(p.flatten()); pad to even -> row.
    """
    H, W, C = image_hwc.shape
    patch_h = (tile_h - 1) * stride + ks
    patch_w = (tile_w - 1) * stride + ks
    patch_size_raw = patch_h * patch_w * C
    patch_size = patch_size_raw + (patch_size_raw % 2)
    n_tiles = tiles_h * tiles_w

    # View image as uint16 (bitcast bf16 -> uint16), contiguous HWC.
    img_u16 = bf16_to_uint16(image_hwc.contiguous())  # [H, W, C] uint16

    # Zero-pad with a margin large enough that every patch window lies inside.
    # in_start = t * tile * stride - pad ; window length = patch_h/patch_w.
    # Max needed extent in H = (tiles_h-1)*tile_h*stride - pad + patch_h.
    # We pad symmetrically by `pad` on the low side and enough on the high side.
    step_h = tile_h * stride
    step_w = tile_w * stride
    # Low pad = `pad` (in_start of tile 0 is -pad). In padded coords the window
    # origin for tile tr is tr*step_h (and tc*step_w). The last origin is
    # (tiles_h-1)*step_h; the window needs that origin + patch_h rows present.
    pad_lo_h = pad
    pad_lo_w = pad
    need_h = (tiles_h - 1) * step_h + patch_h   # exclusive end into padded coords
    need_w = (tiles_w - 1) * step_w + patch_w
    pad_hi_h = max(0, need_h - (H + pad_lo_h))
    pad_hi_w = max(0, need_w - (W + pad_lo_w))

    padded = np.zeros((H + pad_lo_h + pad_hi_h, W + pad_lo_w + pad_hi_w, C),
                      dtype=np.uint16)
    padded[pad_lo_h:pad_lo_h + H, pad_lo_w:pad_lo_w + W, :] = img_u16

    # For tile (tr, tc): in_start_h = tr*step_h - pad ; in padded coords this is
    # tr*step_h - pad + pad_lo_h = tr*step_h (since pad_lo_h == pad). Good.
    # Gather windows via sliding_window_view stepped by (step_h, step_w).
    # sliding_window_view over (axis 0, axis 1) with window (patch_h, patch_w).
    swv = np.lib.stride_tricks.sliding_window_view(
        padded, (patch_h, patch_w), axis=(0, 1))
    # swv shape: [Hp-patch_h+1, Wp-patch_w+1, C, patch_h, patch_w]
    # Select the strided tile origins.
    win = swv[:tiles_h * step_h:step_h, :tiles_w * step_w:step_w]
    # win shape: [tiles_h, tiles_w, C, patch_h, patch_w]
    # extract_patch flattens patch as (patch_h, patch_w, C) row-major.
    # Transpose C to last: -> [tiles_h, tiles_w, patch_h, patch_w, C]
    win = np.transpose(win, (0, 1, 3, 4, 2))
    patches = np.ascontiguousarray(win).reshape(n_tiles, patch_size_raw)

    if patch_size != patch_size_raw:
        out = np.zeros((n_tiles, patch_size), dtype=np.uint16)
        out[:, :patch_size_raw] = patches
        return out
    return patches


def pack_input_batch_u16(all_patches_2d, batch_start, patches_per_call):
    """Build one input BO's flat uint16 buffer for a spatial batch.

    all_patches_2d: [n_tiles, patch_size] uint16 (row-major tr,tc order).
    Returns concat of patches [batch_start : batch_start+patches_per_call] in
    j-order, with trailing slots (past the real patch count in this batch)
    padded by repeating this batch's first patch — bit-exact with the old
    per-core concatenation (core*ppc+slot == j makes per-core grouping the
    identity in j).
    """
    n_patches = all_patches_2d.shape[0]
    patch_size = all_patches_2d.shape[1]
    batch_end = min(batch_start + patches_per_call, n_patches)
    real = batch_end - batch_start
    buf = np.empty((patches_per_call, patch_size), dtype=np.uint16)
    buf[:real] = all_patches_2d[batch_start:batch_end]
    if real < patches_per_call:
        buf[real:] = all_patches_2d[batch_start]  # this batch's slot-0 patch
    return buf.reshape(-1)


def reassemble_output_hwc(big_out_u16, n_ocb, tiles_h, tiles_w,
                          tile_h, tile_w, oc_block, ppc,
                          out_h, out_w, out_ch, output_tile_size,
                          output_per_batch):
    """Vectorized reassembly of a flat uint16 output buffer into HWC bf16.

    big_out_u16: flat uint16 of length n_ocb * output_per_batch (or a single
        OCB slice of length output_per_batch when n_ocb==1 paths pass per-OCB).
    Layout: for ocb, for j in [0, tiles_h*tiles_w): tile j occupies
        ocb*output_per_batch + j*output_tile_size (since core*ppc+slot == j),
        as [tile_h, tile_w, oc_block] row-major. Padded trailing slots ignored.

    Returns torch.bfloat16 tensor [out_h, out_w, out_ch].
    """
    n_tiles = tiles_h * tiles_w
    # Slice off only the real tiles per OCB (drop slot-0 padding past n_tiles).
    big = big_out_u16.reshape(n_ocb, output_per_batch)
    real = big[:, :n_tiles * output_tile_size]
    # [n_ocb, n_tiles, tile_h, tile_w, oc_block]
    real = real.reshape(n_ocb, tiles_h, tiles_w, tile_h, tile_w, oc_block)

    edge = (out_h % tile_h != 0) or (out_w % tile_w != 0) or \
           (out_ch % oc_block != 0)
    out_t = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)

    if not edge:
        # Perfect tiling: [n_ocb, th, tw, tile_h, tile_w, oc_block]
        #   -> [th*tile_h, tw*tile_w, n_ocb*oc_block]
        arr = np.transpose(real, (1, 3, 2, 4, 0, 5))  # th,tile_h,tw,tile_w,ocb,ocblk
        arr = np.ascontiguousarray(arr).reshape(out_h, out_w, out_ch)
        out_t[:] = uint16_to_bf16(arr).reshape(out_h, out_w, out_ch)
        return out_t

    # Edge path: place each tile, clipping to valid extent. Done with numpy
    # slicing per OCB (no per-tile python loop over n_tiles).
    out_u16 = bf16_to_uint16(out_t.contiguous())  # [out_h, out_w, out_ch] zeros
    for ocb in range(n_ocb):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        # Build a full padded canvas [tiles_h*tile_h, tiles_w*tile_w, oc_block]
        # then crop to [out_h, out_w, actual_oc].
        canvas = np.transpose(real[ocb], (0, 2, 1, 3, 4))  # th,tile_h,tw,tile_w,ocblk
        canvas = np.ascontiguousarray(canvas).reshape(
            tiles_h * tile_h, tiles_w * tile_w, oc_block)
        out_u16[:out_h, :out_w, oc_start:oc_end] = \
            canvas[:out_h, :out_w, :actual_oc]
    out_t[:] = uint16_to_bf16(out_u16).reshape(out_h, out_w, out_ch)
    return out_t


def pack_gemm_input_batch_u16(input_flat_u16, batch_start, batch_pixels,
                              total_slots, tile_m, input_size):
    """Build one GEMM input BO's flat uint16 buffer for a pixel batch.

    input_flat_u16: [M, IC] uint16 (bitcast of the bf16 input pixels).
    Take rows [batch_start : batch_start+batch_pixels], lay them out into
    ``total_slots`` slots of ``tile_m`` rows each (input_size == tile_m*IC),
    zero-fill any unused rows in the last active slot, then pad the trailing
    (fully inactive) slots by repeating this batch's slot-0 (first tile_m
    rows). Bit-exact with the old per-slot Python loop in
    _run_gemm_oc_blocked_merged / _run_gemm_kblocked_merged.
    """
    IC = input_flat_u16.shape[1]
    buf = np.zeros((total_slots, tile_m, IC), dtype=np.uint16)
    rows = input_flat_u16[batch_start:batch_start + batch_pixels]  # [batch_pixels, IC]
    n_active_slots = (batch_pixels + tile_m - 1) // tile_m
    # Full slots: contiguous blocks of tile_m rows.
    full = batch_pixels // tile_m
    if full:
        buf[:full] = rows[:full * tile_m].reshape(full, tile_m, IC)
    # Partial trailing active slot (if any).
    rem = batch_pixels - full * tile_m
    if rem:
        buf[full, :rem] = rows[full * tile_m:]
    # Pad fully-inactive trailing slots with this batch's slot 0.
    if n_active_slots < total_slots:
        buf[n_active_slots:] = buf[0]
    return buf.reshape(-1)


def reassemble_gemm_output(out_data_u16, batch_start, batch_pixels,
                           total_slots, tile_m, out_ch, output_size,
                           output_flat):
    """Write a GEMM output BO back into output_flat[batch_start:batch_end].

    out_data_u16: flat uint16 of length total_slots*output_size (output_size ==
    tile_m*out_ch). Reshape to [total_slots, tile_m, out_ch], take the first
    batch_pixels rows in slot-major row-major order, convert to bf16, write
    contiguously. Bit-exact with the old per-slot output loop (only active
    pixels written; trailing slot-0 padding ignored).
    """
    out_t = out_data_u16.reshape(total_slots, tile_m, out_ch)
    flat = out_t.reshape(total_slots * tile_m, out_ch)[:batch_pixels]
    tile_out = uint16_to_bf16(np.ascontiguousarray(flat).reshape(-1))
    tile_out = tile_out.reshape(batch_pixels, out_ch)
    output_flat[batch_start:batch_start + batch_pixels, :] = \
        tile_out.to(torch.bfloat16)


def run_tiled_fused_conv(kernel_handle, input_hwc, weights_uint16,
                          out_h, out_w, out_ch, tile_h, tile_w, oc_block,
                          stride=1, kernel_size=3, padding=1):
    """Run a full tiled fused conv layer, returning output HWC tensor."""
    H, W, C = input_hwc.shape
    tiles_h = (out_h + tile_h - 1) // tile_h
    tiles_w = (out_w + tile_w - 1) // tile_w
    n_oc_blocks = (out_ch + oc_block - 1) // oc_block
    output = torch.zeros(out_h, out_w, out_ch, dtype=torch.bfloat16)
    patch_h = (tile_h - 1) * stride + kernel_size
    patch_w = (tile_w - 1) * stride + kernel_size
    patch_size_raw = patch_h * patch_w * C
    patch_size = patch_size_raw + (patch_size_raw % 2)
    output_tile_size = tile_h * tile_w * oc_block
    conv_wt_size = oc_block * C * kernel_size * kernel_size

    # Unpack full weight array: [all_conv_wts (oc*ic*ks*ks), all_bn_w (oc), all_bn_b (oc)]
    total_conv_wts = out_ch * input_hwc.shape[2] * kernel_size * kernel_size
    all_conv_wts = weights_uint16[:total_conv_wts]
    all_bn_w = weights_uint16[total_conv_wts:total_conv_wts + out_ch]
    all_bn_b = weights_uint16[total_conv_wts + out_ch:total_conv_wts + 2 * out_ch]

    for ocb in range(n_oc_blocks):
        oc_start = ocb * oc_block
        oc_end = min(oc_start + oc_block, out_ch)
        actual_oc = oc_end - oc_start
        # Extract per-block weights: [conv_wts_block, bn_w_block, bn_b_block]
        cw_per_oc = input_hwc.shape[2] * kernel_size * kernel_size
        conv_block = all_conv_wts[oc_start * cw_per_oc:oc_end * cw_per_oc]
        bn_w_block = all_bn_w[oc_start:oc_end]
        bn_b_block = all_bn_b[oc_start:oc_end]
        wt_block = np.concatenate([conv_block, bn_w_block, bn_b_block])
        # Pad to expected size if needed
        expected = conv_wt_size + 2 * oc_block
        if len(wt_block) < expected:
            wt_block = np.pad(wt_block, (0, expected - len(wt_block)))

        for tr in range(tiles_h):
            for tc in range(tiles_w):
                patch = extract_patch(input_hwc, tr, tc, tile_h, tile_w,
                                       stride, kernel_size, padding)
                patch_u16 = bf16_to_uint16(patch.flatten())
                if len(patch_u16) < patch_size:
                    patch_u16 = np.pad(patch_u16, (0, patch_size - len(patch_u16)))

                in1 = iron.tensor(patch_u16, dtype=np.uint16)
                in2 = iron.tensor(wt_block, dtype=np.uint16)
                out = iron.zeros(output_tile_size, dtype=np.uint16)
                DefaultNPURuntime.run(kernel_handle, [in1, in2, out])
                tile_out = uint16_to_bf16(out.numpy()[:output_tile_size].copy())
                tile_out = tile_out.reshape(tile_h, tile_w, oc_block)

                oh_s = tr * tile_h; ow_s = tc * tile_w
                oh_e = min(oh_s + tile_h, out_h); ow_e = min(ow_s + tile_w, out_w)
                output[oh_s:oh_e, ow_s:ow_e, oc_start:oc_end] = \
                    tile_out[:oh_e-oh_s, :ow_e-ow_s, :actual_oc]
    return output


def main():
    H, W = 160, 160
    ic, oc = 64, 64
    part_ch = 64
    proc_ch = 32

    print(f"\nTesting ELAN({ic}→{oc}) at {H}×{W} on NPU (host-composed tiled fused)")

    layer = ELAN(ic, oc, part_ch, proc_ch).eval().to(torch.bfloat16)
    torch.manual_seed(42)
    x = torch.randn(1, ic, H, W, dtype=torch.bfloat16)
    with torch.no_grad():
        ref = layer(x)
    print(f"PyTorch ref: {ref.shape}, range [{ref.min():.4f}, {ref.max():.4f}]")

    bd = os.path.join(os.path.dirname(__file__), "..", "conv", "build")

    # Load kernel handles
    kh_conv1 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv1.xclbin", f"{bd}/tf_elan_conv1.bin"))
    kh_conv3 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv3x3.xclbin", f"{bd}/tf_elan_conv3x3.bin"))
    kh_conv4 = DefaultNPURuntime.load(NPUKernel(f"{bd}/tf_elan_conv4.xclbin", f"{bd}/tf_elan_conv4.bin"))

    input_hwc = x.squeeze(0).permute(1, 2, 0).contiguous()
    t0 = time.time()

    # Stage 1: Conv1 (1x1, 64→64)
    print("  Conv1 (1x1 64→64)...", end=" ", flush=True)
    wts1 = fuse_bn(layer.conv1)
    conv1_out = run_tiled_fused_conv(kh_conv1, input_hwc, wts1,
                                      H, W, part_ch, 8, 8, 64, stride=1, kernel_size=1, padding=0)
    print("done")

    # Split: x1=first 32ch, x2=last 32ch
    x1 = conv1_out[:, :, :proc_ch]
    x2 = conv1_out[:, :, proc_ch:]

    # Stage 2: Conv2 (3x3, 32→32) on x2
    print("  Conv2 (3x3 32→32)...", end=" ", flush=True)
    wts2 = fuse_bn(layer.conv2)
    x3 = run_tiled_fused_conv(kh_conv3, x2, wts2,
                               H, W, proc_ch, 16, 16, 32, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 3: Conv3 (3x3, 32→32) on x3
    print("  Conv3 (3x3 32→32)...", end=" ", flush=True)
    wts3 = fuse_bn(layer.conv3)
    x4 = run_tiled_fused_conv(kh_conv3, x3, wts3,
                               H, W, proc_ch, 16, 16, 32, stride=1, kernel_size=3, padding=1)
    print("done")

    # Stage 4: Concat [x1, x2, x3, x4] → Conv4 (1x1, 128→64)
    concat = torch.cat([x1, x2, x3, x4], dim=2)  # 128ch
    print("  Conv4 (1x1 128→64)...", end=" ", flush=True)
    wts4 = fuse_bn(layer.conv4)
    result = run_tiled_fused_conv(kh_conv4, concat, wts4,
                                   H, W, oc, 8, 8, 64, stride=1, kernel_size=1, padding=0)
    print("done")

    total = time.time() - t0
    print(f"\n  Total time: {total:.1f}s")

    aie_nchw = result.float().permute(2, 0, 1).unsqueeze(0)
    print(f"AIE output range: [{aie_nchw.min():.4f}, {aie_nchw.max():.4f}]")

    diff = torch.abs(ref.float() - aie_nchw)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    tol = 0.5
    if max_diff < tol:
        print(f"✓ PASS (max diff < {tol})")
    else:
        print(f"✗ FAIL (max diff >= {tol})")
    return max_diff < tol


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
