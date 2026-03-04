# CERT Conversion Status

This document tracks which AIEX operations are converted to CERT format by the `--aie-npu-to-cert` pass.

## Converted Operations

The following NPU operations are successfully converted to CERT format:

| NPU Operation | CERT Operation | Status |
|--------------|----------------|--------|
| `aiex.npu.write32` | `aiex.cert.write32` | ✅ Converted |
| `aiex.npu.maskwrite32` | `aiex.cert.maskwrite32` | ✅ Converted |
| `aiex.npu.blockwrite` | `aiex.cert.uc_dma_write_des_sync` + `aiex.cert.uc_dma_chain` | ✅ Converted |
| `aiex.npu.sync` | `aiex.cert.wait_tcts` | ✅ Converted |
| `aiex.npu.address_patch` | `aiex.cert.apply_offset57` | ✅ Converted |
| `aiex.npu.load_pdi` | `aiex.cert.load_pdi` | ✅ Converted |
| `aiex.configure` | `aiex.cert.section` + `aiex.cert.load_pdi` | ✅ Converted |
| `aiex.run` | Inlined into caller | ✅ Converted |
| `aie.runtime_sequence` | `aiex.cert.page` + `aiex.cert.job` | ✅ Converted |

## NOT Yet Converted Operations

The following NPU operations are **NOT** yet converted to CERT format and remain in their original form:

| NPU Operation | Expected CERT Operation | Status | Notes |
|--------------|------------------------|--------|-------|
| `aiex.npu.dma_memcpy_nd` | TBD (likely DMA-related CERT ops) | ❌ Not Converted | Remains as `aiex.npu.dma_memcpy_nd` in output |
| `aiex.npu.rtp_write` | TBD | ❌ Not Converted | Used for runtime parameter updates |
| `aiex.set_lock` | TBD | ❌ Not Converted | Lock synchronization operations |

## Test Coverage

### Tests with Known Unconverted Operations

1. **inline_symbols.mlir**: Contains `aiex.npu.dma_memcpy_nd` - test updated to expect unconverted form
2. **inline_ssa.mlir**: Contains `aiex.npu.rtp_write` and `aiex.set_lock` - currently failing
3. **inline_locks.mlir**: Contains `aiex.set_lock` - currently failing

### Passing Tests

1. **inline_and_load_pdi.mlir**: Tests `aiex.configure`/`aiex.run` with `aiex.npu.write32` operations
2. **cross_device_inline_load_pdi.mlir**: Tests cross-device PDI loading
3. **inline_multiple_ssa.mlir**: Tests multiple SSA value inlining

## Implementation Notes

### CERT Pass Structure

The `AIENpuToCert` pass processes operations in multiple phases:

1. **Phase 0**: Create cert.sections from referenced devices' "configure" sequences
2. **Phase 0a**: Inline `aiex.run` operations
3. **Phase 0b**: Convert `aiex.configure` to cert.section + cert.load_pdi
4. **Phase 1**: Convert runtime_sequence to cert.job
5. **Phase 2**: Convert NPU operations to CERT operations
6. **Phase 3**: Merge consecutive uc_dma operations
7. **Phase 4**: Move referenced devices' cert.jobs into cert.sections
8. **Phase 5**: Clean up unused block arguments
9. **Phase 6**: Remove absorbed devices (keep only main device)

### File Locations

- Pass implementation: `lib/Dialect/AIEX/Transforms/AIENpuToCert.cpp`
- Test directory: `test/Passes/materialize-to-cert/`
- Roadmap document: `cert_roadmap.md`

## Future Work

### Required for Complete CERT Conversion

1. **DMA Operations**: Convert `aiex.npu.dma_memcpy_nd` to appropriate CERT DMA operations
   - May require new CERT dialect operations for DMA configuration
   - Should handle shim DMA allocations and metadata references

2. **RTP Operations**: Convert `aiex.npu.rtp_write` to CERT runtime parameter write
   - Needs to handle buffer references and address calculations

3. **Lock Operations**: Convert `aiex.set_lock` to CERT lock synchronization
   - Must preserve lock value semantics
   - Handle lock ID references

### Test Updates Needed

Once the above operations are converted:
- Update `inline_ssa.mlir` CHECK lines to expect CERT format
- Update `inline_locks.mlir` CHECK lines to expect CERT format
- Verify all tests in the materialize-to-cert suite pass

## References

- CERT specification: See architecture documentation
- Implementation roadmap: `cert_roadmap.md`
- Related pass: `AIEMaterializeRuntimeSequences` (handles inlining before CERT conversion)
