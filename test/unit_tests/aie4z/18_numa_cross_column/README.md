# Test 18: NUMA Cross-Column Memory Access

## Purpose

This test demonstrates and validates the NUMA memory segmentation feature by intentionally creating cross-column memory accesses between two columns.

## Test Design

### Memory Layout
External memory base address: 0x80000000 (configured by simulator)

**NUMA Segments (relative to base):**
- **Column 0**: 0x00000000-0x01FFFFFF (32MB) → Absolute: 0x80000000-0x81FFFFFF
- **Column 1**: 0x02000000-0x03FFFFFF (32MB) → Absolute: 0x82000000-0x83FFFFFF

### Data Flow
```
Column 0:
  - Reads from 0x80000000 (local - Column 0's memory) ✓
  - Writes to 0x82000000 (CROSS-COLUMN to Column 1's memory) ⚠️

Column 1:
  - Reads from 0x82000400 (local - Column 1's memory) ✓
  - Writes to 0x80000400 (CROSS-COLUMN to Column 0's memory) ⚠️
```

### Expected Results
1. ✅ All accesses complete successfully (functional correctness maintained)
2. ✅ NUMA statistics show cross-column accesses: Column 0→1 and Column 1→0
3. ✅ If warnings enabled, see warning on first cross-column access

## How to Run

## How to Configure NUMA

The NUMA configuration must be added to the device model JSON file (e.g., `aie4z_4r_9c_1m.json`).

Add to the `ExternalMemory` section:

```json
"ExternalMemory": {
  "num_ddr_ports": 2,
  "mem_size": 536870912,
  ...
  "numa_segmentation": {
    "enabled": true,
    "log_warnings": true,
    "segments": [
      {"column": 0, "base_addr": "0x00000000", "size": "0x02000000"},
      {"column": 1, "base_addr": "0x02000000", "size": "0x02000000"}
    ]
  }
}
```

**Important:** The `base_addr` values are **relative offsets** from the external memory base (0x80000000), NOT absolute addresses!

## Run the Test

```bash
cd mlir-aie/test/unit_tests/aie4z/18_numa_cross_column
xchesscc_wrapper aie4 -c kernel.cc
aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host \
         --aie-generate-txn --txn-name=transaction.bin --aiesim aie4.mlir test.cpp
./aie4.mlir.prj/mesim.sh
```

### Step 4: Review Output

Look for:
1. `PASS!` - Test functional correctness
2. NUMA statistics section showing cross-column accesses

Example output:
```
NUMA Cross-Column Memory Access Statistics

Cross-Column Access Matrix:
  Column 0 → Column 1: 128 accesses
  Column 1 → Column 0: 128 accesses
  
Column 0 Summary:
  Total cross-column accesses: 128
  Total cross-column bytes: 32768
    Write bytes: 32768

Column 1 Summary:
  Total cross-column accesses: 128
  Total cross-column bytes: 32768
    Write bytes: 32768

Overall Summary:
  Total cross-column accesses: 256
  Total cross-column bytes: 65536
```

## Files

- `aie4.mlir` - AIE configuration with 2 columns, cross-column memory buffers
- `kernel.cc` - Simple passthrough kernel (copies input to output)
- `test.cpp` - Host code that validates cross-column data movement
- `README.md` - This file

## Notes

- This test exercises the NUMA feature by design - it SHOULD show cross-column accesses
- The test passes if data moves correctly AND NUMA statistics are reported
- To disable NUMA and run as normal test, set `enable_numa_segmentation = false`
