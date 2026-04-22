//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
// Test for NUMA cross-column memory access detection
// Each of the 9 columns reads from its own segment and writes to the next
// column's segment, creating a cross-column access per column.
//
// REQUIRES: mesimulator
//
// RUN: xchesscc_wrapper aie4 -c %S/kernel.cc
// RUN: aiecc.py --alloc-scheme=basic-sequential --xchesscc --xbridge --no-compile-host --aie-generate-txn --txn-name=transaction.bin --aiesim %s %test_utils_flags %S/test.cpp
// RUN: python3 -c "import json,os; cfg=open('aie4.mlir.prj/sim/mesim_config.py').read(); path=[l.split('\"')[1] for l in cfg.splitlines() if l.startswith('device_json')][0]; d=json.load(open(path)); p=json.load(open('%S/device_config_numa.json')); d['Device']['ExternalMemory']['numa_segmentation']=p['Device']['ExternalMemory']['numa_segmentation']; out=os.path.join('aie4.mlir.prj/sim',os.path.basename(path)); json.dump(d,open(out,'w'),indent=2)"
// RUN: sed -i 's|^device_json = .*|device_json = "aie4.mlir.prj/sim/aie4z_4r_9c_1m.json"|' aie4.mlir.prj/sim/mesim_config.py
// RUN: aie4.mlir.prj/mesim.sh | FileCheck --check-prefix=MESIM %s
//
// MESIM: device model: {{.*}}aie4z
// MESIM: PASS!
// MESIM: Column 0 {{.*}} Column 1:
// MESIM: Column 1 {{.*}} Column 2:
// MESIM: Column 2 {{.*}} Column 3:
// MESIM: Column 3 {{.*}} Column 4:
// MESIM: Column 4 {{.*}} Column 5:
// MESIM: Column 5 {{.*}} Column 6:
// MESIM: Column 6 {{.*}} Column 7:
// MESIM: Column 7 {{.*}} Column 8:
// MESIM: Column 8 {{.*}} Column 0:
// MESIM: Total cross-column accesses: 18
// MESIM: Total cross-column bytes: 9216

// NUMA address map (each column has a 32MB segment):
//   Column 0: 0x80000000 - 0x81FFFFFF
//   Column 1: 0x82000000 - 0x83FFFFFF
//   Column 2: 0x84000000 - 0x85FFFFFF
//   Column 3: 0x86000000 - 0x87FFFFFF
//   Column 4: 0x88000000 - 0x89FFFFFF
//   Column 5: 0x8A000000 - 0x8BFFFFFF
//   Column 6: 0x8C000000 - 0x8DFFFFFF
//   Column 7: 0x8E000000 - 0x8FFFFFFF
//   Column 8: 0x90000000 - 0x91FFFFFF
//
// Each column C:
//   Input  (local read):        base[C]       = 0x80000000 + C * 0x2000000
//   Output (cross-column write): base[(C+1)%9] + 0x400

module @test_18_numa_cross_column {
  aie.device(npu3z) {

    func.func private @passthrough(%A: memref<32xi32>, %B: memref<32xi32>) -> () attributes { link_with="kernel.o" }

    // =====================================================================
    // Column 0
    // =====================================================================
    %tile_0_2 = aie.tile(0, 2)
    %ping_in_0  = aie.buffer(%tile_0_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_0"} : memref<32xi32>
    %pong_in_0  = aie.buffer(%tile_0_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_0"} : memref<32xi32>
    %ping_out_0 = aie.buffer(%tile_0_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_0"} : memref<32xi32>
    %pong_out_0 = aie.buffer(%tile_0_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_0"} : memref<32xi32>
    %in_rd_0  = aie.lock(%tile_0_2, 0) { init = 0 : i32 }
    %in_wr_0  = aie.lock(%tile_0_2, 1) { init = 1 : i32 }
    %out_rd_0 = aie.lock(%tile_0_2, 2) { init = 0 : i32 }
    %out_wr_0 = aie.lock(%tile_0_2, 3) { init = 1 : i32 }
    %core_0 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_0, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_0, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_0, %ping_out_0) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_0, Release, 1)
        aie.use_lock(%out_rd_0, Release, 1)
        aie.use_lock(%in_rd_0, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_0, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_0, %pong_out_0) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_0, Release, 1)
        aie.use_lock(%out_rd_0, Release, 1)
      }
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_0 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_0, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_0 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_0, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_0 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_0, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_0 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_0, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 1
    // =====================================================================
    %tile_1_2 = aie.tile(1, 2)
    %ping_in_1  = aie.buffer(%tile_1_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_1"} : memref<32xi32>
    %pong_in_1  = aie.buffer(%tile_1_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_1"} : memref<32xi32>
    %ping_out_1 = aie.buffer(%tile_1_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_1"} : memref<32xi32>
    %pong_out_1 = aie.buffer(%tile_1_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_1"} : memref<32xi32>
    %in_rd_1  = aie.lock(%tile_1_2, 0) { init = 0 : i32 }
    %in_wr_1  = aie.lock(%tile_1_2, 1) { init = 1 : i32 }
    %out_rd_1 = aie.lock(%tile_1_2, 2) { init = 0 : i32 }
    %out_wr_1 = aie.lock(%tile_1_2, 3) { init = 1 : i32 }
    %core_1 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_1, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_1, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_1, %ping_out_1) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_1, Release, 1)
        aie.use_lock(%out_rd_1, Release, 1)
        aie.use_lock(%in_rd_1, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_1, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_1, %pong_out_1) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_1, Release, 1)
        aie.use_lock(%out_rd_1, Release, 1)
      }
      aie.end
    }
    %mem_1_2 = aie.mem(%tile_1_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_1 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_1, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_1 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_1, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_1 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_1, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_1 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_1, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 2
    // =====================================================================
    %tile_2_2 = aie.tile(2, 2)
    %ping_in_2  = aie.buffer(%tile_2_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_2"} : memref<32xi32>
    %pong_in_2  = aie.buffer(%tile_2_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_2"} : memref<32xi32>
    %ping_out_2 = aie.buffer(%tile_2_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_2"} : memref<32xi32>
    %pong_out_2 = aie.buffer(%tile_2_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_2"} : memref<32xi32>
    %in_rd_2  = aie.lock(%tile_2_2, 0) { init = 0 : i32 }
    %in_wr_2  = aie.lock(%tile_2_2, 1) { init = 1 : i32 }
    %out_rd_2 = aie.lock(%tile_2_2, 2) { init = 0 : i32 }
    %out_wr_2 = aie.lock(%tile_2_2, 3) { init = 1 : i32 }
    %core_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_2, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_2, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_2, %ping_out_2) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_2, Release, 1)
        aie.use_lock(%out_rd_2, Release, 1)
        aie.use_lock(%in_rd_2, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_2, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_2, %pong_out_2) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_2, Release, 1)
        aie.use_lock(%out_rd_2, Release, 1)
      }
      aie.end
    }
    %mem_2_2 = aie.mem(%tile_2_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_2 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_2, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_2 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_2, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_2 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_2, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_2 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_2, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 3
    // =====================================================================
    %tile_3_2 = aie.tile(3, 2)
    %ping_in_3  = aie.buffer(%tile_3_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_3"} : memref<32xi32>
    %pong_in_3  = aie.buffer(%tile_3_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_3"} : memref<32xi32>
    %ping_out_3 = aie.buffer(%tile_3_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_3"} : memref<32xi32>
    %pong_out_3 = aie.buffer(%tile_3_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_3"} : memref<32xi32>
    %in_rd_3  = aie.lock(%tile_3_2, 0) { init = 0 : i32 }
    %in_wr_3  = aie.lock(%tile_3_2, 1) { init = 1 : i32 }
    %out_rd_3 = aie.lock(%tile_3_2, 2) { init = 0 : i32 }
    %out_wr_3 = aie.lock(%tile_3_2, 3) { init = 1 : i32 }
    %core_3 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_3, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_3, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_3, %ping_out_3) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_3, Release, 1)
        aie.use_lock(%out_rd_3, Release, 1)
        aie.use_lock(%in_rd_3, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_3, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_3, %pong_out_3) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_3, Release, 1)
        aie.use_lock(%out_rd_3, Release, 1)
      }
      aie.end
    }
    %mem_3_2 = aie.mem(%tile_3_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_3 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_3, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_3 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_3, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_3 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_3, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_3 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_3, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 4
    // =====================================================================
    %tile_4_2 = aie.tile(4, 2)
    %ping_in_4  = aie.buffer(%tile_4_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_4"} : memref<32xi32>
    %pong_in_4  = aie.buffer(%tile_4_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_4"} : memref<32xi32>
    %ping_out_4 = aie.buffer(%tile_4_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_4"} : memref<32xi32>
    %pong_out_4 = aie.buffer(%tile_4_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_4"} : memref<32xi32>
    %in_rd_4  = aie.lock(%tile_4_2, 0) { init = 0 : i32 }
    %in_wr_4  = aie.lock(%tile_4_2, 1) { init = 1 : i32 }
    %out_rd_4 = aie.lock(%tile_4_2, 2) { init = 0 : i32 }
    %out_wr_4 = aie.lock(%tile_4_2, 3) { init = 1 : i32 }
    %core_4 = aie.core(%tile_4_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_4, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_4, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_4, %ping_out_4) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_4, Release, 1)
        aie.use_lock(%out_rd_4, Release, 1)
        aie.use_lock(%in_rd_4, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_4, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_4, %pong_out_4) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_4, Release, 1)
        aie.use_lock(%out_rd_4, Release, 1)
      }
      aie.end
    }
    %mem_4_2 = aie.mem(%tile_4_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_4 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_4, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_4 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_4, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_4 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_4, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_4 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_4, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 5
    // =====================================================================
    %tile_5_2 = aie.tile(5, 2)
    %ping_in_5  = aie.buffer(%tile_5_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_5"} : memref<32xi32>
    %pong_in_5  = aie.buffer(%tile_5_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_5"} : memref<32xi32>
    %ping_out_5 = aie.buffer(%tile_5_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_5"} : memref<32xi32>
    %pong_out_5 = aie.buffer(%tile_5_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_5"} : memref<32xi32>
    %in_rd_5  = aie.lock(%tile_5_2, 0) { init = 0 : i32 }
    %in_wr_5  = aie.lock(%tile_5_2, 1) { init = 1 : i32 }
    %out_rd_5 = aie.lock(%tile_5_2, 2) { init = 0 : i32 }
    %out_wr_5 = aie.lock(%tile_5_2, 3) { init = 1 : i32 }
    %core_5 = aie.core(%tile_5_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_5, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_5, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_5, %ping_out_5) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_5, Release, 1)
        aie.use_lock(%out_rd_5, Release, 1)
        aie.use_lock(%in_rd_5, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_5, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_5, %pong_out_5) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_5, Release, 1)
        aie.use_lock(%out_rd_5, Release, 1)
      }
      aie.end
    }
    %mem_5_2 = aie.mem(%tile_5_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_5 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_5, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_5 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_5, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_5 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_5, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_5 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_5, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 6
    // =====================================================================
    %tile_6_2 = aie.tile(6, 2)
    %ping_in_6  = aie.buffer(%tile_6_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_6"} : memref<32xi32>
    %pong_in_6  = aie.buffer(%tile_6_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_6"} : memref<32xi32>
    %ping_out_6 = aie.buffer(%tile_6_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_6"} : memref<32xi32>
    %pong_out_6 = aie.buffer(%tile_6_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_6"} : memref<32xi32>
    %in_rd_6  = aie.lock(%tile_6_2, 0) { init = 0 : i32 }
    %in_wr_6  = aie.lock(%tile_6_2, 1) { init = 1 : i32 }
    %out_rd_6 = aie.lock(%tile_6_2, 2) { init = 0 : i32 }
    %out_wr_6 = aie.lock(%tile_6_2, 3) { init = 1 : i32 }
    %core_6 = aie.core(%tile_6_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_6, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_6, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_6, %ping_out_6) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_6, Release, 1)
        aie.use_lock(%out_rd_6, Release, 1)
        aie.use_lock(%in_rd_6, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_6, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_6, %pong_out_6) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_6, Release, 1)
        aie.use_lock(%out_rd_6, Release, 1)
      }
      aie.end
    }
    %mem_6_2 = aie.mem(%tile_6_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_6 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_6, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_6 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_6, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_6 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_6, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_6 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_6, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 7
    // =====================================================================
    %tile_7_2 = aie.tile(7, 2)
    %ping_in_7  = aie.buffer(%tile_7_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_7"} : memref<32xi32>
    %pong_in_7  = aie.buffer(%tile_7_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_7"} : memref<32xi32>
    %ping_out_7 = aie.buffer(%tile_7_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_7"} : memref<32xi32>
    %pong_out_7 = aie.buffer(%tile_7_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_7"} : memref<32xi32>
    %in_rd_7  = aie.lock(%tile_7_2, 0) { init = 0 : i32 }
    %in_wr_7  = aie.lock(%tile_7_2, 1) { init = 1 : i32 }
    %out_rd_7 = aie.lock(%tile_7_2, 2) { init = 0 : i32 }
    %out_wr_7 = aie.lock(%tile_7_2, 3) { init = 1 : i32 }
    %core_7 = aie.core(%tile_7_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_7, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_7, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_7, %ping_out_7) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_7, Release, 1)
        aie.use_lock(%out_rd_7, Release, 1)
        aie.use_lock(%in_rd_7, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_7, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_7, %pong_out_7) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_7, Release, 1)
        aie.use_lock(%out_rd_7, Release, 1)
      }
      aie.end
    }
    %mem_7_2 = aie.mem(%tile_7_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_7 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_7, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_7 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_7, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_7 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_7, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_7 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_7, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Column 8
    // =====================================================================
    %tile_8_2 = aie.tile(8, 2)
    %ping_in_8  = aie.buffer(%tile_8_2) {address = 2048 : i32, bank = 0 : i32, sym_name = "ping_in_8"} : memref<32xi32>
    %pong_in_8  = aie.buffer(%tile_8_2) {address = 2304 : i32, bank = 0 : i32, sym_name = "pong_in_8"} : memref<32xi32>
    %ping_out_8 = aie.buffer(%tile_8_2) {address = 2560 : i32, bank = 0 : i32, sym_name = "ping_out_8"} : memref<32xi32>
    %pong_out_8 = aie.buffer(%tile_8_2) {address = 2816 : i32, bank = 0 : i32, sym_name = "pong_out_8"} : memref<32xi32>
    %in_rd_8  = aie.lock(%tile_8_2, 0) { init = 0 : i32 }
    %in_wr_8  = aie.lock(%tile_8_2, 1) { init = 1 : i32 }
    %out_rd_8 = aie.lock(%tile_8_2, 2) { init = 0 : i32 }
    %out_wr_8 = aie.lock(%tile_8_2, 3) { init = 1 : i32 }
    %core_8 = aie.core(%tile_8_2) {
      %c0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c128 step %c1 {
        aie.use_lock(%in_rd_8, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_8, AcquireGreaterEqual, 1)
        func.call @passthrough(%ping_in_8, %ping_out_8) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_8, Release, 1)
        aie.use_lock(%out_rd_8, Release, 1)
        aie.use_lock(%in_rd_8, AcquireGreaterEqual, 1)
        aie.use_lock(%out_wr_8, AcquireGreaterEqual, 1)
        func.call @passthrough(%pong_in_8, %pong_out_8) : (memref<32xi32>, memref<32xi32>) -> ()
        aie.use_lock(%in_wr_8, Release, 1)
        aie.use_lock(%out_rd_8, Release, 1)
      }
      aie.end
    }
    %mem_8_2 = aie.mem(%tile_8_2) {
      %s0 = aie.dma_start("MM2S", 0, ^mm2s0, ^dma0)
    ^dma0:
      %s1 = aie.dma_start("S2MM", 0, ^s2mm0, ^end)
    ^mm2s0:
      aie.use_lock(%out_rd_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_out_8 : memref<32xi32>, 0, 32) {bd_id = 0 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%out_wr_8, Release, 1)
      aie.next_bd ^mm2s1
    ^mm2s1:
      aie.use_lock(%out_rd_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_out_8 : memref<32xi32>, 0, 32) {bd_id = 1 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%out_wr_8, Release, 1)
      aie.next_bd ^mm2s0
    ^s2mm0:
      aie.use_lock(%in_wr_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%ping_in_8 : memref<32xi32>, 0, 32) {bd_id = 2 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%in_rd_8, Release, 1)
      aie.next_bd ^s2mm1
    ^s2mm1:
      aie.use_lock(%in_wr_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%pong_in_8 : memref<32xi32>, 0, 32) {bd_id = 3 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%in_rd_8, Release, 1)
      aie.next_bd ^s2mm0
    ^end:
      aie.end
    }

    // =====================================================================
    // Shim DMAs — each column reads local, writes to next column (cross-column)
    // =====================================================================

    // Column 0 shim: input from 0x80000000 (col 0), output to 0x82000400 (col 1)
    %tile_0_0 = aie.tile(0, 0)
    %ext_in_0  = aie.external_buffer{address = 0x80000000 : i64, sym_name = "ext_in_0"}  : memref<256xi32>
    %ext_out_0 = aie.external_buffer{address = 0x82000400 : i64, sym_name = "ext_out_0"} : memref<256xi32>
    %shim_in_wr_0  = aie.lock(%tile_0_0, 0) { init = 1 : i32 }
    %shim_in_rd_0  = aie.lock(%tile_0_0, 1) { init = 0 : i32 }
    %shim_out_wr_0 = aie.lock(%tile_0_0, 2) { init = 1 : i32 }
    %shim_out_rd_0 = aie.lock(%tile_0_0, 3) { init = 0 : i32 }
    %shim_0 = aie.shim_dma(%tile_0_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_0 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_0, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_0 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_0, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 1 shim: input from 0x82000000 (col 1), output to 0x84000400 (col 2)
    %tile_1_0 = aie.tile(1, 0)
    %ext_in_1  = aie.external_buffer{address = 0x82000000 : i64, sym_name = "ext_in_1"}  : memref<256xi32>
    %ext_out_1 = aie.external_buffer{address = 0x84000400 : i64, sym_name = "ext_out_1"} : memref<256xi32>
    %shim_in_wr_1  = aie.lock(%tile_1_0, 0) { init = 1 : i32 }
    %shim_in_rd_1  = aie.lock(%tile_1_0, 1) { init = 0 : i32 }
    %shim_out_wr_1 = aie.lock(%tile_1_0, 2) { init = 1 : i32 }
    %shim_out_rd_1 = aie.lock(%tile_1_0, 3) { init = 0 : i32 }
    %shim_1 = aie.shim_dma(%tile_1_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_1 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_1, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_1 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_1, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 2 shim: input from 0x84000000 (col 2), output to 0x86000400 (col 3)
    %tile_2_0 = aie.tile(2, 0)
    %ext_in_2  = aie.external_buffer{address = 0x84000000 : i64, sym_name = "ext_in_2"}  : memref<256xi32>
    %ext_out_2 = aie.external_buffer{address = 0x86000400 : i64, sym_name = "ext_out_2"} : memref<256xi32>
    %shim_in_wr_2  = aie.lock(%tile_2_0, 0) { init = 1 : i32 }
    %shim_in_rd_2  = aie.lock(%tile_2_0, 1) { init = 0 : i32 }
    %shim_out_wr_2 = aie.lock(%tile_2_0, 2) { init = 1 : i32 }
    %shim_out_rd_2 = aie.lock(%tile_2_0, 3) { init = 0 : i32 }
    %shim_2 = aie.shim_dma(%tile_2_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_2 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_2, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_2 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_2, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 3 shim: input from 0x86000000 (col 3), output to 0x88000400 (col 4)
    %tile_3_0 = aie.tile(3, 0)
    %ext_in_3  = aie.external_buffer{address = 0x86000000 : i64, sym_name = "ext_in_3"}  : memref<256xi32>
    %ext_out_3 = aie.external_buffer{address = 0x88000400 : i64, sym_name = "ext_out_3"} : memref<256xi32>
    %shim_in_wr_3  = aie.lock(%tile_3_0, 0) { init = 1 : i32 }
    %shim_in_rd_3  = aie.lock(%tile_3_0, 1) { init = 0 : i32 }
    %shim_out_wr_3 = aie.lock(%tile_3_0, 2) { init = 1 : i32 }
    %shim_out_rd_3 = aie.lock(%tile_3_0, 3) { init = 0 : i32 }
    %shim_3 = aie.shim_dma(%tile_3_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_3 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_3, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_3 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_3, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 4 shim: input from 0x88000000 (col 4), output to 0x8A000400 (col 5)
    %tile_4_0 = aie.tile(4, 0)
    %ext_in_4  = aie.external_buffer{address = 0x88000000 : i64, sym_name = "ext_in_4"}  : memref<256xi32>
    %ext_out_4 = aie.external_buffer{address = 0x8A000400 : i64, sym_name = "ext_out_4"} : memref<256xi32>
    %shim_in_wr_4  = aie.lock(%tile_4_0, 0) { init = 1 : i32 }
    %shim_in_rd_4  = aie.lock(%tile_4_0, 1) { init = 0 : i32 }
    %shim_out_wr_4 = aie.lock(%tile_4_0, 2) { init = 1 : i32 }
    %shim_out_rd_4 = aie.lock(%tile_4_0, 3) { init = 0 : i32 }
    %shim_4 = aie.shim_dma(%tile_4_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_4 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_4, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_4 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_4, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 5 shim: input from 0x8A000000 (col 5), output to 0x8C000400 (col 6)
    %tile_5_0 = aie.tile(5, 0)
    %ext_in_5  = aie.external_buffer{address = 0x8A000000 : i64, sym_name = "ext_in_5"}  : memref<256xi32>
    %ext_out_5 = aie.external_buffer{address = 0x8C000400 : i64, sym_name = "ext_out_5"} : memref<256xi32>
    %shim_in_wr_5  = aie.lock(%tile_5_0, 0) { init = 1 : i32 }
    %shim_in_rd_5  = aie.lock(%tile_5_0, 1) { init = 0 : i32 }
    %shim_out_wr_5 = aie.lock(%tile_5_0, 2) { init = 1 : i32 }
    %shim_out_rd_5 = aie.lock(%tile_5_0, 3) { init = 0 : i32 }
    %shim_5 = aie.shim_dma(%tile_5_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_5 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_5, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_5 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_5, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 6 shim: input from 0x8C000000 (col 6), output to 0x8E000400 (col 7)
    %tile_6_0 = aie.tile(6, 0)
    %ext_in_6  = aie.external_buffer{address = 0x8C000000 : i64, sym_name = "ext_in_6"}  : memref<256xi32>
    %ext_out_6 = aie.external_buffer{address = 0x8E000400 : i64, sym_name = "ext_out_6"} : memref<256xi32>
    %shim_in_wr_6  = aie.lock(%tile_6_0, 0) { init = 1 : i32 }
    %shim_in_rd_6  = aie.lock(%tile_6_0, 1) { init = 0 : i32 }
    %shim_out_wr_6 = aie.lock(%tile_6_0, 2) { init = 1 : i32 }
    %shim_out_rd_6 = aie.lock(%tile_6_0, 3) { init = 0 : i32 }
    %shim_6 = aie.shim_dma(%tile_6_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_6 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_6, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_6 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_6, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 7 shim: input from 0x8E000000 (col 7), output to 0x90000400 (col 8)
    %tile_7_0 = aie.tile(7, 0)
    %ext_in_7  = aie.external_buffer{address = 0x8E000000 : i64, sym_name = "ext_in_7"}  : memref<256xi32>
    %ext_out_7 = aie.external_buffer{address = 0x90000400 : i64, sym_name = "ext_out_7"} : memref<256xi32>
    %shim_in_wr_7  = aie.lock(%tile_7_0, 0) { init = 1 : i32 }
    %shim_in_rd_7  = aie.lock(%tile_7_0, 1) { init = 0 : i32 }
    %shim_out_wr_7 = aie.lock(%tile_7_0, 2) { init = 1 : i32 }
    %shim_out_rd_7 = aie.lock(%tile_7_0, 3) { init = 0 : i32 }
    %shim_7 = aie.shim_dma(%tile_7_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_7 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_7, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_7, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_7 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_7, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // Column 8 shim: input from 0x90000000 (col 8), output to 0x80000400 (col 0 — wrap!)
    %tile_8_0 = aie.tile(8, 0)
    %ext_in_8  = aie.external_buffer{address = 0x90000000 : i64, sym_name = "ext_in_8"}  : memref<256xi32>
    %ext_out_8 = aie.external_buffer{address = 0x80000400 : i64, sym_name = "ext_out_8"} : memref<256xi32>
    %shim_in_wr_8  = aie.lock(%tile_8_0, 0) { init = 1 : i32 }
    %shim_in_rd_8  = aie.lock(%tile_8_0, 1) { init = 0 : i32 }
    %shim_out_wr_8 = aie.lock(%tile_8_0, 2) { init = 1 : i32 }
    %shim_out_rd_8 = aie.lock(%tile_8_0, 3) { init = 0 : i32 }
    %shim_8 = aie.shim_dma(%tile_8_0) {
      %mm2s = aie.dma_start(MM2S, 0, ^mm2s, ^s2mm)
    ^s2mm:
      %s2mm_ch = aie.dma_start(S2MM, 0, ^s2mm_bd, ^end)
    ^mm2s:
      aie.use_lock(%shim_in_rd_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_in_8 : memref<256xi32>, 0, 256) {bd_id = 1 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%shim_in_wr_8, Release, 1)
      aie.next_bd ^mm2s
    ^s2mm_bd:
      aie.use_lock(%shim_out_wr_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%ext_out_8 : memref<256xi32>, 0, 256) {bd_id = 0 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%shim_out_rd_8, Release, 1)
      aie.next_bd ^s2mm_bd
    ^end:
      aie.end
    }

    // =====================================================================
    // Flows: shim <-> compute tile per column
    // =====================================================================
    aie.flow(%tile_0_0, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_0, DMA : 0)
    aie.flow(%tile_1_0, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_0, DMA : 0)
    aie.flow(%tile_2_0, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_3_0, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_0, DMA : 0)
    aie.flow(%tile_4_0, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %tile_4_0, DMA : 0)
    aie.flow(%tile_5_0, DMA : 0, %tile_5_2, DMA : 0)
    aie.flow(%tile_5_2, DMA : 0, %tile_5_0, DMA : 0)
    aie.flow(%tile_6_0, DMA : 0, %tile_6_2, DMA : 0)
    aie.flow(%tile_6_2, DMA : 0, %tile_6_0, DMA : 0)
    aie.flow(%tile_7_0, DMA : 0, %tile_7_2, DMA : 0)
    aie.flow(%tile_7_2, DMA : 0, %tile_7_0, DMA : 0)
    aie.flow(%tile_8_0, DMA : 0, %tile_8_2, DMA : 0)
    aie.flow(%tile_8_2, DMA : 0, %tile_8_0, DMA : 0)
  }
}
