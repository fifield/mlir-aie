//===- configure.mlir ------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

module {

// aie-opt --convert-aie-to-transaction='device-name=configure_cascade' configure.mlir
aie.device (npu2) @configure_cascade {
    // Compute tiles - 3x3 grid (rows 2, 3, 4)
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)

    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)

    %t04 = aie.tile(0, 4)
    %t14 = aie.tile(1, 4)
    %t24 = aie.tile(2, 4)

    // Configure cascade direction (West to East) for all compute tiles
    aie.configure_cascade(%t02, West, East)
    aie.configure_cascade(%t12, West, East)
    aie.configure_cascade(%t22, West, East)

    aie.configure_cascade(%t03, West, East)
    aie.configure_cascade(%t13, West, East)
    aie.configure_cascade(%t23, West, East)

    aie.configure_cascade(%t04, West, East)
    aie.configure_cascade(%t14, West, East)
    aie.configure_cascade(%t24, West, East)
}

// aie-opt --convert-aie-to-transaction='device-name=release_locks' configure.mlir
aie.device (npu2) @release_locks {
    // Compute tiles - 3x3 grid (rows 2, 3, 4)
    %t02 = aie.tile(0, 2)
    %t12 = aie.tile(1, 2)
    %t22 = aie.tile(2, 2)

    %t03 = aie.tile(0, 3)
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)

    %t04 = aie.tile(0, 4)
    %t14 = aie.tile(1, 4)
    %t24 = aie.tile(2, 4)

    // Locks for synchronization
    %lock02 = aie.lock(%t02, 10) { init = 1 : i32 }
    %lock12 = aie.lock(%t12, 10) { init = 1 : i32 }
    %lock22 = aie.lock(%t22, 10) { init = 1 : i32 }

    %lock03 = aie.lock(%t03, 10) { init = 1 : i32 }
    %lock13 = aie.lock(%t13, 10) { init = 1 : i32 }
    %lock23 = aie.lock(%t23, 10) { init = 1 : i32 }

    %lock04 = aie.lock(%t04, 10) { init = 1 : i32 }
    %lock14 = aie.lock(%t14, 10) { init = 1 : i32 }
    %lock24 = aie.lock(%t24, 10) { init = 1 : i32 }
}
}