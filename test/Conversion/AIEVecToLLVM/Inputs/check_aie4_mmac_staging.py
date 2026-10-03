# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check every staging chunk's index and column pair in lowered MMAC IR."""

import sys

# Register the custom AIEVec types and attributes used elsewhere in the input.
from aie.dialects import aievec  # noqa: F401
from aie.extras.context import mlir_mod_ctx
from aie.ir import IntegerAttr, Module, VectorType


def check_block(block):
    chunks = 0
    for op in block.operations:
        name = op.operation.name
        if name.startswith("xllvm.intr.aie4.insert.staging."):
            chunk, index = op.operands
            index = IntegerAttr(index.owner.attributes["value"]).value
            assert index == chunks, f"expected chunk {chunks}, got {index}"
            shuffle = chunk.owner
            assert shuffle.name == "vector.shuffle"
            columns = []
            for half in shuffle.operands:
                extract = half.owner
                if extract.name == "vector.bitcast":
                    extract = extract.operands[0].owner
                assert extract.name == "vector.extract"
                columns.append(extract.operands[0])
            assert columns[0] == columns[1], "columns must come from the same B tile"
            assert columns[0].owner.name == "vector.transpose"
            assert list(columns[0].owner.attributes["permutation"]) == [1, 0]
            for j, half in enumerate(shuffle.operands):
                extract = half.owner
                if extract.name == "vector.bitcast":
                    extract = extract.operands[0].owner
                assert list(extract.attributes["static_position"]) == [2 * index + j]
            lanes = VectorType(chunk.type).shape[0]
            mask = list(shuffle.attributes["mask"])
            assert mask == list(range(lanes)), "column halves must concatenate in order"
            chunks += 1
        elif name.startswith("xllvm.intr.aie4.staging.to.matrix."):
            assert chunks == 32, f"commit after {chunks} chunks, expected 32"
            chunks = 0
        for region in op.regions:
            for nested in region.blocks:
                check_block(nested)
    assert chunks == 0, "staging inserts without a commit"


with open(sys.argv[1]) as source, mlir_mod_ctx(allow_unregistered_dialects=True):
    # -split-input-file emits multiple modules separated by this marker.
    modules = [
        Module.parse(part) for part in source.read().split("// -----") if part.strip()
    ]
    for module in modules:
        module.operation.verify()
        check_block(module.body)
