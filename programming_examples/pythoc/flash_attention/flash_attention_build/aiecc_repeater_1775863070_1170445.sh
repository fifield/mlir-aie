#!/bin/bash
set -e
# Repeater script for: NPU lowering
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
'aiex.dma_await_task' op Cannot wait on a BD that is not configured to issue a token.
failed to legalize operation 'aiex.dma_await_task' that was explicitly marked illegal: "aiex.dma_await_task"(%327) : (index) -> ()
'aiex.dma_configure_task' op Cannot lower while op still has uses.
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='/work/npu-dev/mlir-aie/programming_examples/pythoc/flash_attention/flash_attention_build/aiecc_failure_1775863070_1170445.mlir'
PASS_PIPELINE='any(aie-materialize-runtime-sequences,aie.device(aie-materialize-bd-chains,aie-substitute-shim-dma-allocations,aie-assign-runtime-sequence-bd-ids,aie-dma-tasks-to-npu,aie-dma-to-npu,aie-lower-set-lock))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
