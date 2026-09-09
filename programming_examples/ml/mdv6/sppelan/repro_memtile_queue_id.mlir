// Compile-only reproducer. Do not execute this incomplete program on hardware.
// Run: aie-opt --aie-dma-to-npu repro_memtile_queue_id.mlir
// NPU2 memtile MM2S5 supports BD 24; START_BD_ID is six bits (mask 0x3f).
// Expected queue write at 0x1a065c: value 24. The affected generic lowering
// masks the ID with 0xf and emits value 8. The numerical phase-A/gather
// generator uses explicit memtile queue writes until the compiler is fixed.
module {
  aie.device(npu2) {
    %mt = aie.tile(0, 1)
    aie.runtime_sequence() {
      aiex.npu.push_queue(0, 1, MM2S : 5) {bd_id = 24 : i32, issue_token = false, repeat_count = 0 : i32}
    }
  }
}
