
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "cert_utils.h"
#include "memory_allocator.h"
#include "test_utils.h"

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;

  printf("Testing cert packets\n");

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  const uint32_t column = 0;
  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0);

  // dummy allocation so that subsequent allocations are not paddr=zero
  ext_mem_model_t dummy_h;
  (void)mlir_aie_mem_alloc(&xaie, dummy_h, 64);

  ext_mem_model_t queue_h;
  void *queue = cert_utils::setup_queue(&xaie, column, queue_h, 64);

  // allocate control code buffer

  // two jobs of 3 NOP each
  std::vector<uint16_t> control_code = {
      0xffff, 0x0000, 0x0000, 0x0000, 0x0044, 0x0000, 0x0000, 0x0000, 0x0000,
      0x0000, 0x0018, 0x0000, 0x0016, 0x0000, 0x0016, 0x0000, 0x0016, 0x0000,
      0x0016, 0x0000, 0x0007, 0x0000, 0x0000, 0x0001, 0x0018, 0x0000, 0x0016,
      0x0000, 0x0016, 0x0000, 0x0016, 0x0000, 0x0016, 0x0000, 0x0007, 0x0000,
      0x00ff // eof
  };

  ext_mem_model_t signal_h =
      cert_utils::enqueue_control_code(&xaie, queue, control_code);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the completion signal to be set to state 'completed'
  uint32_t completion = 0;
  while (completion != 4 /* HSA_CMD_STATE_COMPLETED */) {
    mlir_aie_sync_mem_cpu(signal_h);
    completion = *reinterpret_cast<uint32_t *>(signal_h.virtualAddr);
    std::cout << "completion signal: " << completion << "\n";
    if (completion == 5 /* HSA_CMD_STATE_ERROR */)
      break;
    if (completion == 6 /* HSA_CMD_STATE_ABORT */)
      break;
    if (completion >= 8 /* timeout, noresponse, skerror, skcrashed */)
      break;
    sleep(1);
  }
  if (completion == 4) {
    std::cout << "PASS!\n";
    return 0;
  }
  std::cout << "failed.\n";
  return 1;
}