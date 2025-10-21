
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

  printf("Hello, cert world.\n");

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));
  const uint32_t column = 0;

  uint32_t fw_state = 0;
  do {
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  } while (fw_state != 0xffe);

  ext_mem_model_t queue_h;
  cert_utils::setup_queue(&xaie, column, queue_h, 64);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the firmware to reach state FW_STATE_LEADER_HOST_QUEUE_POP
  // (0x1001) where it waiting to pop something off the queue
  while (fw_state != 0x1001) {
    sleep(1);
    fw_state = cert_utils::get_fw_state(column, 0);
    std::cout << "fw_state: " << fw_state << "\n";
  }
  std::cout << "CERT reached FW_STATE_LEADER_HOST_QUEUE_POP\n";
  std::cout << "PASS!\n";
  return 0;
}