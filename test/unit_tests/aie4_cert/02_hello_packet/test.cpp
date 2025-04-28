
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

struct host_queue_header_t {
  uint64_t read_index;
  struct {
    uint16_t major;
    uint16_t minor;
  } version;
  uint32_t capacity; // Queue capacity, must be a power of two.
  uint64_t write_index;
  uint64_t data_address;
};

// from host_queue.h

enum host_queue_packet_type {
  HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  HOST_QUEUE_PACKET_TYPE_INVALID = 1,
};

struct common_header {
  union {
    struct {
      uint16_t type : 8;
      uint16_t barrier : 1;
      uint16_t acquire_fence_scope : 2;
      uint16_t release_fence_scope : 2;
    };
    uint16_t header;
  };
  uint16_t opcode;
  uint16_t count;
  uint8_t distribute;
  uint8_t indirect;
};

struct xrt_packet_header {
  struct common_header common_header;
  uint64_t completion_signal;
};

typedef struct xrt_packet_header xrt_packet_header_t;

/*
 * hsa pkt payload of exec_buf
 * this cmd has to be after the cu_config cmd, with which the cu_index in
 * this cmd can locate the pdi info corresponding to the cu, CERT then can
 * decide whether pdi load is required.
 * dpu_control_code_host_addr* is used to load control code interpreted by
 * the job runner in CERT
 * args contains the info of input/output frame, parameter of network
 * etc, which are all transparent to CERT
 */
struct exec_buf {
  uint16_t cu_index;
  uint16_t reserved0;
  uint32_t dpu_control_code_host_addr_low;
  uint32_t dpu_control_code_host_addr_high;
  uint16_t args_len;
  uint16_t reserved1;
  uint32_t args_host_addr_low;
  uint32_t args_host_addr_high;
};
typedef exec_buf exec_buf_t;

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

  ext_mem_model_t signal_h;
  uint32_t *completion_signal =
      (uint32_t *)mlir_aie_mem_alloc(&xaie, signal_h, 64);
  *completion_signal = 0;
  uint64_t completion_signal_paddr = signal_h.physicalAddr - 0x80000000;

  ext_mem_model_t queue_h;
  host_queue_header_t *host_queue = reinterpret_cast<host_queue_header_t *>(
      cert_utils::setup_queue(&xaie, column, queue_h, 64));
  void *packet_buffer = reinterpret_cast<void *>(
      reinterpret_cast<uint64_t>(host_queue) + sizeof(host_queue_header_t));

  // create an xrt hsa packet in the buffer
  xrt_packet_header_t *xrt_header = (xrt_packet_header_t *)packet_buffer;
  xrt_header->common_header.type = HOST_QUEUE_PACKET_TYPE_VENDOR_SPECIFIC;
  xrt_header->common_header.opcode = 1; // exec_buf
  xrt_header->common_header.count = sizeof(exec_buf_t);
  xrt_header->completion_signal = completion_signal_paddr;

  // allocate control code buffer
  ext_mem_model_t control_code_h;
  uint16_t *control_code = reinterpret_cast<uint16_t *>(
      mlir_aie_mem_alloc(&xaie, control_code_h, 256 /*FIXME*/));

  // two jobs of 3 NOP each
  control_code[0] = 0xffff;
  control_code[1] = 0x0000;
  control_code[2] = 0x0000;
  control_code[3] = 0x0000;
  control_code[4] = 0x0044;
  control_code[5] = 0x0000;
  control_code[6] = 0x0000;
  control_code[7] = 0x0000;
  control_code[8] = 0x0000;
  control_code[9] = 0x0000;
  control_code[10] = 0x0018; // start job
  control_code[11] = 0x0000;
  control_code[12] = 0x0016; // nop
  control_code[13] = 0x0000;
  control_code[14] = 0x0016;
  control_code[15] = 0x0000;
  control_code[16] = 0x0016;
  control_code[17] = 0x0000;
  control_code[18] = 0x0016;
  control_code[19] = 0x0000;
  control_code[20] = 0x0007; // end job
  control_code[21] = 0x0000;
  control_code[22] = 0x0000;
  control_code[23] = 0x0001;
  control_code[24] = 0x0018;
  control_code[25] = 0x0000;
  control_code[26] = 0x0016;
  control_code[27] = 0x0000;
  control_code[28] = 0x0016;
  control_code[29] = 0x0000;
  control_code[30] = 0x0016;
  control_code[31] = 0x0000;
  control_code[32] = 0x0016;
  control_code[33] = 0x0000;
  control_code[34] = 0x0007;
  control_code[35] = 0x0000;
  control_code[36] = 0x00ff; // eof

  exec_buf_t *execution_buffer = reinterpret_cast<exec_buf_t *>(
      (uint8_t *)packet_buffer + sizeof(xrt_packet_header_t));

  uint64_t control_code_paddr = control_code_h.physicalAddr - 0x80000000;
  execution_buffer->dpu_control_code_host_addr_high =
      reinterpret_cast<uint64_t>(control_code_paddr) >> 32;
  execution_buffer->dpu_control_code_host_addr_low =
      reinterpret_cast<uint64_t>(control_code_paddr) & 0xffffffff;
  std::cout << "control_code_paddr: " << std::hex << control_code_paddr
            << std::dec << "\n";

  host_queue->read_index = 0;
  host_queue->write_index = 1;

  mlir_aie_sync_mem_dev(signal_h);
  mlir_aie_sync_mem_dev(control_code_h);
  mlir_aie_sync_mem_dev(queue_h);

  const uint32_t partition_size = 3;
  cert_utils::set_partition_size(column, partition_size);

  cert_utils::set_alive(column, 0);

  // wait for the completion signal to be set to state 'completed'
  uint32_t completion = 0;
  while (completion != 4 /* HSA_CMD_STATE_COMPLETED */) {
    mlir_aie_sync_mem_cpu(signal_h);
    completion = *completion_signal;
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