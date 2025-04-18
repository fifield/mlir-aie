
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "memory_allocator.h"
#include "test_utils.h"

// copied from "cert/handshake/handshake.h"
typedef struct {
  volatile uint32_t mpaie_alive;   // 0
  uint32_t partition_base_address; // 4
  uint32_t partition_size : 7;     // 8
  uint32_t reserved : 23;          // 8
  uint32_t mode : 1;               // 8
  uint32_t uc_b : 1;               // 8
  volatile uint32_t hsa_addr_high; // c
  volatile uint32_t hsa_addr_low;  // 10
  uint32_t ctx_switch_req;         // 14
  uint32_t hsa_location;           // 18
  uint32_t hsa_lite_status;        // 1c
  uint32_t misc_status;            // 20
  volatile uint32_t log_addr_high; // 24
  volatile uint32_t log_addr_low;  // 28
  uint32_t log_buf_size;           // 2c
  uint32_t host_time_high;         // 30
  uint32_t host_time_low;          // 34
  struct {
    volatile uint32_t dtrace_addr_high; // 38
    volatile uint32_t dtrace_addr_low;  // 3c
  } trace;
  struct {
    uint32_t restore_page; // 40
    uint32_t pdi_id;       // 44
    struct {
      uint16_t page_index;
      uint16_t page_len;
    } pdi_page; // 48
  } ctx_save;
  struct {
    volatile uint32_t hsa_addr_high; // 4c
    volatile uint32_t hsa_addr_low;  // 50
  } dbg;
  struct {
    volatile uint32_t dbg_buf_addr_high; // 54
    volatile uint32_t dbg_buf_addr_low;  // 58
    uint32_t size;                       // 5c
  } dbg_buf;
  volatile struct {
    uint32_t c_job_readiness_checked; // number of checks whether there are jobs
                                      // ready
    uint32_t c_opcode;                // number of opcode run
    uint32_t c_job_launched;
    uint32_t c_job_finished;
    uint32_t c_hsa_pkt;       // number of hsa pkt handled
    uint32_t c_page;          // number of pages loaded
    uint32_t c_doorbell;      // number of hsa doorbell ring
    uint32_t c_uc_scrub;      // number of uc memory(PM) scrub
    uint32_t c_tct_requested; // number of tct requested
    uint32_t c_tct_received;  // number of tct received
    uint16_t
        c_preemption_ucdma; // run out of wait handle UC_DMA_WRITE_DES opcode
    uint16_t c_preemption_ucdma_sync;     // run out of wait handle
                                          // UC_DMA_WRITE_DES_SYNC opcode
    uint16_t c_preemption_poll;           // POLL_32 opcode retry times
    uint16_t c_preemption_mask_poll;      // MASK_POLL_32 opcode retry times
    uint16_t c_preemption_remote_barrier; // run out of physical barrier
                                          // REMOTE_BARRIER opcode
    uint16_t c_preemption_wait_tct;  // actor entry overflow or run out of wait
                                     // handle WAIT_TCTS opcode
    uint16_t c_block_ucdma;          // block UC_DMA_WRITE_DES opcode
    uint16_t c_block_ucdma_sync;     // block UC_DMA_WRITE_DES_SYNC opcode
    uint16_t c_block_local_barrier;  // block local_barrier opcode
    uint16_t c_block_remote_barrier; // block REMOTE_BARRIER opcode
    uint16_t c_block_wait_tct;       // block WAIT_TCTS opcode
    uint16_t c_actor_hash_conflict;  // number of slow actor entry lookup
  } counter;
  volatile struct {
    uint32_t fw_state;
    uint32_t abs_page_index; // absolute index of page where current control
                             // code are in
    uint32_t ppc; // previous pc (relative address to current page) that drives
                  // current_job_context to NULL
  } vm;
  volatile struct {
    uint32_t ear; // exception address
    uint32_t esr; // exception status
    uint32_t pc;  // exception pc
  } exception;
#ifdef PDI_LOAD_TEST
  uint32_t test_pdi_addr_high;
  uint32_t test_pdi_addr_low;
#endif
} handshake_t;

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

int main(int argc, char *argv[]) {

  aie_libxaie_ctx_t xaie;

  printf("Hello, cert world.\n");

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t *>(data.data()),
                                       data.size() * sizeof(uint32_t));

  uint64_t fw_state_offset = offsetof(handshake_t, vm);
  uint64_t p_dm_addr = 0x28000;
  uint32_t fw_state;
  do {
    fw_state = mlir_aie_sim_read32(p_dm_addr + fw_state_offset);
    std::cout << "fw_state: " << fw_state << " at address: 0x" << std::hex
              << (p_dm_addr + fw_state_offset) << std::dec << "\n";
  } while (fw_state != 0);

  // dummy allocation so that subsequent allocations are not paddr=zero
  ext_mem_model_t dummy;
  (void)mlir_aie_mem_alloc(&xaie, dummy, 128);

  // allocate a power of two buffer
  uint32_t buffer_capacity = 64; // example capacity, must be a power of two
  ext_mem_model_t buffer_h;
  uint64_t *buffer = reinterpret_cast<uint64_t *>(
      mlir_aie_mem_alloc(&xaie, buffer_h, buffer_capacity * 64));

  // allocate host_queue
  ext_mem_model_t host_queue_h;
  host_queue_header_t *host_queue = reinterpret_cast<host_queue_header_t *>(
      mlir_aie_mem_alloc(&xaie, host_queue_h, sizeof(host_queue_header_t)));

  // initialize the buffer to zero
  std::memset(buffer, 0, buffer_capacity * 64);

  // set host_queue address field to be the buffer address
  host_queue->capacity = buffer_capacity;
  host_queue->data_address = reinterpret_cast<uint64_t>(buffer);

  uint64_t hsa_addr_high_offset = 0xc;
  uint64_t hsa_addr_low_offset = 0x10;
  // The cert firmware sets the axi offset register to 0x80000000 out of the box
  // and in order to make that work the external memory base address is the same
  // in the simulator. The allocator also starts at that offset (so that sync
  // works), so the offset must be subtracted here because the controller will
  // dereference it and the axi offset will get added.
  uint64_t host_queue_paddr = host_queue_h.physicalAddr - 0x80000000;
  mlir_aie_sim_write32(p_dm_addr + hsa_addr_high_offset,
                       host_queue_paddr >> 32);
  mlir_aie_sim_write32(p_dm_addr + hsa_addr_low_offset,
                       host_queue_paddr & 0xffffffff);

  uint64_t partition_size_offset = 0x8;
  uint32_t partition_size = 3;
  mlir_aie_sim_write32(p_dm_addr + partition_size_offset, partition_size);

  uint64_t mpaie_alive_offset = 0x0;
  uint32_t alive_magic = 0x404C5645;
  mlir_aie_sim_write32(p_dm_addr + mpaie_alive_offset, alive_magic);

  // while (fw_state != 3) {
  //   sleep(1);
  //   fw_state = mlir_aie_sim_read32(p_dm_addr + fw_state_offset);
  //   std::cout << "fw_state: " << fw_state << " at address: 0x" << std::hex
  //             << (p_dm_addr + fw_state_offset) << std::dec << "\n";
  // }
  // std::cout << "CERT reached FW_STATE_HSA_CONFIG\n";

  // wait for the firmware to reach state FW_STATE_LEADER_HOST_QUEUE_POP
  // (0x1001) where it waiting to pop something off the queue
  while (fw_state != 0x1001) {
    sleep(1);
    fw_state = mlir_aie_sim_read32(p_dm_addr + fw_state_offset);
    std::cout << "fw_state: " << fw_state << " at address: 0x" << std::hex
              << (p_dm_addr + fw_state_offset) << std::dec << "\n";
  }
  std::cout << "CERT reached FW_STATE_LEADER_HOST_QUEUE_POP\n";
  std::cout << "PASS!\n";
  return 0;
}