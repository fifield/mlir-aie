// test.cpp

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <stdio.h>

#include "test_utils.h"

extern "C" {

void NpuWrite32(uint64_t Addr, unsigned int Data);
uint NpuRead32(uint64_t Addr);
void NpuWriteTransactionBinary(unsigned char *data, size_t nbytes);

}


// Function to print each field in a 32-bit register
void print_dma_status_register_fields(char * reg_name, uint32_t reg_value) {
    printf("\n%s register Value: 0x%08X\n\n", reg_name, reg_value);

    // Extract fields using bit masks and shifts
    uint8_t cur_bd = (reg_value >> 25) & 0xF;          // 28:25 - 4 bits
    uint8_t task_queue_size = (reg_value >> 20) & 0x1F; // 24:20 - 5 bits
    uint8_t channel_running = (reg_value >> 19) & 0x1; // Bit 19 - 1 bit
    uint8_t task_queue_overflow = (reg_value >> 18) & 0x1; // Bit 18 - 1 bit
    uint8_t error_fot_bds_per_task = (reg_value >> 13) & 0x1; // Bit 13 - 1 bit
    uint8_t error_fot_length_exceeded = (reg_value >> 12) & 0x1; // Bit 12 - 1 bit
    uint8_t stalled_tct_fifo_full = (reg_value >> 5) & 0x1;  // Bit 5 - 1 bit
    uint8_t stalled_stream_starvation = (reg_value >> 4) & 0x1; // Bit 4 - 1 bit
    uint8_t stalled_lock_rel = (reg_value >> 3) & 0x1; // Bit 3 - 1 bit
    uint8_t stalled_lock_acq = (reg_value >> 2) & 0x1; // Bit 2 - 1 bit
    uint8_t status = reg_value & 0x3;                  // 1:0 - 2 bits

    // Print fields
    printf("Field Breakdown:\n");
    printf("Cur_BD (28:25):                   0x%X\n", cur_bd);
    printf("Task_Queue_Size (24:20):          0x%X\n", task_queue_size);
    printf("Channel_Running (19):             %u\n", channel_running);
    printf("Task_Queue_Overflow (18):         %u\n", task_queue_overflow);
    printf("Error_FoT_BDs_per_Task (13):      %u\n", error_fot_bds_per_task);
    printf("Error_FoT_Length_Exceeded (12):   %u\n", error_fot_length_exceeded);
    printf("Stalled_TCT_or_Count_FIFO_Full(5):%u\n", stalled_tct_fifo_full);
    printf("Stalled_Stream_Starvation (4):    %u\n", stalled_stream_starvation);
    printf("Stalled_Lock_Rel (3):             %u\n", stalled_lock_rel);
    printf("Stalled_Lock_Acq (2):             %u\n", stalled_lock_acq);
    printf("Status (1:0):                     0x%X ", status);

    switch (status) {
        case 0: printf("IDLE\n"); break;
        case 1: printf("STARTING\n"); break;
        case 2: printf("RUNNING\n"); break;
        default: printf("UNKNOWN\n"); break;
    }
    printf("\n");
}


void print_mem_tile_status(uint64_t col) {
  uint64_t row = 1;
  uint64_t mem_tile_s2mm_status_0 = ((col << 25) | (row  << 20)) + 0x00009ED0;
  uint64_t mem_tile_s2mm_status_1 = ((col << 25) | (row  << 20)) + 0x00009ED4;
  uint64_t mem_tile_mm2s_status_0 = ((col << 25) | (row  << 20)) + 0x00009F00;
  uint64_t mem_tile_mm2s_status_1 = ((col << 25) | (row  << 20)) + 0x00009F04;

  uint32_t s2mm_status_0 = NpuRead32(mem_tile_s2mm_status_0);
  uint32_t s2mm_status_1 = NpuRead32(mem_tile_s2mm_status_1); 

  uint32_t mm2s_status_0 = NpuRead32(mem_tile_mm2s_status_0);
  uint32_t mm2s_status_1 = NpuRead32(mem_tile_mm2s_status_1);

  print_dma_status_register_fields("Mem tile S2MM0", s2mm_status_0);
  print_dma_status_register_fields("Mem tile MM2S0", mm2s_status_0);
  printf("s2mm_status_0 = 0x%08x, s2mm_status_1 = 0x%08x, mm2s_status_0 = 0x%08x, mm2s_status_1 = 0x%08x\n", s2mm_status_0, s2mm_status_1, mm2s_status_0, mm2s_status_1);

  for (uint32_t i=0;i<6;i++) {
    uint64_t mem_tile_lock_addr = ((col << 25) | (row  << 20)) + 0x0010000 + (i * 0x0010);
    uint32_t lock = NpuRead32(mem_tile_lock_addr);
    printf("lock[%d] = 0x%08x\n", i, lock);
  }
}
void print_tile_status(uint64_t col, uint64_t row) {
  uint64_t tile_ping_in_a_buffer  = ((col << 25) | ((row + 4) << 20)) + 2048;
  uint64_t tile_pong_in_a_buffer  = ((col << 25) | ((row + 4) << 20)) + 2304;
  uint64_t tile_ping_in_b_buffer  = ((col << 25) | ((row + 4) << 20)) + 2560;
  uint64_t tile_pong_in_b_buffer  = ((col << 25) | ((row + 4) << 20)) + 2816;
  uint64_t tile_ping_out_buffer   = ((col << 25) | ((row + 4) << 20)) + 3072;
  uint64_t tile_pong_out_buffer   = ((col << 25) | ((row + 4) << 20)) + 3328;

  printf("Tile status for col %d, row %d:\n", col, row);
  for (int i = 0; i < 4; i++) {
    uint32_t ping_in_a = NpuRead32(tile_ping_in_a_buffer + (4 * i));
    uint32_t pong_in_a = NpuRead32(tile_pong_in_a_buffer + (4 * i));
    uint32_t ping_in_b = NpuRead32(tile_ping_in_b_buffer + (4 * i));
    uint32_t pong_in_b = NpuRead32(tile_pong_in_b_buffer + (4 * i));
    uint32_t ping_out  = NpuRead32(tile_ping_out_buffer + (4 * i));
    uint32_t pong_out  = NpuRead32(tile_pong_out_buffer + (4 * i));
    printf("ping_in_a[%d] = 0x%08x, pong_in_a[%d] = 0x%08x, ping_in_b[%d] = 0x%08x, pong_in_b[%d] = 0x%08x,ping_out[%d] = 0x%08x, pong_out[%d] = 0x%08x\n", i, ping_in_a, i, pong_in_a, i, ping_in_b, i, pong_in_b, i, ping_out, i, pong_out);
  }

  for (int i = 0; i < 6; i++) {
    uint64_t lock_addr = ((col << 25) | ((row + 4) << 20)) + 0xae000 + (i * 0x0010);
    uint32_t lock = NpuRead32(lock_addr);
    printf("lock[%d] = 0x%08x\n", i, lock);
  }

  uint64_t tile_s2mm_status_0 = ((col << 25) | ((row + 4)  << 20)) + 0x000ACF00;
  uint64_t tile_s2mm_status_1 = ((col << 25) | ((row + 4)  << 20)) + 0x000ACF04;
  uint64_t tile_mm2s_status_0 = ((col << 25) | ((row + 4)  << 20)) + 0x000ACF10;

  uint32_t s2mm_status_0 = NpuRead32(tile_s2mm_status_0);
  uint32_t s2mm_status_1 = NpuRead32(tile_s2mm_status_1);
  uint32_t mm2s_status_0 = NpuRead32(tile_mm2s_status_0);

  print_dma_status_register_fields("Tile S2MM0", s2mm_status_0);
  print_dma_status_register_fields("Tile S2MM1", s2mm_status_1);
  print_dma_status_register_fields("Tile MM2S0", mm2s_status_0);



}





int main(int argc, char *argv[]) {
  printf("Hello, world.  I really do see you two there\n");

  uint64_t buffer_size = 2048;
  uint64_t tile_size = 32;
  uint64_t output_size = 4;
  uint64_t col = 0;

  uint32_t lock_id;
  uint64_t row;

  // We want to leave some space for the overrun pattern on A and B
  uint64_t input_a_offset = 0;
  uint64_t input_b_offset = 8448; // 8192+256;
  uint64_t output_offset = 16896; // 16384 + 512;

  uint32_t input_a_pattern_seed = 0x10000042;
  uint32_t input_b_pattern_seed = 0x02200001;
  uint32_t input_a_pattern_stomp = 0xbaadbaad;
  uint32_t input_b_pattern_stomp = 0xdaabdaab;
  uint32_t output_pattern_stomp = 0x00badbad;


  
  // Stomp over the data in the memory tiles and an overrun pattern
  row = 1;
  uint64_t input_a_memtile_data_mem = (col << 25) | (row << 20) + 0x00100000 + input_a_offset;
  uint64_t input_b_memtile_data_mem = (col << 25) | (row << 20) + 0x00100000 + input_b_offset;
  uint64_t output_memtile_data_mem  = (col << 25) | (row << 20) + 0x00100000 + output_offset;

  printf("The buffer size is %ld\n", buffer_size);

  for (int i = 0; i < buffer_size; i++) {
    NpuWrite32(input_a_memtile_data_mem  + (4 * i), input_a_pattern_seed +i);
    NpuWrite32(input_b_memtile_data_mem  + (4 * i), input_b_pattern_seed +i);
    NpuWrite32(output_memtile_data_mem + (4 * i), output_pattern_stomp);
  }
  for (int i = 0; i <4; i++) {
    NpuWrite32(input_a_memtile_data_mem  + (4 * (buffer_size+i)), input_a_pattern_stomp);
    NpuWrite32(input_b_memtile_data_mem  + (4 * (buffer_size+i)), input_b_pattern_stomp);
    NpuWrite32(output_memtile_data_mem  + (4 * (buffer_size+i)), output_pattern_stomp);
  }

  printf("Memory tile has been stomped.\n");
  printf("The buffer size is %ld\n", buffer_size);

  // Check the stomps
  row = 1;
  for (int i = 0; i < output_size; i++) {
    uint32_t in_a = NpuRead32(input_a_memtile_data_mem + (4 * i));
    uint32_t in_b = NpuRead32(input_b_memtile_data_mem + (4 * i));
    uint32_t out = NpuRead32(output_memtile_data_mem + (4 * i));
    printf("input_a[%d] = 0x%08x, input_b[%d] = 0x%08x, output[%d] = 0x%08x\n", i, in_a, i, in_b, i, out);
  }

  printf("The buffer size is %ld\n", buffer_size);

  // Stomp over the AIE tile memory
  row = 2;
  uint64_t tile_ping_in_a_buffer  = ((col << 25) | ((row + 4) << 20)) + 2048;
  uint64_t tile_pong_in_a_buffer  = ((col << 25) | ((row + 4) << 20)) + 2304;
  uint64_t tile_ping_in_b_buffer  = ((col << 25) | ((row + 4) << 20)) + 2560;
  uint64_t tile_pong_in_b_buffer  = ((col << 25) | ((row + 4) << 20)) + 2816;
  uint64_t tile_ping_out_buffer   = ((col << 25) | ((row + 4) << 20)) + 3072;
  uint64_t tile_pong_out_buffer   = ((col << 25) | ((row + 4) << 20)) + 3328;
  for (int i = 0; i < tile_size; i++) {
    NpuWrite32(tile_ping_in_a_buffer  + (4 * i), 0xbad11100 + i);
    NpuWrite32(tile_pong_in_a_buffer  + (4 * i), 0xbad22200 + i);
    NpuWrite32(tile_ping_in_b_buffer  + (4 * i), 0xbad33300 + i);
    NpuWrite32(tile_pong_in_b_buffer  + (4 * i), 0xbad44400 + i);
    NpuWrite32(tile_ping_out_buffer   + (4 * i), 0xbad55500 + i);
    NpuWrite32(tile_pong_out_buffer   + (4 * i), 0xbad66600 + i);
  }

  printf("Tile memory stomped.\n");

  print_tile_status(col, 2);
  print_mem_tile_status(col); 

  printf("The buffer size is %ld\n", buffer_size);

  // Configure the array
  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  NpuWriteTransactionBinary(reinterpret_cast<uint8_t *>(data.data()),
                            data.size() * sizeof(uint32_t));

  printf("Configuration written.\n");
  printf("The buffer size is %ld\n", buffer_size);

  // At this point, we need to toggle the memtile locks to indicate that data has been written
  // and can be read by the DMA
  // This is the mem_tile_input_read_lock which the MM2S channel0 0 is blocked on
  // aie.use_lock(%mem_tile_input_read_lock, AcquireGreaterEqual, 1)

  row = 1;
  uint64_t mem_tile_lock_1 = ((col << 25) | (row  << 20)) + 0x00010010;
  NpuWrite32(mem_tile_lock_1, 1);  // There's 1 tile of data there now, go for it
  uint64_t mem_tile_lock_3 = ((col << 25) | (row  << 20)) + 0x00010030;
  NpuWrite32(mem_tile_lock_3, 1);  // There's 1 tile of data there now, go for it

  printf("Locks toggled!\n");
  printf("-------------\n\n");
  printf("The buffer size is %ld\n", buffer_size);

  /*
  print_tile_status(col, 2);
  print_mem_tile_status(col);

  for (int poll=0;poll<10;poll++) {
    print_tile_status(col, 2);
  }

  printf("AFter the poll\n");
  print_mem_tile_status(col);
*/


  row = 1;
  while (1) {
    uint64_t mem_tile_lock_3 = ((col << 25) | (row  << 20)) + 0x00010050;
    uint32_t l = NpuRead32(mem_tile_lock_3);
    if (l >= 1)
      break;
//    else {
//      printf("lock = %d\n", l);
//      print_mem_tile_status(col);
//    }
  }
  printf("The memory tile lock has toggled\n");
  printf("The buffer size is %ld\n", buffer_size);
  
  // Check the memory tile for the ascending pattern
  row = 1;
  uint32_t output[buffer_size];
  int errs = 0;
  for (int i = 0; i < buffer_size; i++) {
    output[i] = NpuRead32(output_memtile_data_mem + (4 * i));
    uint32_t expected = input_a_pattern_seed + i + input_b_pattern_seed + i;
    if (output[i] != expected) {
      errs++;
      printf("error @ %d : %x != %x (actual != expected)\n", i, output[i], expected);
    }
    else {
      printf(" good @ %d : %x == %x (actual == expected)\n", i, output[i], expected);

    }
  }
  printf("I'm here now!");
  printf("The buffer size is %ld\n", buffer_size);


  for (int i=0;i<4;i++) {
    uint32_t over = NpuRead32(output_memtile_data_mem + (4 * (buffer_size+i)));
    if (over != output_pattern_stomp) {
      errs++;
      printf("error @ over %d : %x\n", i, over);
    }
    else {
      printf(" good @ over %d : %x\n", i, over);
    }
  }


  if (!errs) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail whale.\n");
    return -1;
  }
  

  return 0;
}