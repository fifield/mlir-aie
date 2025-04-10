
#include <fstream>
#include <iostream>
#include "memory_allocator.h"
#include "test_utils.h"

using namespace std;


int main(int argc, char ** argv)
{
  constexpr int input_size = 256;
  constexpr int output_size = 256;
  constexpr int rows = 16;
  constexpr int cols = 16;
  
  uint8_t* inputArray = (uint8_t*) malloc(input_size * sizeof(uint8_t));
  uint8_t* outputArray = (uint8_t*) malloc(output_size * sizeof(uint8_t));

  for (int i=0; i < input_size; i++){ 
    inputArray[i] = i; // inputArray is row-major
  }

  std::string txn_path = "transaction.bin";
  std::vector<uint32_t> data = test_utils::load_instr_binary(txn_path);
  test_utils::write_transaction_binary(reinterpret_cast<uint8_t*> (data.data()),
				       data.size() * sizeof(uint32_t));

  uint64_t col = 1;
  uint64_t row = 3;

  uint64_t tile_data_mem = ( (col << 25) | ((row + 4) << 20)) + 1024;
  
  for(unsigned int i = 0; i < 64; i++){
    int32_t val = *((int32_t*) inputArray + i);
    mlir_aie_sim_write32(tile_data_mem + (i * 4), val);
  }

  uint64_t tile_lock_1 = ((col << 25) | ((row + 4) << 20)) + 0x000AE010;
  uint64_t tile_lock_3 = ((col << 25) | ((row + 4) << 20)) + 0x000AE030;

  std::cout << "Read buffer lock \n";
  uint32_t lock1 = mlir_aie_sim_read32(tile_lock_1);
  if (lock1 != 0) {
    std::cout << "Unexpected initial lock value: " << lock1 << "\n";
    return -1;
  }
  
  std::cout << "Assign buffer lock\n";
  mlir_aie_sim_write32(tile_lock_1, 1);

  std::cout << "Wait for completion\n";
  while (1) {
    uint32_t lock3 = mlir_aie_sim_read32(tile_lock_3);
    if (lock3 >= 1)
      break;
  }
  std::cout << "Done running\n";
  

  tile_data_mem = ((col << 25) | ((row + 4) << 20)) + 32768;
  for (int i = 0; i < 64; i++) {
    unsigned int val = mlir_aie_sim_read32(tile_data_mem + (4 * i));
    *((uint32_t*) outputArray + i) = val;
  }
  
  cout << "- Matrix before AIE 16x16 transpose \n";
  matrix_utils::print_matrix(inputArray, rows, cols);

  cout<<endl;
  cout<<endl;
    
  cout << "+ Matrix After AIE 16x16 transpose \n";
  matrix_utils::print_matrix(outputArray, rows, cols);

  uint8_t *cpuResult = (uint8_t *) malloc(output_size * sizeof(uint8_t));
  matrix_utils::transpose_cpu(inputArray, cpuResult, rows, cols);

  cout << "\n\n+ CPU Matrix After AIE 16x16 transpose \n";
  matrix_utils::print_matrix(cpuResult, rows, cols);
    
  if (matrix_utils::tensor_equal(outputArray, cpuResult, output_size)){
    cout << "PASS \n";
  }
  else{
   cout << "Fail. CPU Results Do NOT Match... \n";
  }

  free(inputArray);
  free(outputArray);

  return 0;
}


