all: eltwise_add_placed.elf test.exe

%.exe: %.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

%.mlir: %.py
	python $< -d npu3 -i1s 2048 -i2s 2048 -os 2048 > $@

%.elf: %.mlir 
	aiecc.py -v --tmpdir=project --xchesscc --xbridge --no-compile-host \
			--generate-full-elf --full-elf-name=$@ $<
clean:
	rm -rf project eltwise_add_placed.mlir eltwise_add_placed.elf test.exe
