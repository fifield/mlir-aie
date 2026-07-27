all: passthrough_dmas_placed.elf test.exe

%.exe: %.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

%.mlir: %.py
	python $< > $@

%.elf: %.mlir 
	aiecc -v --tmpdir=project --xchesscc --xbridge \
			--get-full-elf --full-elf-name=$@ $<
clean:
	rm -rf project passthrough_dmas_placed.mlir passthrough_dmas_placed.elf test.exe
