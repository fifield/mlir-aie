all: vector_vector_mul.elf test.exe

%.exe: %.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

%.mlir: %.py
	python $< -o $@

%.elf: %.mlir 
	aiecc.py -v --tmpdir=project --xchesscc --xbridge --no-compile-host \
			--generate-full-elf --full-elf-name=$@ $<
clean:
	rm -rf project vector_vector_mul.mlir vector_vector_mul.elf test.exe
