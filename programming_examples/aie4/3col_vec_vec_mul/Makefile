srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: vector_vector_mul.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

%.mlir: ${srcdir}/%.py
	python $< -o $@

%.elf: %.mlir
	aiecc -v --tmpdir=project \
			--get-full-elf --full-elf-name=$@ $<
clean:
	rm -rf project vector_vector_mul.mlir vector_vector_mul.elf test.exe
