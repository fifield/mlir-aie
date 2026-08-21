srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

designs := cert_reg_snapshot cert_reg_profile
hosts := test_snapshot test_profile

all: $(addsuffix .elf,$(designs)) $(addsuffix .exe,$(hosts))

test_%.exe: ${srcdir}/test_%.cpp ${srcdir}/cert_debug_buffer.h
	clang -std=c++17 -Wall -lstdc++ -I/opt/xilinx/xrt/include \
		-L/opt/xilinx/xrt/lib -luuid -lxrt_coreutil $< -o $@

%.elf: ${srcdir}/%.mlir
	aiecc --tmpdir=project_$* --get-full-elf --full-elf-name=$@ $<

run: all
	./test_snapshot.exe cert_reg_snapshot.elf
	./test_profile.exe cert_reg_profile.elf

clean:
	rm -rf *.elf *.exe project_*

.PHONY: all run clean
