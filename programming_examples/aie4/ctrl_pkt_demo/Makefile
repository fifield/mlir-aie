srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

all: ctrl_pkt_demo.elf test.exe

%.exe: ${srcdir}/%.cpp
	clang -std=c++17 -lstdc++ -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
		-luuid -lxrt_coreutil $< -o $@

# `--generate-ctrl-pkt-overlay` installs the column control overlay (shim
# TileControl -> per-tile TileControl packet flows plus the
# `ctrlpkt_col0_ctrl_mm2s_chan0` shim DMA allocation) and lowers the design's
# `aiex.control_packet` ops into a shim control-MM2S transfer. aiecc then
# serializes the packets into `.ctrlpkt-<arg>.<grp>` and emits the patch-info
# that relocates the control-MM2S BD onto that section, so the payload travels
# inside the ELF and the host never sees it as an argument.
#
# The design has no aie.core, so no core compiler (chess or peano) is involved
# and no --xchesscc/--no-xchesscc selection is needed.
ctrl_pkt_demo.elf: ${srcdir}/ctrl_pkt_demo.mlir
	aiecc --tmpdir=project \
		--generate-ctrl-pkt-overlay \
		--get-full-elf --full-elf-name=$@ $<
	@echo "--- expect one .ctrlpkt-<arg>.<grp> section:"
	@readelf -SW $@ 2>/dev/null | grep ctrlpkt

run: all
	./test.exe ctrl_pkt_demo.elf

clean:
	rm -rf project ctrl_pkt_demo.elf test.exe
