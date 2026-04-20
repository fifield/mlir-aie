srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
include ${srcdir}/../../../makefile-common

all: rep_elan_bf16.o

rep_elan_bf16.o: rep_elan_bf16.cc
	${PEANO_INSTALL_DIR}/bin/clang ${PEANOWRAP2P_FLAGS} -c ${srcdir}/$< -o $@

clean:
	rm -f rep_elan_bf16.o

.PHONY: all clean
