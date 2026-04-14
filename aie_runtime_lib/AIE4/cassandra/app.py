# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import argparse
import os
import re
import sys
import time

from aie.compiler.aiecc.configure import install_path

# A typical command line passed to this script is something like:
# -p 51375 --device-cfg aie4_4r_3c_1m_S3.json --pkg-dir . --input-dir . --output-dir mesim_output --cycle-timeout 0
parser = argparse.ArgumentParser()
parser.add_argument("--qdma")
parser.add_argument("--pkg-dir")

args, extra_args = parser.parse_known_args()

from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    void acquire_cassandra_handle(char* server_path, int port);
    void release_cassandra_handle();

    uint32_t mesim_Read32(uint64_t addr);
    void mesim_Write32(uint64_t addr, uint32_t Data);
    uint32_t qdma_Read32(uint64_t addr);
    void qdma_Write32(uint64_t addr, uint32_t Data);
    void cold_reset_npu();

    int run_main(int argc, char *argv[]);
"""
)

ffibuilder.set_source(
    "_ps_cffi",
    """
    extern int run_main(int argc, char *argv[]);
    extern void acquire_cassandra_handle(char* server_path, int port);
    extern void release_cassandra_handle();

    extern uint32_t mesim_Read32(uint64_t addr);
    extern void mesim_Write32(uint64_t addr, uint32_t Data);
    extern uint32_t qdma_Read32(uint64_t addr);
    extern void qdma_Write32(uint64_t addr, uint32_t Data);

    extern void cold_reset_npu();

""",
    libraries=["app_cassandra", "stdc++", "test_utils", "boost_program_options"],
    library_dirs=[
        args.pkg_dir if args.pkg_dir else ".",
        os.path.join(install_path(), "runtime_lib", "x86_64", "test_lib", "lib"),
    ],
)


ffibuilder.compile(tmpdir=args.pkg_dir, verbose=False)

sys.path.append(args.pkg_dir)
from _ps_cffi import ffi, lib


lib.acquire_cassandra_handle(bytes('localhost', 'utf-8'), 3335)
lib.cold_reset_npu()

user_args = ["app"] + os.getenv("MLIR_AIE_MESIM_PROGRAM_ARGS", "").split()
char_array = ffi.new("char*[]", len(user_args))
char_pointers = []
for i, s in enumerate(user_args):
    # Convert each Python string to a C-style string
    encoded_string = s.encode("utf-8")
    c_string = ffi.new("char[]", encoded_string)
    char_pointers.append(c_string)  # Keep a reference
    char_array[i] = c_string

exit_code = lib.run_main(len(user_args), char_array)
lib.release_cassandra_handle()
sys.exit(exit_code)
