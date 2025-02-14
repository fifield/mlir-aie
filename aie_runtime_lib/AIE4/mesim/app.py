# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import aie_runtime as rt
from aie_runtime.driver import TCPDriver

import argparse
import os
import sys

from aie.compiler.aiecc.configure import install_path

# A typical command line passed to this script is something like:
# -p 51375 --device-cfg aie4_4r_3c_1m_S3.json --pkg-dir . --input-dir . --output-dir mesim_output --cycle-timeout 0
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", type=int)
parser.add_argument("--pkg-dir")
args, extra_args = parser.parse_known_args()

# connect to the port from the command line
tcpdriver = TCPDriver("localhost", args.port, debug=False)

# unlock NPI
tcpdriver.write_word(0xF70A000C, 0xF9E8D7C6, secure=True)
# PROTECTED_REG_EN = 1
tcpdriver.write_word(0xF70A0200, (0x7F << 8) | 0x1, secure=True)

from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """
    extern "Python" uint32_t _mesim_read32_cb(uint64_t);
    extern "Python" void _mesim_write32_cb(uint64_t, uint32_t);
    extern "Python" void _mesim_readGM_cb(uint64_t, uint64_t, char*);
    extern "Python" void _mesim_writeGM_cb(uint64_t, uint64_t, char*);
    int run_main(int argc, char *argv[]);
    void set_read32_callback(uint32_t (*read32_cb)(uint64_t addr));
    void set_write32_callback(void(*write32_cb)(uint64_t addr, uint32_t data));
    void set_readGM_callback(void(*readGM_cb)(uint64_t addr, uint64_t size, char *filename));
    void set_writeGM_callback(void(*writeGM_cb)(uint64_t addr, uint64_t size, char *filename));
"""
)

ffibuilder.set_source(
    "_ps_cffi",
    """
    extern int run_main(int argc, char *argv[]);
    extern void set_read32_callback(uint32_t (*read32_cb)(uint64_t addr));
    extern void set_write32_callback(void(*write32_cb)(uint64_t addr, uint32_t data));
    extern void set_readGM_callback(void(*read32_cb)(uint64_t addr, uint64_t size, char *filename));
    extern void set_writeGM_callback(void(*write32_cb)(uint64_t addr, uint64_t size, char *filename));
""",
    libraries=["app", "stdc++", "test_utils", "boost_program_options"],
    library_dirs=[
        args.pkg_dir if args.pkg_dir else ".",
        os.path.join(install_path(), "runtime_lib", "x86_64", "test_lib", "lib"),
    ],
)


ffibuilder.compile(tmpdir=args.pkg_dir, verbose=False)

sys.path.append(args.pkg_dir)
from _ps_cffi import ffi, lib


def onerror(exception, exc_value, traceback):
    print(f"MESIM raised exception {exception}: '{exc_value}'\n")
    os._exit(-1)


@ffi.def_extern(onerror=onerror)
def _mesim_read32_cb(addr):
    return tcpdriver.read_int32(addr)


@ffi.def_extern(onerror=onerror)
def _mesim_write32_cb(addr, data):
    tcpdriver.write_int32(addr, data)


@ffi.def_extern(onerror=onerror)
def _mesim_readGM_cb(addr, size, _filename):
    filename = ffi.string(_filename).decode("utf-8")
    tcpdriver.send_command(f"EXT_MEM_DUMP {addr} {size} {filename}")


@ffi.def_extern(onerror=onerror)
def _mesim_writeGM_cb(addr, size, _filename):
    filename = ffi.string(_filename).decode("utf-8")
    tcpdriver.send_command(f"EXT_MEM_FILL {addr} {size} {filename}")


lib.set_read32_callback(lib._mesim_read32_cb)
lib.set_write32_callback(lib._mesim_write32_cb)
lib.set_readGM_callback(lib._mesim_readGM_cb)
lib.set_writeGM_callback(lib._mesim_writeGM_cb)

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
sys.exit(exit_code)
