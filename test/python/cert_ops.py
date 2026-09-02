# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Python bindings for the `aiex.cert` ops: the AIE2PS microcontroller control-code
# dialect. Every op in CERTOps.td is constructed here and its printed form
# checked, so the generated builders and the region-op decorators in
# `aie.dialects.aiex` stay usable from Python.
#
# `aiex.cert_reg` is a plain 0..23 integer attribute, so registers are passed as
# ints (`cert_mov(3, ...)` targets `$r3`).

# RUN: %PYTHON %s | FileCheck %s

import numpy as np

from aie.dialects.aie import device, AIEDevice, tile, buffer
from aie.dialects.aiex import *
from util import construct_and_print_module


# CHECK-LABEL: TEST: certRegions
# CHECK: aie.device(xcve3858)
# CHECK:   aiex.cert.section @sec {
# CHECK:     aiex.cert.page {
# CHECK:       aiex.cert.job(20) {
# CHECK:         aiex.cert.nop
# CHECK:   aiex.cert.attach_to_group(2) {
# CHECK:     aiex.cert.page {
# CHECK:       aiex.cert.job(0) {
# CHECK:         aiex.cert.write32(4096, 42)
# CHECK:       aiex.cert.job(1) {
# CHECK:         aiex.cert.nop
# CHECK:   aiex.cert.page {
# CHECK:     aiex.cert.job(2) {
# CHECK:       aiex.cert.nop
# CHECK:   } {placement = 4 : i32}
@construct_and_print_module
def certRegions():
    @device(AIEDevice.xcve3858)
    def device_body():
        # A named section, referenced by load_pdi/preempt below.
        @cert_section("sec")
        def _():
            @cert_page()
            def _():
                @cert_job(20)
                def _():
                    cert_nop()

        # A group is one microcontroller's stream; two jobs in one page are
        # cooperatively scheduled, which is what makes overlap possible.
        @cert_attach_to_group(2)
        def _():
            @cert_page()
            def _():
                @cert_job(0)
                def _():
                    cert_write32(0x1000, 42)

                @cert_job(1)
                def _():
                    cert_nop()

        # `placement` is the resolved uC group id (aie2ps: one uC per column); a later
        # pass lowers it to an enclosing attach_to_group.
        @cert_page(placement=4)
        def _():
            @cert_job(2)
            def _():
                cert_nop()


# CHECK-LABEL: TEST: certScalarOps
# CHECK: aiex.cert.write32(4096, 42)
# CHECK: aiex.cert.maskwrite32(4096, 255, 16)
# CHECK: aiex.cert.write32_d(4096, 42)
# CHECK: aiex.cert.write32_d(3, 7) {address_is_reg, value_is_reg}
# CHECK: aiex.cert.read32(3, 4096)
# CHECK: aiex.cert.read32_d(3, 4)
# CHECK: aiex.cert.add(3, 1)
# CHECK: aiex.cert.mov(3, 99)
# CHECK: aiex.cert.poll32(4096, 42)
# CHECK: aiex.cert.maskpoll32(4096, 255, 16)
# CHECK: aiex.cert.nop
# CHECK: aiex.cert.sleep(100)
# CHECK: aiex.cert.yield
@construct_and_print_module
def certScalarOps():
    @device(AIEDevice.xcve3858)
    def device_body():
        @cert_page()
        def _():
            @cert_job(0)
            def _():
                cert_write32(0x1000, 42)
                cert_maskwrite32(0x1000, 0xFF, 0x10)
                # Both operands constant...
                cert_write32_d(0x1000, 42)
                # ...and both taken from registers instead.
                cert_write32_d(3, 7, address_is_reg=True, value_is_reg=True)
                cert_read32(3, 0x1000)
                cert_read32_d(3, 4)
                cert_add(3, 1)
                cert_mov(3, 99)
                cert_poll32(0x1000, 42)
                cert_maskpoll32(0x1000, 0xFF, 0x10)
                cert_nop()
                cert_sleep(100)
                cert_yield()


# CHECK-LABEL: TEST: certSyncOps
# CHECK: aiex.cert.local_barrier(1, 2)
# CHECK: aiex.cert.remote_barrier(1, 5)
# CHECK: aiex.cert.wait_tcts(0, 0, 1)
@construct_and_print_module
def certSyncOps():
    @device(AIEDevice.xcve3858)
    def device_body():
        @cert_page()
        def _():
            @cert_job(0)
            def _():
                # 2 jobs on this uC meet here.
                cert_local_barrier(1, 2)
                # uCs 0 and 2 (mask bits 0 and 2) meet here.
                cert_remote_barrier(1, 0x5)
                cert_wait_tcts(0, 0, 1)


# CHECK-LABEL: TEST: certDmaOps
# CHECK: aiex.cert.uc_dma_chain @chain {
# CHECK:   aiex.cert.uc_dma_bd @bd_data, 36864, 9, true
# CHECK:   aiex.cert.uc_dma_bd @bd_data, 35389440, 8, false
# CHECK: aiex.cert.uc_dma_write_des_sync(@chain)
# CHECK: aiex.cert.uc_dma_write_des(5, @chain)
# CHECK: aiex.cert.wait_uc_dma(5)
# CHECK: aiex.cert.apply_offset_57(@bd_data, 1, 0)
@construct_and_print_module
def certDmaOps():
    @device(AIEDevice.xcve3858)
    def device_body():
        t = tile(0, 2)
        buffer(t, np.ndarray[(9,), np.dtype[np.int32]], name="bd_data")

        @cert_uc_dma_chain("chain")
        def _():
            cert_uc_dma_bd("bd_data", 0x00009000, 9, True)
            cert_uc_dma_bd("bd_data", 0x021C0000, 8, False)

        @cert_page()
        def _():
            @cert_job(0)
            def _():
                # Blocking form...
                cert_uc_dma_write_des_sync("chain")
                # ...and the asynchronous pair, which lets configuration DMA
                # overlap with the scalar writes around it. The handle is a
                # register id.
                cert_uc_dma_write_des(5, "chain")
                cert_wait_uc_dma(5)
                cert_apply_offset_57("bd_data", 1, 0)


# CHECK-LABEL: TEST: certControlOps
# CHECK: aiex.cert.section @sec {
# CHECK: aiex.cert.load_pdi(1, @sec)
# CHECK: aiex.cert.preempt(0, @sec, @sec)
# CHECK: aiex.cert.save_register(4096, 7)
# CHECK: aiex.cert.save_timestamps(101)
@construct_and_print_module
def certControlOps():
    @device(AIEDevice.xcve3858)
    def device_body():
        @cert_section("sec")
        def _():
            @cert_page()
            def _():
                @cert_job(20)
                def _():
                    cert_nop()

        # load_pdi and preempt each take a whole job in a whole page, per the
        # ISA, so they get one page apiece rather than sharing.
        @cert_page()
        def _():
            @cert_job(0)
            def _():
                cert_load_pdi(1, "sec")

        @cert_page()
        def _():
            @cert_job(1)
            def _():
                cert_preempt(0, "sec", "sec")

        @cert_page()
        def _():
            @cert_job(2)
            def _():
                cert_save_register(0x1000, 7)
                cert_save_timestamps(101)
