//===- CERTOps.td ------------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CERT_OPS
#define CERT_OPS

// A job-runner register index. isa-spec.yaml defines the `register` type
// as "8-bit unsigned integer 0..23 used to reference job-runner registers
// r0..r23"; r0..r7 are job-private locals and r8..r23 are globals (also
// spelled g0..g15 in asm source, which the assembler folds to r8..r23,
// so there is no separate $gN form in the IR).
// A plain integer attribute, printed as `$rN` only by the asm emitter.
// Verified against aiebu-asm: `$r23` assembles rc=0, `$r24` fails with
// "Register number $r24 out of range".
def AIEX_CertRegisterAttr :
    ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<23>]>;

def AIE_CertUcDmaChainOp: AIEX_Op<"cert.uc_dma_chain", [Symbol,
                                                    SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "uC DMA Chain operation";
  let description = [{
    This operation defines a chain of uC DMA buffer descriptors. It contains a
    single region where the buffer descriptors can be specified.

    Example:
    ```
      cert.uc_dma_chain @dma_chain {
        aiex.uc_dma_bd 0x600400, 0x20400, 128, true, false, false
        ...
      }
    ```
  }];
  let arguments = (
    ins OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    ($sym_name^)? $body attr-dict
  }];
}

def AIE_CertUcDmaBdOp : AIEX_Op<"cert.uc_dma_bd", [HasParent<"CertUcDmaChainOp">]> {
  let summary = "A DMA Buffer Descriptor for uC-DMA";
  let description = [{
    Defines a uC-DMA buffer descriptor within a chain. The descriptor specifies
    how the uC-DMA should transfer data from a remote address to a local address.

    - `remote_address`: Symbol reference to the data to be written
    - `local_address`: Address in the AIE array register that uC-DMA will access
    - `length`: Number of 32-bit words to transfer
    - `next_bd`: Flag indicating if this is the last uC-DMA BD in the chain
      (true if there are more BDs following, false if this is the last)

    Example:
    ```mlir
      cert.uc_dma_chain @dma_chain {
        aiex.cert.uc_dma_bd @mem01_bd0, 0x20400, 8, true
        aiex.cert.uc_dma_bd @mem01_bd1, 0x20420, 8, true
        aiex.cert.uc_dma_bd @mem11_bd0, 0x40400, 8, false
      }
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$remote_address,
        I32Attr:$local_address,
        I32Attr:$length,
        BoolAttr:$next_bd
  );
  let results = (outs);
  let assemblyFormat = [{
    $remote_address `,` $local_address `,` $length `,` $next_bd  attr-dict
  }];
}

def AIE_CertPageOp: AIEX_Op<"cert.page",
                            [ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp", "CertSectionOp"]>,
                             SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "A control code page";
  let description = [{
    This operation represents a page of control code. Pages are the basic unit
    of control code organization, with a maximum size of 8192 bytes. A page may
    contain one or more jobs along with their associated data; jobs within a
    page are cooperatively scheduled (see `cert.job`), while pages of a given
    microcontroller execute strictly in order.

    Corresponds to the `.eop` (end of page) directive in assembly.

    The optional `placement` attribute selects the microcontroller (CERT group /
    uC id) that runs this page. It is a resolved group id — use the target
    model's `getUcGroupId(col, half)` to compute it rather than deriving it by
    hand. When absent, the page defaults to group 0 (col0/uC0). A later pass lowers
    `placement` to an enclosing `cert.attach_to_group`.

    Example:
    ```
      cert.page {
        cert.job(0) {
          cert.write32(0x1000, 42)
        }
      }

      // placed on microcontroller (group) 2:
      cert.page {
        cert.job(1) {
          cert.write32(0x2000, 43)
        }
      } {placement = 2 : i32}
    ```
  }];
  let arguments = (ins OptionalAttr<I32Attr>:$placement);
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{ $body attr-dict }];
  let builders = [
      // Convenience builder for an unplaced page (defaults to group 0), keeping
      // the previous no-argument call sites valid.
      OpBuilder<(ins), [{
      build($_builder, $_state, /*placement=*/nullptr);
    }]>];
}

def AIE_CertJobOp: AIEX_Op<"cert.job", [ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp",
                                                      "CertPageOp", "CertSectionOp"]>,
                                        SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Start a job with a given job ID";
  let description = [{
    This operation starts a job identified by the given job ID.

    The job ID is a *label*, not an execution or emission order: it must be
    unique among jobs on the same page, and the CERT firmware uses it as a
    job-table key (and as the handle referenced by deferred/launch jobs). It is
    NOT a priority. Emission order and execution order are both determined by
    textual (IR) order: within a page the firmware starts jobs in textual order
    (running each until it blocks or ends), and between pages on one uC
    execution is strictly sequential. Emitters must therefore preserve IR order
    and must not sort jobs by job_id.

    Example:
    ```
      cert.job(42) {
        ...
      }
    ```
  }];
  let arguments = (
    ins I32Attr:$job_id
  );
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{ `(` $job_id `)` $body attr-dict }];
  let extraClassDeclaration = [{
    int getJobID() { return getJobId(); }
  }];
}

def AIE_CertWrite32Op: AIEX_Op<"cert.write32", [HasParent<"CertJobOp">]> {
  let summary = "Write a 32-bit value to a specified address";
  let description = [{
    This operation writes a 32-bit value to a specified address.

    Example:
    ```
      cert.write32(0x1000, 42)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict
  }];
}

def AIE_CertMaskWrite32Op: AIEX_Op<"cert.maskwrite32", [HasParent<"CertJobOp">]> {
  let summary = "Write a masked 32-bit value to a specified address";
  let description = [{
    This operation writes a masked 32-bit value to a specified address.

    Example:
    ```
      cert.maskwrite32(0x1000, 0xFF, 42)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$mask,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $mask `,` $value `)` attr-dict
  }];
}

def AIE_CertApplyOffset57Op: AIEX_Op<"cert.apply_offset_57", [HasParent<"CertJobOp">]> {
  let summary = "Apply offset 57 operation";
  let description = [{
    This operation adds a 57-bit offset to Shim DMA BD base address fields.
    The offset is loaded from the 32-bit registers arg[offset] and arg[offset+1]
    from the current argument list.
    'symbol' - A symbol pointing to the data to be patched
    'num_entries' - the number of Shim DMA BD entries to patch
    'offset' - index into the argument list to the start of offset data

    Example:
    ```
      memref.global "private" constant @bd_table : memref<9xi32> = dense<[0, -2147483648, 256, 117440512, 4351, 1, 1, 1, 1024]>
      ...
      cert.apply_offset_57(@bd_table, 1, 2)
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol,
        I16Attr:$num_entries,
        I16Attr:$offset
  );
  let assemblyFormat = [{
    `(` $symbol `,` $num_entries `,` $offset `)` attr-dict
  }];
}

def AIE_CertWaitTCTSOp: AIEX_Op<"cert.wait_tcts", [HasParent<"CertJobOp">]> {
  let summary = "Wait for a specified number of TCTs (Task Completion Tokens)";
  let description = [{
    This operation waits for a specified number of Task Completion Tokens (TCTs)
    for a given shim tile and channel.

    Example:
    ```
      cert.wait_tcts(1, 2, 3) // Wait for 3 TCTs on shim column 1, channel 2
    ```
  }];
  let arguments = (
    ins I32Attr:$tile_id,
        I32Attr:$channel_id,
        I32Attr:$target_tcts
  );
  let assemblyFormat = [{
    `(` $tile_id `,` $channel_id `,` $target_tcts `)` attr-dict
  }];
}

def AIE_CertUcDmaWriteDesSyncOp: AIEX_Op<"cert.uc_dma_write_des_sync",
                                        [HasParent<"CertJobOp">]> {
  let summary = "uC DMA Write Descriptor Sync operation";
  let description = [{
    This operation enqueues a uC DMA transfer. The symbol operand refers to a
    cert.uc_dma_bd_chain operation.

    Example:
    ```
      cert.uc_dma_write_des_sync(@dma_descriptor)
    ```
  }];
  let arguments = (
    ins FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    `(` $symbol `)` attr-dict
  }];
}

def AIE_CertLocalBarrierOp: AIEX_Op<"cert.local_barrier", [HasParent<"CertJobOp">]> {
  let summary = "Local barrier operation";
  let description = [{
    This operation represents a local barrier that synchronizes execution based on a barrier ID and the number of participants.

    Example:
    ```
      cert.local_barrier(1, 2)
    ```
  }];
  let arguments = (
    ins I32Attr:$local_barrier_id,
        I32Attr:$num_participants
  );
  let assemblyFormat = [{
    `(` $local_barrier_id `,` $num_participants `)` attr-dict
  }];
}

def AIE_CertRemoteBarrierOp: AIEX_Op<"cert.remote_barrier", [HasParent<"CertJobOp">]> {
  let summary = "Remote barrier operation";
  let description = [{
    This operation represents a remote barrier that synchronizes execution
    based on a barrier ID `remote_barrier_id` and `party_mask`, a bit map of
    participant uC IDs.

    Example:
    ```
      cert.remote_barrier(1, 0x3)
    ```
  }];
  let arguments = (
    ins I32Attr:$remote_barrier_id,
        I32Attr:$party_mask
  );
  let assemblyFormat = [{
    `(` $remote_barrier_id `,` $party_mask `)` attr-dict
  }];
}

def AIE_CertPreemptOp: AIEX_Op<"cert.preempt", [HasParent<"CertJobOp">]> {
  let summary = "Preemption point with save/restore control code references";
  let description = [{
    This operation defines a preemption point in control code. If preemption is
    required, the runtime will execute either the save or restore control code
    section depending on the context switch direction.

    - `id`: Preemption point index (must be consistent across all uCs in multi-uC case)
    - `save_section`: Symbol reference to control code section for saving state
    - `restore_section`: Symbol reference to control code section for restoring state

    This opcode should take one whole job which in turn should take one whole page.

    Example:
    ```
      cert.preempt(0, @save0, @restore0)
    ```
  }];
  let arguments = (
    ins I32Attr:$id,
        FlatSymbolRefAttr:$save_section,
        FlatSymbolRefAttr:$restore_section
  );
  let assemblyFormat = [{
    `(` $id `,` $save_section `,` $restore_section `)` attr-dict
  }];
}

def AIE_CertUcDmaWriteDesOp: AIEX_Op<"cert.uc_dma_write_des", [HasParent<"CertJobOp">]> {
  let summary = "uC DMA Write Descriptor operation";
  let description = [{
    Enqueues a DM2MM uC-DMA transfer and returns a wait handle in
    `wait_handle`, to be consumed by `cert.wait_uc_dma`. The symbol operand
    refers to a `cert.uc_dma_chain` (isa-spec.yaml).

    Example:
    ```
      cert.uc_dma_write_des(2, @dma_chain)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$wait_handle,
        FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    `(` $wait_handle `,` $symbol `)` attr-dict
  }];
}

def AIE_CertWaitUcDmaOp: AIEX_Op<"cert.wait_uc_dma", [HasParent<"CertJobOp">]> {
  let summary = "Wait for a uC-DMA wait handle to finish";
  let description = [{
    Waits for the specified uC-DMA wait handle to finish.

    The wait handle must be a job-private local register r0..r7 whose reaching
    definition inside the same job is a `cert.uc_dma_write_des`. The ISA also
    permits a global register r8..r23 to carry a handle across jobs
    (isa-spec.yaml), but this dialect deliberately imposes the stricter
    same-job rule and rejects a global wait handle.

    Example:
    ```
      cert.wait_uc_dma(2)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$wait_handle
  );
  let assemblyFormat = [{
    `(` $wait_handle `)` attr-dict
  }];
}

def AIE_CertRead32Op: AIEX_Op<"cert.read32", [HasParent<"CertJobOp">]> {
  let summary = "Read a 32-bit value from a constant address into a register";
  let description = [{
    Reads the value of a constant AXI-MM address and stores it in a register
    (isa-spec.yaml).

    Example:
    ```
      cert.read32(1, 0x02100000)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$value,
        UI32Attr:$address
  );
  let assemblyFormat = [{
    `(` $value `,` $address `)` attr-dict
  }];
}

def AIE_CertRead32DOp: AIEX_Op<"cert.read32_d", [HasParent<"CertJobOp">]> {
  let summary = "Read a 32-bit value from a register-held address into a register";
  let description = [{
    Reads the value of an AXI-MM address stored in register `address` and
    stores it into destination register `value` (isa-spec.yaml).

    Example:
    ```
      cert.read32_d(0, 1)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$address,
        AIEX_CertRegisterAttr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict
  }];
}

def AIE_CertAddOp: AIEX_Op<"cert.add", [HasParent<"CertJobOp">]> {
  let summary = "Add a constant to a register";
  let description = [{
    Adds `value` to the content of register `dest` and stores the result back
    in `dest`. `value` is UI32Attr, so to add -1 write `4294967295`/`0xffffffff`.

    Example:
    ```
      cert.add(0, 1)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$dest,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $dest `,` $value `)` attr-dict
  }];
}

def AIE_CertMovOp: AIEX_Op<"cert.mov", [HasParent<"CertJobOp">]> {
  let summary = "Move a constant into a register";
  let description = [{
    Stores the constant `value` in register `dest`.

    Example:
    ```
      cert.mov(0, 16)
    ```
  }];
  let arguments = (
    ins AIEX_CertRegisterAttr:$dest,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $dest `,` $value `)` attr-dict
  }];
}

def AIE_CertYieldOp: AIEX_Op<"cert.yield", [HasParent<"CertJobOp">]> {
  let summary = "Yield to another job";
  let description = [{
    Tells the scheduler to perform a context switch to the next available job.
    The current job will be re-scheduled in the next scheduling cycle.

    Example:
    ```
      cert.yield
    ```
  }];
  let assemblyFormat = [{ attr-dict }];
}

def AIE_CertWrite32DOp: AIEX_Op<"cert.write32_d", [HasParent<"CertJobOp">]> {
  let summary = "Dynamic write of a 32-bit value to an address";
  let description = [{
    Dynamic version of `cert.write32`: `address`/`value` may each be an
    immediate or come from a register.

    The ISA `flags` byte encodes is-REGISTER: bit0 `1 = address is a register
    number`, bit1 `1 = value is a register number` (confirmed against the
    CERT firmware, which contradicts isa-spec.yaml's prose but matches the
    spec's own worked example). These attributes map directly: present means
    the field is a register NUMBER, absent means it is an immediate.

    When `address_is_reg` is present, `$address` must be in [0,23]; same for
    `value_is_reg`/`$value`.

    Examples:
    ```
      cert.write32_d(68157440, 5)
      cert.write32_d(3, 5) {address_is_reg}
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$value,
        UnitAttr:$address_is_reg,
        UnitAttr:$value_is_reg
  );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict
  }];
}

def AIE_CertPoll32Op: AIEX_Op<"cert.poll32", [HasParent<"CertJobOp">]> {
  let summary = "Poll a constant address until it equals a value";
  let description = [{
    This poll is a busy poll: it blocks the current job (other jobs continue to
    execute) until the value read at `address` equals `value`.

    Example:
    ```
      cert.poll32(0x02100000, 1)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $value `)` attr-dict
  }];
}

def AIE_CertMaskPoll32Op: AIEX_Op<"cert.maskpoll32", [HasParent<"CertJobOp">]> {
  let summary = "Poll a constant address, masked, until it equals a value";
  let description = [{
    This poll is a busy poll: it blocks the current job until the value read at
    `address`, after `mask` is applied, equals `value`.

    Example:
    ```
      cert.maskpoll32(0x02100000, 0xf, 1)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$mask,
        UI32Attr:$value
  );
  let assemblyFormat = [{
    `(` $address `,` $mask `,` $value `)` attr-dict
  }];
}

def AIE_CertSleepOp: AIEX_Op<"cert.sleep", [HasParent<"CertJobOp">]> {
  let summary = "Busy-sleep for a number of microseconds";
  let description = [{
    Makes the current job busy-wait `target` microseconds without yielding to
    another job. Per the spec this opcode is for test purposes only
    (isa-spec.yaml).

    Example:
    ```
      cert.sleep(100)
    ```
  }];
  let arguments = (
    ins UI32Attr:$target
  );
  let assemblyFormat = [{
    `(` $target `)` attr-dict
  }];
}

def AIE_CertSaveTimestampsOp: AIEX_Op<"cert.save_timestamps", [HasParent<"CertJobOp">]> {
  let summary = "Save a timestamp with a unique id";
  let description = [{
    Saves the timestamp whenever control code encounters this opcode, tagged
    with the compiler-assigned unique identifier `unq_id`. The timestamps are
    saved in shared data memory.

    Example:
    ```
      cert.save_timestamps(7)
    ```
  }];
  let arguments = (
    ins UI32Attr:$unq_id
  );
  let assemblyFormat = [{
    `(` $unq_id `)` attr-dict
  }];
}

def AIE_CertSaveRegisterOp: AIEX_Op<"cert.save_register", [HasParent<"CertJobOp">]> {
  let summary = "Save a register's value with a unique id";
  let description = [{
    Reads the value from the specified register address and saves it, tagged
    with the compiler-assigned unique identifier `unq_id`, in shared data
    memory.

    Example:
    ```
      cert.save_register(0x02100000, 9)
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$unq_id
  );
  let assemblyFormat = [{
    `(` $address `,` $unq_id `)` attr-dict
  }];
}

def AIE_CertNopOp: AIEX_Op<"cert.nop", []> {
  let summary = "No operation (NOP)";
  let description = [{
    This operation represents a no-op instruction. It does nothing.

    Example:
    ```
      cert.nop
    ```
  }];
  let assemblyFormat = [{ attr-dict }];
}

def AIE_CertSectionOp: AIEX_Op<"cert.section",
                               [Symbol,
                                ParentOneOf<["AIE::DeviceOp", "CertAttachToGroupOp"]>,
                                SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Named control code section";
  let description = [{
    This operation defines a named section of control code that can be referenced
    by other operations (like cert.load_pdi or cert.preempt). It corresponds to the
    assembly pattern of `label: ... .endl label`.

    The section contains one or more jobs and can be loaded/executed by referencing
    its symbol.

    Example:
    ```
      cert.section @pdi0 {
        cert.job(10) {
          cert.write32(0x6A3000, 0x2)
        }
      }

      cert.job(0) {
        cert.load_pdi(0, @pdi0)
      }
    ```
  }];
  let arguments = (
    ins SymbolNameAttr:$sym_name
  );
  let regions = (region
    AnyRegion:$body
  );
  let assemblyFormat = [{
    $sym_name $body attr-dict
  }];
}

def AIE_CertLoadPdiOp: AIEX_Op<"cert.load_pdi", [HasParent<"CertJobOp">]> {
  let summary = "Load PDI (Platform Device Image)";
  let description = [{
    This operation loads a PDI (Platform Device Image), which is a piece of control code.
    It can be loaded by other control code at any time. The `pdi_id` is an ELF-wide unique
    identifier that specifies a unique PDI. Consecutive loading of the same PDI results in
    subsequent loads being skipped by the microcontroller. The `symbol` operand refers to
    a cert.section containing the PDI jobs.

    This opcode should take one whole job which in turn should take one whole page.

    Example:
    ```
      cert.load_pdi(0, @pdi_section)
    ```
  }];
  let arguments = (
    ins UI32Attr:$pdi_id,
        FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    `(` $pdi_id `,` $symbol `)` attr-dict
  }];
}

def AIE_CertAttachToGroupOp: AIEX_Op<"cert.attach_to_group", [SingleBlockImplicitTerminator<"AIE::EndOp">]> {
  let summary = "Attach to a group operation";
  let description = [{
    This operation attaches enclosed jobs to a specified group identified by an
    integer ID.

    Example:
    ```
      cert.attach_to_group(1) {
        ...
        cert.job(2) {
          ...
        }
        cert.job(3) {
          ...
        }
        ...
      }
    ```
  }];
  let arguments = (
    ins I32Attr:$group_id
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $group_id `)` $body attr-dict }];
  let extraClassDeclaration = [{
    int getGroupID() { return getGroupId(); }
  }];
}

#endif // CERT_OPS
