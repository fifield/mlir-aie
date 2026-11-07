# AIE4 configuration delivery

How an AIE4 (npu3) full ELF configures the array, and what each dispatch must
guarantee for that to work. Applies to `aiecc --get-full-elf` builds of AIE4
devices; AIE2 and the other outputs are unchanged.

## Model

As on aie2p, the configuration of a device is loaded by a skippable `LOAD_PDI`.
aie2p attaches the kernel's PDI in `full_elf_config.json`; aiebu does not accept
PDIs for AIE4 control code, so on AIE4 the configuration is a cert section of
the control code instead:

```
program (all runtime sequences of the entry device, one cert program):
  job 1:      LOAD_PDI <id>, @main__site0      ; {cert.configure}, first on uC 0
  sequences:  the runtime sequences, in order; aiex.configure @dev becomes
              LOAD_PDI <id(dev)>, @dev__site<n>
section @main__site0: reset (Q0, QB, S, Q1, Q2, Q2p, Q2s, Q3), then the
                      configuration (core ELFs, locks, buffers, DMAs, switches,
                      core enables)
```

- `aie-npu-to-cert` turns the entry's generated configuration
  (`convert-aie-to-transaction`'s `@configure`) into a section named after
  the entry, and job 1 into a `LOAD_PDI` of it. An `aiex.npu.load_pdi` of the
  entry heading its first runtime sequence (IRON emits one) is that load and
  is not repeated.
- Each load site gets its own copy of the section (`@<dev>__site<n>`, own
  `.asm` file), cloned from one legalized template, so every copy of a device
  has the same PDI id.
- PDI ids are hashed from the section content, job ids included
  (`aie-cert-assign-pdi-ids`), so equal sections get equal ids across ELFs,
  whatever device or name they came from, and different sections different
  ids.
- The firmware skips reconfiguration when a `LOAD_PDI` id equals the id it
  loaded last on that uC. The instruction still updates recovery state and
  takes part in multi-uC synchronization. A repeated dispatch skips the
  configuration: nothing is reset or written again. A to B to A reloads A
  (only an immediate repeat is skipped). The first `LOAD_PDI` after a uC reset
  always loads its configuration.
- A dispatch that does load (the first dispatch of a hardware context, a load
  of another configuration, another ELF on the same context) runs the reset at
  the top of the section, then the configuration.

A pure dispatcher, whose program only configures other devices
(`aiex.configure`, `aiex.npu.load_pdi` or `cert.load_pdi` of another device,
or configuration written in line under `--expand-load-pdis`) and that
configures nothing beyond its own stream switch routing, gets no
configuration: the devices it configures set up their own routes after a
reset of the partition. It has no section, and the first load of a device it
dispatches leads its program, and is skipped on a repeated dispatch like any
other. An entry that configures resources of its own and also configures
other devices is an error: every configuration starts with a reset of the
partition, which would undo the entry's resources. Move them into a device
the entry configures.

For example, a program that loads A, then B, then A: on its first dispatch
all three loads run, each with its reset. On a repeated dispatch on the same
hardware context the first load of A is skipped (A is the configuration the
previous dispatch ended with) and B and A reload. A program that ends with a
different configuration than it starts with reloads everything on every
dispatch.

## Contract

A repeated dispatch is correct only if the previous dispatch left the array in
the post-configuration state of the section loaded last. This is the AIE2
contract (the AIE2 firmware also skips a reload of the same PDI). A design
must meet all of:

- cores loop forever (IRON `Worker(while_true=True)`, the default, or
  `range_(sys.maxsize)` in placed designs);
- cyclic DMA chains are balanced, so they end each dispatch where they started;
- bounded tasks are pushed again by the sequence on every dispatch;
- acquire-once fifos are re-armed with `aiex.dma_channel_reset_for`;
- outgoing streams are drained and shim tasks have completed before every
  configuration change: at the end of a dispatch, and before every load of
  another device inside a program. The stream switch clear (S) does not remove
  data already in switch FIFOs;
- re-arms do not race running cores: `dma_channel_reset_for` rewrites locks
  without stopping the cores. This is the same exposure as on AIE2.

The compiler does not check these conditions.

A design dispatched once per hardware context needs none of this: a new
hardware context loads the section again (checked on npu9 with a core that
runs to completion, one ELF dispatched once in each of two hardware contexts
created one after the other in one process). Designs dispatched more than
once on one context must loop their cores or use `--expand-load-pdis`.

## `--expand-load-pdis`

Every dispatch writes each configuration in line (`aiex.npu.write_config`)
instead of loading it, starting with the configure-time reset of the
partition, so designs with cores that finish, or other state a repeat must
rebuild, run correctly on every dispatch. On aie2p the expansion loads an
empty device ahead of each configuration so that the firmware resets the
array; the AIE4 firmware does not reset on a load, so on AIE4 there is no
empty device: an `aiex.npu.write_reset` writes the reset in line before each
configuration (on AIE4 an `aiex.npu.write_config` must follow a `write_reset`,
or a `write_teardown` standing in for it). `--expand-load-pdis` cannot be
combined with `cert.preempt` (rule 5), nor with a program on more than one
microcontroller (uC 0 would reset columns the other uCs may be using, with
nothing ordering them).

Per-sequence outputs (`--get-npu-insts`, `--get-npu-cpp`, `--get-elf`,
`--get-txn`, `--get-ctrlpkt`, `--get-locmap`) may accompany a full ELF: they
use their own lowering, which the full ELF's programs do not change.

## Preemption

On restore the firmware brings memtile data back from DDR, then
`LOAD_LAST_PDI` replays the whole section the uC loaded last, reset included,
then resumes after the `PREEMPT` job. Only memtile data is saved.
`aie-cert-verify` checks:

1. **No memtile data writes in a loaded section** of a design with
   `cert.preempt` (buffer initializers): the replay would overwrite the restored
   data. A `cert.write32_d` with `address_is_reg` in such a section is an error
   too, since its address is only known at run time.
2. *(No rule.)* A reset spanning several columns is safe on restore in a
   multi-uC program: every uC passes the control-code barrier after its
   memtile restore and before the first replayed section page, and again
   after the last one (cert `dpu/dpu_control.c`, the page loop's `need_sync`),
   so one uC's reset cannot abort another uC's memtile restore. This follows
   from the firmware source; preemption has not been exercised on hardware.
3. **Every `cert.preempt` needs an earlier `cert.load_pdi` on its uC**, in
   program order: `LOAD_LAST_PDI` uses the page index the last load recorded,
   which the firmware neither validates nor clears, so without a load it
   replays page 0 of the command or a stale page of another ELF.
4. **Preempt points sit at post-configuration state.** Not checked. The replay
   rebuilds only section state.
5. **`--expand-load-pdis` and `--load-pdi-to-ctrl-pkt` cannot be combined with
   `cert.preempt`**: restore replays the last loaded section (or the
   overlay), not the configuration written in line.

Not checked: `LOAD_CORES` metadata. The firmware records it even for a skipped
load and never clears it on `LOAD_PDI`. mlir-aie emits no `LOAD_CORES`, but a
command that used `LOAD_CORES`/`LOAD_CORES_CP` earlier on the same hardware
context (for example an aiebu production ELF) can leave a core replay target
that is wrong for a later command. Preemptible commands must not share a
hardware context with `LOAD_CORES` commands.

## Multi-uC programs

A program on more than one microcontroller (a `cert.attach_to_group` or
`cert.page` placement other than 0) keeps the entry's configuration as job
1, which runs on every dispatch: the firmware's post-load barrier counts
every uC of the command, and matching loads on the other uCs are not
generated yet.

Preemption in a multi-uC program needs an explicit `cert.load_pdi` on every
uC before its `cert.preempt` (rule 3 is checked per uC). Each such load must
hold the complete configuration of its uC's columns: restore replays only the
section loaded last, never job 1. And it is possible only if every uC with
code is a first uC of its column (group 0, 2 or 4 on npu3): aiebu attaches
save and restore code only to those groups, every uC with code must have the
same `cert.preempt` ids (G-preempt), and the firmware makes every uC of the
command meet at a barrier at each `cert.preempt`. So with code on group 1, 3
or 5, `cert.preempt` cannot be used at all, on any uC.

## What is reset

The reset, at the top of every configuration section and as
`aiex.npu.write_reset` before configuration written in line, visits every tile
of the partition, not only the device's, since a previous design may have left
state on tiles this one does not use. It has no option to turn it off, as the
aie2p firmware's reset on a load has none:

| Phase | Effect |
| --- | --- |
| Q0 | hold every core in reset, enable cleared |
| QB | read back each `Core_Control`, so the core resets land before any channel reset |
| S | zero every stream switch configuration register (512-bit and 32-bit switches: port configuration, deterministic merge) of every shim, memtile and core tile, and the packet slots of the tiles where the configuration that follows has packet rules (elsewhere S leaves no port in packet mode, so a stale slot is inert); about 6.4 KB of zeros on a full npu3 without packet routes |
| Q1, Q2 | assert, then release, the reset of every core-tile and memtile DMA channel, writing each channel's default mode |
| Q2p | clear every memtile MM2S constant pad value |
| Q2s | return the shim data channels to their default modes (they have no reset field) |
| Q3 | release the cores' reset, leaving them disabled; the configuration enables the cores it loads |

The configuration then zeroes every lock of a tile with a core, and every
declared memtile lock without an initializer, before writing initializers.

Not reset:

- memtile and core data memory: the DMA reset domain excludes BDs and memory,
  and core reset clears only PC and core state. Buffers with initializers are
  written again; everything else keeps its contents;
- BDs the configuration does not write;
- memtile locks the design does not declare, and shim locks without an
  initializer;
- trace, event and performance-counter state;
- data already in stream switch FIFOs.

`disableOtherDeviceSwitches` (which disables the connections of the module's
other devices of the same type) still runs in every AIE4 configuration: the
configuration is generated without the reset, which is added to it later.
After S its writes are redundant; it matters for control-packet output (which
never gets the reset) and for CDO output.

## Costs

Measured on npu9 with `test/npu-xrt/back_to_back_dispatch`
(`out = in + 1` over 1024 x i32, one column), host wall-clock per dispatch,
medians. The numbers drift from day to day; compare only runs taken together.

| Build | Each dispatch | Time per dispatch |
| --- | --- | --- |
| default | skips the load | 32 us (`seq`), 10 us (`runlist`), 12 us (`startall`) |
| `--expand-load-pdis` | reset + configuration, every dispatch | 92 us, measured with an earlier, larger reset; not re-measured |
| default, two ELFs alternating | reloads | 84 us (80 without S) |
| default, A, B, A in one program | skips the first A, reloads B and A | 130 us (127 without S) |

S adds about 2 to 4 us per reloading dispatch (the last two rows were
measured together, with and without S). On AIE4 a CERT page over the limit is
cut where the earlier page is as full as fits, so S adds as few pages as
possible. The first dispatch of a context took 130 to 533 us across runs, too
variable to measure the reset with. A dispatch that skips its load runs no
reset. See `test/npu-xrt/stream_reuse_elfs` for S on two separately compiled
ELFs sharing one hardware context.

## Known gaps

- Multi-uC programs stay on job 1 (no matching per-uC loads yet).
- Preemption deadlocks that only hand-written cert IR can reach are not
  checked: a `cert.preempt` in a section that is never loaded still counts
  for G-preempt's id consistency but never runs, and the order of preempts
  and user `remote_barrier`s on each uC (within one load phase) must match
  across the uCs that take part, since both pair up across uCs.
- The contract is not checked: cores that finish, unbalanced DMA chains,
  undrained streams and re-arm races all go undiagnosed.
- `LOAD_CORES` mixing on one hardware context is documented, not checked.
- 32-bit PDI ids: a hash collision between two ELFs on one context would skip
  a load that should run (about n^2 / 2^33 for n configurations). Two
  different sections loaded on the same uC of one program that get one id are
  an error.
- Per-site section copies grow the ELF with each extra load site of a device.
- Trace, event and performance-counter state survive a reload.
- A context idle for longer than its hysteresis (1 ms, the driver default)
  is torn down, and the uC is reset, which clears its last loaded id. That
  the teardown also resets the array comes from the driver's firmware
  interface, not from management firmware source. The design does not depend
  on it: the first load on a reset uC is never skipped and begins with the
  reset.
