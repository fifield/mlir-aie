# AIE4 Shim DMA Memtile Bandwidth

This benchmark measures host-memory read bandwidth through NPU3 shim MM2S DMA
channels into simple double-buffered memtile sinks. It generates a placed
design, builds a full ELF, and runs it from a Python PyXRT host.

The active channel count is configurable from 1 to 12. By default channels are
assigned round-robin across the three NPU3 columns before adding another channel
per column. The fourth stream in each column is routed to memtile S2MM channel
4, which avoids a row-1 switchbox conflict seen when channels 0 through 3 are
used directly.

## Build And Run

```sh
make run
```

Useful overrides:

```sh
make run CHANNELS=1
make run CHANNELS=3
make run CHANNELS=12 BYTES_PER_CHANNEL=16MiB ITERATIONS=8
make run CHANNELS=12 PATTERN=interleaved BLOCK_BYTES=4KiB
make run CHANNELS=6 PATTERN=block-strided BLOCK_BYTES=4KiB GAP_BYTES=4KiB
make run CHANNELS=3 PATTERN=strided STRIDE_ELEMENTS=16 BYTES_PER_CHANNEL=1MiB
```

`BYTES_PER_CHANNEL` is the payload transferred by each active shim channel in
one internal iteration. `ITERATIONS` repeats the same DMA BD inside one kernel
launch to amortize host launch overhead. The host reports bandwidth using:

```text
CHANNELS * BYTES_PER_CHANNEL * ITERATIONS / kernel_time
```

## Access Patterns

`linear` gives each channel a contiguous span.

`interleaved` stripes `BLOCK_BYTES` blocks across active channels so channels
read adjacent blocks in each round.

`block-strided` reads `BLOCK_BYTES` contiguous bytes, skips `GAP_BYTES`, and
repeats.

`strided` reads one 32-bit element every `STRIDE_ELEMENTS`. This can require a
large host buffer span, so reduce `BYTES_PER_CHANNEL` when using large strides.

## Files

`shim_dma_bw_placed.py` emits placed MLIR for `AIEDevice.npu3`.

`host.py` loads the generated full ELF with `pyxrt.elf` and
`pyxrt.ext.kernel`, allocates the host BO with `pyxrt.ext.bo` by default, runs
warmups and samples, and prints best and average bandwidth.

`shim_dma_bw_config.py` is shared between generator and host so the compiled
DMA access pattern and host-side accounting match.

## Historical Results

See the [2026-05-18 benchmark report](reports/shim_dma_bw_report_20260518.md)
for hardware measurements and observed limitations.
