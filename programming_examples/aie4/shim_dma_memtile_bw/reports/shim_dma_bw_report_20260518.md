# Shim DMA Memtile Bandwidth Report

Date: 2026-05-18 UTC

Target: `AIEDevice.npu3`, full-ELF flow, Python PyXRT host

Benchmark directory: `programming_examples/aie4/shim_dma_memtile_bw`

## Summary

The benchmark builds and runs on hardware for the tested stable cases. The best
observed bandwidth in this run was **96.43 GB/s** using 6 shim DMA channels with
linear 512 KiB/channel transfers repeated 64 times per kernel launch.

Higher channel counts are not fully stable for long transfers in the current
implementation/environment. A 12-channel tiny-transfer smoke test passes, but
9-channel long-transfer tests time out in the host run path after the full ELF
loads. This needs follow-up before treating 9/12-channel long-transfer results
as valid bandwidth data.

## Build And Smoke Tests

| Test | Result | Notes |
| --- | ---: | --- |
| Python syntax check | PASS | `python -B -m py_compile shim_dma_bw_config.py shim_dma_bw_placed.py host.py` |
| 1-channel full-ELF smoke | PASS | 4 KiB/channel, 1 iteration, 1 sample |
| 9-channel tiny smoke | PASS | 4 KiB/channel, 1 iteration, 1 sample; 406 us |
| 12-channel tiny smoke | PASS | 4 KiB/channel, 1 iteration, 1 sample; 428 us |
| 12-channel full-ELF compile | PASS | 4 KiB/channel, 1 iteration |

## Linear Scaling

Configuration: `PATTERN=linear`, `BYTES_PER_CHANNEL=512KiB`,
`ITERATIONS=64`, `SAMPLES=5`, `WARMUPS=1`.

| Channels | Host Span | Bytes/Run | Best Time | Best BW | Avg Time | Avg BW | Samples (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 512.00 KiB | 32.00 MiB | 738 us | 45.48 GB/s | 742 us | 45.23 GB/s | 752, 738, 741, 738, 740 |
| 3 | 1.50 MiB | 96.00 MiB | 1435 us | 70.14 GB/s | 1539 us | 65.43 GB/s | 1562, 1558, 1548, 1590, 1435 |
| 6 | 3.00 MiB | 192.00 MiB | 2088 us | 96.43 GB/s | 2228 us | 90.35 GB/s | 2409, 2392, 2088, 2092, 2161 |

## Access Pattern Sweep

Configuration except where noted: `CHANNELS=6`, `ITERATIONS=64`,
`SAMPLES=5`, `WARMUPS=1`.

| Pattern | Per-Channel Payload | Host Span | Bytes/Run | Best Time | Best BW | Avg Time | Avg BW | Samples (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| linear | 512.00 KiB | 3.00 MiB | 192.00 MiB | 2088 us | 96.43 GB/s | 2228 us | 90.35 GB/s | 2409, 2392, 2088, 2092, 2161 |
| interleaved, 4 KiB blocks | 512.00 KiB | 3.00 MiB | 192.00 MiB | 2735 us | 73.60 GB/s | 2805 us | 71.77 GB/s | 2924, 2863, 2762, 2742, 2735 |
| block-strided, 4 KiB block / 4 KiB gap | 512.00 KiB | 5.98 MiB | 192.00 MiB | 2220 us | 90.70 GB/s | 2256 us | 89.26 GB/s | 2314, 2271, 2239, 2234, 2220 |
| strided, stride 16 elements | 4.00 KiB | 383.65 KiB | 1.50 MiB | 95 us | 16.64 GB/s | 103 us | 15.25 GB/s | 126, 104, 96, 95, 95 |

The strided result is not directly comparable to the 512 KiB/channel cases
because the BD dimension encoding requires a much smaller transfer size for the
chosen stride.

## Limits Observed

Host BO allocation: large host-only/ext BOs are limited on this machine. A direct
PyXRT allocation probe succeeded up to 8 MiB and failed at 12 MiB with an XRT
host memory allocation error. The benchmark sweep therefore used payloads whose
host buffer span stayed under that limit.

BD dimension limits: non-linear patterns with 16 MiB/channel exceeded the AIE4
BD dimension encoding range during lowering. The successful pattern sweep uses
512 KiB/channel for block/interleaved patterns and 4 KiB/channel for the
element-strided case.

Higher-channel long transfers: `CHANNELS=9 BYTES_PER_CHANNEL=512KiB
ITERATIONS=64` and `CHANNELS=9 BYTES_PER_CHANNEL=64KiB ITERATIONS=64` both
compiled but timed out in the host run path. `CHANNELS=9` with a 4 KiB single
iteration transfer passed, and `CHANNELS=12` with a 4 KiB single iteration
transfer passed. This points to a remaining runtime/design issue for longer
transfers once the third shim/memtile channel per column is active.

## Raw Logs

- `reports/shim_dma_bw_raw_20260518T212517Z.log`
- `reports/shim_dma_bw_stable_raw_20260518T215345Z.log`
- `reports/shim_dma_bw_raw_20260518T200229Z.log`
