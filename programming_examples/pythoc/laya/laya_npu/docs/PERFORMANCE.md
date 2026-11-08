# laya_npu performance

Measured 2026-10-31 on the current build: the rebased mlir-aie toolchain, with one firmware reset per configuration region. The fast B1/B2/B4 L64 graphs use the m2n2 dense leaf with the branchless GELU epilogue, and M35's native-float backbone attention (`l64_native`).

**M37 decision attention (2026-11-01):** the fast L64 graphs and the fast pair run decision attention on `device_05f` in place of device_08. Device submit time per call (paired with the previous stage, two opposite-order blocks; the tables below predate this):

| graph | before | after |
|---|---:|---:|
| fast/b1_l64 | 57.41 | 51.83 |
| fast/b2_l64 | 116.34 | 104.54 |
| fast/b4_l64 | 177.48 | 153.69 |
| pair/fast/b1_l64 | 140.45 | 128.62 |

This changes numerics: five fixtures were re-recorded. Worst fast screen, hidden L2 / probability: 0.0598 / 0.0034 (B1), 0.0432 / 0.0034 (B2), 0.0305 / 0.0057 (B4), argmax preserved, against gates of 0.1 / 0.01.

**M37 (2026-10-31):** exact B1/L64 and the exact and fast B1/L64 pairs now run the engine on `device_s8`, the B1 split on all 8 columns. Device submit time per call changed as follows:
- exact/b1_l64: 171.9 → 138.4 ms
- pair/exact: 279.2 → 212.1 ms
- pair/fast: 160.2 → 140.8 ms

These are paired measurements of the previous and new stage: 2 opposite-order blocks of 8 calls, outputs bit-identical. The pair/exact time is the graph's device time, not the benchmark's "shared" service row.

**M36 (2026-10-31):** fast B1/L64 now runs its backbone on the combined engine + attention device `device_cb`, one configuration instead of 57. Device submit time dropped from 100.3 to 57.3 ms per call. This is a paired measurement of the previous and new stage: 2 opposite-order blocks of 8 calls in one lock hold, with output bit-identical to the fixture. The tables below predate M36 and were not re-run; the other shapes are unchanged.

M35 sped up the fast L64 graphs; exact is unchanged. Fast B1/L64 went from 138.3 to 105.0 ms, B2 from 173.6 to 121.0 ms and B4 from 279.2 to 184.3 ms. The pair, shared, went from 212.0 to 166.6 ms. The GELU epilogue is bit-identical, and saves about 6/6/13 ms. Native attention changes fast numerics: it passes the fast screen at all three shapes, and the fixtures were re-recorded.

Status: **performance_benchmark_passed**. 2 opposite-order blocks, 5 measured calls per block, one cold call and one additional warmup per configuration/block. CPU threads: 8; memoization disabled; two-context, 2 GiB cache.

Wall time covers tokenization/preparation, graph execution, output readback, and CPU heads/formatting. Hash validation, journal checks, and evidence writes occur outside the timer. Model construction is excluded. Device submission covers runtime submit/wait, not pure kernel cycles. Cold context calls include context/buffer initialization; OS caches are not flushed. Pair time is for both independent requests; mixed time is for all three questions. Service timing uses the Inference API, not HTTP or the public Client wrapper.

All measurements hold the shared device and compiler locks. Every call validates outputs or serial/shared equivalence; the frozen driver journal is unchanged. Reported throughput is units divided by median resident-call latency, not sustained server throughput. P95 is a descriptive sample percentile from only 10 measurements.

## Eligible regular configurations

| Configuration | Median ms | P95 ms | Range ms | Device submit ms | Cold ms | Rows/s | Launches |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact/b1_l64 | 176.38 | 177.22 | 176.10–177.34 | 172.10 | 3467.18 | 5.67 | 1 |
| exact/b2_l64 | 211.37 | 211.90 | 210.61–211.99 | 206.46 | 3449.79 | 9.46 | 1 |
| exact/b4_l64 | 354.81 | 355.77 | 353.64–355.93 | 347.85 | 3646.59 | 11.27 | 1 |
| exact/b8_l64 | 640.20 | 641.60 | 639.15–641.93 | 630.68 | 4011.62 | 12.50 | 2 |
| exact/b1_l128 | 236.62 | 237.27 | 235.83–237.30 | 232.19 | 3518.31 | 4.23 | 1 |
| exact/b2_l128 | 406.00 | 407.15 | 405.07–407.30 | 400.18 | 3799.49 | 4.93 | 1 |
| exact/b4_l128 | 742.34 | 743.08 | 741.55–743.25 | 733.72 | 4131.42 | 5.39 | 1 |
| exact/b8_l128 | 1417.12 | 1418.08 | 1416.32–1418.52 | 1405.18 | 4990.07 | 5.65 | 2 |
| exact/b1_l256 | 524.89 | 526.03 | 524.42–526.06 | 519.22 | 3900.30 | 1.91 | 1 |
| exact/b2_l256 | 983.56 | 984.47 | 983.10–984.48 | 976.25 | 4398.76 | 2.03 | 2 |
| exact/b4_l256 | 1900.98 | 1901.57 | 1900.01–1901.59 | 1889.78 | 5500.14 | 2.10 | 3 |
| exact/b8_l256 | 3735.26 | 3736.53 | 3734.32–3736.54 | 3718.02 | 7656.31 | 2.14 | 5 |
| exact/b1_l512 | 1036.86 | 1037.42 | 1036.01–1037.47 | 1025.57 | 4554.98 | 0.96 | 2 |
| exact/b2_l512 | 2004.14 | 2004.93 | 2002.33–2005.12 | 1987.66 | 5677.29 | 1.00 | 3 |
| exact/b4_l512 | 3965.77 | 3966.43 | 3963.87–3966.45 | 3937.57 | 7985.53 | 1.01 | 5 |
| exact/b8_l512 | 7889.20 | 7891.24 | 7886.55–7891.74 | 7835.22 | 12713.56 | 1.01 | 10 |
| fast/b1_l64 | 104.95 | 105.59 | 104.28–105.69 | 100.69 | 3362.07 | 9.53 | 1 |
| fast/b2_l64 | 120.98 | 121.55 | 120.29–121.59 | 116.57 | 3338.47 | 16.53 | 1 |
| fast/b4_l64 | 184.30 | 185.10 | 183.85–185.35 | 178.12 | 3516.05 | 21.70 | 1 |

## Pair and mixed service comparisons

| Configuration | Median ms | P95 ms | Device submit ms | Cold ms | ms/unit | Launches |
|---|---:|---:|---:|---:|---:|---:|
| pair/exact/serial | 352.49 | 353.89 | 343.70 | 3733.70 | 176.24 | 2 |
| pair/exact/shared | 284.93 | 285.38 | 278.80 | 6923.71 | 142.47 | 1 |
| pair/fast/serial | 209.83 | 211.55 | 200.86 | 3572.62 | 104.92 | 2 |
| pair/fast/shared | 166.55 | 166.89 | 160.18 | 6815.46 | 83.27 | 1 |
| mixed/exact/serial | 358.94 | 360.04 | 349.14 | 6953.71 | 119.65 | 2 |
| mixed/exact/shared | 290.77 | 291.29 | 284.12 | 6857.37 | 96.92 | 1 |

## Resident-call speedups

- Fast B1/L64 versus exact: **1.681×**.
- Fast B2/L64 versus exact: **1.747×**.
- Fast B4/L64 versus exact: **1.925×**.
- Pair exact, shared versus serial: **1.237×**.
- Pair fast, shared versus serial: **1.260×**.
- Mixed exact, shared versus serial: **1.234×**.

The CSV retains both block medians to expose order effects. The 13 rejected fast shapes are outside this eligible-performance matrix.
