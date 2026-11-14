# RX 7600 transducer condensation profile

This experiment compares the same numerical implementation with transducer
condensation explicitly off and on. It does not establish release qualification
or change the default. Full hardware and headless application gates are separate.

## Configuration and evidence

- Windows 11 Pro 25H2; Ryzen 7 5700X; 16 GiB host RAM.
- Radeon RX 7600 (gfx1102, approximately 8 GiB VRAM); ROCm SDK 7.2.
- Julia 1.12.6; AMDGPU.jl 2.7.2; two Julia and two OpenBLAS threads.
- Measured numerical commit: `6c01dad86e68ab878ce439feff4a3e52224e6b27`; its working-tree source hashes were verified against `plan.json`.
- Integration base: `a8f5d94fd67a2146a73d60083ef21d4638027ed8`.
- Benchmark tools: `509707fd340b9ae5d1f1f093c1eb326f1407ed95`.
- Profile evidence: `condensation-profile-20261029-161531`.
- Focused qualification: `condensation-gates-20261029-160945`.

Each workload and precision used three interleaved fresh workers per mode,
with warm-up and three measured sweeps per worker: 24 workers and 72 sweeps.
All workers exited normally. No solver source changed during profiling.
All completed outputs were compared individually at every frequency against
the pristine BEAT 0.6.0 CPU Float64 reference, using the unchanged 5e-4
relative L2 limit. Actual ROCm dispatch, requested precision, Float64 dense
factors and the selected condensation mode were checked in diagnostics.

S218BP uses 50/125 Hz; SAWMOD uses 250/1000 Hz. Both use fixed regular
and singular quadrature orders 2/2. The requests preserve all excitations.
SAWMOD includes its full 7,322 exterior points, three ports, XY symmetry
and an interface-radiation observation.

## Warmed results

Times compare the third sweep across three fresh workers; earlier sweeps
remain in the evidence and are included in accuracy checks. Memory is the
maximum sampled host working set across all nine sweeps, sampled every
20 ms. Dedicated GPU VRAM is excluded; this is not total GPU memory.

| Workload | Precision | Condensation | Time median [min, max] (s) | Peak host working set (MiB) | Worst relative L2 | Dense system order |
|---|---|---|---:|---:|---:|---:|
| s218bp-condensed | float32 | off | 0.734 [0.694, 0.776] | 2231 | 5.39e-05 | 705 |
| s218bp-condensed | float32 | on | 0.631 [0.614, 0.699] | 2197 | 5.47e-05 | 495 |
| s218bp-condensed | float64 | off | 1.070 [1.046, 1.084] | 2212 | 8.68e-14 | 705 |
| s218bp-condensed | float64 | on | 0.965 [0.957, 1.015] | 2220 | 2.08e-14 | 495 |
| sawmod-condensed | float32 | off | 44.930 [44.843, 45.216] | 3304 | 1.31e-05 | 7933 |
| sawmod-condensed | float32 | on | 28.366 [28.036, 28.368] | 2920 | 1.33e-05 | 6320 |
| sawmod-condensed | float64 | off | 72.035 [71.420, 72.130] | 3555 | 5.23e-12 | 7933 |
| sawmod-condensed | float64 | on | 55.005 [54.863, 55.167] | 3321 | 1.43e-13 | 6320 |

## Timing interpretation

- s218bp-condensed, float32: median time changed by -13.9%. Observed ranges overlap; the median difference alone does not establish a robust wall-time improvement.
- s218bp-condensed, float64: median time changed by -9.8%. Observed ranges do not overlap for this workload and precision.
- sawmod-condensed, float32: median time changed by -36.9%. Observed ranges do not overlap for this workload and precision.
- sawmod-condensed, float64: median time changed by -23.6%. Observed ranges do not overlap for this workload and precision.

## Stage timings

Values are medians of summed per-frequency stages in the third sweep.
FEM condensation, Schur extraction and factorization are nested within
assembly. Do not add the nested columns to assembly. A dash means a
stage is absent or below the comparator's 0.01-second reporting threshold.

| Workload | Variant | Assembly | BEM | FEM condensation | Schur extraction | Dense factorization | Field |
|---|---|---:|---:|---:|---:|---:|---:|
| s218bp-condensed | float32-off | 0.316 | 0.056 | 0.223 | 0.122 | 0.021 | 0.034 |
| s218bp-condensed | float32-on | 0.237 | 0.056 | 0.152 | 0.043 | 0.012 | 0.034 |
| s218bp-condensed | float64-off | 0.458 | 0.190 | 0.197 | 0.123 | 0.021 | 0.236 |
| s218bp-condensed | float64-on | 0.364 | 0.189 | 0.118 | 0.041 | 0.012 | 0.233 |
| sawmod-condensed | float32-off | 41.833 | 2.115 | 29.847 | 28.582 | 8.667 | 1.234 |
| sawmod-condensed | float32-on | 25.544 | 2.099 | 17.356 | 15.961 | 4.605 | 1.233 |
| sawmod-condensed | float64-off | 54.410 | 9.332 | 30.288 | 28.779 | 8.569 | 15.576 |
| sawmod-condensed | float64-on | 37.526 | 9.364 | 17.284 | 15.918 | 4.604 | 15.532 |

Automatic-selection qualification is recorded separately in [ROCm transducer condensation](ROCm%20Transducer%20Condensation.md#rx-7600-qualification).
