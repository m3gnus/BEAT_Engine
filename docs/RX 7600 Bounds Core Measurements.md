# RX 7600 regular-pair bounds: core workloads

Baseline `4dadbd8772e1226ee1ad138fed82676ce8992b44`; candidate `4542b562b7b45ad70476ed67ed0e08ea19c68339`.

The comparison uses three interleaved fresh workers for each variant and
precision, with three measured sweeps after warm-up. Julia and OpenBLAS
each use two threads. RGP is not active. All 54 workers completed normally.
The independent audit checked every measured output and frequency against
pristine CPU references at the unchanged 5e-4 relative L2 limit. Source
requests use CPU Float32 references; compiled requests use CPU Float64.

## Workloads

| Case | Frequencies (Hz) | Precision |
|---|---|---|
| source-sample | 125.0, 500.0, 2000.0 | float32 |
| compiled-exterior-sample | 125.0, 500.0, 2000.0 | float32, float64 |
| s218bp-monolithic | 50.0, 125.0 | float32, float64 |
| s218bp-condensed | 50.0, 125.0 | float32, float64 |
| sawmod-condensed | 250.0, 1000.0 | float32, float64 |

## Matched-repeat wall time

Seconds, median [minimum, maximum] across three fresh workers. Overlapping
ranges do not establish a whole-workload improvement or regression.

| Case / precision | Repeat | Baseline | Candidate | Median reduction | Ranges |
|---|---:|---:|---:|---:|---|
| source-sample-float32 | 1 | 0.968 [0.933, 0.973] | 0.771 [0.729, 0.778] | 20.4% | candidate faster, separate |
| source-sample-float32 | 2 | 0.951 [0.935, 0.963] | 0.777 [0.727, 0.786] | 18.3% | candidate faster, separate |
| source-sample-float32 | 3 | 0.951 [0.950, 0.958] | 0.761 [0.734, 0.776] | 19.9% | candidate faster, separate |
| compiled-exterior-sample-float32 | 1 | 0.973 [0.862, 0.979] | 0.810 [0.728, 0.814] | 16.7% | candidate faster, separate |
| compiled-exterior-sample-float32 | 2 | 0.883 [0.866, 0.884] | 0.726 [0.723, 0.728] | 17.8% | candidate faster, separate |
| compiled-exterior-sample-float32 | 3 | 0.896 [0.885, 0.896] | 0.721 [0.716, 0.725] | 19.5% | candidate faster, separate |
| compiled-exterior-sample-float64 | 1 | 4.410 [4.318, 4.439] | 4.276 [4.196, 4.293] | 3.0% | candidate faster, separate |
| compiled-exterior-sample-float64 | 2 | 4.275 [4.271, 4.304] | 4.182 [4.172, 4.198] | 2.2% | candidate faster, separate |
| compiled-exterior-sample-float64 | 3 | 4.294 [4.257, 4.314] | 4.198 [4.193, 4.204] | 2.2% | candidate faster, separate |
| s218bp-monolithic-float32 | 1 | 8.022 [7.966, 8.063] | 8.021 [7.921, 8.054] | 0.0% | overlap |
| s218bp-monolithic-float32 | 2 | 7.924 [7.901, 7.991] | 7.995 [7.955, 8.108] | -0.9% | overlap |
| s218bp-monolithic-float32 | 3 | 7.953 [7.952, 7.959] | 7.955 [7.948, 8.045] | -0.0% | overlap |
| s218bp-monolithic-float64 | 1 | 8.251 [8.231, 8.264] | 8.261 [8.244, 8.341] | -0.1% | overlap |
| s218bp-monolithic-float64 | 2 | 8.265 [8.221, 8.267] | 8.214 [8.193, 8.247] | 0.6% | overlap |
| s218bp-monolithic-float64 | 3 | 8.286 [8.249, 8.287] | 8.260 [8.207, 8.307] | 0.3% | overlap |
| s218bp-condensed-float32 | 1 | 0.702 [0.632, 0.781] | 0.674 [0.634, 0.696] | 4.0% | overlap |
| s218bp-condensed-float32 | 2 | 0.667 [0.633, 0.687] | 0.615 [0.608, 0.676] | 7.8% | overlap |
| s218bp-condensed-float32 | 3 | 0.670 [0.641, 0.728] | 0.631 [0.604, 0.660] | 5.8% | overlap |
| s218bp-condensed-float64 | 1 | 1.056 [1.032, 1.068] | 1.009 [0.988, 1.021] | 4.4% | candidate faster, separate |
| s218bp-condensed-float64 | 2 | 1.032 [1.002, 1.039] | 1.016 [0.982, 1.079] | 1.5% | overlap |
| s218bp-condensed-float64 | 3 | 1.002 [0.992, 1.051] | 0.969 [0.959, 0.976] | 3.2% | candidate faster, separate |
| sawmod-condensed-float32 | 1 | 33.420 [30.676, 34.182] | 32.779 [30.268, 33.625] | 1.9% | overlap |
| sawmod-condensed-float32 | 2 | 32.802 [32.597, 34.245] | 32.067 [30.613, 33.081] | 2.2% | overlap |
| sawmod-condensed-float32 | 3 | 32.408 [32.032, 33.803] | 32.053 [30.841, 32.499] | 1.1% | overlap |
| sawmod-condensed-float64 | 1 | 58.702 [57.469, 60.837] | 58.953 [58.815, 60.737] | -0.4% | overlap |
| sawmod-condensed-float64 | 2 | 60.123 [59.596, 61.928] | 60.291 [57.007, 60.763] | -0.3% | overlap |
| sawmod-condensed-float64 | 3 | 60.552 [59.914, 60.696] | 59.107 [57.336, 60.171] | 2.4% | overlap |

## Third-repeat stages and memory

Stage seconds are summed over the sweep. Solve, field and factorization
columns report medians. Coupled factorization occurs during system preparation
and is included in assembly, not the subsequent solve stage. Host memory is the maximum
sampled working set over all measured repeats and excludes dedicated VRAM.
The JSON audit retains every reported stage and its range for all repeats.

| Case / precision / variant | Assembly median [min, max] | Solve | Field | Coupled factorization | Peak host MiB | Worst relative L2 | Worst dB within 30 dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| source-sample-float32-baseline | 0.524 [0.522, 0.527] | 0.038 | 0.027 | not separately reported | 1388 | 3.270e-06 | 9.760e-05 |
| source-sample-float32-candidate | 0.357 [0.353, 0.359] | 0.038 | 0.028 | not separately reported | 1427 | 3.270e-06 | 9.760e-05 |
| compiled-exterior-sample-float32-baseline | 0.526 [0.523, 0.529] | 0.037 | 0.026 | not separately reported | 2194 | 3.128e-06 | 7.307e-05 |
| compiled-exterior-sample-float32-candidate | 0.355 [0.351, 0.359] | 0.037 | 0.027 | not separately reported | 2189 | 3.128e-06 | 7.307e-05 |
| compiled-exterior-sample-float64-baseline | 3.716 [3.713, 3.741] | 0.119 | 0.132 | not separately reported | 2209 | 5.824e-15 | 1.061e-13 |
| compiled-exterior-sample-float64-candidate | 3.615 [3.605, 3.620] | 0.118 | 0.132 | not separately reported | 2213 | 5.824e-15 | 1.061e-13 |
| s218bp-monolithic-float32-baseline | 7.503 [7.498, 7.504] | 0.011 | 0.035 | 7.379 | 2293 | 5.469e-05 | 9.364e-04 |
| s218bp-monolithic-float32-candidate | 7.498 [7.488, 7.542] | 0.011 | 0.035 | 7.385 | 2294 | 5.469e-05 | 9.364e-04 |
| s218bp-monolithic-float64-baseline | 7.562 [7.551, 7.574] | 0.011 | 0.223 | 7.287 | 2246 | 1.508e-13 | 1.404e-12 |
| s218bp-monolithic-float64-candidate | 7.534 [7.532, 7.569] | 0.011 | 0.223 | 7.291 | 2264 | 1.508e-13 | 1.404e-12 |
| s218bp-condensed-float32-baseline | 0.241 [0.235, 0.300] | 0.005 | 0.036 | 0.012 | 2260 | 5.469e-05 | 9.350e-04 |
| s218bp-condensed-float32-candidate | 0.241 [0.210, 0.264] | 0.005 | 0.035 | 0.024 | 2254 | 5.469e-05 | 9.350e-04 |
| s218bp-condensed-float64-baseline | 0.392 [0.374, 0.396] | 0.005 | 0.235 | 0.012 | 2294 | 2.077e-14 | 2.980e-13 |
| s218bp-condensed-float64-candidate | 0.355 [0.346, 0.370] | 0.005 | 0.237 | 0.012 | 2303 | 2.077e-14 | 2.980e-13 |
| sawmod-condensed-float32-baseline | 29.384 [29.160, 31.015] | 0.245 | 1.265 | 4.585 | 3083 | 1.331e-05 | 8.154e-04 |
| sawmod-condensed-float32-candidate | 29.211 [27.927, 29.574] | 0.237 | 1.259 | 4.578 | 3023 | 1.331e-05 | 8.154e-04 |
| sawmod-condensed-float64-baseline | 43.225 [42.365, 43.277] | 0.243 | 15.549 | 4.589 | 3564 | 1.429e-13 | 9.489e-12 |
| sawmod-condensed-float64-candidate | 41.426 [39.974, 42.571] | 0.240 | 15.574 | 4.581 | 3604 | 1.429e-13 | 9.489e-12 |

These measurements qualify the tested Windows RX 7600 configuration and
listed workloads. They are not claims about all AMD devices or installed
application runtimes. Precision, fixtures and tolerances were unchanged.
Local evidence: `pair-bounds-core-20261030-161411`; generated results remain untracked.
