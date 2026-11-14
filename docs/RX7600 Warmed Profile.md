# RX 7600 warmed solver profile

Measured on 2026-10-29, after the integrated ROCm correctness and real-project
headless gates passed. This is a baseline for the integrated candidate, not an
isolated speedup measurement for any constituent PR or a claim of official AMD
support for this configuration.

## Configuration and provenance

- Windows 11 Pro 25H2, build 26200; Ryzen 7 5700X; 16 GiB host RAM.
- Radeon RX 7600, gfx1102, wave32, approximately 8 GiB dedicated memory.
- AMD driver 32.0.32015.7029; ROCm SDK 7.2; Julia 1.12.6; AMDGPU.jl 2.7.2.
- Two Julia threads and two OpenBLAS threads for every configuration.
- CPU reference: `1bb46da` (pristine BEAT 0.6.0).
- Integrated ROCm candidate: `a8f5d94fd67a2146a73d60083ef21d4638027ed8`.
- Benchmark tools: `509707fd340b9ae5d1f1f093c1eb326f1407ed95` (Windows memory sampling and output-identity fixes).

The machine was restarted to clear two stranded Julia kernel threads before
profiling. Every fresh worker started with no existing Julia process and exited
normally. No solver code changed during the sequence. The runner temporarily
prevented sleep. No competing solver or test workload ran during measurements.

Each configuration used three fresh workers interleaved with the other
configurations for that workload. Each worker performed warm-up followed by
three timed sweeps: 42 workers and 126 measured sweeps in total. The table below
compares only the third sweep from each fresh worker. Earlier sweeps remain in
the evidence; they are not pooled with the third sweeps. In particular, SAWMOD's
third CPU worker had a slower first sweep, and its second ROCm Float64 worker
had slower later sweeps. Those observations were retained. A later host-activity
sample did not identify the cause; it cannot rule out transient interference.

The timing includes per-request setup repeated by the real worker. Peak memory
is the largest sampled host process working set across all nine sweeps, sampled
every 20 ms. **Dedicated GPU VRAM is not measured**; these numbers are not total
accelerator memory or exact allocation peaks.

## Workloads

Source and compiled exterior use the full `sample.msh` at 125, 500 and 2000 Hz,
regular/singular quadrature orders 4/4. S218BP uses 50 and 125 Hz and SAWMOD uses
250 and 1000 Hz, both at fixed quadrature orders 2/2. Coupled runs preserve all
project excitations. SAWMOD includes its full 7,322 exterior observation points,
three ports, XY symmetry, and an interface-radiation observation. This differs
from the earlier small-probe headless correctness runs.

Source requests use Float32 for both reference and candidate. Compiled requests
use a Float64 CPU reference. Both requested ROCm precisions use Float64 dense
factors for coupled systems; Float32 outputs remain Float32. The exterior-only
factorization follows the requested precision. Comparisons between CPU Float64
and ROCm Float32 therefore change both backend and requested precision.

## Warmed results

Times are seconds, median [minimum, maximum] across three fresh workers' third
sweeps. Accuracy is the worst relative L2 error for each individual output at
each frequency across **all nine sweeps**, against the matching CPU-reference
sweep. Every output shape/key and finite value was checked. The limit was
5e-4, unchanged from qualification. Actual backend, compiled precision and
rocSOLVER dispatch were checked from result diagnostics.

| Workload | Backend / requested precision | Time (s) | Peak host working set (MiB) | Worst relative L2 |
|---|---|---:|---:|---:|
| source-sample | cpu-float32 | 2.00 [1.99, 2.05] | 498 | 0 |
| source-sample | rocm-float32 | 0.89 [0.89, 0.91] | 1412 | 3.27e-06 |
| compiled-exterior-sample | cpu-float64 | 4.04 [4.03, 4.27] | 657 | 0 |
| compiled-exterior-sample | rocm-float32 | 0.83 [0.83, 0.84] | 2105 | 3.13e-06 |
| compiled-exterior-sample | rocm-float64 | 4.26 [4.24, 4.27] | 2166 | 5.82e-15 |
| s218bp-monolithic | cpu-float64 | 19.67 [19.46, 20.01] | 3258 | 0 |
| s218bp-monolithic | rocm-float32 | 7.97 [7.95, 7.97] | 2186 | 5.47e-05 |
| s218bp-monolithic | rocm-float64 | 8.24 [8.24, 8.26] | 2185 | 1.51e-13 |
| s218bp-condensed | cpu-float64 | 1.92 [1.89, 1.96] | 648 | 0 |
| s218bp-condensed | rocm-float32 | 0.69 [0.69, 0.74] | 2209 | 5.39e-05 |
| s218bp-condensed | rocm-float64 | 1.05 [1.04, 1.08] | 2196 | 8.68e-14 |
| sawmod-condensed | cpu-float64 | 102.49 [101.38, 102.63] | 3386 | 0 |
| sawmod-condensed | rocm-float32 | 45.03 [44.59, 45.53] | 3244 | 1.31e-05 |
| sawmod-condensed | rocm-float64 | 72.05 [72.00, 79.55] | 3601 | 5.23e-12 |

## Coupled section timings

These are medians of the summed per-frequency stage timings in the third sweep.
Assembly includes BEM operators, FEM condensation and coupled factorization;
Schur extraction is nested inside FEM condensation. **Do not add the nested
columns to assembly.** Setup, output serialization and other stages explain
differences from total wall time. A dash means the stage is absent or its median
is below the comparator's 0.01-second reporting threshold.

| Workload | Backend / precision | Assembly | BEM operators | FEM condensation | Schur extraction | Coupled factorization | Field |
|---|---|---:|---:|---:|---:|---:|---:|
| s218bp-monolithic | cpu-float64 | 18.26 | 0.56 | — | — | 16.73 | 1.00 |
| s218bp-monolithic | rocm-float32 | 7.53 | 0.08 | — | — | 7.41 | 0.03 |
| s218bp-monolithic | rocm-float64 | 7.57 | 0.20 | — | — | 7.31 | 0.22 |
| s218bp-condensed | cpu-float64 | 0.67 | 0.51 | 0.13 | 0.04 | — | 1.00 |
| s218bp-condensed | rocm-float32 | 0.30 | 0.06 | 0.21 | 0.12 | 0.02 | 0.03 |
| s218bp-condensed | rocm-float64 | 0.45 | 0.19 | 0.20 | 0.13 | 0.02 | 0.24 |
| sawmod-condensed | cpu-float64 | 69.64 | 48.07 | 16.90 | 15.81 | 1.28 | 31.40 |
| sawmod-condensed | rocm-float32 | 41.93 | 2.11 | 29.94 | 28.70 | 8.67 | 1.23 |
| sawmod-condensed | rocm-float64 | 55.44 | 9.32 | 31.05 | 29.67 | 8.60 | 14.59 |

## Interpretation and next measurements

Static condensation removes the large monolithic dense-factorization cost on
S218BP. For SAWMOD, the largest remaining ROCm cost is host FEM Schur extraction.
Both CPU and ROCm use the same blocked UMFPACK helper, but their retained systems
differ: CPU condenses transducer surfaces and eliminates interface flux, solving
3,116 unknowns; ROCm retains those variables and solves 7,933. This architecture
difference is not evidence that changing only block width will yield a gain.

The controlled block-size experiment below found no demonstrated benefit from
a 32-column override on S218BP or SAWMOD; the default remains unchanged. Any later
transducer condensation change must preserve its motion/force corrections,
arbitrary-load RHS and full interior reconstruction. Interface elimination and
field batching require separate measurements and qualification. No optimization
gain is claimed here, and differences inside observed timing ranges are not
treated as significant.

This baseline does not qualify Boundary Lab Deploy's array workflow. Deploy
integration and serialized-array qualification remain separate work.

## Schur block-size experiment

The unchanged integrated ROCm Float32 candidate was measured with its default
Schur block selection and `BLAB_SCHUR_BLOCK=32`, using the same requests and
thread counts as the baseline. Three fresh workers per setting were interleaved
on each workload, each with three warmed sweeps. All 12 workers exited normally;
all 36 sweeps passed per-output/per-frequency accuracy checks against the CPU
Float64 reference, and diagnostics confirmed the actual block width and
rocSOLVER dispatch. Default widths were 49 on S218BP and 62 on SAWMOD.

Times below are third-sweep medians [minimum, maximum]. Schur extraction is a
subset of total wall time. Accuracy covers all nine sweeps per setting.

| Workload | Setting | Wall (s) | Schur extraction (s) | Worst relative L2 |
|---|---|---:|---:|---:|
| s218bp-condensed | default | 0.74 [0.69, 0.79] | 0.12 | 5.39e-05 |
| s218bp-condensed | block32 | 0.70 [0.70, 0.73] | 0.13 | 5.39e-05 |
| sawmod-condensed | default | 45.44 [44.81, 45.50] | 29.12 | 1.31e-05 |
| sawmod-condensed | block32 | 44.97 [44.93, 45.22] | 28.38 | 1.31e-05 |

The wall-time ranges overlap on both workloads. This does not establish a
speedup, so it does not justify a new block-size default. The larger remaining
target is the retained-system difference described above, which is being
addressed separately through transducer condensation with independent numerical
qualification and subsequent real-project measurements.

Local evidence: `runs/rx7600-profiling/schur-block-20261029-152244/`, containing
the plan, exact worker commands, logs, all output reports, comparisons and
`probe-audit.json`. Host-memory measurements retain the baseline's sampled
working-set limitation and exclude dedicated VRAM.

## Evidence and reproduction

Local evidence is under `runs/rx7600-profiling/warmed-20261029-134801/`:
`plan.json` pins commits, thread counts, requests, mesh paths and SHA-256 hashes;
`results.json` records every worker command and exit status; per-worker logs and
`*_r{1,2,3}.json` hold timings, diagnostics and full complex outputs;
`baseline-audit.json` records per-output/frequency accuracy and stage aggregates.
Generated run artifacts and machine-local request paths are not committed.

Use `scripts/benchmark_worker.py` with the recorded backend, precision, two
threads, three repeats, request/mesh and frequency arguments. Repeat with three
fresh workers per configuration, interleaved as described above, and compare
matched sweep indices using `scripts/compare_benchmarks.py`. Use the recorded
benchmark-tool revision (or the merged equivalents of PRs #88 and #89) for
Windows memory and output-identity handling. See [Benchmarking](Benchmarking.md)
for the measurement rules. The numerical candidate is an integration revision
of the scoped ROCm parity PRs; these results must not be attributed to one PR.
