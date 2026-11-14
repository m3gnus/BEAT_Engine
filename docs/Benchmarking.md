# Benchmarking

Any change claimed to affect speed, memory or accuracy is measured this way
before the claim goes in a commit message or PR. The tools are
`scripts/benchmark_worker.py` (one run of one configuration through the real
worker) and `scripts/compare_benchmarks.py` (repeated runs, compared).

## Workloads

Use real projects on each solve path the change touches, and say which path:

| Path | Entry point | Workloads |
|---|---|---|
| Exterior radiator (source request) | `solver.jl` | `test_meshes/sample.msh` (1,390 dofs), `sample_detailed.msh` (3,502 dofs) |
| Exterior-only system solve (compiled request) | `coupled_solver.jl` | the same meshes, seeded as an exterior-only system |
| Coupled system solve (compiled request) | `coupled_solver.jl` | a Boundary Lab project, e.g. `examples/Multi_region_SAWMOD` (coupled FEM-BEM-LEM) |

Boundary Lab's GUI sends exterior projects as exterior-only system solves, not
source requests. Capture a Boundary Lab request with
`scripts/capture_boundary_lab_request.py` from a Boundary Lab checkout: pass a
project, or `--exterior-mesh` with a mesh and its radiator tag to seed an
exterior-only system the way the GUI does. The request references that
checkout's meshes, so it is not committed here.

## Runs

- Establish the run-to-run variance once per machine and workload: 3+ runs of
  one configuration, fresh workers, interleaved with other work. After that,
  one run per configuration is enough. Pass the variance to
  `compare_benchmarks.py --variance`; a change inside twice that is "close":
  add runs, and use the per-section timings to see where it comes from.
- `--repeats N` runs N sweeps in one warm worker, paying start-up and warm-up
  once. Sweep 1 matches a fresh worker's sweep (what a user pays on a first
  solve), including one-off costs the short warm-up does not trigger: on an M1
  Pro one frequency of the first sweep carried an extra ~0.3 s that later sweeps
  did not. Later sweeps are steady state. Compare sweep 1 with sweep 1, or
  repeats with repeats, never one with the other.
- Skip a configuration when the diff does not touch the logic on its code path
  and a measurement has confirmed it unchanged. Plumbing for another backend
  passing through the same files does not count as a change.
- Use an otherwise idle machine that does not sleep. Keep the Julia and BLAS
  thread counts fixed across configurations and record them with the chip (CPU
  and GPU core counts). The BLAS count also changes the answer: LU rounds
  differently when its work is split across a different number of threads, so
  a change that alters the count is not bit-identical to one that does not.
- Wall time includes the per-request setup the worker repeats for every sweep
  (mesh caches, maps), because an application pays it once per solve. Before
  calling a change a per-frequency slowdown, compare the first frequency with
  the rest: a one-off setup cost shows up only in the first frequency's
  `assembly_s`, and matters less the longer the sweep.

## What to report

- Wall time per sweep, and the per-section totals the worker reports
  (`assembly_s`, `solve_s`, `field_s`; for coupled solves also `bem_operator_s`,
  `fem_schur_extraction_s`, `coupled_factorization_s`, ...). A gain in one
  section can cost another; the totals show it.
- Memory: peak over the sweep and growth over the warmed worker. On macOS this
  is `phys_footprint`, which includes Metal buffers in unified memory; RSS does
  not. Linux reports RSS. Windows samples current process working-set bytes
  using [GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo).
  These Windows values measure resident host memory, not dedicated GPU VRAM.
  The peak is the maximum sampled within each sweep, not the process lifetime
  high-water mark; short spikes between 20 ms samples can be missed. Reports
  label the metric, process scope, and sampling interval. Unsupported platforms
  or failed reads store `null` and display `unavailable`, rather than zero.
- Accuracy against a reference: Float64 CPU on `main` for compiled requests,
  `main` CPU for source requests (Float32 only). Relative L2 over each output,
  and the worst dB error within 30 dB of each output's peak. The comparison
  checks every repeat and reports the worst of each metric across repeats.
  Quantities, frequencies, shapes and lengths must match the reference, and
  values must be finite; incomplete or invalid outputs stop the comparison.
  Benchmark output-key version 2 retains quantity and output ID, sweep index,
  and the full requested frequency. This preserves multiple pressure outputs,
  repeated frequencies, and closely spaced frequencies. Returned Float32
  frequencies are checked against their requested slot before canonicalization.
  Capture both candidate and reference with the same benchmark-tool version;
  older reports remain readable together but cannot be compared to version 2
  output keys. Re-capture old references rather than discard output identities.

## Overlap, pipelining and offload

Before implementing one, bound its gain from measured section timings. For a
coupled frequency with host work `S` that already overlaps the GPU, GPU work
`G`, and host work `L` that waits for the GPU, a cross-frequency pipeline saves
at most `0` if `G <= S`, otherwise `min(G - S, L)`. Consider other hardware: a
larger GPU shrinks `G`, more CPU cores shrink `S` and `L`. If the bound is near
zero on every realistic configuration, do not implement it.

When it does pay somewhere, the on/off choice comes from calibration or from
measurement during the run, never a constant fixed on one machine: machine
constants with environment overrides and a calibration script, as
`BeatEngineDenseSolve.jl` and `scripts/calibrate_dense_solve.jl` do. Unit-test
the decision function with synthetic timings, and gate that overlapped results
are bit-identical to sequential ones.
