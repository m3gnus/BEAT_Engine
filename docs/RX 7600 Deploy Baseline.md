# RX 7600 Deploy warmed array baseline

These measurements compare CPU and ROCm in the same qualified engine.
They are whole-backend comparisons, not the isolated cost of any one change.

## Configuration

- Windows RX 7600 (gfx1102, 8 GiB dedicated VRAM), Ryzen 7 5700X, 16 GiB RAM.
- ROCm SDK 7.2, Julia 1.12.6, AMDGPU.jl 2.7.2; two Julia and two OpenBLAS threads.
- Candidate engine: `4dadbd8772e1226ee1ad138fed82676ce8992b44`.
- Deploy application: `330863a6411cd68edd2f692897be612aef0005b9`.
- Accuracy reference: pristine CPU `1bb46da833a2a2d330d9b7d915a7ee5c31934812`.
- Deploy uses Float32 source-worker operators/results; ROCm dense factors are ComplexF64.
- Reference evidence: `profile-pilot-20261030-134823`; source commits, packages and payloads match this run.
- Frequencies: 79.31303405761719 and 162.02488708496094 Hz.
- Audience grid: 41 x 41 points, 20 x 20 m, 1.2 m high, starting 5 m from the array.
- Mixed scene: S218BP_LOD and SKHORN, close cabinet surfaces and ground, different level and delay.
- Four-cabinet scene: four S218BP_LOD cabinets at 1.5 m spacing, with different levels/delays and one reversed polarity.

Each sweep submits two independent audience solve requests through public Deploy
preparation and BEAT worker APIs. This includes application preparation and worker
transport; it is not a geometry-cached microphone frequency sweep. Each fresh
worker performs a full warm-up sweep followed by three measured sweeps. Three
fresh workers per backend/path/scene are interleaved: 24 workers, 72 measured
sweeps. Accuracy checks cover warm-up and every measured sweep, not only the
reported timing sample. Event logs and result artifacts are written outside the
timed interval. Source commits, package hashes and payload hashes are frozen.

All complex pressure, diaphragm-velocity and coil-current outputs are checked
against pristine CPU results with the unchanged 5e-4 per-output relative L2
limit. Result shapes, sample indices, frequencies, convergence and actual ROCm
dispatch are checked. The mixed scene also requires direct/ground near pairs.

## Third-sweep timing and accuracy

| Scene/path | Backend | Median [min, max] seconds | Peak host MiB | Worst relative L2 | Order |
|---|---|---:|---:|---:|---:|
| mixed/boundary | cpu | 17.074 [16.949, 19.453] | 745 | 0 | [2072] |
| mixed/boundary | rocm | 1.992 [1.980, 2.060] | 1540 | 2.268e-06 | [2072] |
| mixed/coupled | cpu | 17.522 [17.335, 18.507] | 826 | 0 | [2072] |
| mixed/coupled | rocm | 2.136 [2.107, 2.140] | 1592 | 1.192e-05 | [2072] |
| array4/boundary | cpu | 104.149 [103.837, 105.758] | 2191 | 0 | [5160] |
| array4/boundary | rocm | 7.802 [7.797, 7.837] | 2344 | 8.428e-06 | [5160] |
| array4/coupled | cpu | 104.906 [104.810, 104.969] | 2285 | 0 | [5160] |
| array4/coupled | rocm | 8.020 [8.002, 8.027] | 2379 | 1.181e-05 | [5160] |

Host memory is the maximum sampled Julia working set over all nine measured
sweeps for a configuration. Sampling is every 20 ms and excludes dedicated GPU
VRAM. It is not total CPU+GPU memory; short spikes may be missed.

## Interpretation

- mixed/boundary: ROCm median time changed by -88.3% relative to CPU. Observed ranges do not overlap.
- mixed/coupled: ROCm median time changed by -87.8% relative to CPU. Observed ranges do not overlap.
- array4/boundary: ROCm median time changed by -92.5% relative to CPU. Observed ranges do not overlap.
- array4/coupled: ROCm median time changed by -92.4% relative to CPU. Observed ranges do not overlap.

## Stage medians

Seconds summed across both frequencies in the third sweep. ROM factorization
and GMRES stages are nested within solve; do not add them to solve again.

| Scene/path/backend | Assembly | Solve | Field | ROM factorization | ROM GMRES |
|---|---:|---:|---:|---:|---:|
| mixed-boundary-cpu | 16.329 | 0.253 | 0.067 | 0.000 | 0.000 |
| mixed-boundary-rocm | 0.937 | 0.358 | 0.024 | 0.000 | 0.000 |
| mixed-coupled-cpu | 16.528 | 0.506 | 0.068 | 0.308 | 0.174 |
| mixed-coupled-rocm | 0.951 | 0.367 | 0.022 | 0.288 | 0.068 |
| array4-boundary-cpu | 100.009 | 3.215 | 0.166 | 0.000 | 0.000 |
| array4-boundary-rocm | 3.228 | 3.503 | 0.051 | 0.000 | 0.000 |
| array4-coupled-cpu | 99.801 | 4.108 | 0.166 | 3.035 | 0.928 |
| array4-coupled-rocm | 3.216 | 3.610 | 0.054 | 3.393 | 0.178 |

All repeats, per-frequency timings, end-of-sweep memory samples, full outputs,
commands and provenance remain in local evidence `profile-20261030-141204`. Generated
runs are not committed. These measurements apply to the tested machine and
scenes; they do not establish installed-runtime or all-GPU qualification.

## Profiling priorities

For the four-cabinet Coupled workload, the median assembly stage takes 3.216 s
and system construction/factorization takes 3.393 s. The factorization stage
includes building the dense system, precision conversion and a protective
matrix copy as well as LU; it is not an isolated rocSOLVER kernel measurement.
These are the first targets for separate RGP captures.

GMRES takes 0.178 s of the 8.020 s median wall time. Eliminating that entire
measured stage would save only about 2.2% on this workload, so migrating host
Krylov vectors is not the first optimization priority. Trace overhead must remain
outside these baseline timings. Float64 factors and the qualified numerical
checks remain required during any subsequent optimization.

A subsequent [warmed RGP capture](RX%207600%20RGP%20Capture.md) records static
kernel resource evidence and capture limitations separately from these timings.
