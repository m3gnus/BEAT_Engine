# ROCm regular quadrature bounds checks

The regular SLP/adjoint DLP and hypersingular kernels now mark their nested
quadrature loops `@inbounds`. Cache construction establishes quadrature-array
dimensions and the guarded color-pair lookup provides valid mesh indices.
The scope retains the existing pair guards and singular handling. Arithmetic,
precision, launch geometry and numerical tolerances are unchanged.

## Qualification

The measured source candidate is `4542b562b7b45ad70476ed67ed0e08ea19c68339`,
compared with `4dadbd8772e1226ee1ad138fed82676ce8992b44`. The full standalone
CPU reference gate and 1,198 focused assertions passed: 168 native phasor,
192 coupled phasor, 768 near-correction, 26 ground-image and 44 Deploy checks.
The phasor checks cover both precisions and conventions, symmetry, ground and
near-cavity-pole cases with and without condensation. Fixtures were not changed.

The core comparison completed 54 fresh workers and 162 measured sweeps; the
Deploy comparison completed 24 fresh workers and 72 measured sweeps. Every
measured output and frequency passed the independent CPU-reference comparison
at the unchanged relative L2 limit of 5e-4. Deploy warm-ups were also audited.
The core reference is pristine CPU `1bb46da`; compiled requests use Float64,
while source requests and the Deploy reference use Float32.

## Measured effect

- Exterior source and compiled Float32 workloads improved across all matched
  repeats. Third-repeat median sweep time fell by 19.9% and 19.5%, respectively.
- Compiled exterior Float64 improved across all matched repeats; the third
  repeat fell by 2.2%.
- S218 monolithic, S218 condensed Float32 and SAWMOD in both precisions have
  overlapping wall-time ranges. No whole-workload speedup or regression is
  established for those configurations.
- S218 condensed Float64 has separate, faster ranges in repeats one and three,
  but overlap in repeat two. It does not support an all-repeat improvement claim.
- Deploy's mixed and four-cabinet Boundary/Coupled workloads improved across
  all matched repeats. Third-repeat median reductions span 6.4–10.9%.

Full median/range, stage, memory and accuracy tables are in the
[core measurements](RX%207600%20Bounds%20Core%20Measurements.md) and
[Deploy measurements](RX%207600%20Bounds%20Deploy%20Measurements.md).
Neither comparison has a non-overlapping whole-workload regression.

## Reproduction and limits

Measurements used Windows 11, Ryzen 7 5700X (8 cores/16 threads), RX 7600
(gfx1102, 8 GiB), ROCm SDK 7.2, Julia 1.12.6 and AMDGPU.jl 2.7.2.
Julia and OpenBLAS each used two threads. Three interleaved fresh workers per
configuration each performed warm-up and three measured sweeps; matching repeat
numbers were compared. RGP capture was disabled during timing.

The core benchmark tools were frozen at
`509707fd340b9ae5d1f1f093c1eb326f1407ed95`. Deploy used application
`330863a6411cd68edd2f692897be612aef0005b9`. Saved plans contain input hashes;
the runner verified source identities and input hashes between workers.
Windows memory uses sampled process working set at 20 ms intervals, supplied
by the benchmark-tools branch; it excludes dedicated GPU memory. This differs
from the older benchmark tool's unavailable Windows-memory result.

RGP static register/spill metadata motivated kernel inspection. It does not
establish dynamic stall behavior or the precise cause of the timing gain.
These results cover the listed workloads on this host, not all AMD GPUs or
an installed application runtime. Generated logs and raw results remain local
under `runs/rx7600-profiling/pair-bounds-core-20261030-161411` and
`runs/rx7600-deploy/pair-bounds-comparison-20261030-155013`.

All 81 matched core measured-sweep pairs have identical encoded numerical
outputs. All 48 matched Deploy sweep pairs, including warm-ups, have zero
relative L2 difference between baseline and candidate.
