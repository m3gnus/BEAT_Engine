# ROCm Deploy arrays

Deploy's source worker accepts `beat_engine_backend: "rocm"` for fixed-Neumann
Boundary solves and parity Petrov-Galerkin ROM Coupled solves. The ready event
advertises `rocm` in `deploy_solver_backends`; consumers must also check the
backend's runtime availability. An installed ROCm package alone is insufficient:
the availability handshake requires a functional device, rocBLAS and rocSOLVER.

## Numerical path

Regular, singular and near-corrected boundary operators are assembled on ROCm,
including rigid-ground images. Operators and identity blocks stay on the device.
Both solve modes use a ComplexF64 rocSOLVER factorization of the assembled
Burton-Miller matrix. Each application of the Neumann right-hand side uses
device matrix-vector products, promotes the resulting vector for the dense
solve, then returns pressure in the operator precision.

For Coupled solves, ROM response and GMRES vectors remain on the host. Every
feedback application reuses the same device factors for that frequency. This
preserves the CPU algorithm's physical feedback, convergence and sweep warm-start
contracts without requiring CUDA's projected or compressed RHS experiments.
Those CUDA-specific experiments and cross-request factor reuse remain guarded.

Audience fields are evaluated on ROCm. Retained boundary solutions support field
updates without repeating assembly or factorization; frequency sweeps reuse
geometry caches. Device factors, temporary right-hand sides and backend caches
have explicit release paths. Boundary pressures and Neumann traces retained
between requests are host arrays, so `gpu_resident_field` remains false; that
diagnostic does not imply CPU field evaluation.

Solve diagnostics report:

- `backend: "rocm"`;
- `linear_solver: "rocm_rocsolver_dense_lu"`;
- `dense_factorization_precision: "float64"`;
- `near_correction_backend: "rocm"` and `field_backend: "rocm"`;
- `rom_krylov_backend: "cpu"` for Coupled solves.

## Qualification

The standalone `julia_local/tests/deploy_rocm_tests.jl` gate requires working
ROCm hardware; it does not skip unavailable devices. Its 44 assertions cover
Float32/Float64 operators, both phasor conventions, repeated complex right-hand
sides, and low-rank feedback compared with an independently assembled full
complex system. It also checks the warm-start result and actual Float64 factors.
The test is included in `scripts/qualify_accelerator.py` for ROCm.

On the Windows RX 7600 host (gfx1102, ROCm SDK 7.2, Julia 1.12.6,
AMDGPU.jl 2.7.2), the native gate, standalone CPU numerical references and
focused CPU GMRES gate passed. Engine Python tests passed all 231 cases.

Initial real-package qualification used Boundary Lab Deploy's public request
preparation and BEAT worker APIs with two Julia/OpenBLAS threads. Both Boundary
and Coupled paths passed individual solves, cached field updates and two-frequency
sweeps for:

| Scene | Worst CPU/ROCm pressure relative difference |
|---|---:|
| One S218BP_LOD cabinet | 5.96785e-6 |
| Mixed S218BP_LOD/SKHORN, close surfaces and ground | 3.41061e-6 |

The unchanged comparison limit is 5e-4. Coupled diaphragm velocity and coil
current comparisons also passed, as did GMRES convergence and warm-start checks.
The mixed scene exercised nonzero direct and ground-image near-correction pairs.
Numerical source hashes were verified against both runs; neither fixture data
nor tolerances were changed.

Local artifacts are under `runs/rx7600-deploy/`: `single-initial`,
`mixed-initial`, `initial-qualification-audit.json`, `cpu-reference.log` and
`cpu-gmres.log`. Generated artifacts remain untracked. The single and mixed
comparisons are correctness qualification, not interleaved warmed performance
measurements. No CPU/ROCm speed claim is made from them.

The final `mixed-worker-final` run used Deploy's updated `_worker` and capability
checks, rather than the initial test adapter. Both fidelities, cached fields and
two-frequency sweeps passed again, with worst pressure relative difference
3.31103e-6. It verified native ROCm solve/factor-precision/field diagnostics and
nonzero direct and ground-image near pairs. All workers exited; no Julia process
remained after qualification.

These results qualify the tested GPU/software configuration and workloads. They
do not establish support for every AMD device or replace validation of a future
packaged Deploy runtime. Release dependency pins are handled independently of
this engine change.

The subsequent [RX 7600 warmed array baseline](RX%207600%20Deploy%20Baseline.md)
completed 24 interleaved fresh workers and 72 measured sweeps. Its independent
output audit passed for every warm-up and measured sweep. Assembly and dense
factorization dominate the four-cabinet ROCm path; GMRES and field evaluation
are smaller costs. The report gives matched-repeat medians, ranges, stage
timings, host memory, accuracy and the limits of these measurements.
