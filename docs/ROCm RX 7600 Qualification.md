# RX 7600 native Windows qualification

Qualification date: 2026-10-27. Baseline engine revision:
`2bb0724286083f98e6bde82f98a6fcbd7326ce83` (BEAT 0.5.1).

This record separates executed numerical gates from features that remain
unqualified. No numerical baselines, fixture hashes, solver arithmetic, or
acceptance tolerances were changed for this qualification.

## Environment

| Component | Configuration |
|---|---|
| GPU | AMD Radeon RX 7600, gfx1102, wavefront 32, 7.984 GiB |
| CPU | AMD Ryzen 7 5700X, 8 cores |
| Host memory | 17,098,964,992 bytes (approximately 16 GiB) |
| OS | Windows 11 Pro 25H2, x64, build 26200 |
| Display driver | 32.0.32015.7029 |
| SDK | Native Windows ROCm/HIP SDK 7.2, `C:\Program Files\AMD\ROCm\7.2` |
| Julia | 1.12.6 |
| AMDGPU.jl | 2.7.2, checked-in `julia_rocm` manifest |
| Libraries reported by AMDGPU | HIP 7.2.60201, rocBLAS 5.2.0, rocSOLVER 3.32.0 |
| Qualification threads | Julia 2; `OPENBLAS_NUM_THREADS=2` for the recorded batch |

ROCm manifest SHA-256:
`a30cf4a02545d0c79555f626dbdb2a629c325082c2b425e3eaddc447e701282b`.
AMDGPU reports the runtime, rocBLAS, and rocSOLVER functional. MIOpen is
unavailable; these solver gates do not require it. WSL is not used.

The initial standalone regular gate ran with Julia 2 threads and ambient BLAS
settings; subsequent batch runs explicitly set the BLAS environment above.
Individual tests may override BLAS policy internally.

## Results

**PASS for the executed negative-time Float32 ROCm baseline.** Full logs, worker
source hashes, commands, elapsed times, and machine metadata are stored locally
in `runs/rx7600-qualification-20261027/` (git-ignored). The initial fixture-path
failure is recorded separately from the successful corrected run.

| Gate | Outcome | Evidence |
|---|---|---|
| Native regular assembly | Pass | Maximum operator relative error `1.554866e-7`; existing limit `5e-5` |
| Exterior, q1/s2 | Pass | Maximum operator error `9.437878e-7` against `1e-6`; pressure `3.458864e-6`; field `7.122855e-7`; residual `2.066596e-6` |
| X symmetry | Pass | Maximum operator error `9.166162e-7`; pressure `1.699759e-6`; field `2.543856e-7` |
| XY symmetry | Pass | Maximum operator error `9.044219e-7`; pressure `1.604106e-6`; field `3.559735e-7` |
| Standalone coupled, q1/s1 | Pass after fixture-path correction | Diagnostic, monolithic GPU, and hybrid condensed comparisons passed all existing limits |
| Hardware qualification runner | Pass, exit 0 | 584.4 s including startup; `FP32 ROCm GPU-resident coupled solve`: **39/39 passed**, not skipped |
| Standalone CPU reference gate | Pass, exit 0 | 449.5 s including startup; frozen fixture integrity and full coupled reference checks passed |
| Exterior, q4/s4 | Pass | Maximum operator error `6.949631e-6` against `1e-5`; pressure `8.336152e-5`; field `1.553916e-5`; residual `1.421050e-5` |

Exterior pressure, field and residual limits are `5e-3`. Symmetry uses a `2e-6`
operator limit and `5e-3` pressure/field limits. The exterior fixture has 2,776
faces and 1,390 P1 degrees of freedom; X and XY fixtures have 1,602 and 798
faces respectively. Dedicated validators run at 500 Hz. The native default was
`rocm_native_colored_pair_owned`; no backend/kernel environment overrides were
present at the start of qualification.

The coupled standalone rerun reduced the solved order from 2,164 to 1,485.
Monolithic relative differences versus CPU were `1.059228e-5` for BEM pressure,
`8.220432e-6` for FEM pressure and `1.026130e-5` for the exterior field.
Condensed-versus-monolithic differences were `1.877011e-5`, `1.932456e-5` and
`1.966630e-5` respectively. Both multi-RHS linearity errors were zero on this
fixture; pressure-continuity error was zero and flux-conservation error was
`1.522759e-7`.

The poorly conditioned voltage fixture has limited interface-flux margin:
monolithic ROCm versus CPU measured `1.850488e-3` against the existing `2e-3`
limit; diagnostic versus CPU was `1.276247e-3`, and condensed versus monolithic
ROCm was `1.464121e-3`.
The validator already documents this conditioning-sensitive exception. Physical
pressure, velocity and current comparisons passed their stricter limits. Preserve
these values as a watch point for subsequent changes, not grounds to widen the
gate. The diagnostic voltage residual was `8.654154e-7`, below `2e-6`.

The ordinary suite reported expected CUDA/Metal production-pipeline skips. Its
CPU refinement fallback warning belongs to a deliberately singular test case;
the refinement testset passed. No ROCm corrupt-LAPACK-status retry was reported
in the successful hardware, coupled, or exterior logs.

## Harness correction

The original `validate_rocm_coupled.jl` aborted before solving because its fixture
root resolved to `C:\blab\tests\fixtures`, outside the engine distribution.
It now uses `julia_local/tests/fixtures`, matching the ordinary coupled suite.
The original failure remains in `coupled.log`; the successful rerun is recorded
in `coupled-fixed.log`.
This changes fixture discovery only.

## Scope and next gate

The dedicated ROCm validators exercise Float32 negative-time solves. Their
same-discretization CPU comparisons do not establish arbitrary-geometry accuracy
or full Float64 ROCm qualification. CPU reference success is separate evidence,
not a substitute for GPU execution.

Positive-time ROCm worker requests remain blocked. The existing native phasor
test contains CPU, CUDA, and Metal checks, but no ROCm arm. A passing ordinary
suite must not be interpreted as ROCm positive-time qualification. The next
development stage needs explicit ROCm conjugation tests, complex excitations,
coupled monolithic/condensed coverage, and retained-field checks before changing
the advertised capabilities or request guards.

Other remaining coverage includes ground symmetry, exterior transducers on
ROCm, broader frequencies and geometries, and long-lived worker memory behavior.
These runs are correctness gates, not a benchmark comparison. No speedup or peak
GPU-memory claim is made from compilation-inclusive timings.

## Reproduction

From the engine checkout, use its installed Python environment and Julia 1.12.6:

```powershell
$env:OPENBLAS_NUM_THREADS = '2'
$env:BEAT_BACKEND = 'rocm'
python -m beat_engine doctor --backend rocm --threads 2
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_rocm src/beat_engine/julia_local/scripts/validate_rocm_native_regular.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_rocm src/beat_engine/julia_local/scripts/validate_rocm_exterior.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_rocm src/beat_engine/julia_local/scripts/validate_rocm_symmetry.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_rocm src/beat_engine/julia_local/scripts/validate_rocm_coupled.jl
python scripts/qualify_accelerator.py
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
$env:BLAB_VALIDATE_REGULAR_ORDER = '4'
$env:BLAB_VALIDATE_SINGULAR_ORDER = '4'
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_rocm src/beat_engine/julia_local/scripts/validate_rocm_exterior.jl
Remove-Item Env:BLAB_VALIDATE_REGULAR_ORDER, Env:BLAB_VALIDATE_SINGULAR_ORDER
```

The hardware runner enables `BLAB_RUN_COUPLED_ROCM=1` for `runtests.jl`.
Inspect its output for actual ROCm test execution, not only process success.
