# Coupled ROCm factorization precision

BEAT 0.6.0's projected interface coupling exposed a Float32 ROCm factorization
accuracy gap on the RX 7600. The existing voltage-driven coupled fixture passed
38 of 39 hardware assertions on pristine revision `1bb46da`; condensed versus
monolithic interface-flux error was `2.295359e-3`, exceeding its unchanged `2e-3`
limit. The same value occurred with the independent empty-ground-cache fix.

CPU coupled solves already factor the badly scaled pressure/flux/transducer
matrix in ComplexF64 even for Float32 requests. ROCm now follows that precision
policy while keeping the dense factorization and triangular solves on the GPU:

- BEM assembly and the assembled coupled coefficients retain request precision.
- Before coupled LU, a Float32 device matrix is converted to ComplexF64 and its
  original storage is released. Float64 requests do not need this conversion.
- RHS columns are uploaded in the factorization's precision.
- Returned acoustic and electromechanical traces remain ComplexF32 for Float32
  requests, including reconstructed FEM traces in hybrid condensed solves.
- Diagnostics expose `rocm_dense_factorization_precision`; other backends report
  null for this ROCm-specific field.

The exterior-only solver is unchanged. There is no CPU fallback for the ROCm
dense solve, and no change to quadrature, physical models, fixtures, or tolerance.

## Qualification evidence

On Windows 11 25H2, Radeon RX 7600, HIP SDK 7.2, Julia 1.12.6 and AMDGPU 2.7.2:

| Check | Pristine 0.6.0 | GPU Float64 coupled LU |
|---|---|---|
| Existing coupled hardware assertions | 38 passed, 1 failed | 39 passed |
| Condensed versus monolithic voltage flux relative error | `2.295359e-3` | `3.3344574e-7` |
| Monolithic voltage flux versus CPU relative error | Not recorded in the original failure log | `5.433864e-6` |
| GPU factor storage and requested output types | Not explicitly asserted | 28 additional assertions passed |

The expanded focused run passed 67/67 assertions. Its instrumented wrapper also
completed GPU synchronization, garbage collection, pool reclamation and normal
process exit. An earlier run passed its assertions but did not finish shutting
down promptly and required termination; that lifecycle observation remains a
qualification limitation, not a numerical pass for that earlier process.

The full candidate RX 7600 hardware gate passed with normal exit (611.87 s
including startup), including all 67 coupled assertions. The standalone CPU
reference gate also passed with normal exit (477.19 s). These elapsed times are
qualification records, not warmed performance measurements. The local logs are
under `runs/rx7600-phasor-20261027/`, including the pristine reproduction,
initial factorization experiment, instrumented lifecycle rerun, and full gates.

After a machine restart cleared the stranded Julia threads, warmed real-project
measurements completed on 2026-10-29. The integrated candidate `a8f5d94`, which
includes this precision policy and subsequent correctness fixes, passed all
126 measured sweeps across 42 fresh workers. Every worker exited normally, and
every output at every frequency satisfied the unchanged `5e-4` relative L2 limit.
The complete configuration, stage timings, ranges and accuracy results are in
[the RX 7600 warmed profile](https://github.com/JWSound/BEAT_Engine/pull/93).
Local evidence is under `runs/rx7600-profiling/warmed-20261029-134801/`.

## Cost and follow-up

ComplexF64 dense factors require twice the element storage of ComplexF32, and
Float64 throughput varies substantially by GPU. This change makes no speedup
claim. The completed integrated profile records the following coupled costs:

| Workload | Requested precision | Warmed time, median [min, max] (s) | Dense factorization median (s) |
|---|---|---:|---:|
| S218BP monolithic | Float32 | 7.97 [7.95, 7.97] | 7.41 |
| S218BP monolithic | Float64 | 8.24 [8.24, 8.26] | 7.31 |
| S218BP condensed | Float32 | 0.69 [0.69, 0.74] | 0.02 |
| S218BP condensed | Float64 | 1.05 [1.04, 1.08] | 0.02 |
| SAWMOD condensed | Float32 | 45.03 [44.59, 45.53] | 8.67 |
| SAWMOD condensed | Float64 | 72.05 [72.00, 79.55] | 8.60 |

These are matching third sweeps from three interleaved fresh workers per
configuration, using two Julia and two OpenBLAS threads. Both requested
precisions use Float64 dense factors. These integrated costs do not isolate
the overhead of this PR against the inaccurate Float32-factor implementation.
The profile measures host working set, excluding dedicated GPU VRAM.

Mixed-precision refinement or
equilibration must satisfy these unchanged accuracy gates and be compared with
this GPU Float64 reference, including ill-conditioned cases and multiple RHSs.
