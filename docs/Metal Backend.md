# BEAT Engine Apple Metal

BEAT Engine Apple Metal is the engine's Apple Silicon GPU backend, used by
Boundary Lab and by any other client of the worker. It
uses the same mesh model, Burton-Miller formulation, symmetry rules, and result
protocol as the other BEAT Engine backends while moving the dense BEM operator
assembly and exterior-field evaluation to the GPU through Metal.jl.

The backend supports:

- exterior Burton-Miller BEM solves;
- coupled FEM-BEM-LEM physical-system solves;
- `off`, `x`, and `xy` symmetry;
- GPU-resident regular and singular operator assembly;
- GPU exterior-field evaluation for polar, spherical, and arbitrary observation
  points.

Production solves use `Float32` and `ComplexF32`, which is also the only
floating-point precision Apple GPUs provide. Boundary Lab's [BEAT Engine
Core](https://github.com/JWSound/boundary-lab/blob/main/docs/advanced/beat-engine-core.md)
notes describe the shared boundary-integral formulation.

## Compiled-system requests

Select the backend with `solver_options.bem_backend = "metal"` in a
compiled-system request; Boundary Lab exposes this as **BEAT Engine (Apple
Metal)** (`beat_metal`). The worker advertises `metal` in its ready handshake
when Metal.jl reports a functional device, with both phasor conventions: the
kernels receive the signed outgoing wavenumber at host entry exactly as CUDA
and ROCm do, and the fused Burton-Miller kernels combine the coupling with that
same signed value. See [Phasor Convention](Phasor%20Convention.md).

Exterior solves honour `burton_miller_assembly`. The default `direct_system`
is the fused path described below, which forms the system on the GPU without
the four operators and factorizes once on the host; `operator_matrices`
assembles the four operators and combines them on the host, and the diagnostic
kernel modes (`host_staged` assembly, the `host` singular mode, the reference
regular kernels) fall back to it. Diagnostics report the effective mode and a
`linear_solver` of `metal_assembly_cpu_dense_lu` or `metal_assembly_cpu_dense_gmres`.
The `BLAB_BEAT_FUSED_BM` variable below governs the source-request driver only.

Condensed coupled solves select direct A/C assembly with
`BLAB_METAL_COUPLED_BEM_ASSEMBLY=auto|combined|operators` (see below).
The request option `coupled_bem_assembly` remains the CUDA control; it does not
select this Metal path. Monolithic Metal solves and full-matrix validation
continue to assemble individual operators.

## Deploy exterior and speaker-ROM requests

The source worker supports Metal for `boundary_lab_deploy_solve`,
`boundary_lab_deploy_rom`, microphone sweeps, and cached field requests.
`deploy_solver_backends` in the source and compiled worker handshakes advertises
the engine distribution's Deploy capabilities; device availability is separate.

Deploy assembles four operators on Metal, presents them to CPU dense LU, and
uses the existing host GMRES speaker-ROM feedback solve for Level 3. It retains
Metal geometry across sweep frequencies and a Metal field cache for audience
updates. The effective assembly mode is `operator_matrices`; Deploy does not
yet use fused Metal direct-system assembly.

Metal operator assembly now accepts host `near_correction_cache` and
`image_near_correction_cache` values. CPU quadrature replaces the regular-rule
contribution for those pairs before symmetry row weighting. Shared buffers are
edited in place after GPU synchronization; private buffers are copied back.
Device-only near caches remain rejected. The general source request's automatic
near-pair selection remains CPU-only; this extension serves callers, such as
Deploy, that supply explicit correction caches. Diagnostics identify CPU near
corrections and the host dense solver separately from the Metal field backend.

Run `tests/metal_near_correction_tests.jl` under the Metal Julia project for the
hardware gate: CPU operator agreement for direct and ground-image corrections,
both phasor conventions, both storage modes, and two quadrature rules. Hardware
qualification invokes this gate; it fails if no functional Metal device exists.

## Execution model

Exterior solves take the fused Burton-Miller path described below, which never
forms the four operators. The four-operator path described here remains the
reference for coupled solves and is used by monolithic coupled solves, `host_staged` assembly and
the `host` singular mode. `BLAB_BEAT_FUSED_BM=0` selects it for exterior solves
too; condensed coupled solves have their own control below.

The worker prepares mesh topology, quadrature rules, symmetry transforms, and
frequency-independent cache data on the CPU. The Metal path then:

1. allocates the single-layer, double-layer, adjoint double-layer, and
   hypersingular matrices as `MtlArray` objects;
2. evaluates regular Galerkin pairs in chunks of trial elements: one thread
   per element pair on a two-dimensional grid writes the pair's 3x1 and 3x3
   operator blocks to a device buffer, every Green's-function value used for
   all four operators, and gather kernels with one owner per matrix entry
   sum the buffer into the dense operators (no atomics, fixed summation
   order);
3. evaluates adjacent and coincident pairs with Duffy singular quadrature in
   one fused kernel per pair and scatters their compact correction blocks
   into the dense operators;
4. applies symmetry-image contributions and reduced-domain row weights;
5. wraps the four operators as host arrays in place -- they live in Metal
   shared storage, so nothing is copied -- forms the Burton-Miller system, and
   factors it once per frequency with LAPACK on the CPU, reusing that
   factorization across every channel drive; and
6. evaluates the exterior field with Metal kernels.

### Coupled FEM-BEM-LEM

Coupled solves take the CPU backend's shape with the BEM stage moved to the
GPU: sparse FEM assembly and the UMFPACK interior Schur complement run on the
CPU, the combined BEM matrices are assembled on Metal and wrapped on the host,
and the retained coupled system is factored with the CPU dense LU. The original
four-operator route remains available as a reference and diagnostic fallback. The
condensed formulation is the default, exactly as for the CPU backend, and the
monolithic formulation remains available for validation. Interior-FEM-only
solves have no BEM stage and run on the CPU path unchanged.

The dense factorization stays on the CPU because Metal.jl provides no GPU LU,
and the CPU LU runs through Accelerate-class BLAS. The backend's dense-size
ceiling is therefore the same as the CPU backend's.

There is no device-to-host transfer. The operators are allocated in Metal
shared storage and handed to the CPU as `unsafe_wrap`ped `Array`s over the same
buffers, so `metal_host_operators` costs nothing measurable. This is not what
a private-storage buffer does: `Array(::MtlArray)` on private storage blits
through a staging buffer at a measured 3.5-8 GB/s, which was 1.1-1.5 s per
frequency at 10,230 P1 dofs (five gigabytes of operators). Shared storage does
not slow the assembly kernels -- 4.34-4.41 s against 4.37-4.95 s private on the
same mesh -- and reading the operators back on the CPU is about 1.5x slower
per byte than reading a host copy, which costs about 0.19 s of matrix formation
and is far less than the transfer it removes. `BLAB_METAL_OPERATOR_STORAGE=private`
restores the copying path.

The host arrays alias device memory, so the operator storage is released once,
through whichever tuple the caller still holds: the host tuple returned by
`metal_host_operators` owns the buffers it wrapped, and freeing the device
tuple while those views are still live leaves them dangling.

### Combined coupled Burton-Miller assembly (condensed solves)

`BeatEngineMetalCoupledBurtonMiller.jl` reuses the exterior fused regular and
Duffy pair arithmetic and the cached gather tables, retaining the flux
coefficient rather than multiplying it by a known drive. With P1 pressure
(N rows) and DP0 flux (F columns), it assembles

```text
A = 0.5 Mpp - D + alpha H
C = S + alpha (adjD + 0.5 Mpq)
A p + C Q q_interface + C bem_motion_flux v = -C prescribed_neumann
```

For positive physical k, alpha is +i/k under `exp(-i omega t)` and -i/k under
`exp(+i omega t)`. The kernels receive the signed outgoing k, exactly as the
existing exterior fused path does. Interface orientation, FEM condensation,
MUMPS, dense refinement and pivoted LU remain unchanged.

Regular pairs accumulate 24 real components (nine complex A entries and three
complex -C entries), rather than 48 for S/D/adjD/H. The same fused pair buffer,
trial chunks and deterministic P1 gather are reused. A new gather retains each
DP0 column with one owner per cell. Singular pairs use the existing fused Duffy
blocks, with the existing P1/P1 and P1/DP0 correction maps and deterministic
entry gather. Direct and image-singular pairs use the same skip rules, reflected
normal/curl signs and quadrature as the operator path. Operator row weights
are applied before sparse identity scatter; that pass also changes -C to C.
The identity matrices already carry their symmetry weights. Their sparse device
scatter caches are created once per selected quadrature order on first combined
use, owned by the condensed cache and freed with it. That first-use setup is
included in `fem_system_s`; subsequent frequencies reuse it.

**Variant: retain C, project on the host.** A and C occupy shared Metal storage
by default, so the host wraps them without copying. The host reads C to form
`C*Q`, `C*bem_motion_flux` and `-C*prescribed_neumann`; Q stays sparse. A is
copied once into an owned host matrix before the device buffers are freed,
so all blocks survive assembly storage release. Private storage copies A/C
at host entry and follows the same ownership rule. C is then released; the
condensed build retains only the projected blocks. No four-operator host
combination or dense Q is allocated.

Dense device outputs fall from `2N² + 2NF` to `N² + NF` complex values
(ComplexF32 is eight bytes), while pair scratch uses 96 instead of 192 bytes
per pair at a fixed chunk width. The existing budget-based chunk chooser may
use the same scratch budget with twice the width. During projection the host
also holds one `N²` copy of A and `N*I` interface entries, plus small motion and
prescribed blocks, where I is the interface column count. Retaining C costs
`N*F` rather than only `N*I`; this first step avoids adding a new Metal sparse
projection kernel and keeps the host projections identical to the reference.
Device projection could reduce host bandwidth further when I is much smaller
than F. **Symmetry images still use separate pair launches and gathers**,
following the existing Metal exterior machinery; fusing multiple images into
one accumulator set remains future work. This is a smaller port than CUDA's
fully image-fused, device-projected implementation.

`auto` selects combined for Float32, native assembly, `pair_gather`, native
singular correction, gather write-back, 1/3/6-point triangle rules and `off`,
`x`, `xy` symmetry. Unsupported configurations use operators and record
`coupled_bem_assembly_fallback_reason` plus an optimization fallback reason.
`combined` requires support and errors with the reason otherwise. `operators`
runs the original four-operator path. Results report the effective
`coupled_bem_assembly`; image fusion is false. Full-matrix diagnostics still
require the monolithic formulation and use operators; the existing prohibition
on full diagnostics with static condensation is unchanged. Restart persistent
workers after updating the numerical sources; both new sources enter the
existing source-hash walk.

Expected numerical differences are Float32 summation-order round-off: combination
now occurs before outer-product expansion and gather, rather than after four
independent sums. The qualification gate is `norm(delta) <= 1e-12 +
5e-6*norm(reference)` for A, C and the projected blocks, matching the existing
fused exterior gate's relative tolerance. This is an expected fixture bound,
not a universal bound for cancellation or an ill-conditioned solve. Full
solution gates are 1e-3 relative with 1e-7 absolute, and 2e-3 for interface flux,
following the coupled validation's conditioning allowance. No precision or
pivoting is weakened to meet these gates.

`validate_metal_coupled_combined.jl` compares A/C and all three projection signs,
then complete condensed prescribed-velocity, complex voltage and prescribed-BEM
source solutions on the bundled coupled fixtures for `off/x/xy` and both phasor
conventions. The full fixtures span the mirror planes, so symmetry arms translate
both meshes together into a valid positive fundamental domain (separate mirrored
objects). A small tetrahedral matrix check with vertices on the planes covers
orbit weights and image-singular corrections, shared/private storage, multiple
trial chunks, q1/q2 rules and singular part splits. The script rejects nonfinite
outputs and shape differences and exits nonzero on a failed gate. CPU-only
policy, projection signs, empty maps, gather indexing and output ownership tests are in
`tests/metal_coupled_host_tests.jl`; Metal bundle support probes also run without
a functional GPU in `tests/metal_host_tests.jl`.

Measured on an M1 Max (eight Julia threads), Multi_region_SAWMOD, 40 frequencies
from 20 Hz to 20 kHz, warm worker, MUMPS FEM solver, with the coupled sweep
pipeline and its automatic on/off decision. Three runs per configuration,
interleaved (operators / combined / combined / operators / operators / combined),
on an otherwise idle machine:

| | Four operators | Combined |
| --- | ---: | ---: |
| Sweep, median (range) | 52.6 s (51.8-53.1) | **42.0 s** (41.5-42.0) |
| Peak worker memory | 4.8-5.2 GB | 3.8-3.9 GB |

The sweep pipeline assembles the four-operator BEM stage one frequency ahead;
the combined stage (`bem_operator_s` 0.29 s against 0.82 s for the four
operators on the same tree with the pipeline off) is short enough that the
pipeline decides to stay off.

`validate_metal_coupled_combined.jl` passes for `off`, `x` and `xy` with both
phasor conventions: A and C agree with the four-operator path to about 3e-7
relative (Float32 summation order) and the coupled solutions to 1e-7-5e-6; its
sweep check keeps one system alive while two more frequencies assemble on the
same cache, one of them on another task, and requires the first system's answer
to be unchanged. Under the coupled sweep pipeline the GPU stage is already
hidden; there the gain is the smaller producer and less memory traffic beside
the host stages.

### Fused Burton-Miller assembly (exterior solves)

The coupling eta = i/k is known at assembly time, so the exterior path forms

    lhs = 0.5 M_p1p1 - D + (i/k) H          rhs = (-S - (i/k)(K' + 0.5 M_p1dp0)) q

inside the assembly kernels and never allocates S, K', D or H. That is one
N x N matrix and one right-hand side per drive instead of 6N^2 complex entries,
a measured **6.0x** reduction in operator memory on every mesh tried, and
because the storage is O(N^2) it is sqrt(6) ~ 2.45x more dofs at the same peak.

Every channel's Neumann column is built before assembly and folded in during
the same pass, so one assembly still serves the whole channel set at a
frequency exactly as one factorization does.

The win is *not* the halved stores. The combination is applied to the pair's
3-vectors before the rank-1 expansion, not to four finished blocks afterwards:
one 3x3 expansion per test point instead of two, one 3x1 instead of two, and 24
live accumulator floats instead of 48. Measured on the pair kernel alone,
combining afterwards is 1.00-1.02x and combining before the expansion is
1.82-1.87x. Whole assembly is 2.12-2.77x faster over the ATH ladder from 1,974
to 10,230 dofs. The per-quadrature-point arithmetic is unchanged and cannot
change: D and H carry different geometric prefactors per entry, so both terms
are evaluated whatever they accumulate into. Only the expansion collapses, and
only because the hypersingular curl term carries no basis product and can be
summed as one scalar and expanded after the loop.

The singular correction is fused the same way and for the same reason. The
Duffy/Sauter-Schwab pair carried 50 live accumulator floats and expanded the
rank-1 outer product four times per quadrature point pair; combining inside the
loop and hoisting the loop-invariant curl term leaves 26 live floats and two
expansions. The singular block kernel is **2.7x** faster on both an M1 Max
1,209-dof symmetry-reduced mesh (20.7 to 7.7 ms) and a 4,552-dof full mesh
(84.1 to 30.9 ms), for a 14-17% shorter assembly and a 11-14% shorter
40-frequency sweep. Device memory high-water is unchanged to the byte: the
kernel writes the same `pair_count x part_count x 12` complex scratch through
the same scatter kernel.

`_metal_singular_pair_blocks` in `BeatEngineMetalAtomicKernels.jl` is the
four-operator path's singular pair and is deliberately untouched by the fusion,
so it stays an independent reference rather than a copy of the code under test.

`scripts/validate_metal_fused_burton_miller.jl` gates it by comparing the fused
system against the four-operator system on the same mesh, frequency and
quadrature, where the two differ only by float32 summation order.
`scripts/validate_metal_singular_summation.jl` bounds that summation-order
difference directly, per singular pair, against a Float64 evaluation of the
same algebra.

The Burton-Miller right-hand side is applied matrix-free. The operator
`-S - (i/k)(K' + 0.5 M)` is N x 2N complex -- 1.67 GB at 10,230 P1 dofs -- and
was materialised once per frequency only to be multiplied by a drive vector;
three matrix-vector products replace it. The left-hand side broadcasts the real
identity block directly instead of promoting it to a full complex copy. Between
them these were 0.84-1.16 s of a 6.3-6.7 s solve at 10,230 dofs, and about
2.5 GB of allocation per frequency.

The fused exterior kernels pack quadrature positions, normals and curls into
`float4` loads. Trial quadrature is unrolled from a compile-time rule tuple;
the test loop remains a runtime loop. Each symmetry image accumulates into the
same pair block before one gather per chunk. Singular pairs pack the full
Duffy rule coordinates and triangle vertices, group pairs by rule length, and
specialize point and part counts. Every frequency uses the full singular rule.
These kernels and the packed Float32 field kernel originate in PR #15, commits
`7c8491a` and `7e4a39e`, with packed helpers from `09388b9`.

Pair blocks, singular values, field weights and reduction partials belong to
each assembly or field call. Geometry tables are read-only after locked lazy
initialization, and caches are released after their callers finish. The packed
field API evaluates up to eight drives per geometry pass and validates both the
drive counts and the pressure/Neumann vector lengths before dispatch.

Measured on 2026-10-01 against `e6b3037` on an Apple M1 Max (32 GPU cores, 64 GB), Julia
1.12.6, pinned Metal 1.10.3 / GPUCompiler 2.5.0, OpenBLAS, 10 Julia threads and
9 solve BLAS threads: three interleaved fresh-worker pairs, taking the second
warm 24-frequency sweep. S is the 963-node quarter-symmetric horn, tag 2,
200-1051.357 Hz; C is the 3,719-node `speaker2-lf.msh`, tag 101, 200-2106 Hz.
Both use `xy` symmetry, order 4 regular/singular quadrature and the compiled
exterior entry. The unchanged cost model selected the pipeline for both.

| Case, automatic pipeline | Base sweep seconds, median [range] | Packed sweep seconds, median [range] | Paired base/packed ratio, median [range] |
|---|---:|---:|---:|
| S | 2.045 [1.884-2.255] | 1.952 [1.821-2.197] | 0.965 [0.931-1.238] |
| C | 19.183 [19.179-20.504] | 13.497 [13.356-14.071] | 1.436 [1.421-1.457] |

S has no established wall-time gain within the observed spread; the full
PR #15's reported 2.1x S gain does not carry over to this scoped port. C improves
consistently across the pairs. Median section times per frequency are assembly,
solve and field: S 65.09/49.76/18.66 ms to 59.26/57.60/17.86 ms; C
714.74/506.94/246.63 ms to 409.00/497.81/27.56 ms. These timers include
synchronization waits and overlap, so their sum is not elapsed sweep time.
S uses LU; C uses the unchanged adaptive LU/GMRES route. Median peak physical
footprint is S 1,937 to 1,920 MiB, C 3,303 to 3,352 MiB.

The 37-point arc pressure and radiation impedance preserve all on-axis SPL and
impedance-magnitude peak/dip indices. Maximum relative L2 / absolute dB
differences are S pressure 5.79e-7 / 0.0000217 dB and impedance
6.11e-7 / 0.00000518 dB, with a bit-identical base A/A comparison. C pressure
is 6.67e-5 / 0.000880 dB and impedance 5.36e-5 / 0.000447 dB, both worst at
369.616 Hz, against base A/A noise of 6.64e-5 / 0.000855 dB and
5.34e-5 / 0.000446 dB respectively. C is near this run's noise floor; the full
PR's 1e-3 / 0.0096 dB difference is absent. This comparison excludes both the
singular split and the global BLAS swap and does not isolate either cause.

Four regular-assembly kernel modes exist. The default, `pair_gather`, is
the chunked pair-gather design described above. It exists because the
fused atomic kernel was bound by atomic throughput, not arithmetic: each
pair scatters 48 Float32 atomics (four operators, real and imaginary), and
on an M1 Max those cost as much as the Green's-function evaluations
themselves. Writing the blocks with plain stores and gathering them per
entry removes the atomics and makes the result bit-reproducible run to run.
The trial columns are processed in chunks sized from a device-memory budget
(`BLAB_METAL_GATHER_BUDGET_MB`, 512 MB by default, 192 bytes per pair per
chunk column). `pair_atomic` is the fused kernel with atomic scatter,
non-deterministic in float32 summation order; `pair_owned` is the ROCm
backend's colored pair-owned design, deterministic because no two pairs in
one launch share a matrix entry; `entry_owned`, one thread per dense matrix
entry, is the correctness reference and does roughly nine times the
Green's-function work.

All modes share one pair-arithmetic routine: the fast-math AIR intrinsics
(`air.fast_sin`, `air.fast_cos`, `air.fast_rsqrt`, the arithmetic an
Xcode-compiled Metal shader gets by default), 32-bit indices, a rank-1
accumulation (3-vector inner sums, outer products once per test point),
a compile-time unrolled trial loop, and per-element quadrature points
precomputed once per cache so a point costs three loads instead of nine
vertex loads and nine FMAs. The kernel is register-bound (the 3x3 double
layer and hypersingular accumulators alone are 36 floats), so trial data is
read from cached device arrays rather than hoisted into registers.

The singular corrections use one fused Duffy kernel per (pair, part) that
evaluates the Green's function once per point pair for all four operators;
a pair's rule (512 to 1536 point pairs at singular order 4) is split into
`BLAB_METAL_SINGULAR_PARTS` contiguous ranges. A second kernel then sums the
parts and adds them to the operators. By default that is a gather: one thread
owns one dense-matrix cell and reads the list of block values belonging to it
from a map built on the host once per mesh, so no two threads share a cell and
the summation order is fixed. `BLAB_METAL_SINGULAR_WRITEBACK=scatter` selects
the older write-back instead, one thread per pair adding about 48 atomics, which
is not reproducible.

The map is built by one host pass and cached on the singular correction cache.
That cache lives as long as the request that built it: the exterior driver
builds its caches per request, so every sweep pays for the map once, in its
first frequency. On an M1 Pro the whole build (host pass, sort, upload) takes
16 ms at `sample.msh` (2,776 faces, 37,198 singular pairs) and 38 ms at
`sample_detailed.msh` (7,000 faces, 93,740 pairs), with 3.0 and 7.7 MB of device
memory. It took 195 and 489 ms until the host pass was moved behind a function
barrier: the cache fields it indexes are untyped, so every index was a dynamic
dispatch. In exchange the four-operator write-back drops from 2,249,760 atomic
adds to 311,964 plain ones at the larger mesh, because a corrected cell is
written once instead of once per pair that touches it (3.2 pairs per cell for
the single layer and adjoint, 12.2 for the double layer and hypersingular).

Per frequency, the two write-backs are within a few percent of each other and
the singular stage is under a tenth of assembly, so this is not a speed change.

On an M1 Max at 5,041 P1 dofs (10,078 faces), quadrature order 4, singular
order 4, one frequency: `pair_gather` assembles in about 1.06 s (pair
kernel 0.59 s, gathers 0.31 s, singular 0.12 s, allocation 0.04 s);
`pair_atomic` 1.9 s; the first port's colored `pair_owned` kernels 7.1 s;
`entry_owned` about 25 s. All modes agree with BEAT CPU to the same
tolerances. hornlab-metal-bem's P1 Galerkin kernel, which
assembles one operator with 18 atomics per pair, takes 0.42 s on the same
mesh.

A frequency sweep can overlap the GPU assembly of frequency i+1 with the CPU
solve of frequency i on a second Julia thread, holding two systems at once.
Both exterior entry points do it: the source-request driver, and compiled
exterior-only requests, which is how Boundary Lab's GUI solves exterior
projects. Coupled solves overlap within a frequency instead; see
[Stage overlap](#stage-overlap).

`metal_sweep_overlap_plan` in `BeatEngineSweepOverlap.jl` decides per solve,
from a cost model rather than a mesh size. Per frequency a sequential sweep
costs A + S + F (GPU assembly, CPU solve, GPU field); overlapped it costs
F + max(A + c, (1 + kappa) S), where c is the time the assembly loses beside
the solve and kappa the fraction the solve slows beside the assembly. The sweep
overlaps when the saving, min(S - c, A - kappa S), is positive. S comes from the
dense-solve cost model below and A = a N^2 + b per symmetry copy. a, b, c and
kappa are machine constants: the defaults are an M1 Pro's, each has an
environment override, and `scripts/calibrate_metal_sweep_overlap.jl` measures
them through the pipeline itself. Like the dense-solve calibration, it is run by
hand.

On the compiled path a Metal sweep leaves the assembly producer a core: BLAS
runs one thread below the process default. With BLAS on every performance core
the overlapped solve slowed by more than the assembly it hid. Sequential sweeps
use the same count, because LU rounds differently on a different number of
threads and the overlap must not change the answer;
`validate_metal_exterior_pipeline.jl` gates that it does not. The source-request
driver keeps its BLAS on the Julia thread count.

Compiled exterior requests on an M1 Pro, 4 Julia threads, 12 frequencies from
20 Hz to 20 kHz, first sweep / later sweeps. Before is sequential with BLAS on
all 8 performance cores.

| mesh | P1 dofs | symmetry | before | now | model's choice |
|---|---:|---|---:|---:|---|
| `sample.msh` | 1,390 | off | 1.72 / 1.33 s | 1.45 / 1.17 s | overlap |
| `sample_detailed.msh` | 3,502 | off | 7.94 / 7.61 s | 5.75 / 5.41 s | overlap |
| `sample_half.msh` | 854 | `x` | | 1.50 / 0.76 s | overlap, +0.8 ms a frequency (sequential: 1.54 / 0.82 s) |
| `sample_quarter.msh` | 441 | `xy` | | 1.19 / 0.53 s | sequential, -2.4 ms (overlapped: 1.18 / 0.51 s) |

Where the model calls the choice marginal, the two measure within noise of each
other. The cost is memory: the lookahead holds up to four systems, 530 MB more
at peak on `sample_detailed.msh`, and `sweep_pipeline_depth` caps it by what the
device has free.

This replaces a fixed threshold of 1,900 dofs, fitted on an M1 Max through the
source-request driver, 20 frequencies from 100 Hz to 20 kHz, as sequential over
overlapped wall clock:

| mesh | P1 dofs | symmetry | pipelining |
|---|---:|---|---:|
| ATH `asro68` quarter | 1,209 | `xy` | 0.96x |
| ATH ladder A1 | 1,974 | off | 1.13x |
| ATH ladder A2 | 2,559 | off | 1.19x |
| ATH ladder A3 | 3,898 | off | 1.28x |
| `asro68` full | 4,552 | off | **1.38x** |
| `asro68` quarter, subdivided | 4,692 | `xy` | 1.11x |
| ATH ladder A5 | 5,107 | off | 1.30x |

and six spheres: 1,202 dofs 0.82x, 1,514 0.97x, 1,742 0.96x, 1,986 1.13x,
2,382 1.22x, 3,122 1.19x. On the M1 Pro the same threshold kept `sample.msh`
sequential, where overlapping it is 12-15% faster. The two machines disagree
below 2,000 dofs, which is the case for calibrating rather than moving the
constant; an M1 Max should run the calibration script.

`BLAB_METAL_PIPELINE` forces the choice either way, and each frequency reports
`metal_pipeline`, `metal_pipeline_reason` and `metal_overlap_saving_model_s`.

## FEM static condensation

Coupled solves reduce the FEM interior onto the retained interface before
factoring. `beat_metal` is in both `PHYSICAL_SYSTEM_BACKEND_IDS` and
`CONDENSING_BACKEND_IDS`, so `system_solve.py` requests
`static_condensation: true` and this is the default path.

Metal has no GPU LU, so the driver routes a condensing Metal solve the same
way it routes `beat_cpu`: `coupled_solver.jl` selects the condensed solver for
`:cpu` and `:metal` alike, and `build_condensed_coupled_system` in
`BeatEngineCoupledCondensed.jl` does the work. The Metal-specific BEM stage assembles combined A/C by default and projects
C on the host. Its four-operator reference path hands matrices back through
`metal_host_operators`. The partition, the interior UMFPACK
factorization and the blocked Schur complement are the shared
`_blocked_umfpack_schur_complement` in `BeatEngineCoupled.jl`. Diagnostics
report `fem_condensation_backend: cpu_umfpack` and
`linear_solver: cpu_umfpack_schur_plus_dense_lu`, exactly as `beat_cpu` does,
unless the Metal defaults described in [Coupled condensed optimizations on
Metal](#coupled-condensed-optimizations-on-metal) are active, which they are
by default: the Schur complement then comes from MUMPS (`mumps_seq`).

### Why condensation is on by default

Measured on the curved-interface production fixture -- 19,492 FEM vertices,
94,265 tetrahedra, 5,103 BEM triangles, 1,318 retained interface vertices -- at
1 kHz, `q2/s2`, on an M1 Pro with eight Julia threads. The measurement predates
the merge and ran on the archived host condensation (see below), but that path
used the same UMFPACK Schur complement the condensed solver runs today, so the
ratio carries over:

| | Monolithic | Condensed | Ratio |
| --- | ---: | ---: | ---: |
| System order | 23,433 | 5,259 | 4.5x smaller |
| Build | 96.689 s | 6.305 s | 15.3x |
| Solve | 2.799 s | 2.185 s | 1.28x |
| **Total** | **99.489 s** | **8.490 s** | **11.7x** |

The build gap is the dense coupled matrix: 23,433 squared in `ComplexF32` is
4.4 GB and its LU dominates everything else, against 221 MB condensed.
Agreement between the two formulations is 1.3e-4 on FEM pressure, 1.8e-4 on BEM
pressure, and 1.7e-4 on interface flux -- inside the 5e-4 Float32 gate used
across the coupled validations.

### Stage overlap

Within a frequency the FEM condensation and the BEM operator assembly are
independent: the condensation reads `fem_system`, the interface operators and
the retained vertex list, none of which the BEM assembly touches. On Metal they
also run on different processors -- the condensation is host UMFPACK, the
assembly is on the GPU -- so `build_condensed_coupled_system` spawns the
condensation on a Julia thread before it starts the BEM assembly and collects it
afterwards. On `beat_cpu` both stages are host code competing for the same
cores, so the overlap is off there by default.

Measured through `blab project solve`, `xy` symmetry, eight Julia threads, M1
Pro, warm mean of three frequencies, `BLAB_COUPLED_STAGE_OVERLAP=off` against
the default:

| Fixture | Condensed order | `bem_operator_s` | `assembly_s`, sequential | `assembly_s`, overlapped | Saved |
| --- | ---: | ---: | ---: | ---: | ---: |
| `S218BP` (40-100 Hz) | 2,151 / 19,379 | 0.19-0.22 s | 0.911 s | **0.719 s** | 0.19 s, 21% |
| `F2B_FLH` (100-200 Hz) | 3,081 / 24,410 | 0.27-0.31 s | 1.552 s | **1.366 s** | 0.19 s, 12% |

Outputs are bit-identical with the overlap on and off: the algebra is
unchanged, only the schedule. The saving is bounded by the shorter of the two
stages, which on these fixtures is the GPU assembly at about 0.2-0.3 s; it grows
with the BEM mesh. `beat_cpu` on the same fixtures: 1.362 s and 2.251 s.

When the stages overlap, `fem_condensation_s` is measured from before the BEM
assembly starts, so it spans the concurrent region and must not be added to
`bem_operator_s`; `stage_overlap` in the system timings says which reading
applies.

### Coupled sweep pipeline

The stage overlap hides the FEM condensation behind the GPU, but on a model
with a large exterior it is the other way round: the GPU assembly is the longer
of the two, and the host then still has to combine the operators, assemble and
factor the coupled system, solve and evaluate the field while the GPU idles.
On Multi_region_SAWMOD (M1 Max) the BEM operators take 0.81 s per frequency,
the FEM condensation beside them 0.50 s (`fem_task_s`), and the host work after
both 0.63 s.

A coupled sweep therefore assembles and combines the next frequency's BEM
operators on a producer task -- `assemble_condensed_bem_operators`, the same
code `build_condensed_coupled_system` runs, handed back through its
`bem_operators` argument -- while the host finishes the current frequency. The
producer reuses the exterior sweep's `start_sweep_assembly_pipeline` at depth
one. Outputs are bit-identical with the pipeline on and off.

Whether to run it is decided in the run itself. Frequencies solve sequentially
until `coupled_sweep_pipeline_plan` predicts a saving from the median section
times of the sequential frequencies so far, leaving out the first (one-off
compilation and cache costs) and waiting for at least two; medians keep a
one-off spike -- the first frequency at a new quadrature order -- from deciding:

```text
sequential  max(G + C + R, S) + L
pipelined   max(G + C, max(S, R) + L)
```

with `G` the BEM operator assembly, `C` its combination into the Burton-Miller
blocks (what the producer takes over), `R` the rest of the BEM matrix stage that
stays on the host (motion and prescribed-source products, an interface-radiation
replay LU) and overlaps the FEM task, `S` the FEM condensation task, and `L` the
coupled block assembly and factorization. With `R = 0` the saving is
`min(L, G + C - S)` when `G + C > S`, otherwise zero. It starts the producer when the saving exceeds 10% of a
sequential frequency (`BLAB_COUPLED_SWEEP_PIPELINE_MIN_SAVING`) and two more
combined operator sets fit in half of Metal's free working set. A model with a
large FEM interior behind a small exterior -- `F2B_FLH`, where `S` already
exceeds `G + C` -- stays sequential; forcing the pipeline on there measured 4%
slower. The modelled saving overstates the measured one: the producer and the
host stages share the memory system, and on SAWMOD the host block assembly and
dense LU ran 30-50% slower beside it.

Measured on an M1 Max (eight Julia threads, eight BLAS threads), warm worker,
interleaved runs, Multi_region_SAWMOD:

| Sweep | Sequential | Pipelined | Change |
| --- | ---: | ---: | ---: |
| 40 frequencies, 20 Hz-20 kHz | 64.7-77.0 s, median 70.9 (7 runs) | 54.8-60.0 s, median 57.7 (6 runs) | -19% |
| 200 frequencies, 20 Hz-20 kHz (GUI order), with MUMPS on Accelerate | 321.0 s | **248.7 s** | -23% |

Peak worker memory grows by about 1 GB (5.2 to 6.1 GB on the 200-frequency
sweep), the producer's queued and in-flight operator sets. On `F2B_FLH` the
plan keeps the sweep sequential.

Result diagnostics report `coupled_sweep_pipeline` (whether this frequency's
operators came from the producer), `coupled_sweep_pipeline_reason` and
`coupled_sweep_pipeline_saving_model_s`. `fem_task_s` in the timings is the FEM
condensation task's own duration, which `fem_condensation_s` cannot show when
the stages overlap.

### Schur block balance

The Schur complement hands right-hand-side blocks to worker tasks round-robin,
and every block costs about the same, so the stage's wall time is set by the
*most* blocks any one task gets. `_resolved_schur_block_size` therefore rounds
the block *count* up to a whole multiple of the thread count and derives the
width from that, rather than capping the width alone: on `F2B_FLH` a
1,102-column interface at a 64 cap gave 18 blocks over 8 threads, so two tasks
took three blocks while six took two and idled, and the stage ran 33% longer
than its work required. The width used is reported as `fem_schur_block_size`
(26 on `S218BP` with eight threads). `BLAB_SCHUR_BLOCK` pins the width for
measurement.

### Precision of coupled Float32 solves

Coupled Metal solves run at `precision=float32`. Two host stages lose more than
single precision should, and both are now kept in double precision on Metal by
default:

- **Dense coupled LU.** A Float32 LU of the dense condensed system cannot carry
  its condition number. Against a CPU Float64 reference (every other setting
  equal, ten frequencies from 20 Hz to 20 kHz, 5e-4 relative / 0.01 dB gate),
  the Float32 dense LU missed on every Boundary Lab example tried:
  Multi_region_SAWMOD 0.28 dB, Simple_Sealed 0.71 dB, Vented_Sub 4.4 dB,
  S218BP 1.8 dB, F2B_FLH 0.027 dB, SKRAM 8.9 dB. The system is now assembled in
  `ComplexF64`, factored in `ComplexF32` and solved by iterative refinement
  against the double-precision matrix (`RefinedDenseLU`, LAPACK `zcgesv`'s
  backward-error test). On these examples every frequency converged in one or
  two steps and matched a `ComplexF64` LU. A solve that stalls, a matrix outside
  the Float32 range, or a failed Float32 factorization falls back to a
  `ComplexF64` LU with the reason in `dense_refinement_fallback_reason`.
- **FEM matrices at low frequency.** A sealed cavity with no retained vertices
  has a near-constant pressure mode whose eigenvalue scales like `k²` (the air
  spring). A Float32 stiffness matrix annihilates constants only to about
  `eps(Float32)‖K‖`, and that mode amplifies the defect by about `1/(kh)²`, so
  the transducer's mechanical impedance loses digits at low frequency even
  though `A_II` is factored in `ComplexF64`. On Multi_region_SAWMOD at 20 Hz
  every output was off by 1.2e-4; on S218BP by 1.3e-3 (a gate failure). The
  stiffness, mass and bulk-loss matrices are now assembled in `Float64` (once
  per mesh, on the mesh's own coordinates) and the condensation reads a
  `ComplexF64` dynamic stiffness. The 20 Hz errors drop to 9e-7 and 5e-6.

Neither changes a `precision=float64` solve, and neither applies to `beat_cpu`,
CUDA or ROCm unless the variable is set explicitly. Results stay `Complex{T}`.
Result diagnostics report `dense_solver` (`lu_float32`, `lu_float64`,
`lu_float32_refined`, `lu_float64_fallback`), `dense_refinement_iterations`,
`dense_refinement_backward_error` and `fem_matrix_precision`.

Cost on Multi_region_SAWMOD (4 frequencies, 3 interleaved rounds, M1 Max),
median assembly per frequency: Float32 LU 7.36 s, Float64 LU 10.38 s,
refinement 8.79 s: refinement recovers about half of what a plain Float64 LU
costs.

### Coupled condensed optimizations on Metal

On `beat_metal` the condensed solver reduces the dense coupled system further
and takes the Schur complement from a sparse direct solver's partial
factorization instead of per-column triangular solves. Every step is exact up
to round-off, each is a switch, and on Metal each defaults to `auto`: used when
the model's structure allows it, otherwise the established path is taken and
`coupled_optimization_fallback_reasons` says why. `1` requires a step (refusing
unsupported structure), `0` turns it off.

`beat_cpu` uses the same structural reductions (transducer condensation, flux
elimination, CHOLMOD interface mass, component blocks, demand reconstruction)
and the refined dense LU by default, with UMFPACK for the FEM Schur complement
(MUMPS ships with the Metal environment only) and without the Metal-only stage
overlaps or the Float64 FEM assembly. On Multi_region_SAWMOD (M1 Max, CPU
backend, ten frequencies from 20 Hz to 20 kHz) the earlier CPU defaults, a plain
Float32 dense LU of order 7,933, were up to 3.7 dB from the CPU Float64
reference in pressure within 30 dB of the peak; with these defaults the worst
error is 1.9e-3 dB and a frequency takes 11.0 s instead of 16.1 s. Float64 FEM
assembly stays off on the CPU: it brought diaphragm velocity from 1.1e-3 dB to
1.9e-5 dB but made the UMFPACK interior factorization about ten times slower on
that model. CUDA and ROCm are unchanged unless a variable is set explicitly.
The refined LU holds the dense system in `ComplexF64` next to its `ComplexF32`
factor, three times the memory of the earlier Float32-only LU (about 1.5 GB
instead of 0.5 GB at order 7,933), and a refinement fallback adds a `ComplexF64`
LU on top; `BLAB_COUPLED_DENSE_REFINEMENT=0` restores the earlier footprint.
Because refinement takes precedence, a CPU run that sets only
`BLAB_COUPLED_DENSE_FLOAT64=1` now gets the refined LU; add
`BLAB_COUPLED_DENSE_REFINEMENT=0` for a plain `ComplexF64` LU. A singular
interface mass under `auto` flux elimination solves the unreduced system and
records why.

| Step | Switch | What it does |
| --- | --- | --- |
| Transducer condensation | `BLAB_COUPLED_TRANSDUCER_CONDENSATION` | Eliminates transducer-surface FEM vertices with the interior (rank one per transducer) instead of retaining them in `Γ`; the full transducer × transducer air-spring coupling is kept. Off for the speaker ROM experiment, which reads those surfaces from the Schur block. |
| Interface flux elimination | `BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION` (`_PRESSURE_ELIMINATION` for pressure only, opt-in) | Removes the duplicated interface pressures (unit-selection continuity) and substitutes `q = M_Γ⁻¹(S P p_B + E y − g)` into the BEM rows, leaving only BEM pressures and transducer unknowns. Needs transducer condensation when transducers are present. |
| Interface mass solve | `BLAB_COUPLED_INTERFACE_MASS_SOLVER` (`cholmod` on Metal and CPU, `lu` on CUDA/ROCm) | Cached sparse Cholesky of the real SPD interface mass, residual-checked, LU fallback with a reason. |
| Mass overlap | `BLAB_COUPLED_INTERFACE_MASS_OVERLAP` | Forms `M_Γ⁻¹S` and `M_Γ⁻¹E` inside the FEM task, overlapping the GPU BEM assembly. |
| Per-component blocks | `BLAB_COUPLED_INTERFACE_BLOCKS` | Keeps independent FEM components as separate blocks through the elimination. |
| Demand reconstruction | `BLAB_COUPLED_DEMAND_RECONSTRUCTION` | Skips the interior back-substitution when no requested output reads interior pressure (`fem_interior_reconstruction: skipped`). |
| MUMPS Schur complement | `BLAB_COUPLED_FEM_SOLVER` (`mumps` on Metal, `umfpack` elsewhere) | Sequential MUMPS 5.9.1 (`MUMPS_seq_jll`), complex-symmetric LDLᵀ in Schur mode, analysis cached across a sweep; falls back to UMFPACK with `fem_solver_fallback_reason`. |

`MUMPS_seq_jll` and `OpenBLAS32_jll` are dependencies of `julia_metal` only
(about 25 MB of artifacts on macOS arm64), so CPU, CUDA and ROCm installs do
not download them; there `mumps` falls back with the reason "MUMPS_seq_jll is
not in this Julia environment". Its tests are `tests/mumps_tests.jl`, run under
`julia_metal` in the macOS CI job.

MUMPS calls the LP64 BLAS interface, which Julia's own configuration (ILP64
OpenBLAS) leaves empty. On Apple Silicon the loader forwards Apple Accelerate's
LP64 interface (the macOS 13.3 "new LAPACK" symbols) into it; elsewhere, or if
that forward fails, `OpenBLAS32_jll`. Only the LP64 slots change, so Julia's
own dense LU and products keep OpenBLAS and their thread count, and their
results are bit-identical either way. `BLAB_MUMPS_BLAS=openblas` restores the
OpenBLAS route; result diagnostics report `mumps_blas`. The MUMPS
factorization's rounding changes with the BLAS, far below the Float32 output
precision. On Multi_region_SAWMOD at ten frequencies from 20 Hz to 20 kHz the
Accelerate route agrees with the OpenBLAS route to 8.4e-9 relative L2 in pressure
(2.8e-5 dB worst within 30 dB of each output's peak), 1.9e-12 in diaphragm
velocity and 4.6e-10 in interface velocity, and both measure the same 9.8e-6
relative L2 (1.7e-3 dB) in pressure against the CPU Float64 reference.

Measured on an M1 Max (eight Julia threads, eight BLAS threads), 40 frequencies
from 20 Hz to 20 kHz, warm worker, interleaved runs:

| Fixture | OpenBLAS32 (s per sweep) | Accelerate (s per sweep) | `fem_condensation_factorization_s` |
| --- | ---: | ---: | --- |
| `F2B_FLH` (large FEM interior, small exterior) | 23.0, 23.3, 24.6 | **15.4, 15.6** | 0.25-0.27 s to 0.15 s |
| `Multi_region_SAWMOD` | 70.9, 72.3 | 67.4, 77.1 | 0.42 s to 0.26-0.27 s |

On `F2B_FLH` the FEM condensation is the critical path, and the whole sweep is
33% faster; the host block assembly and dense LU also run faster beside it, as
MUMPS no longer occupies the cores with OpenBLAS threads. On SAWMOD the
condensation already hides behind the GPU BEM assembly, so the faster
factorization does not shorten the sweep (the spread is run-to-run noise).

On Multi_region_SAWMOD (three transducers, three FEM regions, four interfaces,
xy symmetry; M1 Max, eight Julia threads, 4 frequencies, 3 interleaved rounds)
the dense order falls from 7,933 to 3,116. Median assembly per frequency, each
row adding one step to the row above, with the dense precision fixed at
Float64:

| Configuration | s/frequency, median [range] |
| --- | --- |
| Float64 dense LU, no reductions | 10.38 [10.22–11.47] |
| + transducer condensation | 6.12 [5.94–6.42] |
| + flux elimination | 3.60 [3.18–4.59] |
| + CHOLMOD, overlap, blocks | 3.47 [3.03–3.64] |
| + demand reconstruction | 3.18 [3.15–3.72] |
| + MUMPS | 1.72 [1.68–1.80] |
| Metal defaults (+ refinement, Float64 FEM) | 1.76 [1.60–1.83] |

Each step agrees with its predecessor to within 1.6e-7 relative L2.

### Interior solver: UMFPACK, and the Accelerate path that was removed

The condensation factors the FEM interior and then solves it against roughly one
right-hand side per retained interface node; nearly all of the stage is those
triangular solves. The pre-merge Metal branch carried its own condensation
inside `build_coupled_system` with a `ccall` binding to Apple Accelerate's
sparse LU as an optional interior solver (`BLAB_METAL_FEM_CONDENSATION=accelerate`).
Both are gone from this tree. Tag `archive/metal-host-condensation` is the last
commit that carries them, and [Options: speeding up FEM static condensation on
Apple Metal](Metal%20FEM%20Condensation%20Options.md) records the full
argument.

Why: the host condensation duplicated what the condensed solver already does
and was never on the production Metal route, and Accelerate in `ComplexF32`
fails the accuracy standard. On `F2B_FLH` it was 1.87x faster on the
condensation stage but 3.2e-3 relative norm from the UMFPACK result, over the
5e-4 gate. Re-measured on the production route on `S218BP` before removal, it
was 23% faster per frequency and **1.1e-2 to 4.0e-2** relative error against
`beat_cpu` on diaphragm velocity, voice-coil current and probe pressures, where
UMFPACK is at 1.4e-5. The speed was the precision drop, not a better solver:
Accelerate in `ComplexF64` matched UMFPACK on both counts. Overlapping the
condensation with the GPU assembly recovers most of what it offered without
touching the numerics.

## Requirements

- An M-series Mac running macOS 14 or newer.
- Julia 1.10 to 1.12.
- The dedicated `src/beat_engine/julia_metal` environment with Metal.jl.

To prepare the Julia environment from the repository root:

```bash
python -m beat_engine instantiate --backend metal
python -m beat_engine doctor --backend metal --threads 2
```

or directly with Julia:

```bash
julia --project=src/beat_engine/julia_metal -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Verify the runtime:

```bash
julia --project=src/beat_engine/julia_metal -e 'using Metal; Metal.functional() || error("Metal unavailable"); Metal.versioninfo()'
```

## Selecting the backend

A compiled-system request selects Metal with `solver_options.bem_backend =
"metal"`, and a source request with `config.beat_backend = "metal"`; the worker
must run with the `julia_metal` project (`engine_paths("metal")`). Boundary Lab
exposes this in application preferences as **BEAT Engine (Apple Metal)**, backend
identifier `beat_metal`, and only offers it on Apple Silicon macOS.

## Runtime controls

Normal application use does not require these environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `BLAB_METAL_COUPLED_BEM_ASSEMBLY` | `auto` | Condensed coupled BEM: `auto` selects combined A/C when supported, otherwise operators with a recorded reason; `combined` requires support; `operators` runs the original path. |
| `BLAB_METAL_ASSEMBLY_MODE` | `native` | Use `host_staged` to assemble operators on the CPU and upload them as a diagnostic fallback. |
| `BLAB_METAL_REGULAR_KERNEL_MODE` | `pair_gather` | Use `pair_atomic` for the fused atomic kernel, `pair_owned` for the deterministic colored kernels, or `entry_owned` as the correctness reference. |
| `BLAB_METAL_SINGULAR_MODE` | `native` | Use `host` to compute the Duffy singular corrections on the CPU and add them to the device operators, separating kernel defects from rule defects. |
| `BLAB_METAL_KERNEL_GROUPSIZE` | `256` | Threads per threadgroup for the one-dimensional assembly kernels. |
| `BLAB_METAL_ATOMIC_TILE` | `16x16` | Threadgroup shape (test, trial) of the two-dimensional pair kernels. |
| `BLAB_METAL_GATHER_BUDGET_MB` | `512` | Device memory for the pair-block buffer; sets the trial chunk size of `pair_gather`. `BLAB_METAL_GATHER_CHUNK` overrides the chunk size directly. |
| `BLAB_METAL_GATHER_TIMING` | `0` | Set to `1` to synchronize after each `pair_gather` stage and report `metal_native_gather_*` timings (slower). |
| `BLAB_METAL_SINGULAR_PARTS` | `4` | Ranges each singular pair's Duffy rule is split into across threads. |
| `BLAB_METAL_SINGULAR_WRITEBACK` | `gather` | How the singular blocks reach the operators. `gather` owns one dense cell per thread and is reproducible; `scatter` is the older one-thread-per-pair atomic write-back, kept for comparison. |
| `BLAB_METAL_OPERATOR_STORAGE` | `shared` | Use `private` to allocate the operator matrices in private storage and copy them to the host, the pre-2026-09-02 behavior. |
| `BLAB_METAL_PIPELINE` | by cost model | Overlaps the next frequency's GPU assembly with this frequency's CPU solve. Unset, `metal_sweep_overlap_plan` decides per solve; `0` forces sequential, anything else forces the overlap. |
| `BLAB_METAL_PIPELINE_DEPTH` | by memory | Frequencies the assembly may run ahead of the solve, for measurement. |
| `BLAB_METAL_ASSEMBLY_DOF2_SECONDS` | `2.77e-8` | Overlap cost-model constants, calibrated on an Apple M1 Pro: fused assembly seconds per squared dof per symmetry copy, its fixed part, what the assembly loses beside the solve, and the fraction the solve slows beside the assembly. Re-measure with `scripts/calibrate_metal_sweep_overlap.jl` on any other machine. |
| `BLAB_METAL_ASSEMBLY_FIXED_SECONDS` | `0.016` | |
| `BLAB_METAL_OVERLAP_COST_SECONDS` | `0.003` | |
| `BLAB_METAL_OVERLAP_HOST_SLOWDOWN` | `0.1` | |
| `BLAB_METAL_ATOMIC_SCATTER` | `1` | Diagnostic for `pair_atomic` only: `0` skips the atomic scatter to time the pair arithmetic (the operators are then wrong). |
| `BLAB_COUPLED_STAGE_OVERLAP` | `auto` | Coupled solves: `auto` runs the FEM condensation on its own thread while the GPU assembles the BEM operators; `off` runs them in sequence; `on` forces the overlap on `beat_cpu` too. Needs more than one Julia thread. |
| `BLAB_COUPLED_SWEEP_PIPELINE` | `auto` | Coupled sweeps: `auto` assembles the next frequency's BEM operators ahead when the run's own section times predict a saving (see [Coupled sweep pipeline](#coupled-sweep-pipeline)); `on` forces it from the second frequency, `off` never. Needs more than one Julia thread. |
| `BLAB_COUPLED_SWEEP_PIPELINE_MIN_SAVING` | `0.10` | Smallest modelled saving, as a fraction of a sequential frequency, that starts the coupled sweep pipeline under `auto`. |
| `BLAB_COUPLED_DENSE_REFINEMENT` | `auto` on Metal and CPU, `off` on CUDA/ROCm | Coupled Float32 solves: assemble the dense system in `ComplexF64`, factor in `ComplexF32`, refine to the Float64 backward error (falls back to a `ComplexF64` LU with a reason). `1`/`auto`/`0`. |
| `BLAB_COUPLED_DENSE_FLOAT64` | `off` | Coupled Float32 solves: plain `ComplexF64` dense LU instead (refinement takes precedence when both are on). |
| `BLAB_COUPLED_FEM_FLOAT64` | `auto` on Metal, `off` elsewhere | Coupled Float32 solves: assemble the FEM stiffness, mass and bulk-loss matrices in `Float64`. |
| `BLAB_COUPLED_TRANSDUCER_CONDENSATION` | `auto` on Metal and CPU, `off` on CUDA/ROCm | See [Coupled condensed optimizations on Metal](#coupled-condensed-optimizations-on-metal). `1`/`auto`/`0`. |
| `BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION` | `auto` on Metal and CPU, `off` on CUDA/ROCm | As above. |
| `BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION` | `off` | Pressure-only elimination (flux elimination takes precedence). |
| `BLAB_COUPLED_INTERFACE_MASS_SOLVER` | `cholmod` on Metal and CPU, `lu` on CUDA/ROCm | `lu` or `cholmod`. |
| `BLAB_COUPLED_INTERFACE_MASS_OVERLAP` | `auto` on Metal, `off` elsewhere | As above. |
| `BLAB_COUPLED_INTERFACE_BLOCKS` | `auto` on Metal and CPU, `off` on CUDA/ROCm | As above. |
| `BLAB_COUPLED_DEMAND_RECONSTRUCTION` | `auto` on Metal and CPU, `off` on CUDA/ROCm | As above. |
| `BLAB_COUPLED_FEM_SOLVER` | `mumps` on Metal, `umfpack` elsewhere | `umfpack` or `mumps`. |
| `BLAB_MUMPS_BLAS` | `auto` | LP64 BLAS behind MUMPS: `auto` is Apple Accelerate on Apple Silicon and `OpenBLAS32_jll` elsewhere; `accelerate` or `openblas` asks for one. |
| `BLAB_MUMPS_THREADS` / `BLAB_MUMPS_SOLVE_THREADS` | `4` / `1` | OpenBLAS threads for the MUMPS factorization and solve phases (Accelerate schedules its own). |
| `BLAB_SCHUR_BLOCK` | unset | Coupled solves: pins the Schur complement right-hand-side block width, bypassing the thread-count balancing. For measurement only. |
| `BLAB_BEAT_FUSED_BM` | `1` | Set to `0` to assemble the four operators and combine them on the host for exterior solves. Exterior `host_staged` assembly and the `host` singular mode take the four-operator path. Condensed coupled solves use `BLAB_METAL_COUPLED_BEM_ASSEMBLY`. |

The fused system is then solved by the adaptive dense solve described at the
head of [`BeatEngineDenseSolve.jl`](../src/beat_engine/julia_local/src/BeatEngineDenseSolve.jl) — dense LU or
diagonally preconditioned GMRES, chosen per solve. Metal has no GPU LU, so
both routes run on the host; shared storage means the host reads the assembled
matrix in place rather than copying it. Its environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `BLAB_BEAT_DENSE_SOLVE` | `auto` | Force `lu` or `gmres` instead of the cost model. |
| `BLAB_BEAT_GMRES_TOL` | `1e-5` | Tolerance on the true relative residual. |
| `BLAB_BEAT_GMRES_MAX_ITERATIONS` | `min(N, 1000)` | Iteration cap; reaching it reports non-convergence and falls back to the LU. |
| `BLAB_BEAT_GMRES_RESTART` | `0` | Restart length; `0` is unrestarted. |
| `BLAB_BEAT_GMRES_KRYLOV_PRECISION` | `f64` | `f32` reproduces the orthogonality-loss failure on demand. Not for production use. |
| `BLAB_BEAT_GMRES_REORTHOGONALIZE` | `dgks` | `always` or `never`; `never` is the failing variant, kept so the remedies can be compared. |
| `BLAB_BEAT_GMRES_BUDGET` | `1` | Matvecs a model-chosen GMRES may spend, in modelled LUs, before it falls back to the LU. |
| `BLAB_BEAT_GMRES_TIME_CEILING` | `2` | Wall-clock ceiling for a model-chosen GMRES, in modelled LUs. |
| `BLAB_BEAT_LU_GFLOPS` | `480` | Cost-model constants, calibrated on an Apple M1 Max. Re-measure with `scripts/calibrate_dense_solve.jl` on any other machine. |
| `BLAB_BEAT_MATVEC_ENTRY_SECONDS` | `1.071e-10` | |
| `BLAB_BEAT_MATVEC_DOF_SECONDS` | `4.236e-7` | |
| `BLAB_BEAT_TRIANGULAR_GBPS` | `17` | |
| `BLAB_BEAT_GMRES_MODEL_ITERATIONS` | `70` | Expected iterations. A property of the operator, not the machine. |

## Verification

CPU-versus-Metal validation scripts:

| Script | Coverage |
|---|---|
| `validate_metal_fused_burton_miller.jl` | Fused exterior system against the four-operator system, multi-drive. `BLAB_VALIDATE_SYMMETRY` is a comma-separated arm list; the default runs `off,x,xy,ground` and fails if any arm fails. |
| `validate_metal_singular_summation.jl` | The fused singular pair against the four-operator accumulation order, per pair, in Float32 against a Float64 reference: the two orders must agree in Float64, neither Float32 order may exceed the stated bound, and the fused order must not be systematically worse. |
| `validate_gmres_burton_miller.jl` | GMRES against the dense LU on a real assembled operator across the frequency band: true residual, three-way agreement between Krylov variants, restart independence, and that the failure mode the remedies cover is reachable. |
| `validate_metal_packed_exterior.jl` | Concurrent assemblies sharing geometry caches, multi-drive field parity through the eight-drive batch boundary, and input length validation. |
| `validate_metal_exterior.jl` | Operators (both singular modes), boundary pressure, residual, and exterior field for an exterior solve. |
| `validate_metal_symmetry.jl` | X and XY reduced-domain assembly and solve parity, both singular modes. |
| `validate_metal_coupled_combined.jl` | Combined versus operators: A/C, sparse and complex projections, full condensed velocity/voltage/source solutions; `off/x/xy`, both phasors, plane weights and image-singular corrections, shared/private storage and chunk boundaries. |
| `validate_metal_coupled.jl` | Coupled FEM-BEM-LEM assembly, condensation, solution, and field for the monolithic and condensed paths, prescribed-velocity and voltage excitations. |
| `validate_metal_sweep_pipeline.jl` | The sweep assembly pipeline at depths 1-4: steps delivered in order with their own frequency, and pipelined assemblies against sequential ones. |
| `validate_metal_exterior_pipeline.jl` | A compiled exterior request solved sequentially and overlapped at depths 1-4 through the worker: every output bit-identical and labelled with its own frequency. `BLAB_VALIDATE_SYMMETRY` picks the arm. |
| `validate_metal_coupled_pipeline.jl` | A condensed coupled sweep built from operators the sweep pipeline assembled ahead against the same sweep built sequentially: every solution bit-identical, prescribed-velocity and voltage excitations. |

For example:

```bash
julia -t auto --project=src/beat_engine/julia_metal \
  src/beat_engine/julia_local/scripts/validate_metal_exterior.jl
```

`BLAB_VALIDATE_MESH`, `BLAB_VALIDATE_REGULAR_ORDER`,
`BLAB_VALIDATE_SINGULAR_ORDER`, and `BLAB_VALIDATE_FREQUENCY_HZ` select the
fixture, quadrature orders, and frequency. The scripts exit with an error when
CPU-versus-Metal differences exceed their tolerances.

## Operational behavior

- Compiled-system workers load `BeatEngineCompiledMetalBundle`, which caches
  the compiled driver and production Metal kernel signatures in a Julia package
  image. Metal 1.11.1 with GPUCompiler 2.9.0 stores the device code during
  `Pkg.precompile()`; this installation cost is paid once, and the workload
  compiles and links without launching an engine kernel. The precompile log
  reports the signature count and warns on failures. Host compilation and
  device pipeline setup still contribute to the first request. Measure worker
  ready, first request and second request separately; steady-state solve time
  should be evaluated after warm-up.
  The enabled compiled Metal entry loads Metal before JSON to match the bundle
  dependency order and retain its cached worker call graph. Explicit fallback
  and other backend entry orders are unchanged.
  The shared `CompiledCoupledWorkload.jl` also decodes a two-frequency coupled
  request using the packaged `femvolume.msh` and `exterior_conforming.msh`
  fixtures: one FEM air volume, one BEM exterior, a conforming interface and a
  voltage-driven Radiator transducer, with pressure, velocity, current and
  interface-average velocity outputs. It uses CPU BEM assembly while selecting
  the engine's Metal condensed defaults, avoiding engine GPU launches during
  image generation. Diagnostics check MUMPS Schur condensation, CHOLMOD
  interface mass, flux elimination and refined dense LU; failures and refinement
  fallbacks warn. CPU uses a tiny analogue with UMFPACK in place of MUMPS.
  Workload cleanup releases native factors and clears MUMPS pointers/live-solvers,
  provenance and field caches; fresh-process initialization reloads MUMPS and
  forwards LP64 BLAS again. Metal's existing device-state cleanup remains in
  place. Measured on an M1 Max (Julia 1.12.7, fresh worker, package images already
  built, Multi_region_SAWMOD, 10 frequencies): worker start-up plus first request
  was 21.4 + 38.2 s on `main`, 3.8 + 19.7 s with the exterior workload alone, and
  3.8 + 10.0-10.2 s with the coupled workload; `F2B_FLH` 3.6 + 16.7 s against
  3.9 + 7.0 s. The remaining first-request compilation is mostly the Metal side
  of the coupled path, which this host workload does not reach: Metal-typed
  cache structures, `metal_host_operators` and the regular-operator orchestration,
  and the spawned FEM stage (the workload runs it inline). Covering those needs
  compile-only signatures traced from a Metal coupled request.
- The generated kernel inventory is checked against production requests by
  `metal_kernel_coverage_tests.jl` in Metal hardware qualification. See the
  [bundle README](../src/beat_engine/julia_engine/BeatEngineCompiledMetalBundle/README.md)
  for coverage and regeneration. Uncached specializations compile normally.
  `BLAB_BEAT_ENGINE_BUNDLE=0` retains the include fallback for diagnosis; an
  unavailable bundle also uses that fallback. CPU, CUDA and ROCm retain their
  existing behavior.
- Frequency-independent caches remain resident for the worker's lifetime and are
  released when the worker exits.
- The default `pair_gather` kernels are bitwise reproducible run to run, as
  are `pair_owned` and `entry_owned`. `pair_atomic` is not (atomic
  accumulation order); its differences are float32 summation noise.
- That holds for the singular stage as well, but only since the deterministic
  write-back landed. Before it, every mode routed its singular corrections
  through an atomic scatter, so nothing was actually reproducible.
  `BLAB_METAL_SINGULAR_WRITEBACK=scatter` restores the old behavior for
  comparison.
- Assembly being reproducible does not make a sweep reproducible: the CPU LU
  is multithreaded, and two runs of the same solve differ by about 3e-7
  relative in the exterior field. Golden-file comparisons belong on the CPU
  `reference` path, tolerance comparisons everywhere else.

## Resolved: the fused Burton-Miller gate at symmetry `xy`

`validate_metal_fused_burton_miller.jl` used to fail its `xy` arm, and `xy` was
kept out of the default arm list because of it. It passes now, on all four arms,
and the cause was the test fixture rather than any assembly code.

**A symmetry-reduced assembly needs a mesh that is one sector.** It folds mirror
images onto that sector. Hand it a mesh that already spans both sides of the
mirror plane and every contribution is counted twice, leaving an operator that is
close to singular. `sample.msh` spans x in [-0.198, 0.199] and y in
[-0.126, 0.127], so it is a valid fundamental domain for `off` and `ground` only.
The tell was that `snap_symmetry_planes` did nothing to it: identical faces,
vertices and triangle aspect ratios on all four arms, because there was nothing
to snap.

Condition number of the Burton-Miller left-hand side, 2 kHz, Apple M1 Pro:

| arm | mesh | kappa (cpu) | kappa (metal) | backends agree? |
|---|---|---|---|---|
| `off` | `sample.msh` | 4.44e+02 | 4.44e+02 | yes, to 4.2e-6 |
| `x` | `sample.msh` | 1.81e+07 | 2.43e+04 | **no, 100% apart** |
| `xy` | `sample.msh` | 1.33e+09 | 5.52e+04 | **no, 100% apart** |
| `x` | `sample_half.msh` | 5.29e+02 | 5.29e+02 | yes, to 4.2e-6 |
| `xy` | `sample_quarter.msh` | 5.23e+02 | 5.23e+02 | yes, to 4.1e-6 |

On the correct mesh every arm is well conditioned and the two backends agree.
The fold itself is sound. On the wrong mesh both backends produce garbage, and
different garbage, which is why the CPU and Metal columns diverge completely.

The script now picks the fundamental domain per arm — `sample.msh` for `off` and
`ground`, `sample_half.msh` for `x`, `sample_quarter.msh` for `xy` — and calls
`validate_symmetry_fundamental_domain!` after snapping, the same check the
drivers and every other symmetry script already ran. An invalid combination now
stops with a named vertex instead of returning a plausible-looking error:

```
ERROR: Mesh is not in the positive X fundamental domain for XY symmetry.
       Vertex 41 has x=-0.0098830005 m.
```

Results after the fix, M1 Pro, tolerance 5e-6:

| arm | mesh | lhs | rhs | pressure |
|---|---|---|---|---|
| `off` | `sample.msh` | 2.5e-7 | 5.4e-7 | 6.393e-7 |
| `x` | `sample_half.msh` | 2.532e-7 | 3.439e-7 | 6.170e-7 |
| `xy` | `sample_quarter.msh` | 2.559e-7 | 3.275e-7 | **5.397e-7** |
| `ground` | `sample.msh` | 1.6e-7 | 1.0e-6 | 2.839e-6 |

Against 1.6459548e-5 for `xy` on `sample.msh`. All four arms are in the default
list now.

### What this cost, and the note that misdirected it

An earlier version of this section blamed the atomic singular scatter and offered
two candidate fixes: a better-conditioned `xy` fixture, **or** a deterministic
scatter. The first was right. The second was not, and it was the one that got
built first.

The deterministic write-back was worth having on its own merits and is now the
default (see the singular write-back section above). It removed the run-to-run
spread completely. It did not move the arm:

| write-back | run 1 | run 2 | run 3 |
|---|---|---|---|
| `scatter` (atomics) | 2.3376e-5 | 1.7533e-5 | 2.1780e-5 |
| `gather` (deterministic) | 1.6459548e-5 | 1.6459548e-5 | 1.6459548e-5 |

The atomics were adding about plus or minus thirty percent of noise on top of a
real error near 1.6e-5, which was already three times the tolerance. The noise
was the visible symptom; the invalid fixture was the cause. A diagnosis that
explains only the variance and not the magnitude is not finished.
