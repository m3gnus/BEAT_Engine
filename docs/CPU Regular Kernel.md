# CPU regular kernel

The CPU backend integrates regular (non-touching) element pairs with one of two
kernels. `BLAB_BEAT_CPU_REGULAR_KERNEL` selects it; the default is `simd`.

| value | kernel |
|---|---|
| `simd` | `src/BeatEngineCpuSimd.jl`: trial elements batched in a structure-of-arrays layout, inner loop vectorised by LLVM |
| `scalar` | the original per-pair kernels in `BeatEngineCpuAssembly.jl` and `BeatEngineCpuBurtonMiller.jl` |

Each frequency's source-request diagnostics report `cpu_regular_kernel` (`null`
on every other backend).

## What it changes, and what it cannot

Only the CPU source-request driver (`BeatEngineDriver.jl`) selects it, for both
its fused Burton-Miller path and its four-operator path (`BLAB_BEAT_FUSED_BM=0`).

Every library function defaults to the scalar kernel:
`assemble_burton_miller_neumann_system_cpu(...; regular_kernel=:scalar)`,
`assemble_regular_galerkin_operators_cpu(...; regular_kernel=:scalar)` and
`assemble_regular_galerkin_operators(...; cpu_regular_kernel=:scalar)`, whose
keyword is read by the `backend == :cpu` branch only. So the following are bit
for bit what they were:

- CUDA, ROCm and Metal assembly, including the Metal and ROCm host-staged
  paths, which call the CPU assembly directly;
- the CPU references the accelerator validators compare against;
- every compiled-system solve in `coupled_solver.jl`: exterior-only systems and
  coupled FEM-BEM(-LEM), monolithic and condensed, on any BEM backend, CPU
  included. Opting the exterior-only CPU path in is one keyword
  (`cpu_regular_kernel=beat_cpu_regular_kernel()`), left out until it has been
  benchmarked on that path;
- singular, image-singular and near-pair corrections, which keep the scalar
  Duffy kernels under either setting.

## How it works

The regular pass is an all-pairs loop. The scalar kernel takes one element pair
at a time and visits its 9 (order 2) or 36 (order 4) quadrature-point pairs,
each with a `sincos`, a reciprocal and a dozen complex accumulations. The
vectorised kernel fixes the test element and walks the trial elements in blocks
of 256, reading their quadrature points, normals, areas and curls from
contiguous arrays, so the innermost loop is a plain `@simd ivdep` loop over
trial elements. LLVM chooses the vector width; no SIMD package is used and no
dependency is added. Threading is unchanged: coloured test elements, with one
scratch buffer per chunk of a colour group.

The loop has no branch and no call, which needs three departures from the
scalar arithmetic:

- `sincos` is a branch-free polynomial: three-part Cody-Waite reduction by
  pi/2 and Cephes coefficients. Measured maximum absolute error on
  [0, 2500] rad is 0.78 `eps(T)`, against Base's 0.25 (Float32) and 0.39
  (Float64).
- A coincident point pair is masked (`inv_radius = 0`) instead of skipped, and
  an adjacent trial element gets a zero Jacobian and is not scattered.
- The basis expansion is summed over trial points first and the outer product
  with the test basis is taken afterwards.

## Numerical effect

The two kernels agree to rounding, not bitwise. `tests/cpu_simd_kernel_tests.jl`
bounds the assembled operators entrywise at `2e-6` (Float32) and `1e-13`
(Float64) of the largest scalar entry, for regular order 2 and 4, symmetry off,
x, xy and ground, both phasor conventions, element subsets, several right-hand
sides, and block boundaries. Measured differences are 6e-8 to 4e-7 (Float32) and
1e-16 to 2e-15 (Float64); for scale, the scalar Float32 kernel is about 5e-6
from the scalar Float64 one.

Threaded and serial runs of the vectorised regular pass are bitwise identical.

## Measuring it on another machine

```text
julia -t auto --startup-file=no --project=src/beat_engine/julia_local \
  src/beat_engine/julia_local/scripts/benchmark_cpu_regular_kernel.jl --mesh <mesh> --scale <factor>
```

prints the regular-stage time under both kernels, their entrywise difference,
and the widest floating-point vector LLVM emitted for the loop. It has been
measured on x86-64 with AVX2 only.

## Field evaluation and singular corrections

Two further CPU stages use the same design, each with its own switch, and
the same rule: library functions default to scalar and only the CPU
source-request driver opts in.

| variable | default | stage |
|---|---|---|
| `BLAB_BEAT_CPU_FIELD_KERNEL` | `simd` | `evaluate_galerkin_field_cpu(...; kernel=:scalar)` |
| `BLAB_BEAT_CPU_SINGULAR_KERNEL` | `simd` | `assemble_burton_miller_neumann_system_cpu(...; singular_kernel=:scalar)` |

**Field evaluation.** The scalar loop recomputed, for every observation point
and every source quadrature point, the source's pressure density
(`basis . p[face] * weight`) and Neumann density (`q[element] * weight`).
Neither depends on the observation point. They are now formed once per call
into contiguous arrays with the source coordinates and normals, and the sum
over sources is a plain `@simd` reduction with the same polynomial `sincos`
and zero-radius mask as the regular kernel. The compiled-system field paths
in `coupled_solver.jl` keep the scalar loop.

**Singular corrections (fused Burton-Miller).** Each touching pair is
integrated with a Duffy rule of 32 to 1,536 point pairs. A rule stored as
contiguous coordinate arrays (`BeatCpuDuffyRuleSoA`) dispatches
`_beat_cpu_bm_pair_blocks` to a vectorised method: basis values and global
points are formed inline (both are affine in the reference coordinates) and
the loop over the rule is one `@simd` reduction into 26 real accumulators.
The singular drivers are unchanged; the assembly converts the rules when
asked. The four-operator singular path stays scalar.

Both sums have thousands of terms, so in Float32 a changed summation order
moves a result by more than the regular pass does. Their Float32 gates are
therefore against Float64: the vectorised Float32 result must be no further
from the Float64 result than the scalar Float32 result is. In Float64 they are
gated directly at `1e-13` of the largest entry. Measured, the vectorised
singular pass is slightly closer to Float64 than the scalar one.

**Scatter.** With the regular kernel vectorised, writing its blocks into the
dense matrix was about half of the pass at quadrature order 2: a test
element's rows are strided in column-major storage. The pass drivers now
transpose the square operators in place, the kernels write the transpose
(three columns, in trial-dof order), and the drivers transpose back. The
transpose is an involution, so whatever the matrix held before is preserved.
