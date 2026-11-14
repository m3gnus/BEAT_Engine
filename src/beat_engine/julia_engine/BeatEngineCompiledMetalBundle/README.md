# Compiled Metal worker bundle

`coupled_solver.jl` loads this package for Metal compiled-system requests. The
package includes the production driver and engine; its CPU workload caches the
host call graph without an engine GPU launch. CPU uses its existing compiled
bundle; CUDA and ROCm compiled workers retain the include fallback.

Both compiled bundles decode workload requests through `JSON.parse`, matching
the worker's `JSON.Object` specialization. Alongside the tetrahedron they solve
a quadrant plate with non-adjacent and image-singular pairs, xy symmetry,
1/20 kHz, order-4 rules, a 37-by-72 sphere, a diagonal cut and boundary traces.
`CompiledExteriorWorkload.jl` keeps that representative request shared.

`CompiledCoupledWorkload.jl` adds the condensed FEM-BEM-LEM host graph. The
Metal bundle uses the frozen `ENGINE_DIR/tests/fixtures/femvolume.msh` and
`exterior_conforming.msh` (shipped by the wheel's `src/beat_engine` package
selection), with millimeter scaling, a bounded air volume, a rigid exterior,
their conforming interface and an electrodynamic transducer on `Radiator`.
A voltage port drives 500/1000 Hz; outputs are two exterior pressure points,
diaphragm velocity, voice-coil current and interface average normal velocity.
The wire topology is built from the loaded meshes with zero-based indices.

As with the exterior workload, BEM assembly runs on CPU during precompilation.
The helper resolves the engine's Metal optimization defaults with installation
overrides temporarily cleared, then applies those choices to the CPU request.
This reaches FEM assembly, transducer condensation, flux elimination, MUMPS
Schur extraction, CHOLMOD interface mass and `RefinedDenseLU` without an engine
GPU launch. Result diagnostics assert the selected solvers and log any failure;
refinement fallback is allowed but warns with its reason. The CPU bundle uses
the same request graph on four tetrahedra/four BEM triangles, with UMFPACK
instead of MUMPS. CPU CI validates the full fixture request and solves the tiny
variant through the compiled entry and the explicit include fallback.

The driver's `finally` releases systems and frequency-invariant caches. The
workload also releases retained field caches, releases any remaining live MUMPS
solvers, clears `LIBRARY` (native function pointers), `LIVE_SOLVERS` and
`ATEXIT_REGISTERED`, clears mesh/engine/runtime provenance, and collects
unreachable CHOLMOD/UMFPACK factors before image generation finishes. MUMPS
`__init__` clears the pointer cache and exit-hook flag in every fresh process,
so its lazy loader re-requires the JLLs, resolves symbols, self-tests and
forwards LP64 BLAS if needed. The native libblastrampoline forwarding tables
are process-local, not serialized Julia image data; they need no undo in the
build process. No device arrays are created by this host workload. The existing
Metal kernel workload clears process-local device state after compile/link.
Measured on an M1 Max (Julia 1.12.7, fresh worker, package images already
built, Multi_region_SAWMOD, 10 frequencies): worker start-up plus first request
was 21.4 + 38.2 s on `main`, 3.8 + 19.7 s with the exterior workload alone, and
3.8 + 10.0-10.2 s with the coupled workload; `F2B_FLH` 3.6 + 16.7 s against
3.9 + 7.0 s.

The host workload does not reach the Metal side of the coupled path (Metal-typed
cache structures, `metal_host_operators`, the regular-operator gather/scatter
drivers and launch closures, symmetry row weights, the spawned FEM stage), so
`MetalCoupledPrecompile.jl` adds compile-only signatures for it, traced from a
fresh worker's first Metal coupled request (172 compiled statements, 5.6 s; the
inventory targets the 58 that took 4.7 s). Types are spelled out as descriptors
(Float32 Metal caches and shared operator tuples, Float64 FEM and MUMPS
condensation, CHOLMOD mass blocks, refined dense LU, the per-excitation solution
schema) without creating caches, factors or device arrays. Compiler-generated
closures (the solution generator, timed solve/product closures, the FEM stage,
Metal's autorelease launch closure) are found by their exact captured-field
names, never by generated names, and the keyword body through
`Base.bodyfunction`. Kernel argument types come from `metal_kernel_signatures()`.
Left out on purpose: the wire parser's stateless generators, LLVM/ghost-type
compiler helpers, Metal broadcast internals and shutdown/archive callbacks, which
have no robust structural handle. With it, the first request falls to 3.9 +
4.7-4.9 s on the same SAWMOD measurement. Pipelined frequencies (after the
coupled sweep pipeline starts) pass the driver's producer closure as
`bem_operators` and compile that call once; they are not in the inventory.

When a closure's captured fields no longer match (the source changed), the
package build warns, names the capture set and skips the entries that depend on
it; it never fails the build. The strict check runs in CI:
`compiled_metal_worker_tests.jl` (macOS job, no GPU needed) requires every
closure to resolve and every entry to `precompile`, as `metal_host_tests.jl` does
for the runtime list. Matching a closure is not the same as matching what the
worker calls, so hardware qualification (`scripts/qualify_accelerator.py`) also
runs `metal_coupled_precompile_coverage_tests.jl`: the coupled request with Metal
BEM in a fresh worker under `--trace-compile`, failing when coupled-path
compilation exceeds `BLAB_COUPLED_COMPILE_BUDGET_MS` (1,500 ms), with a
source-fallback run that must exceed it. The budget catches a lost inventory
section (seconds), not a single small closure; the CI check covers those.

`MetalHostPrecompile.jl` additionally calls `precompile(f, argtypes)` for the
Float32 native exterior path: fused assembly and gather orchestration, host
launch methods, array construction/conversion, shared-buffer wrapping, dense
solve glue, field evaluation and sweep planning/consumption. It evaluates
concrete types without creating device arrays or executing those methods.
Launch types reuse the generated kernel inventory, restoring the private
temporary and shared destination storage modes used by the production path.
The host workload logs successes and failures; the host test checks every
signature has a compilable method. Other precisions, storage overrides and
diagnostic assembly paths can still need runtime compilation.

Metal 1.11.1 and its resolved GPUCompiler 2.9.0 stack can persist compiled
device code in Julia package images. `MetalKernelPrecompile.jl` calls
`Metal.mtlfunction(f, TT)` inside `@compile_workload`, compiling and linking each
signature without launching it. The workload gates on Apple Silicon rather than
`Metal.functional()`, which is false during package-image generation. A host
without an accessible device receives warnings and retains the host-code cache.
The completion log reports compiled-signature and failure counts. Process-local
Metal objects are cleared using the cleanup performed by Metal 1.11.1 itself;
the dependency pin protects that internal API contract.

The bundle's `__init__` also clears provenance captured by the host workload.
The worker recomputes its source identity, active project and thread count in
the running process rather than reporting the precompile process's metadata.
The Metal entry point loads Metal before JSON, matching the bundle's dependency
order and avoiding invalidation of its cached worker call graph. Other backend
and explicit fallback loading orders are retained.

`MetalKernelSignatures.jl` is generated, not a manually maintained inventory.
The hardware gate `metal_kernel_coverage_tests.jl` observes production
`solve_request` calls using GPUCompiler's scoped debug hook. Metal invokes the
hook even on package-image cache hits. The test fails on any observed signature
missing from the inventory, and never changes the inventory in a normal run.
It covers Float32 fused and four-operator assembly, off/x/xy/ground symmetry,
regular quadrature orders 1/2/4, two frequencies, two drive ports,
37-point exterior pressure and radiation impedance. Diagnostic kernel modes
and custom workloads can still compile additional specializations at runtime.

After reviewing an engine or dependency change, regenerate explicitly on a
Metal GPU, rebuild the bundle, and rerun the ordinary test:

```sh
julia --project=src/beat_engine/julia_metal src/beat_engine/julia_local/tests/metal_kernel_coverage_tests.jl --generate
julia --project=src/beat_engine/julia_metal -e 'using Pkg; Pkg.precompile()'
julia --project=src/beat_engine/julia_metal src/beat_engine/julia_local/tests/metal_kernel_coverage_tests.jl
```

The Metal hardware qualification script runs this coverage test and
`metal_host_tests.jl`. The latter starts the actual compiled worker entry point
in a fresh process and asserts that it dispatched to this bundle rather than
silently taking the include fallback.
It also checks the runtime thread count and active project in the handshake.

No numerical kernel or launch sequence is changed. Native AOT versus JIT host
code can differ at Float32 round-off; compare complete complex outputs and the
unmodified engine's repeatability. `BLAB_BEAT_ENGINE_BUNDLE=0` selects the
existing include fallback for diagnosis.

`MetalRuntimePrecompile.jl` adds first-request specializations observed on
multiple exterior workloads, including wire validation/cleanup, asynchronous
assembly, and Metal argument encoding. Its types are evaluated only inside the
compile workload; the host gate checks that every signature still matches.
Compiler-generated types are resolved structurally (by captured fields or
`Base.bodyfunction`), never by their generated names, which change between
Julia releases. Retrace the inventory after Julia, engine or Metal-stack
changes.
