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
