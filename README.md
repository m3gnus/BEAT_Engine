# BEAT Engine

Numerical BEM, FEM, and lumped electroacoustic solving, extracted from Boundary Lab.
The `beat-engine` Python distribution supplies a standard-library worker client,
versioned wire contracts, Julia CPU/CUDA/ROCm/Metal environments, and numerical
tests. Project authoring, mesh compilation, Qt, and application preferences remain
with the consuming application.

## Local installation

Use Python 3.11+ and install Julia separately (the current qualification uses Julia
1.12). A wheel does not install Julia or GPU drivers.

```text
python -m pip install -e ".[dev]"
python -m beat_engine instantiate --backend cpu
python -m beat_engine doctor --backend cpu --threads 2
python -m beat_engine paths --backend cpu
```

For CUDA or ROCm, instantiate the matching backend and configure the platform SDK
before running `doctor`. For Metal, instantiate the `metal` backend on Apple
Silicon; no SDK beyond macOS is needed. See [the Metal backend](docs/Metal%20Backend.md). Availability is checked by the worker; it is not inferred
from the package being installed. Importing the package does not install packages,
launch Julia, or modify the environment.

## Public Python API

```python
from beat_engine import EngineWorker, engine_paths

paths = engine_paths("cpu")
worker = EngineWorker(
    julia_executable="julia", solver_script=paths.system_solver,
    julia_project=paths.project, julia_threads=2,
)
try:
    worker.ensure_started()
    print(worker.worker_info)
    # request.json contains a validated compiled-system request with local assets.
    # for event in worker.submit(request_path): ...
finally:
    worker.terminate()
```

`EngineWorker`, `WorkerPool`, `EnginePaths`, and `engine_paths` are the public runtime
API. `beat_engine.beat_contract` exposes model-independent contract validators and
version constants. Submission events are raw JSON records; clients own result
models and display conversion. Drain or close event iterators. Worker paths come
from the installed package, including editable contributor checkouts.

Read [the compiled-system contract](docs/BEAT%20Compiled%20System%20Contract.md),
[worker protocol](docs/BEAT%20Worker%20Protocol.md), and
[provenance format](docs/BEAT%20Run%20Provenance.md).
Rapid, subsequent CUDA solves can have memory cleanup policies applied via [bounded worker cleanup reuse](docs/WorkerCleanup.md).

CPU source-request and compiled exterior solves default to
`regular_quadrature_mode = "fixed"`, using the base `quadrature_order` (default
4, six points per triangle), matching accelerator regular quadrature. A single
mesh-wide wavelength rule can underestimate the quadrature needed by coarse
elements when many small elements dominate the mesh statistic, causing inaccurate
high-frequency directivity. `"wavelength"` remains an explicit CPU opt-in and is
unsafe on graded meshes. See [CPU regular quadrature](docs/CPU%20Regular%20Kernel.md#regular-quadrature-mode).

## Tests and releases

```text
python -m pytest
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/runtests.jl
julia --threads=2 --startup-file=no --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/reference_tests.jl
```

The required CPU reference gate has 464 checks including analytical complex
pressure, independent excitations, symmetry, FEM modes, and coupled transducer
comparisons. See its [coverage notes](src/beat_engine/julia_local/tests/README.md).
The manual Accelerator qualification workflow uses self-hosted runners labeled
`cuda`, `rocm`, or `metal`, with Python, Julia, and the matching SDK preinstalled. It requires
the requested backend to be functional before running the numerical gates.
ROCm qualification also checks interface-radiation replay against a monolithic
CPU reference for both precisions and phasor conventions. It verifies complex
excitations, partition ordering, zero input, direct exterior-source remainders,
and reconstruction of boundary traces and observation fields for monolithic
and condensed ROCm solves. Replay uses the shared host factorization; the
coupled solve and observation-field evaluation use ROCm.
GPU qualification remains hardware-specific; CPU success does not qualify GPU
backends. Some historical research scripts and the optional noncubic-cavity test
still require Boundary Lab's extended fixtures; they are not installed runtime
requirements or part of the portable release gate.

The speaker ROM rank experiment accepts ROCm FEM-interface-condensed systems.
It copies the retained Schur block to the host and uses CPU LU for rank analysis;
`interior_factorization_backend` reports this separately from the coupled backend.
The hardware gate compares all four parity sectors on four reflected cavities,
in both precisions and phasor conventions, against CPU condensation with interface
elimination explicitly disabled. It does not qualify exported ROM artifacts or
claim faster rank analysis.

The ROCm ROM export gate reconstructs physical responses from the exported
Petrov-Galerkin matrices, including direct affine terms and padded sector states.
It compares boundary flux, diaphragm velocity, and voice-coil current with full
interior solves and CPU exports for complex voltage and exterior-velocity drives.
This checks numerical ROM contents; serialized package and Deploy execution
qualification are separate.

Exterior-only compiled-system CUDA and Metal solves default to direct
Burton–Miller assembly. CUDA uses the same assembler as Deploy; Metal uses its
fused assembler, which forms the system on the GPU and factorizes once on the
host because Metal.jl has no GPU LU. One pivoted LU factorization serves all
excitation right-hand sides at a frequency. Set the solver option
`burton_miller_assembly` to `operator_matrices` for the previous assembly path;
CPU and ROCm continue to use that path. The result diagnostics report the
effective assembly mode and factorization count. Coupled FEM-BEM solves are
unaffected by this exterior option.

Coupled CUDA solves without full diagnostics now default to combined A/C
Burton-Miller assembly, job-cached sparse interface projection, and fused symmetry
images. The original operator path remains available for comparisons and full
diagnostics. Coupled Metal solves assemble the four operators on the GPU and run
the coupled algebra, including FEM static condensation, on the host. Register caps are explicit hardware tuning options; the compiler
default is preserved. See [coupled CUDA architecture and controls](docs/Coupled%20CUDA%20Assembly.md).

The package version is `0.4.0`. CI builds a wheel after independent
Python and CPU checks on Windows, Linux, and macOS. The manually dispatched release
workflow requires a successful CI run for the tag's exact commit before attaching
wheel/sdist artifacts to a stable GitHub release. No PyPI publication is configured.

History and original authorship were retained from Boundary Lab. See
[extraction provenance](docs/EXTRACTION.md) and the repository [LICENSE](LICENSE).

## Backend catalog

`beat_engine.backend_catalog()` returns immutable `BackendInfo` records for the
backends supplied by the installed engine. Each record contains a stable ID,
display label, Julia project directory, supported platforms and solve capabilities.
`backend_info(id)` looks up one record, and `engine_paths(id)` resolves its assets.
The catalog also drives the CLI: `python -m beat_engine backends` prints JSON.

Listing backends does not launch Julia, inspect hardware, install dependencies or
access the network. A listed backend may not work on the current machine. Clients
should retain the selected ID and let worker negotiation validate device/runtime
availability and request compatibility at solve time. Preferences can therefore
list the catalog immediately without probing or changing options by host.

## Contributing and releases

Use scoped PRs targeting `main`. See [contributing](CONTRIBUTING.md) and the
[release process](docs/development.md). Stable releases are independent of main.
