# RX 7600 retained pressure gradients

The CPU/CUDA `pressure_gradient_v1` feature merged in PR #75 after the
original ROCm qualification baseline. This branch integrates that foundation
from main and adds the equivalent ROCm retained-field evaluator.

The GPU evaluates pressure and its three analytic spatial derivatives with
the shared integral implementation. Near-panel adaptive correction runs on
the host, matching CPU/CUDA. Both Float32 and Float64 are supported with
fourth-order quadrature, both phasor conventions, and existing image symmetry
modes. Gradient requests preserve geometry precision and use separate cache
keys from pressure-only requests. Results retain the existing complex
`(point, 4)` C-order binary contract. The worker advertises the capability
specifically for CPU, CUDA, and ROCm.

Boundary samples too close for reliable adaptive evaluation remain rejected.
Temporary ROCm buffers are freed even when host correction rejects a sample;
the retained cache remains reusable. This is a feature-parity change, with no
performance claim or change to the pressure-only solver algorithms.

## Hardware qualification

RX 7600 (`gfx1102`), Windows 11, Julia 1.12.6, AMDGPU 2.7.2, HIP SDK 7.2.
The following gates are included in `scripts/qualify_accelerator.py`:

- `rocm_field_gradient_tests.jl`: 65 assertions, including the existing
  radiating-monopole and analytic-gradient checks, GPU/CPU comparisons,
  near-panel correction, symmetry, empty fields, and recovery after rejection.
- `rocm_field_gradient_worker_tests.jl`: 96 assertions for retained requests,
  Float32/Float64 coordinates, both phasors, ground images, cache reuse,
  binary layout, unsupported quadrature rejection, and cache release.

The existing numerical fixtures and tolerances are unchanged. The Python
suite passes all 238 tests, including backend-specific capability negotiation;
Ruff and the full standalone Julia CPU reference gate pass. The actual ROCm worker starts with an available backend and
advertises `pressure_gradient_v1`. Local logs are retained under
`runs/gradient/` in the qualification
worktree and are not committed.

An initial rejection test expected an unwrapped Julia error, but threaded
correction produces a composite exception. The test now checks the diagnostic
message. An initial Python run encountered sandbox temporary-directory
permissions; the full rerun used a fresh workspace-local temporary directory.
