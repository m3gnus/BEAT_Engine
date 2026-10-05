# Standalone BEAT numerical references

Run the required CPU extraction gate from any working directory:

```text
julia --threads=2 --startup-file=no --project=<engine>/julia_local <engine>/julia_local/tests/reference_tests.jl
```

It requires the Julia environment to be instantiated. The gate and its fixtures
live entirely under `julia_local`; neither Boundary Lab Python code, Qt, Bempp,
nor application project files are needed. Preserve the sibling `beat_contract`
directory when copying the complete engine/worker distribution. CI runs this gate
from a copied engine tree outside the checkout, after the ordinary Julia suite.

## Coverage and interpretation

| Behavior | Reference | Acceptance |
|---|---|---|
| Exterior BEM and arbitrary complex-pressure probes | Manufactured outgoing Helmholtz point-source solution enclosed by a generated 128-face surface | Relative complex field error below 8% at three exterior points |
| Independent excitations | Centered and displaced enclosed point sources; complex combination of separately solved traces versus a combined solve | Each source matches its analytical field; complex superposition at relative tolerance 1e-12 |
| Rigid-source direction and frame | Compiled v2 driven patch on an asymmetric closed tetrahedron, with the complete problem rotated 90 degrees about x/y and by a generic rotation, in both phasors | Complex Neumann data, boundary pressure, observed field and generalized impedance agree with the unrotated z drive at relative/absolute tolerance 1e-10 on CPU, 8e-4 on Metal |
| Per-component generalized force | Two compiled v2 sources, each owning two differently weighted boundaries and a different global axis; independent pressure-column integration in both port orders and both phasors | Each component's radiation impedance equals its own signed weighted pressure integral; reversing ports reverses pressure/Neumann columns while preserving radiator order, on CPU and available Metal |
| Axis builder and reductions | Float64 maximum and smallest positive subnormal before Float32/Float64 conversion; malformed axes and isolated X/Y reduction violations | Unit direction at both extremes; refuse zero, non-finite, non-numeric and wrong-shape axes; accept in-plane x/xy motion and arbitrary ground motion |
| Translating rigid sphere | Compiled v2 unit velocity along x, y, z and an oblique axis, compared with the outgoing dipole solution on generated 128/512-face spheres in both phasors | Fine-mesh relative complex field error below 8% and less than half the coarse error, real gain above 0.92; both meshes check phase within 0.08 radians, positive gain and signed boundary derivative against Euler |
| Retained fields | Re-evaluate solved pressure/Neumann traces at exterior points; complex scaling and combination | Relative tolerance 1e-12 |
| X and XY symmetry | Reduced field evaluation versus separately constructed full reflected geometry, including normal orientation | Relative tolerance 1e-12, absolute 1e-14 |
| Interior FEM | Generated sealed unit-cube cavity modes, exact affine tetrahedron matrices, and preserved mesh matrix references | Existing mode discretization tolerance 8%; existing matrix/precision tolerances unchanged |
| Coupled FEM-BEM | Full direct solves, pressure continuity, flux conservation, and BEM replay | Existing residual/continuity tolerances, including 1e-8 and 1e-10 checks |
| Coupled FEM-BEM-LEM | Voltage-driven transducer solved using monolithic and condensed formulations | Existing FP64 relative tolerance 1e-9 on acoustic/electromechanical quantities |
| Precision | FP32 versus FP64 coupled and condensed results | Existing relative tolerance 1e-4 on primary fields |

The analytical exterior oracle is `p(x) = exp(i k r) / r`, for the solver's
`exp(-i omega t)` convention. Its facet-normal derivative is
`(i k - 1/r) p(x) dot(n, (x-source)/r)`. The source is enclosed, so the entire
exterior domain satisfies homogeneous Helmholtz. The oracle does not call BEAT's
Green-function implementation. Facet-centroid DP0 data and the coarse P1 mesh
introduce discretization error; the 8% bound is an accuracy guard, not a promise
of production convergence. The assertions compare complex pressure, not SPL.

The translating-sphere oracle in `axial_source_tests.jl` uses the exact outgoing
degree-one spherical wave. For radius `a`, real unit translation velocity along
`u`, and negative time, it is
`p(x) = i*rho*omega*h1(k*r)/(k*h1'(k*a))*dot(u,x/r)` with
`h1(z) = -exp(i*z)*(z+i)/z^2`. The normal derivative at the surface follows
independently from linearized Euler, `dp/dr = i*rho*omega*dot(u,x/r)`.
Positive time conjugates the radial wave and reverses that derivative's sign.
See [NASA-CR-98418, Appendix B, equations B-1–B-4](https://ntrs.nasa.gov/api/citations/19690015869/downloads/19690015869.pdf).
The tests implement explicit Hankel expressions without calling engine phasor
helpers or Green functions, then compare the public compiled solve's complex
outputs. The 8% bound is retained from the monopole check. The translating sphere
also requires error to decrease by at least a factor of two on refinement from
128 to 512 planar facets, distinguishing geometry error from a persistent sign
error. Rotation checks use solver precision, without discretization slack.
These tests run on CPU in both ordinary and reference suites, and additionally
on Metal when the runtime suite's functional-device gate is available.

Condensed-versus-monolithic tests are independent elimination/formulation checks
but share assembly kernels. They complement the analytical checks; they do not
constitute an independently implemented BEM solver. Frozen fixture hashes protect
the input baseline; they are not numerical expected outputs.

The required runner forces `BLAB_RUN_COUPLED_REFERENCE=1` and first-order coupled
quadrature settings, preventing ambient opt-out settings from skipping the dense
comparisons. Those coupled tests compare algebra/formulations at identical
discretization. The analytical exterior tests independently use order 3. Ordinary
`runtests.jl` retains its existing faster default and hardware-dependent checks.

## Accelerator and extended qualification

The existing `BLAB_RUN_COUPLED_CUDA=1`, `BLAB_RUN_COUPLED_ROCM=1` and
`BLAB_RUN_COUPLED_METAL=1` gates in `runtests.jl` still require functioning
hardware and their corresponding Julia projects. CPU reference success does not
qualify CUDA, ROCm, or Metal.

The Metal gate runs with the `julia_metal` project on Apple Silicon. Its
`metal production pipeline` testset solves `off`, `x` and `xy` on the meshes
that are fundamental domains for each, checks the operators land on the GPU
and come back as host matrices, compares the fused Burton-Miller system with
the four-operator one, and asserts two assemblies are bit-identical, which the
gather write-back guarantees and an atomic scatter cannot. The coupled Metal
tests and the Metal arm of the phasor conjugation test run under the same gate.
Hardware release runs must inspect actual test execution; an unavailable-device
skip is not a numerical pass.

The CUDA coupled gate includes `cuda_fem_analysis_tests.jl`. It checks repeated
value updates against fresh cuDSS Schur complements and an FP64 reference,
multiple-right-hand-side forward/backward reconstruction, and cache invalidation
when sparsity, retained-node ordering, or precision changes. A small weakly damped
interior-pole sweep also compares reused and fresh analysis against the same
rounded FP32 coefficients evaluated in FP64. A frequency system
borrows the job's analysis workspace until `release_coupled_system!`; release it
before building the next frequency with the same prepared cache. Job-cache
release destroys the retained cuDSS resources. This reuse runs the ordinary
numerical factorization phase at every frequency, retaining only the analysis.

The larger noncubic-cavity convergence family remains in Boundary Lab's
`tests/fixtures/noncubic_cavity` with its existing optional runner. It is extended
application validation, not a dependency of this portable gate. Preserve that
family if it is included in a future engine release's extended qualification.

The remaining source-request reference harness is retained. Remove it only when
the physical-system replacement covers the comparisons it provides; this gate
creates a portable baseline and does not silently retire additional comparisons.

The CUDA suite also runs `coupled_bem_cuda_tests.jl`: combined A/C versus original
operator matrices, complex/signed sparse projection, separate/fused image
assembly, an explicit register cap, both phasor conventions, symmetry modes,
coupled cache reuse across frequencies, and full-diagnostic fallback. The tests
preserve complex pressure/flux and independent source columns. Run with the
CUDA Julia project and a functional device; an unavailable-device skip does not
qualify this architecture.

Source requests accept `config.symmetry = "ground"` for a rigid image plane at
Y=0. Mesh coordinates (including `meshes[].translation_m`) must put the whole
radiator at Y >= 0, with no triangle flat on the plane. Contact vertices and
edges are allowed. Ground and native mirror modes (`x`, `xy`) are exclusive
values of the same option. The image contributes to pressure, but counts as
one physical radiator for impedance. The optional
`ground_plane_min_clearance_m` must be finite and non-negative and enforces
clearance of the radiator surface in metres; its default of zero allows contact
edges. Compiled requests use the same guard and option in solver options.
Placement and observation-frame mapping remain the caller's responsibility.

`source_ground_contract_tests.jl` runs on CPU in both `runtests.jl` and the
standalone reference gate. It checks domain refusals, clearance, unchanged
native mirror counts, and an active ground image without a 6.02 dB impedance
increase through both source assembly paths. Additional Metal parity cases are
opt-in: set `BLAB_VALIDATE_GROUND=1` when running
`scripts/validate_metal_symmetry.jl` with the `julia_metal` project on a functional
Metal device. They cover lifted `sample_half.msh` and resting-edge
`sample_quarter.msh` with the existing operator, pressure and field bounds.
