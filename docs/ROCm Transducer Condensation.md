# ROCm transducer condensation

ROCm coupled solves can eliminate pressure unknowns on moving transducer
surfaces along with the other interior FEM vertices. Previously these surface
vertices stayed in the dense system even when FEM static condensation was
enabled. The reduced system retains the FEM interface vertices, BEM pressure,
interface flux, diaphragm velocities and coil currents. Full FEM pressures are
reconstructed for every excitation after the solve.

The FEM factorization and Schur extraction remain on the CPU. The reduced dense
factorization and solves remain on ROCm/rocSOLVER, with ComplexF64 factors for
both requested precisions. Returned fields and traces retain the requested
precision. This change does not eliminate interface flux unknowns or change
the field-evaluation algorithm.

## Selection and compatibility

`BLAB_COUPLED_TRANSDUCER_CONDENSATION` accepts `auto` (the default), `on`, and
`off`, with the existing boolean aliases for on/off.

- Automatic selection applies to ROCm static-condensation solves with
  transducers. New compiled-worker caches retain interface vertices only.
- A directly supplied cache that retains transducer-surface vertices keeps
  the previous solve layout in automatic mode. This preserves existing callers.
- Explicit `on`, or the Julia keyword `transducer_condensation=true`, requires
  a matching interface-only cache. Cache-layout validation remains strict.
- Explicit `off`, or `transducer_condensation=false`, selects the previous
  retained-surface formulation.
- Compiled ROM export and rank experiments retain transducer surfaces because
  their state layout requires them. Direct Julia ROM callers must also request
  `transducer_condensation=false`.
- Diagnostics report the actual choice as `rocm_transducer_condensation`,
  together with the solved system order and dense-factor precision.

## Elimination terms

For the acoustic/mechanical block `[A -s*C; -transpose(R) Z]`, split FEM
vertices into eliminated interior vertices I and retained interface vertices G.
Here s is the phasor-dependent normal-derivative scale. With
`X = A_II \ C_I` and `Y = transpose(A_II) \ R_I`, the reduction uses:

- `C_G - A_GI*X` for the motion load;
- `R_G - transpose(A_IG)*Y` for the pressure-force coupling;
- `Z - s*transpose(R_I)*X` for the mechanical correction;
- `transpose(Y)*f_I` added to the mechanical right-hand side;
- `A_II \ (f_I - A_IG*p_G + s*C_I*v)` for interior pressure reconstruction.

These are transposes, not conjugate transposes. The shared host helper preserves
the existing CPU operation order. Independent full complex-system tests include
non-Hermitian matrices, empty interior/retained partitions, both signs of s,
pure mechanical forcing, pure FEM forcing and mixed right-hand sides.

Native qualification also checks both phasor conventions, cached and uncached
solves, both requested precisions, pressure/flux/current/velocity/field outputs,
and comparison with a monolithic CPU solve just below and above a cavity pole.
The compiled-worker tests cover automatic selection and the retained-surface
ROM path.

## Measured implementation and automatic default

The interleaved on/off measurements use commit
`6c01dad86e68ab878ce439feff4a3e52224e6b27`. That commit kept the new path opt-in
during qualification. The subsequent automatic-selection change adds cache
compatibility and chooses between those same measured paths; it does not change
their condensation algebra. The qualification records separately identify that
later source state. Performance measurements do not substitute for its CPU,
hardware and headless application checks.


## RX 7600 qualification

The automatic-selection source state is commit `9da52ae35897f3b2b3df9141611a17a24eb949b8`. Its six
changed-file hashes match the frozen qualification source. The measured opt-in
implementation is `6c01dad86e68ab878ce439feff4a3e52224e6b27`.
See [the interleaved profile](ROCm%20Transducer%20Condensation%20Profile.md)
for workloads, timing ranges, memory scope and accuracy.

Qualification completed on 2026-10-30 with Julia 1.12.6, AMDGPU.jl 2.7.2,
ROCm SDK 7.2 and two Julia/OpenBLAS threads on the Windows RX 7600 host:

- Standalone CPU reference gate: passed (453.13 s), including 34 automatic
  cache-selection assertions and 12 independent elimination-algebra assertions.
- Full hardware qualifier: passed (2265.87 s). This includes the original
  hardware suite and all 12 dedicated ROCm test files, covering phasor handling,
  near corrections, wavelength quadrature, exterior transducers, interface
  radiation, condensation, speaker rank experiments and physical ROM responses.
- New native condensation checks: 2288 forcing/reconstruction assertions,
  1234 automatic-cache assertions, and 2288 assertions against monolithic CPU
  near the first cavity pole (383.895130123 Hz, sampled at 0.999 and 1.001
  times that frequency with loss factor 1e-4).
- Compiled-worker checks: 248 condensation/cache assertions and 48 ROM fallback
  assertions. Automatic mode produces the same decoded outputs as explicit on;
  ROM requests retain the previous layout.
- Headless S218BP and SAWMOD solves: both requested precisions passed project
  validation, actual-backend/precision/dispatch checks and all 128 per-output,
  per-frequency, per-excitation comparisons against pristine CPU Float64.

The headless runs leave the selection environment variable unset, exercising
its automatic default. They use the application precision fix at
`d4e24779bdf2f792b3f9362a51185da82d92313d` and CPU reference engine
`1bb46da833a2a2d330d9b7d915a7ee5c31934812`. S218BP uses 50/125 Hz;
SAWMOD uses 250/1000 Hz. Both use quadrature orders 2/2, all excitation
ports, explicit probes at (0,0,3) and (1,1,3), and retained boundary outputs.

| Application case | Comparisons | Worst relative L2 | Worst dB within 30 dB of peak |
|---|---:|---:|---:|
| s218bp-float32 | 16 | 9.35057e-05 | 0.00107679 |
| s218bp-float64 | 16 | 8.34162e-14 | 7.13988e-12 |
| sawmod-float32 | 48 | 1.99415e-05 | 0.000918908 |
| sawmod-float64 | 48 | 9.19198e-12 | 3.89244e-09 |

The unchanged relative L2 limit is 5e-4; fixture baselines and tolerances were
not regenerated. These checks qualify the tested RX 7600 configuration and
workloads, not every AMD GPU or ROCm version. CPU FEM condensation and GPU dense
solving remain a hybrid path, and requested Float64 field evaluation remains
materially slower than Float32 on this card.

Local evidence is under `runs/rx7600-profiling/`:
`condensation-profile-20261029-161531` for the measurements and
`condensation-default-gates-20261030-122510` for final qualification. Generated
runs are intentionally untracked. The latter contains source hashes, commands,
logs, manifests, comparison reports and the qualified-source commit binding.
The first successful headless process outlived an immediate idle check during
normal worker shutdown. The runner resumed only after verifying no Julia
processes remained and all source hashes matched; subsequent jobs allowed up
to 30 seconds for shutdown. All six jobs returned zero, the resumed runner
completed successfully, and no Julia processes remained afterward.
