# Deploy feedback compression research spike

This opt-in numerical experiment compresses the DP0-flux to P1-pressure
Burton–Miller feedback operator `R` in the Deploy CUDA parity-ROM path.
The dense exterior `L` matrix, its CUDA LU, the package ROM, and the ordinary
solver defaults are unchanged. This is a block-low-rank feasibility prototype,
not a hierarchical LU implementation or a production speed claim.

## Request

Use a validated `boundary_lab_deploy_rom` request with:

```json
{
  "rom_feedback_mode": "compressed",
  "experimental_rhs_compression": {
    "tolerance": 0.0001,
    "leaf_size": 128,
    "admissibility": 1.0,
    "survey_only": false,
    "report_path": "absolute-path-to-rank-report.json"
  },
  "include_complex_pressure": true
}
```

The existing `BEAT_DEPLOY_ROM_FEEDBACK` override also accepts `compressed`.
This mode requires CUDA and a ROM request. Set `survey_only` to assemble four
spatial column slabs and inspect up to eight row leaves per slab; the actual
solve then uses exact matrix-free feedback. Sampled storage is not an estimate
of total operator storage.

## Numerical implementation

* `assemble_burton_miller_rhs_cuda(...; assemble_operator=true,
  operator_columns=indices)` assembles exact selected columns. All existing
  singular, near, reflected-image, identity and row-weight contributions use
  the same scatter mapping. No full `R` allocation is necessary.
* Source-face centers and target nodes are split geometrically. Admissibility
  uses full triangle support, including every triangle supporting a P1 node,
  and checks both direct and reflected distances: maximum support diameter must
  be no larger than `admissibility` times either separation distance. Increasing
  this setting admits more blocks for compression but does not relax the error
  tolerance; overlapping supports remain dense.
* Near blocks remain dense. Admissible row clusters are merged over column
  leaves; complete-pivot cross approximation on each exact dense block stops
  only after its actual Frobenius error is small enough. A failed rank or
  storage-benefit check retains the exact dense block. The rank cap is 64.
* Dense and low-rank blocks are packed into GPU arrays. Two kernels apply the
  right factors and then the dense/left factors, avoiding per-block launches.
  Complex outer products do not conjugate their right factors.
* After GMRES, an independent exact-quadrature feedback application and the
  original LU audit the **exact left-preconditioned** equation. A relative
  residual above `1e-3` rejects the experiment. This is not the unpreconditioned
  physical-equation residual and is not a mesh-convergence test.

The initial drive RHS remains exact. Final pressure feeds the unchanged ROM
velocity/current reconstruction and exterior field evaluator. New settings
must be included in any future application-level solution cache key; the
research harness always submits a fresh solve and bypasses field-only reuse.

## Validation and measurement

Run the standalone CUDA gate:

```powershell
$env:BLAB_BEAT_ENGINE_GPU_BACKEND='cuda'
julia --project=src/beat_engine/julia_cuda --threads=4 src/beat_engine/julia_local/tests/deploy_rhs_compression_tests.jl
```

Also run the required CPU `tests/reference_tests.jl` entrypoint under
`julia_local`, following the repository contributor guide. Do not update frozen
baselines to accept this approximation.

The Deploy checkout contains `scripts/research_deploy_compression.py` and
`scripts/compare_deploy_compression.py`. The replay harness validates a saved
schema-12 scene and package hashes, preserves channel DSP, prepares a real ROM
request, and records source revisions/diffs/snapshots, raw events, complex
results, timings, and sampled device-wide GPU memory. It uses the imported
engine paths and disables bundles so edited source is actually executed.
It does not use the Boundary Lab `.blab.json` CLI for Deploy scenes.

Example from the Deploy checkout (adjust paths for another machine):

```powershell
$env:PYTHONPATH='E:/Code/boundary_lab_deploy/src;E:/Code/BEAT_Engine/src'
python scripts/research_deploy_compression.py 'G:/My Drive/Projects/Boundary Lab Projects/largebattlehawkscene.blabdeploy.json' --output runs/compression-study --counts 16 --modes matrix_free,compressed --leaf-size 128 --admissibility 4 --warmup 1 --repeat 3 --field-planes
python scripts/compare_deploy_compression.py runs/compression-study
```

Use a new output directory for each experiment. `--prepare-only` validates and
stages requests without starting a worker. `--backend cpu --counts 1` supplies
a tractable real-project CPU reference; compression itself is CUDA-only.
`--field-planes` evaluates both saved planes against each mode's final retained
boundary solution without repeating the coupled solve. `--frequencies` accepts
exact exported package frequencies, not arbitrary interpolated ROM frequencies.

Compare at least three interleaved measured runs after warming both modes.
Include construction, packing, audit and output costs in end-to-end time;
do not infer a speedup from apply-only timings. A pair passes the initial
comparison gate only with matching finite outputs, both GMRES residuals at or
below `1e-4`, the independent audit at or below `1e-3`, complex relative L2
errors below 1%, and worst magnitude errors below 0.1 dB within 30 dB of peak.
The harness's `wall_s` measures the worker request, including its repeated
geometry setup and transport. Shared Python request preparation is reported
separately as `prepare_s`; worker startup is outside that timer. Do not label
`wall_s` as desktop click-to-display latency.

## Deliberate limitations

Construction still computes every dense entry and performs host compression;
it is quadratic work, not an ACA entry-oracle implementation. Near corrections
are recomputed for each slab. Packed application is a simple bandwidth-oriented
kernel, not batched BLAS or an H2 nested-basis algorithm. Factors are rebuilt
per solve, and global LU remains. VRAM checks reject oversized operators instead
of silently enabling oversubscription. Host peak memory is not measured by the
current harness; GPU samples include other applications and can miss brief peaks.

This approximation changes numerical results and should be reviewed in a
separate experimental numerical PR before any default or packaging changes.

## Follow-up: exact ROM range projection

`rom_feedback_mode: "projected"` enables a second CUDA-only research path.
For feedback with zero electrical drive, the existing ROM produces
`q = U * state(p)`, where `U` expands each sector's exported `d` matrix into
the scene face ordering. The builder assembles exact RHS columns one cabinet
at a time and computes `T = R * U` on CUDA. GMRES then applies `T * state(p)`.
The exterior LU, initial drive, final ROM reconstruction and field evaluator
remain the same. There is no additional rank truncation or compression tolerance;
floating-point operation ordering changes. The independent exact-quadrature
residual audit is retained and included in solve time.

For the 16-Battlehawk scene, `T` is 27,744 by 1,024 ComplexF32 values
(216.75 MiB). The full `R` is never resident. A cabinet-sized dense slab is
temporary; construction still evaluates every relevant entry and repeats
near/singular correction computation for each cabinet. Factors are rebuilt
per request. This is not hierarchical matrix compression and does not remove
the dense exterior LU. Mixed package and symmetry layouts are supported by
the basis mapping, with synthetic tests; real-scene qualification uses the
Battlehawk package.

Use the same replay command with `--modes matrix_free,projected`, then pass
`--candidate projected` to the comparison script. Run
`tests/deploy_rhs_projection_tests.jl` under `julia_cuda` for the complex basis,
symmetry, offsets and mixed-model projection checks. Diagnostics retain the
experimental `rhs_compression` container for comparison compatibility, recording
`mode: "projected"`, rank, stored bytes, column assembly time, projection time
and audited residual. Compression leaf size and tolerance do not affect this path.

CUDA slab construction explicitly releases the original and intermediate
reinterpret/reshape array aliases once the returned matrix owns its reference.
Otherwise successive large cabinet slabs can remain live until garbage collection
and cause severe Windows device-memory paging. Both selected-column and projection
gates exercise the returned matrix after this cleanup.

The 2026-10-09 Battlehawk follow-up measured a 36.60 s versus 22.65 s median
worker request over three warmed interleaved pairs at 32.474918365478516 Hz,
including a 5.84 s projection build and 2.51 s audit. Both sampled frequencies
(32.474918365478516 and 250 Hz) passed the scene/field accuracy gates. See the
Deploy checkout's `desktop/benchmarks/level3-battlehawk-projection-2026-10-09.md`
for exact scope, ranges, provenance and remaining limitations.

## Follow-up: shared assembly and retained factors

Additional explicit research modes are available:

| `rom_feedback_mode` | Behavior |
|---|---|
| `projected_cached` | Reuse request-local near/singular integral blocks across exterior and slab assembly. |
| `projected_fused` | Assemble exterior regular entries and each RHS slab together, then project the slab. |
| `projected_reuse` | Build with shared assembly on a miss; retain one LU and projected operator for drive updates. |

`projected` remains the original comparison path. All modes keep the independent
exact quadrature audit. Shared assembly accumulates the initial drive RHS from
the same exact slabs; final speaker reconstruction and field evaluation retain
their existing contracts. The full dense RHS operator is never resident.

Correction blocks are scoped to one request, keyed by device-cache identity and
signed wavenumber, and explicitly freed on exit. Their caches must not be reused
across arbitrary geometry/cache mutations. Shared assembly completes the exterior
matrix's corrections, identity and symmetry row weighting once after all regular
slabs have been visited.

The factor cache hashes loaded mesh/ROM numerical content, orbit mappings,
instance offsets, physical settings, quadrature, correction maps and active
phasor convention. Electrical input values are excluded; the next request's
current ROM inputs are used to reconstruct its feedback coordinates and output.
A hit still assembles the exact new drive RHS and independently audits the solve.
A miss, a non-reuse solve, or geometry-cache release frees the old entry. This
experimental mode requires `retain_geometry_cache: false` (the default), so
input geometry is revalidated. The cache is process-local, single-entry, and
retains the large LU allocation; it is not a persistent disk cache.

Diagnostics include `factor_cache_hit`, `factor_signature`, `retained_factor_bytes`,
`shared_assembly` and matrix-finalization time. On a hit, column assembly,
projection and matrix-finalization times are zero. The ordinary assembly timer
then measures the new drive RHS only; it is not a fresh exterior-matrix build.
On a shared-assembly miss, the assembly timer includes both exterior and projected
operator construction, so the separate post-LU operator-build timer is zero.
`rom_factor_key_s` records signature construction separately. Near-pair maps are
normalized into contiguous integer triples using the solver's effective default
quadrature order; this avoids recursively serializing millions of tiny vectors
without discarding numerical inputs. Non-numerical proximity diagnostics are
excluded from the signature.

The replay harness accepts `--drive-update` for deterministic, nonuniform complex
drive changes. It primes `projected_reuse` with the original drive before each
timed update, retaining priming requests/results/times separately. Compare with
`--reference projected --candidate projected_fused` (or `projected_reuse`).
See Deploy's `desktop/benchmarks/level3-battlehawk-shared-assembly-2026-10-09.md`
for qualification, measurements and reproducible commands.

The follow-up's warmed 16-cabinet study measured 22.346 s for the previous
projection, 20.403 s with reused corrections, and 17.249 s with shared assembly.
A separate three-pair compact-key study measured 17.063 s fresh versus 9.365 s
for retained-factor drive updates, with 18.360 s median priming cost recorded
separately. All numbers include the independent audit. The focused gates pass
104 CUDA assertions, and the standalone CPU reference gate passes. These are
scene-specific research results, not a default-selection policy.
