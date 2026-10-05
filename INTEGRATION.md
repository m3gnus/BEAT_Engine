# Slice 1 fork integration

Integrated from upstream main `e6b3037df04d3ac32d2bd4586d2e9f0312e2d91f` on `integration/fork-candidate`, using `git merge --no-ff` in the order below. PR titles and exact heads were verified through GitHub on 2026-10-05; all nine were open. #19 contains #17. No later-slice topic heads were imported.

| PR | Head SHA | Merge SHA | Conflicts and resolution |
| --- | --- | --- | --- |
| #17 | `3d3e0a48af711cf9ca3948d7b5d5d48c0798b27f` | `b6b473b048161cc34f0c5ed2d4f632b0afaf724d` | None; clean merge. |
| #19 | `85bac1329832b366ddf437007cf8a6b1e883f0c2` | `da2858dc0f9115df3ab3f1b0758b22c33231c6cd` | None; clean merge. |
| #20 | `4aa32e7f988a49018f850875895f42258265a07e` | `1599cded15a50ad5c3d75dd7d88f709d283555a3` | None; clean merge. |
| #18 | `9a73cc5e1622183843189db86e2c535270a6cf88` | `4806fb9b2a0d6d9c906f98321d6524e2ba0380a4` | None; clean merge. |
| #25 | `a8da36d8fec1f66b3835fd1a64cdd14a0ceb5d97` | `98d65538736accf205adebc6987c777f6c262cc1` | None; clean merge. |
| #26 | `a004905a5e3a349980118ca99a0d613d3ba81221` | `c1eb640df47b5ec533242c59a44c84775f55a703` | None; clean merge. |
| #30 | `9c9ee9d7027f28cfaece1db9f290f7dbe22c2825` | `14bc540921e2df2672b57ee11976db95f3b23ca5` | None; clean merge. |
| #31 | `9a2c4935baa7dc98a7367ad64b1b149517ef6528` | `b254d6f5f5cf28cecf60157960ccd33231fb200c` | Content conflict in `src/beat_engine/julia_local/tests/README.md`: independent appended sections. Retained the complete #25 ground contract/clearance/Metal-parity text, followed by the complete #31 CPU near-correction invariant and opt-in/refusal documentation. Source changes merged automatically, preserving the SIMD kernels and near-cache collection support. |
| #24 | `b6b388aeb5018cc47da22daedfa131c5b13b877e` | `6a8d20094f9577033eb1b24ba182d52950f78391` | None; clean merge. |

Automatic merges also retained both SIMD and axial-source test includes; #25 and #31 additions remain in the ordinary runner, and #18/#25 additions remain in the reference gate. LICENSE, fixture bytes/hashes, extraction provenance, and numerical baselines were not edited.

Qualification on macOS / Apple Silicon, Python 3.13.1 and Julia 1.12.7. Every Julia invocation and full suite ran through the compute broker, lane `suite`, priority 2, requester `Overseer and landing session`. No performance claim; concurrent broker compute work was active. Power at submission: AC Power, 21% battery; subsequently charging.

CI batch `261005-184827-suite-413c`: exit 0. `python3 -m ruff check src/beat_engine/*.py src/beat_engine/beat_contract tests` passed; `python3 -m ruff format --check pyproject.toml scripts src tests` passed (26 files). `git diff --check` passed.

| Suite | Outcome |
| --- | --- |
| `python3 -m pytest -ra` | 158 passed, no skips |
| CPU `tests/runtests.jl` | 3,698 passed, 2 unavailable-accelerator skips |
| CPU `tests/reference_tests.jl` | 2,010 passed, no skips |
| Metal `tests/metal_host_tests.jl` | 63 passed, functional device |
| Metal `tests/mumps_tests.jl` | 1,024 passed, no skips |
| CPU `scripts/benchmark_sweep.jl` + comparator | 8 frequencies, pressure/field anchors within `1e-3`; passed |
| `python3 -m build` | wheel and sdist built; wheel contains new sources/tests and CUDA bundle assets |

CPU and Metal instantiation passed using `python3 -m beat_engine instantiate --backend <cpu|metal> --julia "$J"`. `$J` was `/Users/magnus/Code/hornlab-workspace/scratchpads/260930-beat-speedups/r3/julia-1.12.7/julia-1.12.7/bin/julia`. The Julia test commands use `$J --threads=2 --startup-file=no --project=src/beat_engine/<julia_local|julia_metal> src/beat_engine/julia_local/<test path above>`.

The sweep command used the CPU project and `--steps 8 --min-freq 100 --max-freq 20000 --eval-points 74 --warmups 0 --repetitions 1 --json /private/tmp/beat-slice1-gate-logs/ci_sweep.json`. Comparison: `python3 scripts/compare_solve_speed.py --baseline src/beat_engine/julia_local/results/baseline_sweep_cpu.json --results /private/tmp/beat-slice1-gate-logs/ci_sweep.json --accuracy-tolerance 1e-3`.

Focused/hardware batch `261005-184938-suite-4e06`: exit 1 solely for the default Metal hardware suite (486 passing assertions, 3 failures, then early termination). Axial validators passed on CPU direct (4 reported PASS groups), Metal direct (3 PASS groups), and Metal operator-matrices (1 PASS group). CPU and Metal direct gates used `--legacy-solver /Users/magnus/Code/hornlab-workspace/BEAT_Engine/src/beat_engine/julia_local/coupled_solver.jl` at unchanged upstream e6b3037, proving bit-identical default v1 outputs and raw-v2 refusal by the old worker. These scripts contain additional assertions but do not report an assertion count. Commands: `python3 src/beat_engine/julia_local/scripts/validate_compiled_axial_source.py --julia "$J" --backend <cpu|metal>`, with `--assembly operator_matrices` for that Metal variant; other variants use the default `direct_system`.

`BLAB_VALIDATE_GROUND=1 $J --threads=2 --startup-file=no --project=src/beat_engine/julia_metal src/beat_engine/julia_local/scripts/validate_metal_symmetry.jl` passed 8 combinations: x/xy mirrors and lifted/resting-edge ground cases, each with host/native singular correction.

Failure classification: **inherited test assumptions**, not a slice-1 regression. `BEAT_BACKEND=metal python3 scripts/qualify_accelerator.py` fails unchanged `coupled_solver_tests.jl:1282,1287,1288`: Metal defaults to `mumps_seq`, but the tests require `cpu_umfpack`, positive Schur block size and a Schur thread count bounded by Julia threads. Observed diagnostics were `mumps_seq`, 0, and 4 with 2 Julia threads; the 41 numerical/other assertions in that testset passed. Broker job `261005-190319-suite-37bd` reproduced the exact three failures on untouched upstream e6b3037 (282 passes, 3 failures, early termination), using `$J --threads=2 --startup-file=no --project=/Users/magnus/Code/hornlab-workspace/BEAT_Engine/src/beat_engine/julia_metal /Users/magnus/Code/hornlab-workspace/BEAT_Engine/src/beat_engine/julia_local/tests/runtests.jl` with `BLAB_RUN_COUPLED_METAL=1`. No test expectations or engine defaults were changed.

Candidate Metal rerun with explicit `BLAB_COUPLED_FEM_SOLVER=umfpack BEAT_BACKEND=metal python3 scripts/qualify_accelerator.py`: exit 0; 4,107 passing assertions, 1 unavailable-CUDA skip, no failures. Diagnostic broker batch exit 1 is expected because it includes the upstream failure reproduction. This diagnoses the test's intended configuration; it does not turn the failed default hardware gate green.

Skipped: #21/#22/#23/#32/#33 by owner scope; #27–#29 outside the requested queue. CUDA/ROCm hardware qualification and clean CUDA device install cannot run on this Mac. Windows/Linux CI matrix execution needs those hosts. Compiled-worker SIMD reachability and startup/performance qualification remain slice 2; this slice makes no claims for them. The ordinary CPU suite's Metal/CUDA skips do not qualify either accelerator.

The candidate was prepared in `/private/tmp/beat-slice1-candidate` because the original checkout's `.git` is read-only in this session. No push or original-branch update has been performed.

Raw commands, per-suite logs and summaries: `/private/tmp/beat-slice1-{gate,focused,diagnostic}-logs/`; runner scripts: `/private/tmp/beat-slice1-run-{gates,focused,diagnostics}.py`.


# Slice 2 fork integration

Prepared from slice 1 `9b6941b` in `/private/tmp/beat-slice2-candidate` because the
original checkout's `.git` remains read-only. No original ref update or push.
GitHub metadata refreshed 2026-10-05: #22/#23 open, #21 open/draft; exact heads
match the supplied refs. All three have no posted reviews or review threads. Their exact-head CI runs
are successful: #22 [36927585598](https://github.com/JWSound/BEAT_Engine/actions/runs/36927585598),
#23 [36835396343](https://github.com/JWSound/BEAT_Engine/actions/runs/36835396343),
#21 [36829097063](https://github.com/JWSound/BEAT_Engine/actions/runs/36829097063).
Each includes successful macOS/Windows/Ubuntu CPU, Metal host/MUMPS and sweep
benchmark jobs. These upstream-head results do not qualify the combined tree
or prove Windows compiled-worker AVX2 dispatch. Combined commit-status lists
are empty; workflow-job evidence was checked separately.

| PR | Head SHA | Merge SHA | Conflicts and resolution |
| --- | --- | --- | --- |
| #22 | `b87f32e37e1e314a31a661822445ebc5f5e88e29` | `83237f1cd8375a76489604e452d026bdcc41a785` | `coupled_solver.jl`: the four-commit stack extracts the driver into `BeatEngineCompiledDriver.jl`. Retained the new loader and moved the complete slice-1 axial-source and ground-clearance diff (134-line patch against e6b3037) into the extracted driver. No source hunks dropped. |
| #23 | `3157fe0710e6b43677ab4d9c3a0485711fd4b58b` | `3e6f81b86731b9384de7211fe66bf8f1eca0e182` | `coupled_solver.jl`: applied the exact upstream driver hunks to `BeatEngineCompiledDriver.jl`, preserving #22's loader and slice-1 changes. Other source changes and runner includes merged automatically. |
| #21 | `4eab387944c55ff00e11d4495c128f64cda182da` | `fbab5c386aaae37c9b24d0854899144630d05f89` | None; clean merge, preserving SIMD and multiple-image near-correction support. |

Bounded reachability repairs use existing numerical kernels, not new algorithms:
compiled exterior CPU direct requests reach the fused assembler, regular and
Duffy SIMD kernels, transposed scatter and SIMD field path. CPU keeps its LU
solve policy; coupled CPU field calls keep their scalar default. Explicit
operator requests and scalar environment controls remain available. CPU
precompile provenance is reset at process initialization as on Metal.

Readiness and result diagnostics identify the requested/loaded compiled bundle,
driver mode, fallback reason, actual assembly/kernels and solve method. Failed
bundle loading warns instead of silently including the source driver. Explicit
bundle disable and unsupported compiled-bundle backends have distinct reasons.

Packed Metal integration requires refreshed compile-only device/host/runtime
inventories: old pair-launch API, removed singular kernel and generated driver
closure numbers were stale. Host launches match #21; runtime closures are
resolved by captured fields; the release callback is named; encoding/batched
launch specializations follow the current device inventory. The GPU-cache
workload now explicitly guards Julia <1.11. The existing generator regenerated
only signatures, never a numerical baseline. An archived A68 host/device patch
bootstrapped the same known #21 repair; the final inventory was freshly generated
on this candidate. Metal's cache-owner caveat for different regular caches
sharing one singular cache remains outside this reachability repair.

Tests now load the extracted driver for axial source checks, expect the requested
CPU direct route, recognize negotiated v1/v2 contracts, and run the CPU host
workload through the Metal compiled bundle when in the Metal project. Physical
and numerical tolerances were not widened. New real-worker qualification script:
`src/beat_engine/julia_local/scripts/validate_compiled_reachability.py`.


## Slice 2 compiled reachability evidence

Apple M1 Max / aarch64, Julia 1.12.7, two Julia threads, Metal 1.11.1. All Julia
processes, including Python-launched workers and instantiation/precompilation,
ran through the compute broker's suite lane, priority 2, requester
`Overseer and landing session`. The exact Julia executable remains the r3
`julia-1.12.7/julia-1.12.7/bin/julia` used by slice 1. AC power recorded; battery
was about 74% at checkpoint submission and 59% during the long reference.

| Actual request | Loaded bundle | Effective numerical route |
| --- | --- | --- |
| CPU exterior, CPU project | `BeatEngineCompiledCpuBundle` | `direct_system`; regular/singular/field `simd`; transposed scatter; CPU dense LU, one factorization for all drive columns |
| Metal exterior, Metal project | `BeatEngineCompiledMetalBundle` | `direct_system`; `packed_fused` regular, `packed_grouped_duffy` singular, `packed_float32` field; calibrated CPU dense LU/GMRES over Metal-assembled matrices |

Real-worker ready records and per-result diagnostics agree on loaded bundle,
project and thread identity and have no fallback reason. Explicit bundle disable
returns `source` / `disabled_by_environment` and agrees numerically. Small off/xy
two-port requests use LU on both backends. CPU Float64 fused SIMD versus scalar
operators has maximum relative L2 `2.34e-14`; Float32 CPU maximum `6.28e-5`;
Metal direct/operator maximum `9.32e-7`. Bundled/source maximum errors are
`2.75e-7` CPU and `3.15e-7` Metal, below the `1e-5` check. Production direct/operator
checks use `1e-4`, Float64 checks `1e-10`; existing suite tolerances are untouched.

The six-frequency real compiled Metal `sample_detailed` request reports a
20 Hz GMRES `stagnated` fallback to LU (56 iterations, true residual `1.18e-5`).
At 80/320/1280/5120/20000 Hz it reports GMRES convergence with warm starts,
35/37/37/39/41 iterations and true residuals below `1e-5`, with no fallback.
The compiled CPU counterpart uses LU at all six frequencies. The library's
GMRES history/backoff tests remain in the full CPU runner; router selection is
measured/calibrated, not replaced with a machine-specific constant.

A brokered native-code probe loaded the compiled CPU bundle and inspected the
actual SIMD regular, field and Duffy methods. LLVM vector mentions / ARM NEON
operand mentions were respectively 376/168, 214/156, 350/226. The `.llvm`, `.asm`,
probe script and machine summary are retained in the evidence. This is Mac
vector-code evidence, not Windows AVX2 evidence or a performance claim.

Fresh compiled Metal workers were instrumented using Metal's own atomic
`compilations` counter, around the real compiled entry and its first negotiated
request. Both `sample.msh` and `sample_detailed.msh` (q4/s4, 1000/2000 Hz)
completed with **0 first-request device compilations**, with the Metal bundle
loaded. This proves the tested device-cache reachability; it does not assert
zero host JIT work or universal signature coverage for all options/geometries.

Regeneration job `261005-195534-suite-59d1` compiled 96 host, 108 runtime and
42 device signatures with zero failures; generator coverage had 62 passing
assertions. The later ordinary kernel-coverage gate adds the inventory-count
assertion and has 63 passes. The old closure-number and removed-singular-symbol
failures are retained as repaired precompile integration failures.

## Slice 2 packed-kernel comparisons

Broker job `261005-200227-suite-0c2c` passed. Compared pre-#21 merge `3e6f81b`
and this candidate, with the same `sample_detailed.msh` bytes, six frequencies
20/80/320/1280/5120/20000 Hz, two threads, q4/s4 and backend stack. Three
interleaved before/after runs per entry, 12 complete sweeps. Both arms explicitly
use `BLAB_BEAT_ENGINE_BUNDLE=0`: these are matched source/compiled-entry
include-mode comparisons, not bundled-startup measurements. Small warm-up
requests precede each measured sweep. Completed records from the two earlier
interrupted harness batches were retained in their original before/after order.

The archived compiled request's obsolete v1 `motion_profile=uniform` property
was removed from the private capture, retaining the default `uniform_normal`
and identical weights. Mesh paths alone were retargeted. Repository fixtures
were not edited. Original capture and its provenance remain available.

| Entry / arm | Sweep wall median [range], s | Assembly median [range], s | Solve median [range], s | Field median [range], s | Peak footprint range, MiB |
| --- | --- | --- | --- | --- | --- |
| Source before | 2.954 [2.914, 2.988] | 1.184 [1.184, 1.187] | 2.204 [2.173, 2.208] | 0.0312 [0.0298, 0.0338] | 2904–2993 |
| Source after | 2.933 [2.916, 2.964] | 1.025 [1.024, 1.043] | 2.222 [2.211, 2.252] | 0.0279 [0.0270, 0.0281] | 2922–2933 |
| Compiled entry before | 3.782 [3.780, 3.785] | 1.181 [1.178, 1.185] | 3.092 [3.076, 3.096] | 0.0804 [0.0784, 0.0839] | 3193–3278 |
| Compiled entry after | 3.733 [3.716, 3.735] | 1.005 [0.998, 1.008] | 3.092 [3.083, 3.137] | 0.0424 [0.0421, 0.0430] | 3251–3298 |

Assembly can overlap solve, so section totals are not summed into wall time.
No general speedup/startup claim is made. Source wall ranges overlap; CPU SIMD
and compiled startup were not performance-benchmarked against shipped HBB.

Worst after/before complex L2 is `3.60e-6`, masked SPL `0.000467 dB`, masked phase
`0.00253°`; repeat numerical noise on each arm was zero in these runs. A scalar
Float64 CPU operator/LU reference for the same compiled request gives pressure
L2 `5.97e-5`, masked SPL `0.000792 dB`, phase `0.00963°`; radiation-impedance L2
`1.43e-6`, SPL `0.000316 dB`. These satisfy the plan's production bounds for
this request. The 30 dB mask is reference-defined; null counts and absolute null
errors are recorded separately in `comparison-summary.json`. Six log-spaced
frequencies cannot qualify dense resonance locations or the full retirement
corpus; those remain later migration gates.


## Slice 2 qualification and failure classification

Reachability repair commit: `239c7bd9c7abd6a60d882358ae200c63aa1b8c96`.
The final CPU/Metal real-worker source fingerprints were checked byte-for-byte
against the candidate after that commit. Python public runtime code is unchanged;
the new qualification script uses only the standard library and existing public
worker client.

| Suite / probe | Final outcome |
| --- | --- |
| Python pytest | 158 passed, no skips |
| Full CPU `tests/runtests.jl` | 3,846 passed, 2 unavailable-accelerator skips; no failures |
| Standalone CPU `tests/reference_tests.jl` | 2,010 passed, no skips |
| Metal `tests/metal_host_tests.jl` | 306 passed, functional device |
| Metal `tests/mumps_tests.jl` | 1,024 passed, no skips |
| Standalone `tests/compiled_metal_worker_tests.jl` | 214 passed; also included in host gate, not an additional distinct assertion total |
| Ordinary `tests/metal_kernel_coverage_tests.jl` | 63 passed; 42 observed / 42 inventoried signatures |
| `scripts/validate_metal_packed_exterior.jl` | 39 passed (5 shared-geometry/independent-scratch assertions, 34 multiple-drive assertions); max relative L2 `3.25e-7`, max SPL `2.61e-6 dB` |
| `scripts/validate_metal_fused.jl` | 4 arms passed: off/x/xy/ground; script does not report an assertion count |
| `scripts/validate_metal_singular.jl` | Passed, four frequency comparisons; no reported assertion count |
| `scripts/validate_metal_pipeline.jl` | 4 frequencies at each lookahead depth 1–4 matched sequential output bit-for-bit; no reported assertion count |
| Real compiled CPU reachability | 11 requests passed, including explicit fallback and Float64 scalar reference; final tightened bounds rerun passed |
| Real compiled Metal reachability | 7 requests passed, including explicit fallback and calibrated six-frequency sweep |
| First-request Metal device compilation counter | 2 fresh-worker mesh cases passed; both compilation deltas 0 |
| Mac native-code SIMD probe | 3 actual kernel methods produced LLVM vectors and NEON instructions |
| Matched packed-kernel sweeps | 12 complete Metal sweeps + 1 scalar Float64 CPU reference; shape/finiteness and the recorded accuracy bounds passed |
| Default Metal hardware checkpoint | 515 passed, 4 failed, early termination; failed gate retained |
| Focused coupled defaults, before/after #21 | Each of 2 runs per arm: 129 passed, 3 failed; all three inherited diagnostics |
| Focused coupled explicit UMFPACK | 132 passed, no failures |
| Full-prelude before/after #21 | Each: 515 passed, 4 failed, identical fourth flux error |
| Full Metal hardware explicit UMFPACK | 518 passed, 1 failed at the flux assertion, early termination; host/coverage tail was not reached by this command |
| Python build / wheel assets | Wheel and sdist built; driver, compiled CPU/Metal bundles, packed sources, new validator and existing CUDA bundle assets present |

CPU ordinary runner includes 221 adaptive dense-solve assertions and 45 pipeline
assertions; these cover true residuals, history, warm starts and fallback backoff.
The full CPU follow-up `261005-201740-suite-002e` passes. The first CPU checkpoint
had 28 stale axial dispatch expectations (3,438 other passing assertions), fixed
by expecting the requested direct path; numerical assertions had passed.

The default Metal hardware checkpoint's lines 1282/1287/1288 are the exact three
slice-1 inherited UMFPACK-diagnostic assumptions: observed `mumps_seq`, Schur
block size 0 and 4 Schur threads with 2 Julia threads. No engine defaults or
expectations were changed to hide these failures.

An additional numerical assertion was newly observed at
`coupled_solver_tests.jl:1355`: FP32 voltage interface-flux relative error
`0.0022033167` exceeds the existing `0.002` bound. The fixture is documented as
ill-conditioned (`cond(A,1) ~= 3.8e9`). Isolated coupled runs before/after #21
both passed that numerical assertion twice; each failed only the three inherited
diagnostics. A full-prelude comparison then reproduced the identical fourth
error with pre-#21 numerical sources. Both arms deliberately use the same final
compiled CPU-workload bootstrap, then their own ordinary-runner prelude and
coupled numerical sources, through the candidate Metal environment. This isolates
#21's numerical changes; it is not a claim that untouched slice 1 had a fourth
failure. The finding is test-sequence-sensitive and remains an observed failed
qualification assertion, not a widened tolerance or a green default gate.

The first temporary focused-UMFPACK harness was invalid (88 passes followed by
undefined `metal_available`): it omitted the ordinary runner's availability
helper. The corrected brokered harness imports Metal and checks `functional()`;
its explicit-UMFPACK result is the 132-pass row above.


Full-prelude job `261005-203940-suite-2dcd` confirms 515 passes / 4 failures on
both arms with the same flux error. Its full explicit-UMFPACK hardware rerun
removes the three diagnostics but still fails the flux bound (518 passes,
1 failure, early termination). Unlike slice 1's explicit-UMFPACK rerun, this
full candidate gate is **not green**. The separate host/coverage gates pass as
reported above. The extra comparison needs later coupled numerical/test-order
investigation; no coupled algorithm, tolerance, fixture or baseline was changed
in this bounded reachability slice.

Cache ownership review: packed pair/source data are read-only and lazily
published under locks; mutable pair blocks, A/b and field partials are per-call.
Shared-geometry independent-scratch and multiple-drive tests pass, and exterior
pipeline depths 1–4 match sequential output. The pre-existing singular-cache
`gather_tables` owner changes with the regular cache and retires old device
maps; sharing one singular cache between different live regular caches is not
qualified and can retire in-flight maps. This was not broadened into a cache
lifecycle redesign. Supported same-geometry scratch/pipeline coverage does not
prove that unsupported cross-cache concurrent case safe.

Required Python ruff checks passed, including the new validator; format check
passed for 27 files. `git diff --check` passed. LICENSE, numerical fixtures and
hashes, numerical baselines, `docs/EXTRACTION.md` and extraction commit map are
byte-identical to slice 1. No numerical baseline was regenerated. Wheel asset
inspection found all requested bundles/inventories/driver/packed files and the
existing portable CUDA bundle assets (193 packaged files).

Broker batches and retained failure records:
- `261005-194823-suite-83f6`: dependency instantiation/precompile; exposed stale
  generated driver closure inventory. `261005-195229-suite-0f40` inspected real
  closure types; `261005-195343-suite-25cd` exposed removed singular symbol.
- `261005-195534-suite-59d1`: repaired precompile and inventory generation passed.
- `261005-195736-suite-0e62`: 26-minute checkpoint held for broker triage and
  cancelled. Scope was reduced by removing a redundant full UMFPACK hardware
  rerun and substituting a focused diagnostic; no self-approval or job sharding.
- `261005-200121-suite-2598`: 17.54-minute narrowed checkpoint. Failed first CPU
  dispatch assertions and default Metal gate; other rows passed. Temporary
  focused harness error separately classified above.
- `261005-195909-suite-e23a` / `261005-200019-suite-a6fc`: comparison harness
  shadowed Python `round`, then old capture's obsolete v1 property was refused.
  Completed source records retained; incomplete work cancelled, repaired runner
  resumed. These are harness/capture failures, not ignored numerical failures.
- `261005-200227-suite-0c2c`: all matched sweeps/reference passed (12.1 minutes).
- `261005-200440-suite-1db1`: real fresh-worker device-cache probes passed.
- `261005-200622-suite-4862`: superseded before execution by a combined final CPU
  follow-up, after tightening the new validator's accuracy bounds.
- `261005-201740-suite-002e`: corrected full CPU runner + final CPU validator passed.
- `261005-202232-suite-915b`: four matched focused defaults each reproduced only
  the inherited three diagnostics; focused explicit UMFPACK passed. Aggregate
  failure is expected for diagnostic reproduction.
- `261005-203940-suite-2dcd`: full-prelude attribution and full explicit-UMFPACK
  hardware rerun; retained failed outcomes described above. This extra rerun
  addressed the newly observed fourth failure, not redundant broad testing.

Skipped / outstanding: #32/#33/#34 deliberately not merged; #27–#29 remain outside
this slice. Windows AVX2 compiled-worker evidence and combined-candidate
Windows/Linux suites require those hosts; exact upstream-head CI is retained
but does not substitute. CUDA/ROCm hardware and clean CUDA device install are
unavailable on this Mac. Julia <1.11 execution was not available: the guard is
implemented but compatibility beyond the pinned 1.12.7 stack is not claimed.
No full dense resonance/HBB-retirement corpus, installed exact-SHA consumer
qualification, shipped-HBB cold/warm startup comparison, or general CPU SIMD /
packed-kernel speedup claim. No two independent integration reviews were carried
out in this session. No fork-main/WG pin move, upstream PR close, remote push,
branch rewrite, release or repository-settings change.

Raw evidence is retained under `/private/tmp/beat-slice2-{gate,final,comparison,
cache,cache-run,cpu-rerun,coupled-diagnostic,coupled-prefix,native}-logs/`; broker
logs retain initial failed commands. Runner/probe scripts are
`/private/tmp/beat-slice2-run-*.py`, `beat-slice2-{native,metal-cache-worker}.jl`,
and `beat-slice2-{cache-probe,analyze-comparisons}.py`. PR metadata is
`/private/tmp/beat-slice2-pr-metadata.json`. The verified history artifact is
`/private/tmp/beat-slice2.bundle`; an evidence archive and SHA-256 sidecars are
provided alongside it. Original checkout remains clean at `9b6941b`.
