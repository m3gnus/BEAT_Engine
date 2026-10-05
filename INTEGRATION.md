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
