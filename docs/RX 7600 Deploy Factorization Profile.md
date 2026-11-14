# ROCm Deploy factorization substages

Six fresh workers completed: three uninstrumented controls and three instrumented
workers, interleaved baseline/instrumented, instrumented/baseline, baseline/instrumented.
Each performed a warm-up and three measured two-frequency sweeps on the same
four-cabinet Coupled audience workload. All outputs, including warm-ups, passed
independent CPU-reference comparisons; worst relative L2 was 1.181411210148693e-5
against the unchanged 5e-4 limit. Every instrumented factorization used one attempt.

Baseline source: a1d3abcb52596179ace730156438141ea18ea946.
Instrumented source: experiment/rocm-factor-profile at 6803d84 (full hash in plan.json).
The instrumentation synchronizes each substage; it is a diagnostic branch.

Third-repeat values are seconds, median [minimum, maximum] over three fresh workers:

| Stage | Time |
|---|---:|
| Control wall time | 7.299 [7.294, 7.385] |
| Instrumented wall time | 7.238 [7.193, 7.317] |
| Identity-cache setup | 0.774 [0.725, 0.805] |
| Burton-Miller matrix construction | 0.0316 [0.0312, 0.0318] |
| Float64 conversion | 0.0474 [0.0469, 0.0478] |
| Protective matrix copy | 0.0510 [0.0509, 0.0516] |
| LU and its status synchronization | 2.537 [2.528, 2.549] |
| Entire factorization wrapper (includes copy and LU) | 2.599 [2.589, 2.611] |

All three matched-repeat wall-time ranges overlap. Instrumentation does not
demonstrate a whole-workload timing change, but this is not proof of zero overhead.
The audit retains all repeat ranges. Stage medians should not be summed as though
they came from one worker; the wrapper overlaps its copy and LU columns by definition.

Even eliminating the protective copy entirely could save only about 0.7% of
the measured control sweep (0.051 / 7.299), before replacement or recovery costs.
Keep the copy: it preserves original coefficients for corrupt-status retries.

Identity-cache setup costs about 10.6% of the control sweep. Inspection shows
full host conversion to Complex{T} followed by device upload for both identity
blocks. Next investigate real-valued transfer followed by device conversion,
or reuse where the existing cache lifetime permits it. This 10.6% is an upper
bound from eliminating the entire measured setup, not an achievable speedup
prediction. Any candidate needs unchanged numerical results, release/lifetime
checks and interleaved whole-workload measurements before adoption.

Dense LU remains the larger cost. Precision and retry policy were preserved.
Raw plans, results and audit are retained locally under
`runs/rx7600-deploy/factor-substages-20261105-124300`; dedicated VRAM was
not measured, and these results do not qualify an installed runtime.
