# RX 7600 regular-pair bounds experiment

This comparison isolates bounds-check elision in the regular SLP/adjoint DLP
and hypersingular quadrature loops. Precision, arithmetic, launch geometry,
singular corrections and solve algorithms are unchanged.

Baseline: `4dadbd8772e1226ee1ad138fed82676ce8992b44`. Candidate: `4542b562b7b45ad70476ed67ed0e08ea19c68339`.
Deploy application: `330863a6411cd68edd2f692897be612aef0005b9`.

The Windows RX 7600 run uses three interleaved fresh workers per variant,
scene and fidelity. Each worker performs a warm-up and three measured sweeps;
each sweep submits two independent 41×41 audience requests at 79.313 and
162.025 Hz. Julia and OpenBLAS each use two threads. RGP is not active.
All saved warm-up and measured outputs passed independent comparison against
pristine CPU Float32 source-worker references at the unchanged 5e-4 limit.

## Matched repeats

Wall times are seconds, median [minimum, maximum] across three fresh workers.
An overlapping range does not establish a whole-workload speedup.

| Scene / fidelity | Repeat | Baseline | Candidate | Median reduction | Range comparison |
|---|---:|---:|---:|---:|---|
| mixed-boundary | 1 | 2.088 [2.034, 2.097] | 1.944 [1.934, 1.960] | 6.9% | candidate faster, separate ranges |
| mixed-boundary | 2 | 2.089 [2.083, 2.165] | 1.946 [1.933, 1.948] | 6.9% | candidate faster, separate ranges |
| mixed-boundary | 3 | 2.070 [2.048, 2.105] | 1.910 [1.907, 1.938] | 7.7% | candidate faster, separate ranges |
| mixed-coupled | 1 | 2.189 [2.167, 2.251] | 2.040 [2.032, 2.048] | 6.8% | candidate faster, separate ranges |
| mixed-coupled | 2 | 2.130 [2.104, 2.189] | 2.050 [2.006, 2.097] | 3.8% | candidate faster, separate ranges |
| mixed-coupled | 3 | 2.185 [2.166, 2.230] | 2.045 [2.033, 2.051] | 6.4% | candidate faster, separate ranges |
| array4-boundary | 1 | 7.976 [7.921, 8.043] | 7.161 [7.158, 7.179] | 10.2% | candidate faster, separate ranges |
| array4-boundary | 2 | 7.944 [7.944, 7.987] | 7.114 [7.104, 7.140] | 10.4% | candidate faster, separate ranges |
| array4-boundary | 3 | 7.975 [7.873, 7.984] | 7.108 [7.055, 7.119] | 10.9% | candidate faster, separate ranges |
| array4-coupled | 1 | 8.179 [8.159, 8.180] | 7.368 [7.345, 7.389] | 9.9% | candidate faster, separate ranges |
| array4-coupled | 2 | 8.198 [8.142, 8.207] | 7.392 [7.297, 7.401] | 9.8% | candidate faster, separate ranges |
| array4-coupled | 3 | 8.163 [8.139, 8.196] | 7.336 [7.311, 7.407] | 10.1% | candidate faster, separate ranges |

## Third-repeat stages, host memory and accuracy

Stage values are median seconds summed over both frequencies. Factorization
includes system construction, conversion and a protective copy; it is nested
within solve. Host peaks cover all measured sweeps and exclude dedicated VRAM.

| Scene / fidelity / variant | Assembly | Solve | Field | Factorization | GMRES | Peak host MiB | Worst relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed-boundary-baseline | 0.947 | 0.300 | 0.024 | — | — | 1570 | 2.268e-06 |
| mixed-boundary-candidate | 0.802 | 0.328 | 0.024 | — | — | 1551 | 2.268e-06 |
| mixed-coupled-baseline | 0.955 | 0.369 | 0.022 | 0.288 | 0.071 | 1583 | 1.192e-05 |
| mixed-coupled-candidate | 0.811 | 0.368 | 0.022 | 0.288 | 0.071 | 1594 | 1.192e-05 |
| array4-boundary-baseline | 3.223 | 3.550 | 0.052 | — | — | 2364 | 8.428e-06 |
| array4-boundary-candidate | 2.405 | 3.543 | 0.053 | — | — | 2361 | 8.428e-06 |
| array4-coupled-baseline | 3.235 | 3.675 | 0.053 | 3.453 | 0.180 | 2395 | 1.181e-05 |
| array4-coupled-candidate | 2.398 | 3.702 | 0.054 | 3.483 | 0.180 | 2408 | 1.181e-05 |

Local evidence: `pair-bounds-comparison-20261030-155013`. The JSON audit retains all matched-repeat
stage ranges, end-of-sweep host memory and accuracy metrics. Generated artifacts
remain untracked. These results concern the measured Deploy workloads only;
the shared-kernel change also requires the separate core workload comparison.

Direct baseline/candidate comparison across all 48 matching sweep pairs
(including warm-ups) found maximum output relative L2
`0.0`.
