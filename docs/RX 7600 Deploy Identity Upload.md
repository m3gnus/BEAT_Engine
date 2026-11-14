# RX 7600 Deploy real identity upload experiment

This comparison isolates uploading real Deploy identity matrices before
conversion to complex values on the GPU. Resident precision, assembly
and solve algorithms are unchanged; the protective LU copy is retained.

The candidate passed 32 exact upload-conversion assertions (both precisions,
signed zero, rectangular blocks, fallback inputs and input preservation), all
44 Deploy factor/feedback assertions, and the full standalone CPU reference
suite. No fixtures or tolerances were changed. The fast path accepts real dense
matrices already in the requested precision; other input types retain the
existing host conversion. Temporary device storage is released after conversion
completes, and partial cache construction has an explicit cleanup path.

[Factorization substage profiling](RX%207600%20Deploy%20Factorization%20Profile.md)
motivated this experiment: identity setup accounted for approximately 10.6% of
the four-cabinet Coupled sweep. Eliminating the protective copy was bounded at
only 0.7%, so its retry-safety behavior was retained.

Host: Windows 11, Ryzen 7 5700X (8 cores/16 threads), RX 7600 (gfx1102, 8 GiB),
ROCm SDK 7.2, Julia 1.12.6 and AMDGPU.jl 2.7.2. Peak host memory is sampled at
20 ms intervals. The conversion uses a temporary real device array; dedicated
VRAM peaks were not measured, so the host-memory reduction is not a VRAM claim.

Baseline: `a1d3abcb52596179ace730156438141ea18ea946`. Candidate: `eb1ce51e8f614a7864f068e2f95728b3859271a3`.
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
| mixed-boundary | 1 | 1.872 [1.862, 1.882] | 1.800 [1.789, 1.803] | 3.9% | candidate faster, separate ranges |
| mixed-boundary | 2 | 1.864 [1.853, 1.916] | 1.786 [1.785, 1.811] | 4.2% | candidate faster, separate ranges |
| mixed-boundary | 3 | 1.851 [1.844, 1.863] | 1.785 [1.775, 1.790] | 3.6% | candidate faster, separate ranges |
| mixed-coupled | 1 | 1.987 [1.984, 2.005] | 1.921 [1.912, 1.932] | 3.3% | candidate faster, separate ranges |
| mixed-coupled | 2 | 1.945 [1.940, 1.953] | 1.914 [1.914, 1.924] | 1.6% | candidate faster, separate ranges |
| mixed-coupled | 3 | 1.980 [1.975, 2.047] | 1.984 [1.902, 2.001] | -0.2% | overlap |
| array4-boundary | 1 | 6.939 [6.884, 6.959] | 6.325 [6.308, 6.339] | 8.9% | candidate faster, separate ranges |
| array4-boundary | 2 | 6.939 [6.909, 6.951] | 6.320 [6.299, 6.320] | 8.9% | candidate faster, separate ranges |
| array4-boundary | 3 | 6.938 [6.917, 6.959] | 6.322 [6.294, 6.329] | 8.9% | candidate faster, separate ranges |
| array4-coupled | 1 | 7.180 [7.170, 7.186] | 6.547 [6.507, 6.552] | 8.8% | candidate faster, separate ranges |
| array4-coupled | 2 | 7.148 [7.126, 7.202] | 6.549 [6.528, 6.555] | 8.4% | candidate faster, separate ranges |
| array4-coupled | 3 | 7.185 [7.182, 7.219] | 6.582 [6.551, 6.594] | 8.4% | candidate faster, separate ranges |

## Third-repeat stages, host memory and accuracy

Stage values are median seconds summed over both frequencies. Factorization
includes system construction, conversion and a protective copy; it is nested
within solve. Host peaks cover all measured sweeps and exclude dedicated VRAM.

| Scene / fidelity / variant | Assembly | Solve | Field | Factorization | GMRES | Peak host MiB | Worst relative L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed-boundary-baseline | 0.797 | 0.361 | 0.024 | — | — | 1539 | 2.268e-06 |
| mixed-boundary-candidate | 0.800 | 0.239 | 0.024 | — | — | 1445 | 2.268e-06 |
| mixed-coupled-baseline | 0.803 | 0.366 | 0.022 | 0.290 | 0.067 | 1586 | 1.192e-05 |
| mixed-coupled-candidate | 0.805 | 0.320 | 0.022 | 0.239 | 0.071 | 1481 | 1.192e-05 |
| array4-boundary-baseline | 2.390 | 3.463 | 0.051 | — | — | 2353 | 8.428e-06 |
| array4-boundary-candidate | 2.382 | 2.893 | 0.052 | — | — | 1738 | 8.428e-06 |
| array4-coupled-baseline | 2.383 | 3.638 | 0.053 | 3.420 | 0.178 | 2385 | 1.181e-05 |
| array4-coupled-candidate | 2.383 | 3.082 | 0.052 | 2.862 | 0.179 | 1786 | 1.181e-05 |

Local evidence: `identity-upload-comparison-20261105-130410`. The JSON audit retains all matched-repeat
stage ranges, end-of-sweep host memory and accuracy metrics. Generated artifacts
remain untracked. These results concern the measured Deploy workloads only;
the changed helper is scoped to Deploy; core solver paths are unchanged.

Direct baseline/candidate comparison across all 48 matching sweep pairs
(including warm-ups) found maximum output relative L2
`0.0`.
