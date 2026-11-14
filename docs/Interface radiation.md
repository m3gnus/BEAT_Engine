# Interface radiation decomposition

The optional coupled output `interface_radiated_pressure` evaluates radiation
from each interface with the original operating flux held fixed. It requires
`options.points_m`, a nonempty list of finite exterior observation coordinates.
The complex result axes are `(excitation, radiation_source, observation)` in Pa.
The worker advertises this quantity in `optional_output_quantities`.

For the exterior equation `A p = C q`, the existing coupled interface block is
`B = -C F`, where `F` maps interface nodal flux to exterior face derivatives.
For interface index set `I`, replay solves `A p_I = -B[:, I] flux[I]` and uses
`q_I = F[:, I] flux[I]` in the full exterior field representation. This retains
nonuniform flux, diffraction, symmetry, and complex phase. Restricting the
original pressure trace to an interface would not perform this decomposition.

The original pressure/flux solution and all electromechanical quantities remain
unchanged. When requested, CUDA coupled solves retain the exterior LU factors
and interface block on the GPU. This is the production path whenever the
coupled linear backend is CUDA; no opt-in flag is required. Source-partitioned interface flux vectors are
uploaded together, and the replay right-hand sides and pressure solves run on
the device. Only the compact pressure traces return to the host field-output
contract. Other coupled paths, including diagnostic CPU linear solves, retain
the host factorization. All interfaces and
excitations share the factorization. Replay owns copies independent of the
temporary assembly storage and releases its CUDA buffers with the frequency
system. No extra assembly or factorization occurs when this output is absent.
Replay field work is included in field timing; `interface_radiation_backend`
records the selected replay backend in worker diagnostics.

Source IDs start with compiled interface IDs in order. An optional
`radiation:other-exterior` remainder groups all directly radiating exterior
components, including their scattering. `radiation:total` is evaluated from the
original complete solution and is not a source to add. Metadata records these
IDs and names, points, and RMS convention. Per-frequency diagnostics retain
normalized boundary reconstruction errors. The remainder is defined by subtracting interface traces from the
original traces; it is not a separate recoupled or blocked-port solution.

Tests reconstruct nonuniform interface fields, exercise source ordering and
zero excitation, and include directly radiating exterior motion. The required
standalone reference gate covers these alongside existing numerical references.
`tests/cuda_interface_radiation_tests.jl` separately qualifies GPU residency,
source ordering, zero excitation, phase conventions, and CPU parity for both
precisions and condensed/monolithic CUDA solves on real hardware.
Real coupled CPU/CUDA checks should compare complex source sums to the original
total, including near cancellation where total-relative errors are sensitive.

## CUDA replay measurement

On 2026-11-10, the Multi_region_SAWMOD project was measured on an RTX 2080 Ti
with Julia 1.12.6, four Julia threads and four OpenBLAS threads. The request
retained all project outputs, XY symmetry and three excitation bases, at
100, 1000 and 5000 Hz. Each implementation received a full warm-up before three
interleaved sweeps. The control used the former host replay factorization and
solve; the candidate used resident CUDA replay. The fused BEM kernel was not
changed.

| Three-frequency sweep | Host replay | CUDA replay |
|---|---:|---:|
| Median wall time | 4.426 s | 2.896 s |
| Min-max wall time | 3.925-4.493 s | 2.877-3.420 s |
| Median BEM-block/replay preparation | 1.427 s | 0.065 s |
| Median field work, including replay solves | 0.280 s | 0.147 s |
| CUDA pool high-water range | 917 MiB | 1172-1393 MiB |
| Sampled Julia host working-set peak range | 2672-2693 MiB | 2489-2677 MiB |

The measured wall-time reduction was 34.6%, with non-overlapping run ranges.
The CUDA pool metric excludes allocations outside that allocator and is not a
total VRAM measurement. Residency trades additional GPU memory for reduced
host factorization and transfer work.

A warmed Nsight Systems capture recorded device-to-host traffic falling from
357.6 MB to 6.75 MB (98.1%). Device-to-device traffic rose from 14.8 MB to
367.4 MB because the resident replay owns its factor/operator copies. These
trace totals corroborate residency; profiled wall times are not benchmark
results. The local Nsight build warned that its bundled tracing support was
older than the installed driver, so the counts describe recorded traffic.

Across all candidate repeats, the largest relative L2 difference from the host
replay control was 2.70e-5 (interface-radiated pressure). Against a Float64 CPU
reference on the identical request, the worst retained-quantity relative L2
error was 1.51e-4; interface-radiated pressure was 1.36e-4. Magnitude differences
were below 0.006 dB within 30 dB of each reference output's peak. These are
numerical comparisons, not mesh-convergence or measurement validation.

The retained local artifacts are in the Boundary Lab checkout under
`runs/sawmod_cuda_replay_20261110`: exact requests, complex outputs, per-section
timings, memory measurements, reference output and qualification logs. This
short sweep establishes the measured gain for this hardware and model, not a
universal speedup across GPU architectures or problem sizes.
