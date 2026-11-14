# RX 7600 RGP capture

Radeon GPU Profiler 2.7.1 and Radeon Developer Panel/Service 3.5.1 captured
a warmed four-cabinet Deploy Coupled workload on 2026-10-30. The engine was
`4dadbd8772e1226ee1ad138fed82676ce8992b44` and the Deploy application was
`330863a6411cd68edd2f692897be612aef0005b9`, matching the
[uninstrumented baseline](RX%207600%20Deploy%20Baseline.md).

The worker completed its warm-up before capture, then passed the same CPU
reference comparisons on the captured repeat. Maximum output relative L2 was
`1.181411210148693e-5`, below the unchanged `5e-4` limit. The worker exited
normally. The 10-dispatch trace is 57,708,778 bytes with SHA-256
`d74b535b889b3d488e6afbd07c483a4860860b0303c263316d82fba35dd854e4`.

## Capture procedure

The tools were installed under `C:\Users\John\Documents\RGP`. Start the
developer service hidden, then configure the panel before starting a fresh
Julia worker. On this host the panel required an explicit loopback connection;
its default local-router connection failed.

```text
RadeonDeveloperPanelCLI.exe --mode profiling --process julia --remote-host 127.0.0.1 --remote-port 27300 --output <trace.rgp> --rgp-capture-mode dispatch --rgp-render-op-count 10 --verbose
```

The capture worker pauses after both warm-up frequencies and writes its Julia
PID to `capture-ready.json`. Confirm the panel identifies that PID, send `c`
to arm capture, and release the worker's gate promptly. Inactive capture clients
can time out. After capture, send `q`, wait for worker teardown and stop the
owned developer service. No driver, SDK or security-policy changes were needed.

RGP reported setting peak clocks during capture and restoring them afterward.
Instrumented timings are excluded from performance comparisons. Instruction
tracing and performance-counter collection were disabled for this first capture.

An earlier full capture attempt failed during warm-up with
`hipErrorLaunchFailure`; its failed output remains separate. An independent
minimal compute/Float64-solve probe passed both with and without capture, and
the subsequent full capture passed. That sequence does not establish that the
initial failure has been resolved generally.

## Static kernel evidence

The native GUI automation connection was unavailable. Inspection used the
[published AMD RDF format](https://github.com/GPUOpen-Drivers/libamdrdf/blob/main/docs/specification.md)
to extract embedded ELF code objects, then the installed ROCm LLVM tools to
read metadata and disassemble gfx1102 instructions.

| Kernel | VGPR count | SGPR count | VGPR spills | SGPR spills | Private bytes |
|---|---:|---:|---:|---:|---:|
| Regular SLP/adjoint DLP pairs | 128 | 107 | 12 | 75 | 76 |
| Regular hypersingular pairs | 114 | 107 | 0 | 140 | 36 |
| Singular SLP/adjoint blocks | 90 | 107 | 0 | 27 | 36 |
| Singular DLP/hypersingular blocks | 99 | 107 | 0 | 64 | 36 |

These are compiler metadata for loaded kernels, not dynamic stall counts.
The code-object inventory does not prove every listed kernel occurred among
the ten captured dispatches. The regular-pair instruction listings also contain
many bounds-check branches and scratch accesses; this motivates a narrowly
scoped bounds-check experiment, but does not establish a speedup or its cause.
Any candidate must pass numerical gates and interleaved uninstrumented timing
on both Deploy and core application paths before promotion.

Local evidence is in
`runs/rx7600-deploy/rgp-capture-20261030-152438/`: `warmed-dispatches.rgp`,
`capture-evidence.json`, RDF metadata, extracted code objects and worker output.
The earlier failed attempt and minimal probe are retained under
`rgp-capture-20261030-151819/`. Generated traces remain untracked.
