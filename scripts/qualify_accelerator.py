"""Manual hardware qualification; unavailable hardware is a failure, never a pass."""

import json
import os
import subprocess
import sys
from pathlib import Path

from beat_engine import engine_paths

backend = os.environ["BEAT_BACKEND"]
if backend not in {"cuda", "rocm", "metal"}:
    raise SystemExit("BEAT_BACKEND must be cuda, rocm, or metal")
info = json.loads(subprocess.check_output([sys.executable, "-m", "beat_engine", "doctor", "--backend", backend]))
Path("doctor.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
if not info["backends"][backend]["available"]:
    raise SystemExit(f"Hardware qualification failed: {info['backends'][backend]}")
paths = engine_paths(backend)
environment = dict(
    os.environ,
    BLAB_RUN_COUPLED_CUDA="1" if backend == "cuda" else "0",
    BLAB_RUN_COUPLED_ROCM="1" if backend == "rocm" else "0",
    BLAB_RUN_COUPLED_METAL="1" if backend == "metal" else "0",
)
subprocess.run(
    [
        "julia",
        "--threads=2",
        "--startup-file=no",
        f"--project={paths.project}",
        str(paths.root / "julia_local/tests/runtests.jl"),
    ],
    env=environment,
    check=True,
)
if backend == "rocm":
    for test in (
        "rocm_phasor_tests.jl",
        "rocm_worker_phasor_tests.jl",
        "rocm_field_gradient_tests.jl",
        "rocm_field_gradient_worker_tests.jl",
        "rocm_near_correction_tests.jl",
        "rocm_near_source_tests.jl",
        "rocm_wavelength_tests.jl",
        "rocm_wavelength_source_tests.jl",
        "rocm_exterior_transducer_tests.jl",
        "rocm_interface_radiation_tests.jl",
        "rocm_transducer_condensation_tests.jl",
        "rocm_transducer_condensation_worker_tests.jl",
        "rocm_speaker_rank_tests.jl",
        "rocm_speaker_rom_export_tests.jl",
        "deploy_rocm_tests.jl",
        "rocm_thermoviscous_tests.jl",
    ):
        subprocess.run(
            [
                "julia",
                "--threads=2",
                "--startup-file=no",
                f"--project={paths.project}",
                str(paths.root / "julia_local/tests" / test),
            ],
            env=environment,
            check=True,
        )
if backend == "metal":
    for test in (
        "metal_host_tests.jl",
        "metal_near_correction_tests.jl",
        "metal_large_panel_tests.jl",
        "metal_kernel_coverage_tests.jl",
        "metal_coupled_precompile_coverage_tests.jl",
    ):
        subprocess.run(
            [
                "julia",
                "--threads=2",
                "--startup-file=no",
                f"--project={paths.project}",
                str(paths.root / "julia_local/tests" / test),
            ],
            env=environment,
            check=True,
        )
