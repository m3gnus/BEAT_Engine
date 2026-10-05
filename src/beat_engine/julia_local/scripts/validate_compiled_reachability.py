"""Qualify effective compiled-worker dispatch with real CPU or Metal solves.

Run through the compute broker. This is a correctness/reachability gate,
not a startup or performance benchmark. Only the standard library is used.
"""

import argparse
import base64
import copy
import json
import math
import os
import struct
from pathlib import Path

from beat_engine import EngineWorker, engine_paths


def decoded(wire):
    raw = base64.b64decode(wire["content_base64"])
    scalar = "f" if wire["dtype"] == "complex64" else "d"
    values = struct.unpack("<" + scalar * (len(raw) // struct.calcsize(scalar)), raw)
    return [complex(r, i) for r, i in zip(values[::2], values[1::2])]


def compare(actual, reference, tolerance):
    assert len(actual) == len(reference)
    worst = 0.0
    for a, b in zip(actual, reference):
        assert a["freq_hz"] == b["freq_hz"]
        assert a["excitation_port_ids"] == b["excitation_port_ids"]
        assert len(a["quantities"]) == len(b["quantities"])
        for qa, qb in zip(a["quantities"], b["quantities"]):
            assert qa["id"] == qb["id"]
            assert qa["values"]["shape"] == qb["values"]["shape"]
            va, vb = decoded(qa["values"]), decoded(qb["values"])
            assert len(va) == len(vb)
            assert all(math.isfinite(v.real) and math.isfinite(v.imag) for v in va + vb)
            error = math.sqrt(sum(abs(x - y) ** 2 for x, y in zip(va, vb)))
            norm = math.sqrt(sum(abs(x) ** 2 for x in vb))
            relative = error / max(norm, 1e-30)
            assert relative <= tolerance, (qa["id"], relative, tolerance)
            worst = max(worst, relative)
    return worst


def run(julia, backend, out):
    paths = engine_paths(backend)
    template = json.loads((paths.root / "beat_contract" / "example-exterior-request.json").read_text())
    bundle_name = "BeatEngineCompiledCpuBundle" if backend == "cpu" else "BeatEngineCompiledMetalBundle"
    evidence = []

    def solve(request, overrides=None, expected_bundle=True):
        env = dict(os.environ, BLAB_BEAT_ENGINE_BUNDLE="1", BLAB_BEAT_ENGINE_GPU_BACKEND=backend)
        env.update(overrides or {})
        worker = EngineWorker(
            julia_executable=julia,
            solver_script=paths.system_solver,
            julia_project=paths.project,
            julia_threads=2,
            environment=env,
        )
        try:
            worker.ensure_started()
            info = worker.worker_info
            load = info["compiled_worker"]
            assert load["loaded_bundle"] == (bundle_name if expected_bundle else None), load
            assert load["driver_mode"] == ("bundle" if expected_bundle else "source")
            if expected_bundle:
                assert load["fallback_reason"] is None, load
            else:
                assert load["fallback_reason"] == "disabled_by_environment", load
            events = list(worker.submit(request))
            assert events[-1]["type"] == "completed", events[-1]
            rows = [event["result"] for event in events if event["type"] == "result"]
            assert len(rows) == len(request["frequencies_hz"])
            assert all(row["diagnostics"]["compiled_worker"] == load for row in rows)
            assert info["runtime"]["julia_threads"] == 2
            assert Path(info["runtime"]["project_file"]).resolve() == paths.project.resolve() / "Project.toml"
            evidence.append(
                {
                    "options": request["solver_options"],
                    "overrides": overrides or {},
                    "ready": info,
                    "diagnostics": [r["diagnostics"] for r in rows],
                }
            )
            return rows
        finally:
            worker.terminate()

    for symmetry, fixture in (("off", "two_tetrahedra.msh"), ("xy", "sample_quarter.msh")):
        request = copy.deepcopy(template)
        system = request["compiled_system"]
        system["meshes"][0]["file"] = str(paths.source_solver.parent / "test_meshes" / fixture)
        system["meshes"][0]["scale_to_m"] = 1.0 if symmetry == "off" else 0.001
        if symmetry == "off":
            system["boundaries"][0]["group"]["tag"] = 1
        port = copy.deepcopy(system["excitation_ports"][0])
        port["id"] = "port:second"
        system["excitation_ports"].append(port)
        request["excitation_port_ids"].append(port["id"])
        request["frequencies_hz"] = [1000, 2000]
        request["solver_options"].update(
            bem_backend=backend,
            symmetry=symmetry,
            quadrature_order=4,
            singular_order=4,
            regular_quadrature_mode="fixed",
        )
        request["outputs"].extend(
            [
                {"id": "surface:p", "quantity": "bem_boundary_pressure", "target_ids": [], "options": {}},
                {"id": "surface:q", "quantity": "bem_boundary_neumann", "target_ids": [], "options": {}},
            ]
        )
        direct = solve(request)
        for row in direct:
            d = row["diagnostics"]
            assert d["burton_miller_assembly"] == "direct_system"
            assert d["factorization_count"] == 1
            assert d["dense_solve_method"] == "lu"
            if backend == "cpu":
                assert [d[key] for key in ("cpu_regular_kernel", "cpu_singular_kernel", "cpu_field_kernel")] == [
                    "simd"
                ] * 3
            else:
                assert d["metal_regular_kernel"] == "packed_fused"
                assert d["metal_singular_kernel"] == "packed_grouped_duffy"
                assert d["metal_field_kernel"] == "packed_float32"
        forced_lu = {"BLAB_BEAT_DENSE_SOLVE": "lu"}
        reference_request = copy.deepcopy(request)
        reference_request["solver_options"]["burton_miller_assembly"] = "operator_matrices"
        scalar = {
            "BLAB_BEAT_CPU_REGULAR_KERNEL": "scalar",
            "BLAB_BEAT_CPU_SINGULAR_KERNEL": "scalar",
            "BLAB_BEAT_CPU_FIELD_KERNEL": "scalar",
            **forced_lu,
        }
        reference = solve(reference_request, scalar)
        error = compare(direct, reference, 1e-4)
        # The explicit source fallback must execute the same request and explain itself.
        fallback = solve(request, {"BLAB_BEAT_ENGINE_BUNDLE": "0"}, expected_bundle=False)
        fallback_error = compare(direct, fallback, 1e-5)
        if backend == "cpu":
            request64 = copy.deepcopy(request)
            request64["solver_options"]["precision"] = "float64"
            reference64 = copy.deepcopy(request64)
            reference64["solver_options"]["burton_miller_assembly"] = "operator_matrices"
            error64 = compare(solve(request64, forced_lu), solve(reference64, scalar), 1e-10)
            print(f"PASS {symmetry}: Float64 fused SIMD/operator scalar relative L2 {error64:.3g}")
        print(
            f"PASS {backend} {symmetry}: bundle dispatch; direct/operator L2 {error:.3g}; bundled/source L2 {fallback_error:.3g}"
        )
    # A production-sized request exercises the calibrated Metal solve router
    # and records per-drive history/fallback diagnostics that scalar summaries omit.
    large = copy.deepcopy(template)
    large["compiled_system"]["meshes"][0]["file"] = str(
        paths.source_solver.parent / "test_meshes" / "sample_detailed.msh"
    )
    large["frequencies_hz"] = [20, 80, 320, 1280, 5120, 20000]
    large["solver_options"].update(
        bem_backend=backend, quadrature_order=4, singular_order=4, regular_quadrature_mode="fixed"
    )
    rows = solve(large)
    print(
        "PASS sample_detailed compiled bundle: "
        + ", ".join(f"{row['freq_hz']:g}Hz:{row['diagnostics']['linear_solver']}" for row in rows)
    )
    out.write_text(json.dumps(evidence, indent=2))
    print(f"PASS compiled {backend} reachability: {len(evidence)} worker requests; evidence {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--julia", required=True)
    parser.add_argument("--backend", choices=("cpu", "metal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.julia, args.backend, args.out)
