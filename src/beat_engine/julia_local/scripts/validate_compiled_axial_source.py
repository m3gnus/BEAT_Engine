"""End-to-end compiled exterior axial source/phasor gate (standard library only).

Run with --julia /path/to/julia and optionally --backend metal.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import struct
import subprocess
import tempfile
from pathlib import Path

from beat_engine.client import EngineWorker
from beat_engine.worker import WorkerProcess

MESH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
5
1 -0.1 -0.1 0
2 0.1 -0.1 0
3 0.1 0.1 0
4 -0.1 0.1 0
5 0 0 0.08
$EndNodes
$Elements
4
1 2 2 2 2 1 2 5
2 2 2 2 2 2 3 5
3 2 2 2 2 3 4 5
4 2 2 2 2 4 1 5
$EndElements
"""


def decoded(quantity):
    wire = quantity["values"]
    raw = base64.b64decode(wire["content_base64"])
    scalar = "f" if wire["dtype"] == "complex64" else "d"
    numbers = struct.unpack("<" + scalar * (len(raw) // struct.calcsize(scalar)), raw)
    return [complex(real, imaginary) for real, imaginary in zip(numbers[::2], numbers[1::2])]


def close(actual, expected, rtol=8e-4, atol=2e-4):
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected):
        assert abs(observed - wanted) <= atol + rtol * abs(wanted), (observed, wanted)


def result_quantities(result):
    return {item["quantity"]: decoded(item) for item in result["quantities"]}


def run(julia, backend, legacy_solver=None, assembly="direct_system"):
    root = Path(__file__).resolve().parents[1]
    if legacy_solver is not None:
        legacy_solver = legacy_solver.resolve()
    # The backend projects are siblings of julia_local.
    project = root.parent / ("julia_metal" if backend == "metal" else "julia_local")
    solver = root / "coupled_solver.jl"
    template = json.loads((root.parent / "beat_contract" / "example-exterior-request.json").read_text())
    vertices = [(-0.1, -0.1, 0), (0.1, -0.1, 0), (0.1, 0.1, 0), (-0.1, 0.1, 0), (0, 0, 0.08)]
    faces = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)]
    axis = (1 / math.sqrt(2), 0, 1 / math.sqrt(2))
    projections = []
    areas = []
    for i, j, k in faces:
        a = [vertices[j][d] - vertices[i][d] for d in range(3)]
        b = [vertices[k][d] - vertices[i][d] for d in range(3)]
        cross = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
        length = math.sqrt(sum(value * value for value in cross))
        projections.append(sum(cross[d] * axis[d] for d in range(3)) / length)
        areas.append(length / 2)
    with tempfile.TemporaryDirectory(prefix="beat-axial-gate-") as directory:
        mesh = Path(directory) / "axial.msh"
        mesh.write_text(MESH)
        template["compiled_system"]["meshes"][0]["file"] = str(mesh)
        template["compiled_system"]["contract_version"] = 2
        template["compiled_system"]["meshes"][0]["scale_to_m"] = 1
        template["frequencies_hz"] = [500]
        template["solver_options"].update(
            bem_backend=backend,
            quadrature_order=2,
            singular_order=2,
            regular_quadrature_mode="fixed",
            burton_miller_assembly=assembly,
        )
        template["outputs"] = [
            {
                "id": "field",
                "quantity": "exterior_pressure",
                "target_ids": [],
                "options": {"points_m": [[0, 0, 1], [0.2, 0, 1], [0, 0.2, 1]]},
            },
            {"id": "q", "quantity": "bem_boundary_neumann", "target_ids": [], "options": {}},
            {"id": "p", "quantity": "bem_boundary_pressure", "target_ids": [], "options": {}},
            {"id": "z", "quantity": "radiation_impedance", "target_ids": [], "options": {}},
        ]
        env = dict(os.environ, BLAB_BEAT_ENGINE_BUNDLE="0")
        process = subprocess.Popen(
            [julia, "--threads=2", "--startup-file=no", f"--project={project}", str(solver), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            ready = json.loads(process.stdout.readline())
            assert ready["backends"][backend]["available"], ready["backends"][backend]
            assert "rigid_translation" in ready["exterior_source_profiles"]

            def solve(
                parameters,
                convention="exp(-i omega t)",
                expected_error=None,
                contract_version=2,
                mutate=None,
                all_results=False,
            ):
                request = copy.deepcopy(template)
                request["compiled_system"]["contract_version"] = contract_version
                request["compiled_system"]["components"][0]["parameters"] = parameters
                request["solver_options"]["phasor_convention"] = convention
                if mutate is not None:
                    mutate(request)
                request_file = Path(directory) / "request.json"
                request_file.write_text(json.dumps(request))
                command = {
                    "protocol_version": 1,
                    "operation": "solve",
                    "request": str(request_file),
                    "result_schema_version": 2,
                    "phasor_convention": convention,
                }
                process.stdin.write(json.dumps(command) + "\n")
                process.stdin.flush()
                events = []
                while True:
                    event = json.loads(process.stdout.readline())
                    events.append(event)
                    if event["type"] in ("completed", "failed"):
                        break
                if expected_error is not None:
                    assert events[-1]["type"] == "failed" and expected_error in events[-1]["error"], events[-1]
                    return None
                if events[-1]["type"] == "failed":
                    raise AssertionError(events[-1]["error"])
                results = [event["result"] for event in events if event["type"] == "result"]
                return results if all_results else results[0]

            uniform_result = solve({})
            if backend == "metal":
                assert uniform_result["diagnostics"]["burton_miller_assembly"] == assembly
            uniform = result_quantities(uniform_result)
            source = {"motion_profile": "rigid_translation", "motion_axis": [10, 0, 10]}
            negative = result_quantities(solve(source))
            positive = result_quantities(solve(source, "exp(+i omega t)"))
            scale = 1j * 1.21 * 2 * math.pi * 500
            close(uniform["bem_boundary_neumann"], [scale] * 4)
            close(negative["bem_boundary_neumann"], [scale * value for value in projections])
            assert max(abs(a - b) for a, b in zip(negative["exterior_pressure"], uniform["exterior_pressure"])) > 1e-2
            for quantity in (
                "bem_boundary_neumann",
                "bem_boundary_pressure",
                "exterior_pressure",
                "radiation_impedance",
            ):
                close(positive[quantity], [value.conjugate() for value in negative[quantity]])
            pressure = negative["bem_boundary_pressure"]
            expected_force = sum(
                sum(pressure[index] for index in face) / 3 * area * projection
                for face, area, projection in zip(faces, areas, projections)
            )
            close(negative["radiation_impedance"], [expected_force])
            reversed_source = dict(source, motion_axis=[-10, 0, -10])
            reversed_result = result_quantities(solve(reversed_source))
            for quantity in ("bem_boundary_neumann", "bem_boundary_pressure", "exterior_pressure"):
                close(reversed_result[quantity], [-value for value in negative[quantity]])
            close(reversed_result["radiation_impedance"], negative["radiation_impedance"])
            weighted = result_quantities(solve(dict(source, boundary_motion_weights={"boundary:source": 2})))
            for quantity in ("bem_boundary_neumann", "bem_boundary_pressure", "exterior_pressure"):
                close(weighted[quantity], [2 * value for value in negative[quantity]])
            close(weighted["radiation_impedance"], [4 * value for value in negative["radiation_impedance"]])
            sideways = result_quantities(solve(dict(source, motion_axis=[1, 0, 0])))
            side_projections = [0, 0.8 / math.sqrt(1.64), 0, -0.8 / math.sqrt(1.64)]
            close(sideways["bem_boundary_neumann"], [scale * value for value in side_projections])
            for magnitude in (1e100, 1e-100):
                rescaled = result_quantities(solve(dict(source, motion_axis=[magnitude, 0, magnitude])))
                close(rescaled["bem_boundary_neumann"], negative["bem_boundary_neumann"])
            solve(source, contract_version=1, expected_error="contract version 2")
            legacy_uniform = solve({}, contract_version=1)
            assert legacy_uniform["quantities"] == uniform_result["quantities"]
            ignored_parameters = {"legacy_extension": 1, "boundary_motion_weights": {"unrelated": 2}}
            assert solve(ignored_parameters, contract_version=1)["quantities"] == legacy_uniform["quantities"]
            assert solve({"motion_profile": "uniform_normal"})["quantities"] == uniform_result["quantities"]
            solve(
                {"motion_profile": "rigid_translation", "motion_axis": [0, 0, 0]}, expected_error="finite and nonzero"
            )
            solve({"motion_profile": "rigid_translation", "motion_axis": [0, 1]}, expected_error="three-component")
            solve({"motion_profile": "rigid_translation"}, expected_error="three-component")
            solve({"motion_axis": [0, 0, 1]}, expected_error="motion_axis requires")
            solve(dict(source, motion_axis=[True, 0, 1]), expected_error="finite numbers")
            solve({"motion_profile": "not_a_profile"}, expected_error="motion_profile must")
            solve(dict(source, unexpected=1), expected_error="unsupported parameters")
            solve(dict(source, boundary_motion_weights=[]), expected_error="must be an object")
            solve(dict(source, boundary_motion_weights={"unrelated": 1}), expected_error="unrelated boundaries")
            for weight in (0, -1):
                solve(
                    dict(source, boundary_motion_weights={"boundary:source": weight}),
                    expected_error="finite and positive",
                )
            for symmetry, translation in (("x", [0.2, 0, 0]), ("y", [0, 0.2, 0]), ("xy", [0.2, 0.2, 0])):

                def reduced(request):
                    request["solver_options"]["symmetry"] = symmetry
                    request["compiled_system"]["meshes"][0]["translation_m"] = translation

                expected_error = "Unsupported image mode: y" if symmetry == "y" else "physical symmetry planes"
                refused_axis = [0, 1, 0] if symmetry == "xy" else [1, 0, 0]
                solve(dict(source, motion_axis=refused_axis), expected_error=expected_error, mutate=reduced)

            def bounded(request):
                region = copy.deepcopy(request["compiled_system"]["regions"][0])
                region.update(id="region:interior", kind="bounded_air")
                request["compiled_system"]["regions"].append(region)

            solve(source, expected_error="exterior BEM solve", mutate=bounded)

            def interface_output(request):
                request["outputs"][0]["quantity"] = "interface_radiated_pressure"

            solve(source, expected_error="coupled system", mutate=interface_output)
            mesh.write_text(MESH.replace("3 2 2 2 2", "3 2 2 3 3").replace("4 2 2 2 2", "4 2 2 3 3"))

            def two_sources(request):
                system = request["compiled_system"]
                boundary = copy.deepcopy(system["boundaries"][0])
                boundary["id"] = "boundary:second"
                boundary["group"]["tag"] = 3
                system["boundaries"].append(boundary)
                component = copy.deepcopy(system["components"][0])
                component.update(id="component:second", boundary_ids=[boundary["id"]])
                component["parameters"] = dict(
                    source, motion_axis=[0, 0, -1], boundary_motion_weights={boundary["id"]: 1.3}
                )
                system["components"].append(component)
                port = dict(system["excitation_ports"][0], id="excitation:second", component_id=component["id"])
                system["excitation_ports"].append(port)
                request["excitation_port_ids"] = [system["excitation_ports"][index]["id"] for index in port_order]

            forward = None
            component_projections = (side_projections, [-1 / math.sqrt(1.64)] * 4)
            component_faces = ((0, 1), (2, 3))
            component_weights = (0.7, 1.3)
            for port_order in ([0, 1], [1, 0]):
                result = solve(
                    dict(source, motion_axis=[1, 0, 0], boundary_motion_weights={"boundary:source": 0.7}),
                    mutate=two_sources,
                )
                assert result["excitation_port_ids"] == [
                    "excitation:source" if index == 0 else "excitation:second" for index in port_order
                ]
                multiple = result_quantities(result)
                expected_q = []
                expected_force = []
                for component in range(2):
                    column = port_order.index(component)
                    pressure = multiple["bem_boundary_pressure"][5 * column : 5 * (column + 1)]
                    expected_force.append(
                        sum(
                            sum(pressure[index] for index in faces[face])
                            / 3
                            * areas[face]
                            * component_weights[component]
                            * component_projections[component][face]
                            for face in component_faces[component]
                        )
                    )
                for component in port_order:
                    expected_q.extend(
                        scale * component_weights[component] * component_projections[component][face]
                        if face in component_faces[component]
                        else 0
                        for face in range(4)
                    )
                close(multiple["bem_boundary_neumann"], expected_q)
                close(multiple["radiation_impedance"], expected_force)
                if forward is None:
                    forward = multiple
                else:
                    close(multiple["radiation_impedance"], forward["radiation_impedance"])
                    for quantity, size in (("bem_boundary_pressure", 5), ("bem_boundary_neumann", 4)):
                        close(multiple[quantity], forward[quantity][size:] + forward[quantity][:size])
            mesh.write_text(MESH)

            mesh.write_text(MESH.replace("5 0 0 0.08", "5 0 0 0"))
            flat_uniform = solve({})
            flat_axial = solve(dict(source, motion_axis=[0, 0, 1]))
            assert flat_uniform["quantities"] == flat_axial["quantities"], "flat axial source differs from normal"
            mesh.write_text(MESH)

            def sweep(request):
                request["frequencies_hz"] = [500, 1000]

            sweep_results = solve(source, mutate=sweep, all_results=True)
            assert [result["freq_hz"] for result in sweep_results] == [500, 1000]
            close(result_quantities(sweep_results[0])["exterior_pressure"], negative["exterior_pressure"])
            close(
                result_quantities(sweep_results[1])["bem_boundary_neumann"],
                [2 * value for value in negative["bem_boundary_neumann"]],
            )

            def single_1000(request):
                request["frequencies_hz"] = [1000]

            single = solve(source, mutate=single_1000)
            for quantity, values in result_quantities(sweep_results[1]).items():
                close(values, result_quantities(single)[quantity])
            if backend == "metal" and assembly == "direct_system" and os.environ.get("BLAB_METAL_PIPELINE") == "1":
                assert sweep_results[0]["diagnostics"]["metal_pipeline"]
            if legacy_solver is not None:
                request = copy.deepcopy(template)
                request["compiled_system"]["contract_version"] = 1
                request["compiled_system"]["components"][0]["parameters"] = ignored_parameters
                worker = EngineWorker(
                    julia_executable=julia,
                    solver_script=legacy_solver,
                    julia_project=legacy_solver.parent
                    if backend == "cpu"
                    else legacy_solver.parent.parent / "julia_metal",
                    julia_threads=2,
                    environment=env,
                )
                try:
                    events = list(worker.submit(request))
                    assert not [event for event in events if event["type"] == "failed"], events
                    result = next(event["result"] for event in events if event["type"] == "result")
                    assert result["quantities"] == legacy_uniform["quantities"], "default results changed from upstream"
                finally:
                    worker.terminate()
                print(f"PASS default exterior outputs bit-identical to upstream v1: {backend} {assembly}")
            if backend == "cpu":
                deployment_gate(julia, root, directory, template, source, solve, env)
            print(
                f"PASS compiled axial source {backend} {assembly}: signed projections, force, phasors, 2 independent axes, refusals"
            )
        finally:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=20)


def deployment_gate(julia, root, directory, template, source, solve, env):
    """Deployment already accepts the signed face trace produced by a compiled source."""

    def ground(request):
        request["solver_options"]["symmetry"] = "ground"
        request["compiled_system"]["meshes"][0]["translation_m"] = [0, 1, 0]

    reference = result_quantities(solve(source, mutate=ground))
    mesh = Path(directory) / "deployed.msh"
    mesh.write_text(
        MESH.replace("-0.1 -0.1 0", "-0.1 0.9 0")
        .replace("0.1 -0.1 0", "0.1 0.9 0")
        .replace("0.1 0.1 0", "0.1 1.1 0")
        .replace("-0.1 0.1 0", "-0.1 1.1 0")
        .replace("5 0 0 0.08", "5 0 1 0.08")
    )

    def trace(values):
        return {"real": [value.real for value in values], "imag": [value.imag for value in values]}

    request = {
        "schema": "boundary_lab_deploy_solve",
        "schema_version": 1,
        "solution_key": "compiled-axial-deployment-validation",
        "beat_engine_backend": "cpu",
        "frequency_hz": 500,
        "mesh_file": str(mesh),
        "mesh_is_world_space": True,
        "mesh_scale_factor": 1,
        "boundary_neumann": trace(reference["bem_boundary_neumann"]),
        "reference_boundary_pressure": trace(reference["bem_boundary_pressure"]),
        "observation_points_m": template["outputs"][0]["options"]["points_m"],
        "quadrature_order": 2,
        "singular_order": 2,
        "include_complex_pressure": True,
    }
    worker = WorkerProcess(
        julia_executable=julia, solver_script=root / "solver.jl", julia_project=root, julia_threads=2, environment=env
    )
    try:
        path = Path(directory) / "deploy.json"
        path.write_text(json.dumps(request))
        events = list(worker.submit(path))
        assert not [event for event in events if event["type"] == "failed"], events
        result = next(event["result"] for event in events if event["type"] == "result")
        observed = [complex(r, i) for r, i in zip(result["field_pressure"]["real"], result["field_pressure"]["imag"])]
        close(observed, reference["exterior_pressure"], rtol=2e-3)
        print("PASS CPU deployment consumes compiled signed axial Neumann trace")
    finally:
        worker.terminate()


def old_worker_raw_v2_refusal(julia, legacy_solver):
    """Send raw v2 axial JSON to a v1-only Julia worker, bypassing Python negotiation."""
    old_root = legacy_solver.resolve().parent
    request = json.loads((old_root.parent / "beat_contract" / "example-exterior-request.json").read_text())
    request["compiled_system"]["contract_version"] = 2
    request["compiled_system"]["components"][0]["parameters"] = {
        "motion_profile": "rigid_translation",
        "motion_axis": [0, 0, 1],
    }
    with tempfile.TemporaryDirectory(prefix="beat-old-worker-v2-") as directory:
        request_file = Path(directory) / "request.json"
        request_file.write_text(json.dumps(request))
        process = subprocess.Popen(
            [julia, "--threads=2", "--startup-file=no", f"--project={old_root}", str(legacy_solver), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(os.environ, BLAB_BEAT_ENGINE_BUNDLE="0"),
        )
        assert process.stdin is not None and process.stdout is not None
        try:
            ready = json.loads(process.stdout.readline())
            assert ready["contracts"]["compiled_system"] == [1], ready["contracts"]
            command = {
                "protocol_version": 1,
                "operation": "solve",
                "request": str(request_file),
                "result_schema_version": 2,
            }
            process.stdin.write(json.dumps(command) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            assert response["type"] == "failed" and "unsupported version" in response["error"], response
            print("PASS old v1 Julia worker rejects raw v2 axial JSON before solving")
        finally:
            process.stdin.close()
            process.terminate()
            process.wait(timeout=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--julia", required=True)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="cpu")
    parser.add_argument("--legacy-solver", type=Path)
    parser.add_argument("--assembly", choices=("direct_system", "operator_matrices"), default="direct_system")
    arguments = parser.parse_args()
    if arguments.legacy_solver is not None:
        old_worker_raw_v2_refusal(arguments.julia, arguments.legacy_solver)
    run(arguments.julia, arguments.backend, arguments.legacy_solver, arguments.assembly)
