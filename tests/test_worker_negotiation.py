"""Compatibility rejection must happen before a worker receives a job."""

import copy
import json
import sys
from pathlib import Path

import pytest

from beat_engine import EngineWorker as BeatEngineWorkerProcess
from beat_engine.beat_contract.worker import (
    WorkerCompatibilityError,
    negotiate_submission,
    validate_worker_ready,
)

CONTRACT = Path(__file__).resolve().parents[1] / "src/beat_engine/beat_contract"


@pytest.fixture
def ready():
    info = json.loads((CONTRACT / "worker-v1.json").read_text())
    info["backends"] = {"cpu": {"available": True}, "cuda": {"available": False, "reason": "no device"}}
    return info


@pytest.fixture
def payload():
    return json.loads((CONTRACT / "example-exterior-request.json").read_text())


def enable_wall_loss(payload):
    system = payload["compiled_system"]
    system["regions"][0]["kind"] = "bounded_air"
    wall = copy.deepcopy(system["boundaries"][0])
    wall.update(id="boundary:wall", kind="rigid", parameters={"thermoviscous_wall_losses": "thin_boundary_layer"})
    wall["group"]["tag"] = 3
    system["boundaries"].append(wall)
    return wall


@pytest.mark.parametrize("capability", ["fem_wall_loss_models", "fem_wall_loss_scopes"])
@pytest.mark.parametrize("advertised", [None, [], ["future_value"]])
def test_thermoviscous_model_requires_explicit_capability(ready, payload, advertised, capability):
    wall = enable_wall_loss(payload)
    if advertised is None:
        ready.pop(capability)
    else:
        ready[capability] = advertised
    with pytest.raises(WorkerCompatibilityError, match="thermoviscous wall loss model"):
        negotiate_submission(ready, payload, "solve")
    wall["parameters"]["thermoviscous_wall_losses"] = "off"
    negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize("capability", ["fem_wall_loss_models", "fem_wall_loss_scopes"])
def test_thermoviscous_capability_accepts_enabled_model_and_rejects_invalid_handshake(ready, payload, capability):
    enable_wall_loss(payload)
    negotiate_submission(ready, payload, "solve")
    ready[capability] = "thin_boundary_layer"
    with pytest.raises(WorkerCompatibilityError, match=f"invalid {capability}"):
        negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize("capability", ["fem_wall_loss_models", "fem_wall_loss_scopes"])
def test_thermoviscous_incompatible_request_never_reaches_worker(tmp_path, ready, payload, capability):
    ready.pop(capability)
    enable_wall_loss(payload)
    worker, received = fake_worker(tmp_path, ready)
    path = tmp_path / "thermoviscous.json"
    path.write_text(json.dumps(payload))
    try:
        with pytest.raises(WorkerCompatibilityError, match="thermoviscous wall loss model"):
            worker.submit(path)
        assert not received.exists()
    finally:
        worker.terminate()


def test_selects_current_formats_and_accepts_future_advertised_versions(ready, payload):
    ready["contracts"]["system_result"].append(3)
    ready["future_extension"] = True
    assert negotiate_submission(ready, payload, "solve") == {"protocol_version": 1, "result_schema_version": 2}
    field = {"binary_array_schema_version": 1, "bem_backend": "cpu", "precision": "float32"}
    assert negotiate_submission(ready, field, "bem_field") == {"protocol_version": 1, "field_array_schema_version": 1}


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
def test_gradient_field_requires_explicit_backend_capability(ready, backend):
    ready["backends"][backend] = {"available": True}
    field = {
        "binary_array_schema_version": 1,
        "bem_backend": backend,
        "precision": "float64",
        "field_quantity": "pressure_gradient_v1",
    }
    with pytest.raises(WorkerCompatibilityError, match="field quantity"):
        negotiate_submission(ready, field, "bem_field")
    ready["field_quantities"] = {backend: ["pressure", "pressure_gradient_v1"]}
    assert negotiate_submission(ready, field, "bem_field")["field_array_schema_version"] == 1


@pytest.mark.parametrize("capabilities", [None, [], "pressure_gradient_v1", {"cpu": "pressure_gradient_v1"}])
def test_gradient_field_rejects_malformed_capabilities(ready, capabilities):
    ready["field_quantities"] = capabilities
    field = {
        "binary_array_schema_version": 1,
        "bem_backend": "cpu",
        "precision": "float64",
        "field_quantity": "pressure_gradient_v1",
    }
    with pytest.raises(WorkerCompatibilityError, match="invalid field_quantities"):
        negotiate_submission(ready, field, "bem_field")
    ready["field_quantities"] = {"cpu": ["pressure", "pressure_gradient_v1"]}
    assert negotiate_submission(ready, field, "bem_field")["field_array_schema_version"] == 1
    field["field_quantity"] = "future_gradient"
    with pytest.raises(WorkerCompatibilityError, match="field quantity"):
        negotiate_submission(ready, field, "bem_field")


def test_axial_source_requires_explicit_worker_capability(ready, payload):
    source = payload["compiled_system"]["components"][0]
    payload["compiled_system"]["contract_version"] = 2
    source["parameters"] = {"motion_profile": "rigid_translation", "motion_axis": [0, 0, 1]}
    old_ready = copy.deepcopy(ready)
    old_ready.pop("exterior_source_profiles")
    old_ready["contracts"]["compiled_system"] = [1]
    with pytest.raises(WorkerCompatibilityError, match="compiled_system version 2"):
        negotiate_submission(old_ready, payload, "solve")
    old_ready["contracts"]["compiled_system"] = [1, 2]
    with pytest.raises(WorkerCompatibilityError, match="source profile"):
        negotiate_submission(old_ready, payload, "solve")
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    payload["compiled_system"]["contract_version"] = 1
    source["parameters"] = {}
    old_ready["contracts"]["compiled_system"] = [1]
    assert negotiate_submission(old_ready, payload, "solve")["result_schema_version"] == 2


@pytest.mark.parametrize(
    "profiles",
    [
        "not_a_list_but_contains_rigid_translation",
        None,
        {},
        ("rigid_translation",),
        [""],
        [" "],
        ["rigid_translation", 1],
        ["rigid_translation", True],
    ],
)
def test_rejects_malformed_source_profile_announcements(ready, payload, profiles):
    payload["compiled_system"]["contract_version"] = 2
    payload["compiled_system"]["components"][0]["parameters"] = {
        "motion_profile": "rigid_translation",
        "motion_axis": [0, 0, 1],
    }
    ready["exterior_source_profiles"] = profiles
    with pytest.raises(WorkerCompatibilityError, match="exterior_source_profiles"):
        negotiate_submission(ready, payload, "solve")


def test_missing_profiles_preserves_old_worker_normal_fallback(ready, payload):
    ready.pop("exterior_source_profiles")
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    ready["exterior_source_profiles"] = []
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2


@pytest.mark.parametrize("kind", ["interior_fem", "coupled_fem_bem_lem"])
def test_rigid_ideal_source_rejects_interior_boundaries_before_submission(ready, payload, kind):
    system = payload["compiled_system"]
    system["contract_version"] = 2
    system["components"][0]["parameters"] = {"motion_profile": "rigid_translation", "motion_axis": [1, 2, 3]}
    if kind == "interior_fem":
        system["regions"][0]["kind"] = "bounded_air"
    else:
        interior = copy.deepcopy(system["regions"][0])
        interior.update(id="region:interior", kind="bounded_air")
        system["regions"].append(interior)
        system["boundaries"][0]["region_id"] = interior["id"]
    with pytest.raises(WorkerCompatibilityError, match="exterior-only boundary assignments"):
        negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize("profiles", [None, {}, "rigid_translation", [1], [""]])
def test_rejects_malformed_coupled_source_profiles(ready, profiles):
    ready["coupled_exterior_source_profiles"] = profiles
    with pytest.raises(WorkerCompatibilityError, match="coupled_exterior_source_profiles"):
        validate_worker_ready(ready)


def test_coupled_exterior_axial_source_requires_worker_capability(ready, payload):
    system = payload["compiled_system"]
    system["contract_version"] = 2
    system["components"][0]["parameters"] = {"motion_profile": "rigid_translation", "motion_axis": [0, 0, 1]}
    interior = copy.deepcopy(system["regions"][0])
    interior.update(id="region:interior", kind="bounded_air")
    system["regions"].append(interior)
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    ready.pop("coupled_exterior_source_profiles")
    with pytest.raises(WorkerCompatibilityError, match="restart the worker"):
        negotiate_submission(ready, payload, "solve")
    system["components"][0]["parameters"] = {}
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2


@pytest.mark.parametrize("version", [None, True, 1.0, 2, "1"])
def test_rejects_incompatible_protocol_versions(ready, version):
    ready["protocol"]["version"] = version
    with pytest.raises(WorkerCompatibilityError, match="protocol version"):
        validate_worker_ready(ready)


@pytest.mark.parametrize("contract", ["system_request", "compiled_system", "system_result"])
def test_rejects_incompatible_contracts(ready, payload, contract):
    ready["contracts"][contract] = [99]
    with pytest.raises(WorkerCompatibilityError, match=contract):
        negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize(
    "capability,value,match",
    [
        ("operations", [], "operation"),
        ("precisions", ["float64"], "precision"),
        ("solve_kinds", [], "solve kind"),
        ("cancellation", "unsupported", "cancellation"),
    ],
)
def test_rejects_missing_capabilities(ready, payload, capability, value, match):
    ready[capability] = value
    payload["cancel_path"] = "cancel"
    with pytest.raises(WorkerCompatibilityError, match=match):
        negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize("backend,match", [("cuda", "no device"), ("metal", "not advertised")])
def test_rejects_unavailable_backend_without_fallback(ready, payload, backend, match):
    payload["solver_options"]["bem_backend"] = backend
    with pytest.raises(WorkerCompatibilityError, match=match):
        negotiate_submission(ready, payload, "solve")


def fake_worker(tmp_path, info):
    script = tmp_path / "worker.py"
    received = tmp_path / "received.jsonl"
    script.write_text(
        f"""
import json, pathlib, sys
print(json.dumps({info!r}), flush=True)
for line in sys.stdin:
    with pathlib.Path({str(received)!r}).open('a') as stream:
        stream.write(line)
    print(json.dumps({{"type": "result", "result": {{"schema_version": 2}}}}), flush=True)
    print(json.dumps({{"type": "completed"}}), flush=True)
""",
        encoding="utf-8",
    )
    worker = BeatEngineWorkerProcess(
        julia_executable=sys.executable, solver_script=script, julia_threads=1, julia_project=None
    )
    return worker, received


def test_incompatible_request_never_reaches_worker_and_compatible_job_reuses_it(tmp_path, ready, payload):
    worker, received = fake_worker(tmp_path, ready)
    path = tmp_path / "payload.json"
    bad = copy.deepcopy(payload)
    bad["solver_options"]["bem_backend"] = "cuda"
    path.write_text(json.dumps(bad))
    try:
        with pytest.raises(WorkerCompatibilityError, match="no device"):
            worker.submit(path)
        assert not received.exists()
        process = worker._process
        path.write_text(json.dumps(payload))
        assert list(worker.submit(path))[-1]["type"] == "completed"
        assert worker._process is process
        command = json.loads(received.read_text())
        assert command["protocol_version"] == 1
        assert command["result_schema_version"] == 2
        snapshot = worker.worker_info
        snapshot["operations"].clear()
        assert worker.worker_info["operations"] == ["solve", "bem_field"]
    finally:
        worker.terminate()
    assert worker.worker_info is None


def test_bare_ready_cannot_accept_physical_requests(tmp_path, payload):
    worker, received = fake_worker(tmp_path, {"type": "ready"})
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    try:
        with pytest.raises(WorkerCompatibilityError, match="missing versioned handshake"):
            worker.submit(path)
        assert not received.exists()
    finally:
        worker.terminate()


def test_invalid_handshake_is_discarded_and_restart_renegotiates(tmp_path, ready, payload):
    bad = copy.deepcopy(ready)
    bad["protocol"]["version"] = 99
    worker, received = fake_worker(tmp_path, bad)
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    try:
        with pytest.raises(WorkerCompatibilityError):
            worker.submit(path)
        assert worker._process is None
        assert worker.worker_info is None
        assert not received.exists()
        fake_worker(tmp_path, ready)  # Replace the executable's announcement.
        assert list(worker.submit(path))[-1]["type"] == "completed"
    finally:
        worker.terminate()


def test_abandoned_stream_discards_process_and_handshake(tmp_path, ready, payload):
    worker, _ = fake_worker(tmp_path, ready)
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    try:
        stream = worker.submit(path)
        assert next(stream)["type"] == "result"
        process = worker._process
        stream.close()
        assert process.poll() is not None
        assert worker.worker_info is None
        assert list(worker.submit(path))[-1]["type"] == "completed"
        assert worker._process is not process
    finally:
        worker.terminate()


def test_response_version_mismatch_discards_process(tmp_path, ready, payload):
    worker, _ = fake_worker(tmp_path, ready)
    script = worker.solver_script
    script.write_text(script.read_text().replace('"schema_version": 2', '"schema_version": 1'))
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))
    try:
        with pytest.raises(WorkerCompatibilityError, match="selected system_result"):
            list(worker.submit(path))
        assert worker._process is None
    finally:
        worker.terminate()


def test_interface_velocity_requires_advertised_output(ready, payload):
    payload["outputs"][0]["quantity"] = "interface_average_normal_velocity"
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    ready.pop("optional_output_quantities")
    with pytest.raises(WorkerCompatibilityError, match="interface-average velocity"):
        negotiate_submission(ready, payload, "solve")


def test_reclamation_requires_advertisement_but_no_result_contract(ready):
    with pytest.raises(WorkerCompatibilityError, match="unavailable"):
        negotiate_submission(ready, {}, "reclaim")
    ready["operations"].append("reclaim")
    assert negotiate_submission(ready, {}, "reclaim") == {"protocol_version": 1}


def test_interface_radiation_requires_advertised_output(ready, payload):
    payload["outputs"][0]["quantity"] = "interface_radiated_pressure"
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    ready["optional_output_quantities"].remove("interface_radiated_pressure")
    with pytest.raises(WorkerCompatibilityError, match="interface radiation"):
        negotiate_submission(ready, payload, "solve")


def test_impedance_matrix_requires_advertised_output(ready, payload):
    payload["outputs"] = [{"id": "z", "quantity": "radiation_impedance_matrix", "target_ids": [], "options": {}}]
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    for missing_list in (False, True):
        old_ready = copy.deepcopy(ready)
        if missing_list:
            old_ready.pop("optional_output_quantities")
        else:
            old_ready["optional_output_quantities"].remove("radiation_impedance_matrix")
        with pytest.raises(WorkerCompatibilityError, match="radiation_impedance_matrix"):
            negotiate_submission(old_ready, payload, "solve")
    payload["outputs"] = []
    assert negotiate_submission(old_ready, payload, "solve")["result_schema_version"] == 2


@pytest.mark.parametrize("capabilities", ["radiation_impedance_matrix", None, {}, [1], [""]])
def test_rejects_malformed_optional_output_announcements(ready, capabilities):
    ready["optional_output_quantities"] = capabilities
    with pytest.raises(WorkerCompatibilityError, match="optional_output_quantities"):
        validate_worker_ready(ready)


def exterior_transducer_payload(payload):
    payload["compiled_system"]["contract_version"] = 3
    payload["compiled_system"]["components"][0]["kind"] = "electrodynamic_transducer"
    payload["compiled_system"]["excitation_ports"][0]["kind"] = "voltage"
    payload["solver_options"]["precision"] = "float64"
    return payload


def test_exterior_transducers_require_kind_capability_even_with_v3(ready, payload):
    exterior_transducer_payload(payload)
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
    for kinds in (None, ["ideal_velocity_source"], []):
        old = copy.deepcopy(ready)
        if kinds is None:
            old.pop("exterior_component_kinds")
        else:
            old["exterior_component_kinds"] = kinds
        with pytest.raises(WorkerCompatibilityError, match="electrodynamic_transducer"):
            negotiate_submission(old, payload, "solve")
    old = copy.deepcopy(ready)
    old["contracts"]["compiled_system"] = [1, 2]
    with pytest.raises(WorkerCompatibilityError, match="compiled_system version 3"):
        negotiate_submission(old, payload, "solve")


def test_undriven_exterior_transducer_also_requires_capability(ready, payload):
    driver = copy.deepcopy(payload["compiled_system"]["components"][0])
    driver.update(id="undriven", kind="electrodynamic_transducer")
    payload["compiled_system"]["components"].append(driver)
    payload["compiled_system"]["contract_version"] = 3
    payload["solver_options"]["precision"] = "float64"
    ready.pop("exterior_component_kinds")
    with pytest.raises(WorkerCompatibilityError, match="electrodynamic_transducer"):
        negotiate_submission(ready, payload, "solve")


@pytest.mark.parametrize("kinds", [None, {}, "ideal_velocity_source", [1], [""], [" "]])
def test_invalid_exterior_kind_advertisement(ready, kinds):
    ready["exterior_component_kinds"] = kinds
    with pytest.raises(WorkerCompatibilityError, match="exterior_component_kinds"):
        validate_worker_ready(ready)


def test_missing_exterior_kinds_keeps_ideal_sources_compatible(ready, payload):
    ready.pop("exterior_component_kinds")
    assert negotiate_submission(ready, payload, "solve")["result_schema_version"] == 2
