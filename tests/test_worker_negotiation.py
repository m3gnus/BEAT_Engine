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


def test_selects_current_formats_and_accepts_future_advertised_versions(ready, payload):
    ready["contracts"]["system_result"].append(3)
    ready["future_extension"] = True
    assert negotiate_submission(ready, payload, "solve") == {"protocol_version": 1, "result_schema_version": 2}
    field = {"binary_array_schema_version": 1, "bem_backend": "cpu", "precision": "float32"}
    assert negotiate_submission(ready, field, "bem_field") == {"protocol_version": 1, "field_array_schema_version": 1}


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
def test_rigid_ideal_source_rejects_bounded_regions_before_submission(ready, payload, kind):
    system = payload["compiled_system"]
    system["contract_version"] = 2
    system["components"][0]["parameters"] = {"motion_profile": "rigid_translation", "motion_axis": [1, 2, 3]}
    if kind == "interior_fem":
        system["regions"][0]["kind"] = "bounded_air"
    else:
        interior = copy.deepcopy(system["regions"][0])
        interior.update(id="region:interior", kind="bounded_air")
        system["regions"].append(interior)
    with pytest.raises(WorkerCompatibilityError, match="exterior BEM solve"):
        negotiate_submission(ready, payload, "solve")


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
