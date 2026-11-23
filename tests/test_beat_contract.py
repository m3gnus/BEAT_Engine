"""BEAT's model-independent contract and the Boundary Lab projection."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from beat_engine.beat_contract import COMPILED_SYSTEM_VERSION, validate_compiled_system, validate_solve_request

CONTRACT = Path(__file__).resolve().parents[1] / "src/beat_engine/beat_contract"
CORPUS = json.loads((CONTRACT / "conformance.json").read_text())


def test_schema_uses_only_keywords_supported_by_both_validators():
    supported = {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "minimum",
        "exclusiveMinimum",
    }

    def check(schema):
        assert set(schema) <= supported
        if "$ref" in schema:
            assert schema["$ref"].startswith("#/$defs/")
            assert set(schema) <= {"$ref", "title", "description", "$schema", "$id", "$defs"}
        if "additionalProperties" in schema:
            assert isinstance(schema["additionalProperties"], bool)
        for key in ("properties", "$defs"):
            for child in schema.get(key, {}).values():
                check(child)
        if "items" in schema:
            check(schema["items"])

    check(json.loads((CONTRACT / "system-v1.schema.json").read_text()))


@pytest.mark.parametrize("case", CORPUS["cases"], ids=lambda case: case["name"])
def test_engine_conformance_cases(case):
    request = copy.deepcopy(CORPUS["base_request"])
    for change in case["changes"]:
        parent = request
        for key in change["path"][:-1]:
            parent = parent[key]
        key = change["path"][-1]
        if change.get("remove"):
            del parent[key]
        else:
            parent[key] = change["value"]
    if case["valid"]:
        validate_solve_request(request)
    else:
        with pytest.raises(ValueError, match="BEAT contract") as error:
            validate_solve_request(request)
        if "expected_message" in case:
            assert case["expected_message"] in str(error.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_or_nonfinite_extension_values_are_rejected(value):
    request = copy.deepcopy(CORPUS["base_request"])
    request["solver_options"]["custom"] = value
    with pytest.raises(ValueError, match="finite JSON"):
        validate_solve_request(request)


@pytest.mark.parametrize("value", [0, 0.0, 0.005, 1])
def test_wavenumber_shift_accepts_non_negative_numbers(value):
    request = copy.deepcopy(CORPUS["base_request"])
    request["solver_options"]["wavenumber_shift"] = value
    validate_solve_request(request)


@pytest.mark.parametrize("value", [-0.001, True, "0.005", None])
def test_wavenumber_shift_rejects_negative_or_non_numeric(value):
    request = copy.deepcopy(CORPUS["base_request"])
    request["solver_options"]["wavenumber_shift"] = value
    with pytest.raises(ValueError, match="wavenumber_shift"):
        validate_solve_request(request)


def test_contract_loads_standalone_and_validates_without_application_or_numpy():
    code = """
import importlib.abc
import importlib.util
import json
import pathlib
import sys

class StandardLibraryOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            raise AssertionError(fullname)

sys.meta_path.insert(0, StandardLibraryOnly())
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('independent_contract', path / '__init__.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
request = json.loads((path / 'example-exterior-request.json').read_text())
module.validate_solve_request(request)
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(CONTRACT)], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr


def test_compiled_contract_requires_version_even_without_request():
    system = copy.deepcopy(CORPUS["base_request"]["compiled_system"])
    del system["contract_version"]
    with pytest.raises(ValueError, match="contract_version"):
        validate_compiled_system(system)


def test_axial_source_requires_compiled_v2_and_v1_normal_remains_valid():
    request = copy.deepcopy(CORPUS["base_request"])
    source = request["compiled_system"]["components"][0]
    validate_solve_request(request)
    request["compiled_system"]["contract_version"] = 2
    validate_solve_request(request)
    source["parameters"] = {"motion_profile": "rigid_translation", "motion_axis": [0, 0, 1]}
    validate_solve_request(request)
    validate_compiled_system(request["compiled_system"])
    request["compiled_system"]["contract_version"] = 1
    with pytest.raises(ValueError, match="contract version 2"):
        validate_solve_request(request)
    with pytest.raises(ValueError, match="contract version 2"):
        validate_compiled_system(request["compiled_system"])
    source["parameters"] = {"motion_axis": [0, 0, 1]}
    with pytest.raises(ValueError, match="contract version 2"):
        validate_solve_request(request)


@pytest.mark.parametrize("bounded", [False, True])
def test_passive_radiator_rejected_in_every_region_kind(bounded):
    request = copy.deepcopy(CORPUS["base_request"])
    system = request["compiled_system"]
    if bounded:
        system["regions"][0]["kind"] = "bounded_air"
    system["components"][0]["kind"] = "passive_radiator"
    with pytest.raises(ValueError, match="passive_radiator is not implemented"):
        validate_compiled_system(system)
    with pytest.raises(ValueError, match="passive_radiator is not implemented"):
        validate_solve_request(request)


def test_matrix_refused_for_bounded_request():
    request = copy.deepcopy(CORPUS["base_request"])
    request["compiled_system"]["regions"][0]["kind"] = "bounded_air"
    request["outputs"] = [{"id": "z", "quantity": "radiation_impedance_matrix", "target_ids": [], "options": {}}]
    with pytest.raises(ValueError, match="requires an exterior system"):
        validate_solve_request(request)


def test_current_compiled_contract_version():
    assert COMPILED_SYSTEM_VERSION == 3
