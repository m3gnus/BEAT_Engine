"""BEAT-owned JSON contract; independent of application models and numerical libraries.

The bundled schema defines syntax; reference/topology checks define graph invariants.
Mesh contents, formulation support, and numerical options are checked by the solver.
This validator implements only the keywords used by the bundled schema, not a
general-purpose JSON Schema service.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

COMPILED_SYSTEM_VERSION = 3
SYSTEM_SOLVE_REQUEST_VERSION = 1
SYSTEM_RESULT_VERSION = 2
SUPPORTED_SYSTEM_RESULT_VERSIONS = frozenset({1, SYSTEM_RESULT_VERSION})


@lru_cache(maxsize=1)
def _schema() -> dict:
    return json.loads(Path(__file__).with_name("system-v1.schema.json").read_text(encoding="utf-8"))


def _fail(path: str, message: str) -> None:
    raise ValueError(f"BEAT contract {path}: {message}")


def _finite_json(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, item in value.items():
            _finite_json(item, f"{path}.{key}")
        return
    _fail(path, "must contain finite JSON values")


def _matches(value: Any, kind: str) -> bool:
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "null": value is None,
        "number": number,
        "integer": number and (isinstance(value, int) or value.is_integer()),
    }[kind]


def _validate(value: Any, schema: dict, path: str) -> None:
    if "$ref" in schema:
        schema = _schema()["$defs"][schema["$ref"].removeprefix("#/$defs/")]
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if kinds and not any(_matches(value, kind) for kind in kinds):
        _fail(path, f"expected {' or '.join(kinds)}")
    if "const" in schema and value != schema["const"]:
        _fail(path, f"unsupported version; expected {schema['const']}")
    if "enum" in schema and value not in schema["enum"]:
        _fail(path, f"expected one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                _fail(f"{path}.{key}", "required field is missing")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                _fail(f"{path}.{key}", "unknown field; use metadata/options for extensions")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            _fail(path, "invalid array length")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            _fail(path, "string is empty")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            _fail(path, f"must be >= {schema['minimum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            _fail(path, f"must be > {schema['exclusiveMinimum']}")


def _unique(values: list, path: str) -> None:
    if len(values) != len(set(values)):
        _fail(path, "duplicate identifiers")


def _references(values: list, available: dict, path: str) -> None:
    _unique(values, path)
    for value in values:
        if value not in available:
            _fail(path, f"unknown reference {value!r}")


@lru_cache(maxsize=1)
def _mesh_validator():
    # Keep the contract loadable directly from its distribution directory.
    import importlib.util

    spec = importlib.util.spec_from_file_location("beat_mesh_contract", Path(__file__).with_name("mesh.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_mesh_sources


def _source_profile_version(system: dict) -> None:
    if system["contract_version"] >= 2:
        return
    for component in system["components"]:
        if component["kind"] != "ideal_velocity_source":
            continue
        parameters = component["parameters"]
        if "motion_profile" in parameters or "motion_axis" in parameters:
            _fail(
                f"compiled_system.components.{component['id']}.parameters",
                "source motion profiles require compiled-system contract version 2",
            )


def _component_support(system: dict) -> None:
    exterior = not any(region["kind"] == "bounded_air" for region in system["regions"])
    for component in system["components"]:
        path = f"compiled_system.components.{component['id']}"
        if component["kind"] == "passive_radiator":
            _fail(path, "passive_radiator is not implemented")
        if exterior and component["kind"] == "electrodynamic_transducer" and system["contract_version"] != 3:
            _fail(path, "exterior electrodynamic_transducer requires compiled-system contract version 3")


def _impedance_outputs(request: dict) -> None:
    system = request["compiled_system"]
    excited = {
        port["component_id"] for port in system["excitation_ports"] if port["id"] in request["excitation_port_ids"]
    }
    components = {component["id"]: component for component in system["components"]}
    exterior_transducers = not any(region["kind"] == "bounded_air" for region in system["regions"]) and any(
        component["kind"] == "electrodynamic_transducer" for component in components.values()
    )
    options = request["solver_options"]
    if "transducer_reference_voltage_v" in options:
        voltage = options["transducer_reference_voltage_v"]
        if (
            isinstance(voltage, bool)
            or not isinstance(voltage, (int, float))
            or not math.isfinite(voltage)
            or voltage <= 0
        ):
            _fail(
                "request.solver_options.transducer_reference_voltage_v",
                "transducer_reference_voltage_v must be finite and positive",
            )
    if exterior_transducers:
        if (
            str(options.get("bem_backend", "cpu")).lower() == "metal"
            and str(options.get("precision", "float32")).lower() == "float64"
        ):
            _fail(
                "request.solver_options.bem_backend",
                "exterior electrodynamic_transducers cannot use Metal: float64 BEM is unsupported; use CPU",
            )
        if str(options.get("symmetry", "off")).strip().lower() not in ("off", "ground", "x", "xy"):
            _fail(
                "request.solver_options.symmetry",
                "exterior electrodynamic_transducers support only off, ground, x and xy symmetry",
            )
    for output in request["outputs"]:
        if output["quantity"] == "radiation_impedance_matrix":
            if any(region["kind"] == "bounded_air" for region in system["regions"]) or any(
                component["kind"] not in ("ideal_velocity_source", "electrodynamic_transducer")
                for component in components.values()
            ):
                _fail(
                    "request.outputs",
                    "radiation_impedance_matrix requires an exterior system with supported component kinds",
                )
            targets = output["target_ids"]
            _references(targets, components, "request.outputs.target_ids")
            if any(
                target not in excited and components[target]["kind"] != "electrodynamic_transducer"
                for target in targets
            ):
                _fail("request.outputs.target_ids", "radiation_impedance_matrix targets must be excited")
        elif output["quantity"] == "radiation_impedance" and not any(
            region["kind"] == "bounded_air" for region in system["regions"]
        ):
            if any(
                component_id not in excited
                for component_id, component in components.items()
                if component["kind"] == "ideal_velocity_source"
            ):
                _fail(
                    "request.outputs",
                    "radiation_impedance requires every compiled component to be excited; radiator axis remains compiled component order",
                )


def _graph(system: dict) -> None:
    _mesh_validator()(system)
    collections = {}
    for name in ("meshes", "regions", "boundaries", "interfaces", "components", "excitation_ports"):
        _unique([item["id"] for item in system[name]], f"compiled_system.{name}")
        collections[name] = {item["id"]: item for item in system[name]}
    meshes, regions, boundaries, components = (
        collections[key] for key in ("meshes", "regions", "boundaries", "components")
    )
    for region in regions.values():
        if "thermoviscous_wall_losses" in region["loss_model"]:
            _fail(f"region {region['id']}.loss_model", "assign thermoviscous wall losses to boundary parameters")
        _references(region["mesh_ids"], meshes, f"region {region['id']}.mesh_ids")
        for group in region["volume_groups"]:
            if group["mesh_id"] not in region["mesh_ids"] or group["dimension"] != 3:
                _fail(f"region {region['id']}.volume_groups", "must reference a volume group on a region mesh")
    for boundary in boundaries.values():
        _references([boundary["region_id"]], regions, f"boundary {boundary['id']}.region_id")
        if boundary["parameters"].get("thermoviscous_wall_losses", "off") != "off" and (
            regions[boundary["region_id"]]["kind"] != "bounded_air"
            or boundary["kind"] != "rigid"
            or "wall_impedance" in boundary["parameters"]
            or any(boundary["id"] in c["boundary_ids"] for c in components.values())
        ):
            _fail(
                f"boundary {boundary['id']}.parameters",
                "thermoviscous wall losses require an unlined, stationary rigid wall in bounded air",
            )
        group = boundary["group"]
        if group["mesh_id"] not in regions[boundary["region_id"]]["mesh_ids"] or group["dimension"] != 2:
            _fail(f"boundary {boundary['id']}.group", "must reference a surface group on a region mesh")
    for component in components.values():
        _references(component["boundary_ids"], boundaries, f"component {component['id']}.boundary_ids")
    for port in collections["excitation_ports"].values():
        _references([port["component_id"]], components, f"port {port['id']}.component_id")
    for interface in collections["interfaces"].values():
        _references(
            [interface["bounded_boundary_id"], interface["unbounded_boundary_id"]],
            boundaries,
            f"interface {interface['id']}",
        )
        topology = interface["topology"]
        if len(topology["fem_vertex_indices"]) != len(topology["fem_to_bem_vertex_indices"]):
            _fail(f"interface {interface['id']}.topology", "vertex mapping lengths differ")
        if len({len(topology[key]) for key in ("fem_face_indices", "bem_face_indices", "normal_sign")}) != 1:
            _fail(f"interface {interface['id']}.topology", "face mapping lengths differ")


def validate_compiled_system(raw: dict) -> None:
    _finite_json(raw, "compiled_system")
    _validate(raw, _schema()["$defs"]["compiled_system"], "compiled_system")
    _source_profile_version(raw)
    _graph(raw)
    _component_support(raw)


def validate_solve_request(raw: dict) -> None:
    _finite_json(raw, "request")
    _validate(raw, _schema()["$defs"]["solve_request"], "request")
    _source_profile_version(raw["compiled_system"])
    _graph(raw["compiled_system"])
    _component_support(raw["compiled_system"])
    _references(
        raw["excitation_port_ids"],
        {port["id"]: port for port in raw["compiled_system"]["excitation_ports"]},
        "request.excitation_port_ids",
    )
    _unique([output["id"] for output in raw["outputs"]], "request.outputs")
    _impedance_outputs(raw)
