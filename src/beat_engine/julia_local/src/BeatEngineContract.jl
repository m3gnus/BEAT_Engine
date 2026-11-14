module BeatEngineContract

using JSON
include(joinpath(@__DIR__, "BeatEngineMeshData.jl"))
using .BeatEngineMeshData
include(joinpath(@__DIR__, "BeatEngineProvenance.jl"))
using .BeatEngineProvenance

export validate_system_request, worker_ready, validate_worker_submission

# The same engine-owned schema is loaded by the standalone Python validator.
# JSON inputs must invalidate compiled bundles just like included Julia sources.
include_dependency(joinpath(@__DIR__, "..", "..", "beat_contract", "system-v1.schema.json"))
include_dependency(joinpath(@__DIR__, "..", "..", "beat_contract", "worker-v1.json"))
const SCHEMA = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "system-v1.schema.json"))
const WORKER = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "worker-v1.json"))

function worker_ready(backends)
    info = deepcopy(WORKER)
    info["backends"] = backends
    merge!(info["engine"], engine_identity())
    info["runtime"] = runtime_identity()
    return info
end

function validate_worker_submission(submission)
    submission isa AbstractDict || error("BEAT worker command must be an object.")
    version = get(submission, "protocol_version", nothing)
    version isa Integer && !(version isa Bool) && version == WORKER["protocol"]["version"] ||
        error("Incompatible BEAT worker protocol; client must select protocol version 1.")
    convention = get(submission, "phasor_convention", "exp(-i omega t)")
    convention in WORKER["phasor_conventions"] || error("Unsupported phasor convention: $convention")
    operation = get(submission, "operation", nothing)
    (operation == "reclaim" || operation in WORKER["operations"]) || error("Unsupported BEAT worker operation: $operation")
    operation == "reclaim" && return nothing
    request = get(submission, "request", nothing)
    inline = get(submission, "request_inline", nothing)
    if inline !== nothing
        operation == "solve" && request === nothing && inline isa AbstractDict || error("Invalid inline worker submission.")
    else
        request isa AbstractString && !isempty(request) || error("BEAT worker command requires a request filename.")
    end
    field, contract = operation == "solve" ? ("result_schema_version", "system_result") : ("field_array_schema_version", "field_array")
    selected = get(submission, field, nothing)
    selected isa Integer && !(selected isa Bool) && selected in WORKER["contracts"][contract] ||
        error("Unsupported BEAT worker $field; select an advertised version.")
    return nothing
end

fail(path, message) = error("BEAT contract $path: $message")

function finite_json(value, path)
    if value === nothing || value isa AbstractString || value isa Bool || value isa Integer
        return
    elseif value isa Real && isfinite(value)
        return
    elseif value isa AbstractVector
        for (i, item) in enumerate(value)
            finite_json(item, "$path[$(i - 1)]")
        end
    elseif value isa AbstractDict && all(key isa AbstractString for key in keys(value))
        for (key, item) in value
            finite_json(item, "$path.$key")
        end
    else
        fail(path, "must contain finite JSON values")
    end
end

function matches(value, kind)
    kind == "object" && return value isa AbstractDict
    kind == "array" && return value isa AbstractVector
    kind == "string" && return value isa AbstractString
    kind == "null" && return value === nothing
    number = value isa Real && !(value isa Bool)
    kind == "number" && return number
    kind == "integer" && return number && isinteger(value)
    error("Unknown BEAT schema type: $kind")
end

function validate(value, schema, path)
    if haskey(schema, "\$ref")
        schema = SCHEMA["\$defs"][replace(schema["\$ref"], "#/\$defs/" => "")]
    end
    kinds = get(schema, "type", [])
    kinds = kinds isa AbstractString ? [kinds] : kinds
    isempty(kinds) || any(matches(value, kind) for kind in kinds) || fail(path, "unexpected JSON type")
    haskey(schema, "const") && value != schema["const"] && fail(path, "unsupported version")
    haskey(schema, "enum") && !(value in schema["enum"]) && fail(path, "unsupported enum value")
    if value isa AbstractDict
        for key in get(schema, "required", [])
            haskey(value, key) || fail("$path.$key", "required field is missing")
        end
        properties = get(schema, "properties", Dict())
        for (key, item) in value
            if haskey(properties, key)
                validate(item, properties[key], "$path.$key")
            elseif get(schema, "additionalProperties", true) === false
                fail("$path.$key", "unknown field; use metadata/options for extensions")
            end
        end
    elseif value isa AbstractVector
        get(schema, "minItems", 0) <= length(value) <= get(schema, "maxItems", Inf) || fail(path, "invalid array length")
        if haskey(schema, "items")
            for (i, item) in enumerate(value)
                validate(item, schema["items"], "$path[$(i - 1)]")
            end
        end
    elseif value isa AbstractString
        length(value) >= get(schema, "minLength", 0) || fail(path, "string is empty")
    elseif value isa Real && !(value isa Bool)
        haskey(schema, "minimum") && value < schema["minimum"] && fail(path, "below minimum")
        haskey(schema, "exclusiveMinimum") && value <= schema["exclusiveMinimum"] && fail(path, "must exceed minimum")
    end
end

function unique_ids(values, path)
    length(values) == length(Set(values)) || fail(path, "duplicate identifiers")
end

function references(values, available, path)
    unique_ids(values, path)
    for value in values
        haskey(available, value) || fail(path, "unknown reference $value")
    end
end

function source_profile_version(system)
    system["contract_version"] >= 2 && return nothing
    for component in system["components"]
        component["kind"] == "ideal_velocity_source" || continue
        parameters = component["parameters"]
        haskey(parameters, "motion_profile") || haskey(parameters, "motion_axis") || continue
        fail("compiled_system.components.$(component["id"]).parameters",
            "source motion profiles require compiled-system contract version 2")
    end
    return nothing
end

function component_support(system)
    exterior = !any(region["kind"] == "bounded_air" for region in system["regions"])
    for component in system["components"]
        path = "compiled_system.components.$(component["id"])"
        component["kind"] == "passive_radiator" && fail(path, "passive_radiator is not implemented")
        exterior && component["kind"] == "electrodynamic_transducer" && system["contract_version"] != 3 &&
            fail(path, "exterior electrodynamic_transducer requires compiled-system contract version 3")
    end
    return nothing
end

function impedance_outputs(request)
    system = request["compiled_system"]
    excited = Set(port["component_id"] for port in system["excitation_ports"]
        if port["id"] in request["excitation_port_ids"])
    components = Dict(component["id"] => component for component in system["components"])
    exterior = !any(region["kind"] == "bounded_air" for region in system["regions"])
    options = request["solver_options"]
    if haskey(options, "transducer_reference_voltage_v")
        voltage = options["transducer_reference_voltage_v"]
        voltage isa Real && !(voltage isa Bool) && isfinite(voltage) && voltage > 0 ||
            fail("request.solver_options.transducer_reference_voltage_v",
                "transducer_reference_voltage_v must be finite and positive")
    end
    if exterior && any(component["kind"] == "electrodynamic_transducer" for component in values(components))
        lowercase(String(get(options, "bem_backend", "cpu"))) == "metal" &&
            lowercase(String(get(options, "precision", "float32"))) == "float64" &&
            fail("request.solver_options.bem_backend",
                "exterior electrodynamic_transducers cannot use Metal: float64 BEM is unsupported; use CPU")
        lowercase(strip(String(get(options, "symmetry", "off")))) in ("off", "ground", "x", "xy") ||
            fail("request.solver_options.symmetry", "exterior electrodynamic_transducers support only off, ground, x and xy symmetry")
    end
    for output in request["outputs"]
        quantity = output["quantity"]
        if quantity == "radiation_impedance_matrix"
            exterior && all(component["kind"] in ("ideal_velocity_source", "electrodynamic_transducer") for component in values(components)) ||
                fail("request.outputs", "radiation_impedance_matrix requires an exterior system with supported component kinds")
            targets = output["target_ids"]
            references(targets, components, "request.outputs.target_ids")
            all(target in excited || components[target]["kind"] == "electrodynamic_transducer" for target in targets) ||
                fail("request.outputs.target_ids", "radiation_impedance_matrix targets must be excited")
        elseif quantity == "radiation_impedance" && exterior
            all(id in excited for (id, component) in components if component["kind"] == "ideal_velocity_source") || fail("request.outputs",
                "radiation_impedance requires every compiled component to be excited; radiator axis remains compiled component order")
        end
    end
    return nothing
end

function graph(system)
    validate_mesh_sources(system)
    collections = Dict()
    for name in ("meshes", "regions", "boundaries", "interfaces", "components", "excitation_ports")
        unique_ids([item["id"] for item in system[name]], "compiled_system.$name")
        collections[name] = Dict(item["id"] => item for item in system[name])
    end
    meshes, regions, boundaries, components = (collections[key] for key in ("meshes", "regions", "boundaries", "components"))
    for region in values(regions)
        if haskey(region["loss_model"], "thermoviscous_wall_losses")
            fail("region.loss_model", "assign thermoviscous wall losses to boundary parameters")
        end
        references(region["mesh_ids"], meshes, "region $(region["id"]).mesh_ids")
        for group in region["volume_groups"]
            group["mesh_id"] in region["mesh_ids"] && group["dimension"] == 3 || fail("volume_groups", "must reference a volume group on a region mesh")
        end
    end
    for boundary in values(boundaries)
        references([boundary["region_id"]], regions, "boundary $(boundary["id"]).region_id")
        if get(boundary["parameters"], "thermoviscous_wall_losses", "off") != "off" && (
            regions[boundary["region_id"]]["kind"] != "bounded_air" ||
            boundary["kind"] != "rigid" || haskey(boundary["parameters"], "wall_impedance") ||
            any(boundary["id"] in c["boundary_ids"] for c in values(components)))
            fail("boundary.parameters", "thermoviscous wall losses require an unlined, stationary rigid wall in bounded air")
        end
        group = boundary["group"]
        group["mesh_id"] in regions[boundary["region_id"]]["mesh_ids"] && group["dimension"] == 2 || fail("boundary.group", "must reference a surface group on a region mesh")
    end
    for component in values(components)
        references(component["boundary_ids"], boundaries, "component $(component["id"]).boundary_ids")
    end
    for port in values(collections["excitation_ports"])
        references([port["component_id"]], components, "port $(port["id"]).component_id")
    end
    for interface in values(collections["interfaces"])
        references([interface["bounded_boundary_id"], interface["unbounded_boundary_id"]], boundaries, "interface $(interface["id"])")
        topology = interface["topology"]
        length(topology["fem_vertex_indices"]) == length(topology["fem_to_bem_vertex_indices"]) || fail("topology", "vertex mapping lengths differ")
        length(Set(length(topology[key]) for key in ("fem_face_indices", "bem_face_indices", "normal_sign"))) == 1 || fail("topology", "face mapping lengths differ")
    end
end

function validate_system_request(request)
    finite_json(request, "request")
    validate(request, SCHEMA["\$defs"]["solve_request"], "request")
    source_profile_version(request["compiled_system"])
    graph(request["compiled_system"])
    component_support(request["compiled_system"])
    references(request["excitation_port_ids"], Dict(port["id"] => port for port in request["compiled_system"]["excitation_ports"]), "request.excitation_port_ids")
    unique_ids([output["id"] for output in request["outputs"]], "request.outputs")
    impedance_outputs(request)
    return nothing
end

end # module
