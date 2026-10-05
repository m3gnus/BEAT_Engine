"""
    BeatEngineCompiledCpuBundle

A precompilable home for the compiled-contract worker (`coupled_solver.jl`)
on the CPU backend. See BeatEngineCpuBundle for why a package and not a script:
`coupled_solver.jl` used to `include` the whole engine and its 3,900-line
driver from source on every worker start. The driver lives in
`BeatEngineCompiledDriver.jl`; this package includes it and runs a compiled
exterior request as its precompile workload so the compiled call graph is
cached in the pkgimage.

`BEAT_ENGINE_BACKEND` names the backend for `BeatEngineCore` (a child module
of this package), because the cache is built in a different environment than
the one a solve runs in.
"""
module BeatEngineCompiledCpuBundle

using PrecompileTools: @compile_workload

const BEAT_ENGINE_BACKEND = "cpu"

const ENGINE_DIR = let root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    found = nothing
    for name in ("julia_local", "julia")
        candidate = joinpath(root, name)
        if isfile(joinpath(candidate, "BeatEngineCompiledDriver.jl"))
            found = candidate
            break
        end
    end
    found === nothing && error("No BEAT engine sources found under $(root).")
    found
end

include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))

# A closed tetrahedron with every face tagged 2 (symmetry off). The second
# workload is a quadrant plate, defined in the shared workload helper.
const WORKLOAD_NODES = """
\$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
\$EndNodes
"""
const WORKLOAD_HEAD = """
\$MeshFormat
2.2 0 8
\$EndMeshFormat
\$PhysicalNames
1
2 2 "warmup"
\$EndPhysicalNames
"""
const WORKLOAD_MESH_OFF = WORKLOAD_HEAD * WORKLOAD_NODES * """
\$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
\$EndElements
"""

function workload_request(mesh, symmetry)
    points = [[1.0 * sin(a), 0.0, 1.0 * cos(a)] for a in range(0.0, pi; length=5)]
    return Dict{String,Any}(
        "schema_version" => 1,
        "compiled_system" => Dict{String,Any}(
            "id" => "system:warmup", "name" => "warmup", "contract_version" => 1,
            "meshes" => [Dict("id" => "mesh:surface", "name" => "Surface", "file" => mesh,
                "purpose" => "bem_surface", "scale_to_m" => 1.0, "translation_m" => [0.0, 0.0, 0.0])],
            "regions" => [Dict("id" => "region:air", "name" => "Air", "kind" => "unbounded_air",
                "mesh_ids" => ["mesh:surface"], "volume_groups" => [], "sound_speed_m_per_s" => 343.0,
                "density_kg_per_m3" => 1.2041, "loss_model" => Dict())],
            "boundaries" => [Dict("id" => "boundary:source", "name" => "Source", "kind" => "moving",
                "region_id" => "region:air",
                "group" => Dict("mesh_id" => "mesh:surface", "dimension" => 2, "tag" => 2, "name" => nothing),
                "parameters" => Dict())],
            "interfaces" => [],
            "components" => [Dict("id" => "component:source", "name" => "Source", "kind" => "ideal_velocity_source",
                "boundary_ids" => ["boundary:source"], "parameters" => Dict())],
            "excitation_ports" => [Dict("id" => "port:source", "name" => "Source",
                "component_id" => "component:source", "kind" => "normal_velocity")],
        ),
        "frequencies_hz" => [1000.0],
        "excitation_port_ids" => ["port:source"],
        "outputs" => [
            Dict("id" => "p", "quantity" => "exterior_pressure", "target_ids" => [],
                "options" => Dict("points_m" => points)),
            Dict("id" => "z", "quantity" => "radiation_impedance", "target_ids" => [], "options" => Dict()),
        ],
        "solver_options" => Dict("precision" => "float32", "bem_backend" => "cpu",
            "symmetry" => symmetry),
    )
end

include(joinpath(@__DIR__, "..", "..", "CompiledExteriorWorkload.jl"))

@compile_workload begin
    # One compiled exterior request per symmetry mode, through the same
    # `solve_request(...; event_mode=true)` the worker loop calls. The CPU
    # backend needs no device on the build machine.
    directory = mktempdir()
    try
        for (name, text, symmetry) in (("off.msh", WORKLOAD_MESH_OFF, "off"), ("xy.msh", workload_plate_mesh(), "xy"))
            mesh = joinpath(directory, name)
            write(mesh, text)
            # Match JSON.parse in run_worker, including nested JSON.Object values.
            request = JSON.parse(JSON.json(symmetry == "xy" ?
                representative_workload_request(mesh) : workload_request(mesh, symmetry)))
            redirect_stdout(devnull) do
                try
                    solve_request(request; event_mode=true)
                catch exception
                    @warn "BEAT compiled exterior workload failed" symmetry exception=(exception, catch_backtrace())
                end
            end
        end
    finally
        rm(directory; force=true, recursive=true)
    end
    precompile(run_worker, ())
end

end
