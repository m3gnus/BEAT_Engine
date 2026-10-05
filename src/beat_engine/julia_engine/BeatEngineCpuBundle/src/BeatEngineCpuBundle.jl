"""
    BeatEngineCpuBundle

A precompilable home for the BEAT engine's CPU build.

The engine (`julia_local/src`) and the worker driver
(`julia_local/BeatEngineDriver.jl`) were, until this package existed, pulled
into `Main` with `include` on every worker start. `include` compiles from
source every time: neither Julia's pkgimage cache nor a PackageCompiler
sysimage can see code that is not in a package, which is why the sysimage in
`docker/` never captured the engine, and why a cold worker spent most of its
start-up compiling before it could accept a job.

Loading the same sources from a package fixes that, and fixes it without
anything that can go stale. Julia records every `include`d file in the cache
header and rechecks them on load, so editing an engine source invalidates the
pkgimage automatically and the next process rebuilds it. A hand-built sysimage
has no such property, which is why this is the primary mechanism and a
sysimage is an optional extra layered on top of it.

There is one package per accelerator because a package gets one precompile
cache and the engine chooses its backend while it loads. `BEAT_ENGINE_BACKEND`
is what `BeatEngineCore` consults in place of the process environment: the
environment is read when the cache is *built*, so a bundle has to state its
backend rather than inherit whatever the building process happened to have set.
"""
module BeatEngineCpuBundle

using PrecompileTools: @compile_workload

const BEAT_ENGINE_BACKEND = "cpu"

#: Where the engine and driver sources are, found rather than assumed.
#: Boundary Lab calls that directory `julia_local`; the Python wrapper that
#: vendors these sources flattens it to `julia`. Looking for the directory
#: that actually holds `BeatEngineCore.jl` means re-vendoring is a file copy
#: and not a path rewrite -- and a path rewrite that was missed would fail at
#: precompile time in the shipped package, which is the worst place to find it.
const ENGINE_DIR = let root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    found = nothing
    for name in ("julia_local", "julia")
        candidate = joinpath(root, name)
        if isfile(joinpath(candidate, "src", "BeatEngineCore.jl"))
            found = candidate
            break
        end
    end
    found === nothing && error("No BEAT engine sources found under $(root).")
    found
end

include(joinpath(ENGINE_DIR, "src", "BeatEngineCore.jl"))

using .BeatEngineCore

include(joinpath(ENGINE_DIR, "BeatEngineDriver.jl"))

#: Four triangles, one closed tetrahedron, every face tagged as the source.
#: The workload below only needs a mesh that solves; nothing it compiles
#: depends on the geometry.
const WORKLOAD_MESH = """
\$MeshFormat
2.2 0 8
\$EndMeshFormat
\$PhysicalNames
1
2 2 "warmup"
\$EndPhysicalNames
\$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
\$EndNodes
\$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
\$EndElements
"""

#: A 3x3 plate in the positive x/y quadrant, every face tagged as
#: the source. It is what the second workload request below solves with the
#: `xy` symmetry, so the symmetry-image, image-singular and non-adjacent regular
#: paths are compiled too; the tetrahedron above has none of the three.
const WORKLOAD_QUADRANT_MESH = """
\$MeshFormat
2.2 0 8
\$EndMeshFormat
\$Nodes
16
1 0.000 0.000 0.100
2 0.013 0.000 0.100
3 0.026 0.000 0.100
4 0.039 0.000 0.100
5 0.000 0.013 0.100
6 0.013 0.013 0.100
7 0.026 0.013 0.100
8 0.039 0.013 0.100
9 0.000 0.026 0.100
10 0.013 0.026 0.100
11 0.026 0.026 0.100
12 0.039 0.026 0.100
13 0.000 0.039 0.100
14 0.013 0.039 0.100
15 0.026 0.039 0.100
16 0.039 0.039 0.100
\$EndNodes
\$Elements
18
1 2 2 2 2 1 2 6
2 2 2 2 2 1 6 5
3 2 2 2 2 2 3 7
4 2 2 2 2 2 7 6
5 2 2 2 2 3 4 8
6 2 2 2 2 3 8 7
7 2 2 2 2 5 6 10
8 2 2 2 2 5 10 9
9 2 2 2 2 6 7 11
10 2 2 2 2 6 11 10
11 2 2 2 2 7 8 12
12 2 2 2 2 7 12 11
13 2 2 2 2 9 10 14
14 2 2 2 2 9 14 13
15 2 2 2 2 10 11 15
16 2 2 2 2 10 15 14
17 2 2 2 2 11 12 16
18 2 2 2 2 11 16 15
\$EndElements
"""

@compile_workload begin
    # Solve one frequency on the CPU backend. Running a whole request is the
    # only way to reach the driver's real call graph, and that graph -- not
    # loading the engine -- was the largest term in a cold start.
    #
    # The CPU backend even in a GPU bundle, deliberately. It compiles
    # everything up to and including the backend branch, it is the same code a
    # GPU solve runs to get there, and it needs no device: precompilation runs
    # in a sandboxed worker on a build machine that may have no accelerator at
    # all. What a GPU workload would add is its own kernel compilation, and
    # that cannot be cached to disk in any case -- it is why the worker is kept
    # alive between solves.
    directory = mktempdir()
    try
        mesh = joinpath(directory, "workload.msh")
        write(mesh, WORKLOAD_MESH)
        request = Dict{String,Any}(
            "schema_version" => 2,
            "beat_engine_backend" => "cpu",
            "frequencies_hz" => [1000.0],
            "config" => Dict{String,Any}(
                "mesh_file" => mesh,
                "scale_factor" => 1.0,
                "distance" => 1.0,
                "axial_offset" => 0.0,
                "step_size" => 90.0,
                "min_angle" => 0.0,
                "max_angle" => 90.0,
                "freq_min" => 1000.0,
                "freq_max" => 1000.0,
                "freq_count" => 1,
                "tag_throat" => 2,
                "rho" => 1.2041,
                "sound_speed" => 343.0,
                "symmetry" => "off",
                "source_motion" => "normal",
            ),
        )
        redirect_stdout(devnull) do
            try
                solve_request(request)
                # The worker hands `solve_request` what
                # `JSON.parse` returns, not a Dict, and the driver specialises
                # on the request type. Solving the parsed form too caches the
                # native code the worker actually calls.
                solve_request(JSON.parse(JSON.json(request)))
            catch
                # A workload that cannot solve still leaves everything it did
                # reach compiled, and a build must not fail over an
                # optimisation.
            end
            try
                # A request shaped like a real application solve, as the
                # worker receives it. Two frequencies, so both the
                # order-2 and the order-4 regular rule run; the xy symmetry,
                # sphere grid, diagonal cut and surface traces are each code
                # the tetrahedron request never reaches. Measured on a real
                # first solve: most of what was still compiled at run time.
                quadrant = joinpath(directory, "quadrant.msh")
                write(quadrant, WORKLOAD_QUADRANT_MESH)
                realistic = Dict{String,Any}(
                    "schema_version" => 2,
                    "beat_engine_backend" => "cpu",
                    "frequencies_hz" => [1000.0, 20000.0],
                    "config" => Dict{String,Any}(
                        "mesh_file" => quadrant,
                        "scale_factor" => 1.0,
                        "meshes" => [Dict{String,Any}(
                            "name" => "mesh", "file" => quadrant, "scale_factor" => 1.0,
                            "translation_m" => [0.0, 0.0, 0.0],
                        )],
                        "distance" => 2.0,
                        "axial_offset" => 0.0,
                        "step_size" => 45.0,
                        "min_angle" => 0.0,
                        "max_angle" => 180.0,
                        "freq_min" => 1000.0,
                        "freq_max" => 20000.0,
                        "freq_count" => 2,
                        "tag_throat" => 2,
                        "rho" => 1.2041,
                        "sound_speed" => 343.0,
                        "symmetry" => "xy",
                        "flat_target_normalization_enabled" => false,
                        "spherical_sampling_enabled" => false,
                        "source_motion" => "normal",
                        "quadrature_order" => 4,
                        "singular_order" => 4,
                        "diagonal_enabled" => true,
                        "diagonal_inclination_deg" => 45.0,
                        "spherical_grid" => Dict{String,Any}(
                            "theta_count" => 3, "phi_count" => 4, "theta_max_deg" => 180.0,
                        ),
                        "surface_traces_enabled" => true,
                    ),
                )
                solve_request(JSON.parse(JSON.json(realistic)))
            catch
                # A workload that cannot solve still leaves everything it did
                # reach compiled, and a build must not fail over an
                # optimisation.
            end
        end
    finally
        rm(directory; force=true, recursive=true)
    end

    # The worker's own entry path never runs here -- it reads stdin -- so ask
    # for it by signature. Compiling `worker_loop` is what forces inference
    # through the dynamic `solve_request` call it makes.
    precompile(worker_loop, ())
    precompile(main, (Vector{String},))
end

end
