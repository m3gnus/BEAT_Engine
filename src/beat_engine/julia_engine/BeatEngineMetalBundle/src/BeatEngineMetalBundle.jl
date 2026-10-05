"""
    BeatEngineMetalBundle

A precompilable home for the BEAT engine's Metal build.

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
module BeatEngineMetalBundle

using PrecompileTools: @compile_workload

const BEAT_ENGINE_BACKEND = "metal"

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

@compile_workload begin
    # Solve one frequency on the CPU backend. Running a whole request is the
    # only way to reach the driver's real call graph, and that graph -- not
    # loading the engine -- was the largest term in a cold start.
    #
    # The CPU backend even in a GPU bundle, deliberately. It compiles
    # everything up to and including the backend branch, it is the same code a
    # GPU solve runs to get there, and it needs no device: precompilation runs
    # in a sandboxed worker on a build machine that may have no accelerator at
    # all. This source-entry bundle retains its host-only workload. The
    # compiled-contract Metal bundle also caches device code with Metal 1.11.1;
    # see BeatEngineCompiledMetalBundle for its compile-only inventory.
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
