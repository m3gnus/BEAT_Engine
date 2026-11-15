# Hardware gate; run under julia_metal on Apple Silicon. Guards the compile-only coupled Metal
# coverage against drift: the signature inventory is hand-built from production types, so if a
# production cache, system or solution type changes, the old signatures still `precompile` but no
# longer match what the worker calls, and the first coupled request quietly compiles again. This
# runs the packaged coupled workload request with Metal BEM assembly and production defaults through
# the worker entry (`coupled_solver.jl`) in a fresh process under `--trace-compile`, and fails when
# coupled-path compilation exceeds a budget. A second run through the source fallback must exceed it.
using Test, JSON
using BeatEngineCompiledMetalBundle

const Bundle = BeatEngineCompiledMetalBundle
# Milliseconds of first-request compilation allowed for the coupled/Metal-host statements below.
# With the inventory current it measured well under this on an M1 Max (the whole first request
# compiled ~0.9 s outside this filter); a stale inventory puts several seconds back.
const COUPLED_COMPILE_BUDGET_MS = parse(Float64, get(ENV, "BLAB_COUPLED_COMPILE_BUDGET_MS", "1500"))
const COUPLED_PATTERNS = (
    "BeatEngineCoupledCondensed.", "BeatEngineInterfaceRadiation.", "metal_host_operators",
    "burton_miller_neumann_matrices", "_launch_metal_", "_apply_metal_operator_p1_row_weights!",
    "release_operator_storage!",
)

# Trace one fresh worker solving the packaged coupled request with Metal BEM assembly. `bundle=true`
# loads the compiled bundle (what a worker does); `bundle=false` takes the supported source fallback
# (`BLAB_BEAT_ENGINE_BUNDLE=0`), which compiles the coupled path at run time.
function coupled_compile_trace(directory, tag; bundle::Bool)
    # Boundary Lab coupled projects also request interface-radiated pressure.
    request = Bundle.coupled_workload_request(; bem_backend="metal", radiation=true)
    request_path = joinpath(directory, "request-$tag.json")
    write(request_path, JSON.json(request))
    trace = joinpath(directory, "trace-$tag.jl")
    project = dirname(Base.active_project())
    entry = normpath(joinpath(@__DIR__, "..", "coupled_solver.jl"))
    command = `$(Base.julia_cmd()) --threads=2 --startup-file=no --project=$project --trace-compile=$trace --trace-compile-timing $entry`
    output = read(pipeline(addenv(command, "BLAB_BEAT_ENGINE_GPU_BACKEND" => "metal",
            "BLAB_BEAT_ENGINE_BUNDLE" => bundle ? "1" : "0"); stdin=request_path), String)
    results = count(line -> startswith(line, "{") && occursin("\"freq_hz\"", line), split(output, '\n'))
    results == 2 || error("coupled request returned $results of 2 results ($tag)")
    all_statements = 0
    offenders = Tuple{Float64,String}[]
    for line in eachline(trace)
        m = match(r"^#=\s*([\d.]+)\s*ms\s*=#\s*precompile\((.*)\)\s*$", strip(line))
        m === nothing && continue
        all_statements += 1
        any(pattern -> occursin(pattern, m[2]), COUPLED_PATTERNS) || continue
        push!(offenders, (parse(Float64, m[1]), m[2]))
    end
    sort!(offenders; rev=true)
    return (all_statements=all_statements, offenders=offenders, total=sum(first, offenders; init=0.0))
end

@testset "coupled Metal first request compiles within budget" begin
    mktempdir() do directory
        cached = coupled_compile_trace(directory, "cached"; bundle=true)
        for (ms, statement) in cached.offenders[1:min(end, 10)]
            println(round(Int, ms), " ms  ", first(statement, 200))
        end
        println("coupled-path first-request compilation: ", round(Int, cached.total), " ms in ",
            length(cached.offenders), " of ", cached.all_statements, " traced statements (budget ",
            round(Int, COUPLED_COMPILE_BUDGET_MS), " ms)")
        # The trace must not be empty: a fresh worker always compiles something.
        @test cached.all_statements > 0
        @test cached.total <= COUPLED_COMPILE_BUDGET_MS
        # Negative control: the source fallback compiles the coupled path at run time, so the filter
        # must catch it there; otherwise this gate could never fail.
        uncached = coupled_compile_trace(directory, "source"; bundle=false)
        println("control through the source fallback: ", round(Int, uncached.total), " ms in ",
            length(uncached.offenders), " coupled statements")
        @test uncached.total > COUPLED_COMPILE_BUDGET_MS
    end
end
