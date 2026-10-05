#!/usr/bin/env julia
"""
The compiled-contract worker entry point.

Nothing but loading and dispatch lives here. The driver is in
`BeatEngineCompiledDriver.jl`, which a bundle package under `julia_engine/`
`include`s so Julia can cache its native code in a pkgimage. A script cannot be
cached -- it is parsed, lowered and compiled from source in every process.

Loading follows solver.jl: `BLAB_BEAT_ENGINE_BUNDLE=0`, an un-instantiated
checkout, or an unresolved bundle fall back to including the driver from source
(same code, compiled from source again).
"""

const BEAT_COMPILED_BUNDLE_NAME = let
    hint = lowercase(strip(get(ENV, "BLAB_BEAT_ENGINE_GPU_BACKEND", "")))
    if isempty(hint)
        active = Base.active_project()
        directory = active === nothing ? "" : lowercase(basename(dirname(active)))
        hint = directory == "julia_cuda" ? "cuda" :
            directory == "julia_rocm" ? "rocm" :
            directory == "julia_metal" ? "metal" : "cpu"
    end
    # Only the CPU and Metal compiled bundles exist so far; others take the fallback.
    hint == "metal" ? :BeatEngineCompiledMetalBundle :
        hint == "cpu" ? :BeatEngineCompiledCpuBundle : nothing
end

# Match the compiled Metal bundle's dependency load order. Loading JSON first
# invalidates its cached run_worker call graph when Metal's extensions load.
# Other backends and the explicit include fallback retain their existing order.
if BEAT_COMPILED_BUNDLE_NAME === :BeatEngineCompiledMetalBundle && get(ENV, "BLAB_BEAT_ENGINE_BUNDLE", "1") != "0"
    try
        @eval import Metal
    catch
        # An unresolved environment still takes the existing fallback below.
    end
end

using JSON

const BEAT_COMPILED_BUNDLE = if get(ENV, "BLAB_BEAT_ENGINE_BUNDLE", "1") == "0" || BEAT_COMPILED_BUNDLE_NAME === nothing
    nothing
else
    try
        @eval using $BEAT_COMPILED_BUNDLE_NAME
        @eval $BEAT_COMPILED_BUNDLE_NAME
    catch
        nothing
    end
end

if BEAT_COMPILED_BUNDLE === nothing
    include(joinpath(@__DIR__, "BeatEngineCompiledDriver.jl"))
end

const DRIVER = BEAT_COMPILED_BUNDLE === nothing ? Main : BEAT_COMPILED_BUNDLE

if "--worker" in ARGS
    try
        DRIVER.run_worker()
    catch exception
        DRIVER.reclaim_accelerator_memory!()
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
else
    try
        request = JSON.parse(read(stdin, String))
        DRIVER.solve_request(request)
    catch exception
        showerror(stderr, exception, catch_backtrace())
        println(stderr)
        exit(1)
    end
end
