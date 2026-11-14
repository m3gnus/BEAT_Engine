using Test

const METAL_LOADER = normpath(joinpath(@__DIR__, "../src/BeatEngineMetalLoad.jl"))

@testset "Failed required Metal import cannot poison a precompile cache" begin
    mktempdir() do root
        packages = joinpath(root, "packages")
        depot = joinpath(root, "depot")
        mkpath(joinpath(packages, "Metal", "src"))
        mkpath(joinpath(packages, "CacheProbe", "src"))
        write(joinpath(packages, "Metal", "src", "Metal.jl"), """
        module Metal
        get(ENV, "TEST_METAL_IMPORT_FAIL", "0") == "1" && error("injected Metal import failure")
        functional() = true
        end
        """)
        write(joinpath(packages, "CacheProbe", "src", "CacheProbe.jl"), """
        module CacheProbe
        const BEAT_ACCELERATOR_HINT = "metal"
        const METAL_MODULE = include($(repr(METAL_LOADER)))
        end
        """)
        env = copy(ENV)
        env["JULIA_LOAD_PATH"] = packages * (Sys.iswindows() ? ";" : ":") * "@stdlib"
        env["JULIA_DEPOT_PATH"] = depot
        env["JULIA_PKG_OFFLINE"] = "true"
        env["TEST_METAL_IMPORT_FAIL"] = "1"
        command = `$(Base.julia_cmd()) --startup-file=no -e 'using CacheProbe; @assert CacheProbe.METAL_MODULE.functional()'`
        output = IOBuffer()
        failed = run(pipeline(ignorestatus(setenv(command, env)), stdout=output, stderr=output))
        @test !success(failed)
        @test occursin("injected Metal import failure", String(take!(output)))
        # The exact same sources/depot recover without any manual cache deletion.
        env["TEST_METAL_IMPORT_FAIL"] = "0"
        @test success(run(ignorestatus(setenv(command, env))))
        @test success(run(ignorestatus(setenv(command, env))))
    end
end

@testset "CPU-only loading does not require Metal" begin
    cpu = Module(:CpuProbe)
    Core.eval(cpu, :(const BEAT_ACCELERATOR_HINT = "none"))
    @test Base.include(cpu, METAL_LOADER) === nothing
end

module AvailabilityProbe
const NEGATIVE_TIME_PHASOR = "negative"
const POSITIVE_TIME_PHASOR = "positive"
module Device
const state = Ref(:working)
functional() = state[] === :throw ? error("device initialization failed") : state[] === :working
end
module BeatEngineCore
using ..AvailabilityProbe: Device
const CUDA_MODULE = nothing
const AMDGPU_MODULE = nothing
const METAL_MODULE = Device
end
include(joinpath(@__DIR__, "../worker_backend_availability.jl"))
end

@testset "Readiness reflects current device state rather than cached support" begin
    for (state, ready) in ((:working, true), (:unavailable, false), (:throw, false))
        AvailabilityProbe.Device.state[] = state
        result = AvailabilityProbe.worker_backend_availability()
        @test result["cpu"]["available"]
        @test result["metal"]["available"] == ready
        @test ready || !isempty(result["metal"]["reason"])
    end
end
