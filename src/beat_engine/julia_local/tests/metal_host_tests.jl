# Everything about the Metal backend that can be checked without a Metal GPU.
#
# GitHub's macOS runners are Apple Silicon but virtualized under Apple's
# Virtualization Framework, which exposes no GPU: `Metal.functional()` is false
# there and no kernel can run. That leaves the Metal backend with no CI at all,
# so a renamed entry point or a broken control string would reach a device only
# on someone's desk. The checks below are the part that does not need a device:
# the bundle precompiles (which compiles every `BeatEngineMetal*.jl` source),
# the entry points the solver dispatches to exist, and the environment controls
# parse and reject what they should.
#
# Run this under the Metal project, not `julia_local`:
#
#     julia --project=src/beat_engine/julia_metal .../tests/metal_host_tests.jl
#
# Nothing here asserts `Metal.functional()`. Real kernel coverage needs a
# self-hosted Apple Silicon runner; see `hardware.yml`.

using Test
import Metal

include(joinpath(@__DIR__, "compiled_metal_worker_tests.jl"))

using BeatEngineMetalBundle
const Engine = BeatEngineMetalBundle.BeatEngineCore

@testset "metal bundle loads" begin
    # Metal.jl logs an error and returns rather than throwing when it cannot
    # reach a device, so a non-Apple-Silicon runner would sail through every
    # test below while testing nothing at all. Fail loudly instead.
    @test Sys.isapple()
    @test Sys.ARCH === :aarch64

    # The bundle picks the backend at precompile time; if this is not "metal"
    # then `BeatEngineCore` never included the Metal sources and the rest of
    # this file is checking the CPU build.
    @test BeatEngineMetalBundle.BEAT_ENGINE_BACKEND == "metal"

    @info "Metal device availability" functional = Metal.functional()
end

@testset "metal entry points are defined" begin
    # The solver reaches these by name from the `backend == :metal` branches.
    # A rename that misses one is a MethodError on a device and nothing here.
    for name in (
        :assemble_regular_galerkin_operators_metal_regular,
        :build_metal_regular_assembly_cache,
        :release_metal_regular_assembly_cache!,
        :build_metal_singular_correction_cache,
        :release_metal_singular_correction_cache!,
        :assemble_burton_miller_neumann_system_metal,
        :release_metal_burton_miller_system!,
        :metal_host_operators,
    )
        @test isdefined(Engine, name)
    end
end

@testset "metal operator storage mode" begin
    withenv("BLAB_METAL_OPERATOR_STORAGE" => nothing) do
        @test Engine.metal_operator_storage_mode() === Metal.SharedStorage
    end
    withenv("BLAB_METAL_OPERATOR_STORAGE" => "shared") do
        @test Engine.metal_operator_storage_mode() === Metal.SharedStorage
    end
    withenv("BLAB_METAL_OPERATOR_STORAGE" => "private") do
        @test Engine.metal_operator_storage_mode() === Metal.PrivateStorage
    end
    withenv("BLAB_METAL_OPERATOR_STORAGE" => "PRIVATE") do
        @test Engine.metal_operator_storage_mode() === Metal.PrivateStorage
    end
    withenv("BLAB_METAL_OPERATOR_STORAGE" => "unified") do
        @test_throws ErrorException Engine.metal_operator_storage_mode()
    end
end

@testset "metal kernel groupsize" begin
    withenv("BLAB_METAL_KERNEL_GROUPSIZE" => nothing) do
        @test Engine._metal_kernel_groupsize() == 256
    end
    for size in (32, 64, 128, 256, 512, 1024)
        withenv("BLAB_METAL_KERNEL_GROUPSIZE" => string(size)) do
            @test Engine._metal_kernel_groupsize() == size
        end
    end
    # Apple executes in SIMD groups of 32; a non-multiple must be refused
    # rather than silently rounded, because a wrong launch geometry shows up
    # as wrong numbers and not as an error.
    withenv("BLAB_METAL_KERNEL_GROUPSIZE" => "100") do
        @test_throws ErrorException Engine._metal_kernel_groupsize()
    end
    withenv("BLAB_METAL_KERNEL_GROUPSIZE" => "2048") do
        @test_throws ErrorException Engine._metal_kernel_groupsize()
    end
end

@testset "metal regular kernel mode aliases" begin
    withenv("BLAB_METAL_REGULAR_KERNEL_MODE" => nothing) do
        @test Engine._normalized_metal_regular_kernel_mode() === :pair_gather
    end
    aliases = (
        "gather" => :pair_gather,
        "pair_gather" => :pair_gather,
        "chunked" => :pair_gather,
        "chunked_pair_gather" => :pair_gather,
        "atomic" => :pair_atomic,
        "pair_atomic" => :pair_atomic,
        "fused_atomic" => :pair_atomic,
        "pair" => :pair_owned,
        "pair_owned" => :pair_owned,
        "colored" => :pair_owned,
        "colored_pair_owned" => :pair_owned,
        "entry" => :entry_owned,
        "entry_owned" => :entry_owned,
    )
    for (value, expected) in aliases
        @test Engine._normalized_metal_regular_kernel_mode(value) === expected
        @test Engine._normalized_metal_regular_kernel_mode(uppercase(value)) === expected
    end
    @test_throws ErrorException Engine._normalized_metal_regular_kernel_mode("warp")
end

@testset "metal singular mode and writeback" begin
    withenv("BLAB_METAL_SINGULAR_MODE" => nothing) do
        @test Engine._normalized_metal_singular_mode() === :native
    end
    for (value, expected) in ("native" => :native, "device" => :native,
                              "host" => :host, "cpu" => :host)
        @test Engine._normalized_metal_singular_mode(value) === expected
    end
    @test_throws ErrorException Engine._normalized_metal_singular_mode("duffy")

    # The gather is the reproducible write-back and so the default; the scatter
    # is kept only so the two can be compared on one mesh.
    withenv("BLAB_METAL_SINGULAR_WRITEBACK" => nothing) do
        @test Engine._normalized_metal_singular_writeback() === :gather
    end
    @test Engine._normalized_metal_singular_writeback("gather") === :gather
    @test Engine._normalized_metal_singular_writeback("scatter") === :scatter
    @test_throws ErrorException Engine._normalized_metal_singular_writeback("atomic")
end
