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

@testset "coupled host launches restore the production storage modes" begin
    bundle = BeatEngineCompiledMetalBundle
    core = bundle.BeatEngineCore
    launches = bundle.metal_coupled_launch_types()
    runtime = bundle.metal_coupled_runtime_signatures()
    rules = bundle.metal_launch_rules()
    # Four-operator path, then the combined A/C path (the Metal coupled default). Pinned
    # here, not read from the table, so a rule dropped from the table fails this test.
    expected_coupled = Set((
        core._metal_regular_pair_blocks_kernel!, core._metal_gather_slp_adjoint_kernel!,
        core._metal_gather_dlp_hyp_kernel!, core._metal_singular_fused_blocks_kernel!,
        core._metal_singular_pair_gather_kernel!,
        core._metal_fused_pair_blocks_kernel!, core._metal_coupled_flux_gather_kernel!,
        core._metal_fused_lhs_gather_kernel!, core._metal_fused_singular_packed_kernel!,
        core._metal_singular_entry_gather_kernel!))
    coupled_rules = Set(f for (f, rule) in rules if :coupled in rule.paths)
    # Every coupled kernel in the shared launch table has generated device signatures.
    @test Set(f for (f, _, _, _) in launches) == coupled_rules
    @test coupled_rules == expected_coupled
    @test core._metal_coupled_flux_gather_kernel! in keys(rules)
    # The exterior host inventory takes its launches from the same table. Some exterior
    # rules cover diagnostic modes the generated inventory does not record, so this is a subset.
    exterior = Set(f for (f, _, _, _) in bundle.metal_launch_types(:exterior))
    exterior_rules = Set(f for (f, rule) in rules if :exterior in rule.paths)
    @test !isempty(exterior) && exterior ⊆ exterior_rules
    # Production exterior kernels must be both in the table and in the generated inventory.
    production_exterior = Set((
        core._metal_fused_pair_blocks_kernel!, core._metal_fused_lhs_gather_kernel!,
        core._metal_fused_singular_packed_kernel!, core._metal_fast_field_kernel!))
    @test production_exterior ⊆ exterior_rules
    @test production_exterior ⊆ exterior
    pair_orders = Set{Int}()
    for (f, tt, args, size) in launches
        @test length(args) == fieldcount(tt)
        @test all(isconcretetype, args)
        rule = rules[f]
        for (index, device_type) in enumerate(fieldtypes(tt))
            host_type = args[index]
            if device_type <: Metal.MtlDeviceArray
                storage = index in rule.shared ? Metal.SharedStorage : Metal.PrivateStorage
                @test host_type === Metal.MtlArray{eltype(device_type),ndims(device_type),storage}
            else
                @test host_type === device_type
            end
        end
        @test size === rule.grid
        f === core._metal_regular_pair_blocks_kernel! && push!(pair_orders, args[17].parameters[1])
        kernel = Metal.HostKernel{typeof(f),tt}
        captures = (; groups=size, threads=size, queue=Nothing, submit=Bool,
                    kernel=kernel, args=Tuple{args...})
        closure = bundle.metal_captured_closure_type(Metal, captures)
        @test Set(fieldnames(closure)) == Set(keys(captures))
        @test all(name -> fieldtype(closure, name) === getproperty(captures, name), keys(captures))
        signature = Tuple{Type{Metal.ObjectiveC.Foundation.NSAutoreleasePool},closure}
        @test signature in runtime
    end
    @test pair_orders == Set((1, 3, 6))
end

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
        :evaluate_galerkin_field_metal_multi,
        :assemble_coupled_burton_miller_metal,
        :metal_host_coupled_burton_miller,
        :release_metal_coupled_burton_miller!,
        :metal_coupled_combined_support_reason,
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

@testset "metal multi-drive input validation without a device" begin
    for T in (Float32, Float64)
        mesh = Engine.load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", "two_tetrahedra.msh"), one(T))
        # Validation and the empty-point fallback must not dereference device data.
        cache = Engine.MetalFieldEvaluationCache{T}(nothing, nothing, nothing, nothing, nothing, nothing, 0)
        pressure = zeros(Complex{T}, length(mesh.vertices))
        neumann = zeros(Complex{T}, length(mesh.faces))
        for points in ([], [(one(T), zero(T), one(T))]), nd in (1, 2)
            pressures, neumanns = fill(pressure, nd), fill(neumann, nd)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(points, mesh, pressures, neumanns[1:end-1], one(T), cache)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(points, mesh, [pressure[1:end-1]], [neumann], one(T), cache)
            @test_throws DimensionMismatch Engine.evaluate_galerkin_field_metal_multi(points, mesh, [pressure], [neumann[1:end-1]], one(T), cache)
        end
        @test isempty(Engine.evaluate_galerkin_field_metal_multi([], mesh, [], [], one(T), cache))
        @test length(Engine.evaluate_galerkin_field_metal_multi([], mesh, fill(pressure, 9), fill(neumann, 9), one(T), cache)) == 9
    end
end

include(joinpath(@__DIR__, "metal_coupled_host_tests.jl"))

@testset "combined coupled Metal support probes need no GPU" begin
    @test Engine.metal_coupled_combined_support_reason(nothing, Float64) ==
          "combined Metal assembly requires Float32"
    withenv("BLAB_METAL_ASSEMBLY_MODE" => "host_staged") do
        @test Engine.metal_coupled_combined_support_reason(nothing, Float32) == "host-staged assembly requested"
    end
    withenv("BLAB_METAL_ASSEMBLY_MODE" => "native", "BLAB_METAL_REGULAR_KERNEL_MODE" => "entry_owned") do
        @test Engine.metal_coupled_combined_support_reason(nothing, Float32) == "reference regular kernel mode requested"
    end
    withenv("BLAB_METAL_ASSEMBLY_MODE" => "native", "BLAB_METAL_REGULAR_KERNEL_MODE" => "pair_gather",
            "BLAB_METAL_SINGULAR_MODE" => "host") do
        @test Engine.metal_coupled_combined_support_reason(nothing, Float32) == "host singular mode requested"
    end
    withenv("BLAB_METAL_ASSEMBLY_MODE" => "native", "BLAB_METAL_REGULAR_KERNEL_MODE" => "pair_gather",
            "BLAB_METAL_SINGULAR_MODE" => "native", "BLAB_METAL_SINGULAR_WRITEBACK" => "scatter") do
        @test Engine.metal_coupled_combined_support_reason(nothing, Float32) == "singular scatter write-back requested"
    end
end
