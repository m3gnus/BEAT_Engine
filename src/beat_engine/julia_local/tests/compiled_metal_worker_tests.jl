using Test, JSON
import BeatEngineCompiledMetalBundle

# First, before anything in this process can look it up: the package image must not carry the
# precompile workload's "already looked up" flag without its (nulled) pointer.
@testset "Accelerate zgemm is looked up afresh in a new worker" begin
    cc = BeatEngineCompiledMetalBundle.BeatEngineCoupledCondensed
    @test !cc._ACCELERATE_ZGEMM_LOOKED_UP[]
    @test cc._ACCELERATE_ZGEMM[] == C_NULL
    if Sys.isapple() && Sys.ARCH === :aarch64
        withenv("BLAB_COUPLED_HOST_ZGEMM" => nothing) do
            @test cc.host_zgemm_path() == "accelerate"
        end
        withenv("BLAB_COUPLED_HOST_ZGEMM" => "accelerate") do
            @test cc._host_zgemm_symbol() != C_NULL
        end
    end
end

@testset "Metal runtime inventory follows Channel's task wrapper" begin
    wrapper = BeatEngineCompiledMetalBundle.metal_channel_task_wrapper_type()
    @test fieldnames(Base.unwrap_unionall(wrapper)) == (:func, :chnl)
    taskref = Ref{Task}()
    Channel{Tuple{Int64,Any}}(_ -> nothing; taskref)
    @test Base.typename(typeof(taskref[].code)).wrapper === wrapper
end

@testset "coupled Metal inventory is compile-only and structurally resolved" begin
    bundle = BeatEngineCompiledMetalBundle
    cc = bundle.BeatEngineCoupledCondensed
    mumps = cc.BeatEngineMumps
    library_before = mumps.LIBRARY[]
    solvers_before = copy(mumps.LIVE_SOLVERS)
    types = bundle.metal_coupled_types()
    closures = bundle.metal_coupled_closure_types()
    # The package build skips an unresolved closure; the test insists every one resolves.
    @test all(closure -> closure !== nothing, values(closures))
    host = bundle.metal_coupled_host_signatures()
    runtime = bundle.metal_coupled_runtime_signatures()
    @test !isempty(host) && !isempty(runtime)
    @test all(isconcretetype, values(types))
    @test fieldtype(types.cache, :base) === types.base
    @test fieldtype(types.system, :cache) === types.cache
    @test fieldtype(types.system, :condensation) === types.condensation
    @test fieldtype(types.system, :factorization) === cc.RefinedDenseLU
    @test fieldtype(types.condensation, :mumps_solver) === mumps.MumpsSchurSolver
    @test fieldtype(types.base, :device_cache) ===
          bundle.BeatEngineCore.MetalRegularAssemblyCache{Float32,Nothing}
    @test fieldtype(closures.timed_flux, :flux_rhs_solution) === Core.Box
    @test fieldtype(closures.solution_parts, :system) === types.system
    @test fieldtype(closures.fem_task, :fem_stage) === closures.fem_stage
    @test fieldtype(closures.fem_stage, :dense_type) === Type{Float64}
    @test fieldtype(closures.fem_stage, :fem_system) === Core.Box
    # These must join the inventories that the existing strict precompile gate
    # checks, rather than live in an unused helper.
    all_host = bundle.metal_host_signatures()
    all_runtime = bundle.metal_runtime_signatures()
    @test all(signature -> signature in all_host, host)
    @test all(signature -> signature in all_runtime, runtime)
    @test any(((f, args),) -> f === Core.kwcall && args[2] ===
              typeof(cc.build_condensed_coupled_system), host)
    @test any(((f, args),) -> f === Core.kwcall && args[2] ===
              typeof(cc.solve_condensed_coupled_excitations), host)
    @test (cc.release_condensed_coupled_system!, (types.system,)) in host
    @test Tuple{typeof(collect), Base.Generator{Base.OneTo{Int},closures.solution_parts}} in runtime
    @test mumps.LIBRARY[] === library_before
    @test mumps.LIVE_SOLVERS == solvers_before
end

@testset "warm exterior Metal field calls stay in the runtime inventory" begin
    bundle = BeatEngineCompiledMetalBundle
    core = bundle.BeatEngineCore
    # Derive the batch descriptors from the production builders without a solve
    # or GPU buffers; the inventory must match the types the driver passes.
    outputs = bundle.workload_request("unused.msh", "off")["outputs"]
    points_by_output = bundle.parse_field_output_points(outputs, Float32)
    field_batch = bundle.exterior_field_output_batch(outputs, points_by_output, :metal)
    metal_batch = bundle.exterior_metal_field_output_batch(outputs, points_by_output, :metal)
    @test field_batch === nothing
    @test metal_batch !== nothing
    runtime = bundle.metal_runtime_signatures()
    @test Tuple{typeof(core.prepare_metal_field_output_points), typeof(metal_batch.point_sets), Int} in runtime
    @test Tuple{typeof(core._metal_field_output_partition!), core.MetalFieldOutputPoints, Int} in runtime
    host = bundle.metal_host_signatures()
    for (f, tt) in bundle.metal_kernel_signatures()
        f in (core._metal_fast_field_outputs_kernel!, core._metal_fast_field_outputs_reduce_kernel!) || continue
        @test any(signature -> signature[1] === Core.kwcall &&
                  signature[2][2] === bundle.Metal.HostKernel{typeof(f),tt}, host)
    end
    @test Tuple{typeof(Core.kwcall),
        NamedTuple{(:cpu_kernel, :metal_field_points), Tuple{Symbol, core.MetalFieldOutputPoints}},
        typeof(bundle.exterior_output_fields!), Dict{String,Any}, String, typeof(points_by_output),
        core.BoundaryMesh{Float32}, Vector{ComplexF32}, Vector{ComplexF32}, Float32,
        core.MetalFieldEvaluationCache{Float32}, Symbol, typeof(field_batch), typeof(metal_batch)} in runtime
    @test Tuple{typeof(Core.kwcall),
        NamedTuple{(:point_cache,), Tuple{core.MetalFieldOutputPoints}},
        typeof(core.evaluate_galerkin_field_metal_outputs), typeof(metal_batch.point_sets),
        Vector{ComplexF32}, Vector{ComplexF32}, Float32, core.MetalFieldEvaluationCache{Float32}} in runtime
end

@testset "Metal host workload has matching compile-only methods" begin
    signatures = BeatEngineCompiledMetalBundle.metal_host_signatures()
    @test !isempty(signatures)
    runtime_signatures = BeatEngineCompiledMetalBundle.metal_runtime_signatures()
    @test !isempty(runtime_signatures)
    for signature in runtime_signatures
        @test precompile(signature)
    end
    for (f, args) in signatures
        @test precompile(f, args)
    end
end

@testset "compiled Metal worker uses its bundle" begin
    # Load the actual entry point in a fresh process without preloading the
    # bundle. EOF lets its worker loop finish before checking the dispatch.
    mktempdir() do directory
        entry = normpath(joinpath(@__DIR__, "..", "coupled_solver.jl"))
        wrapper = joinpath(directory, "worker_bundle_test.jl")
        write(wrapper, """
            using Test
            include($(repr(entry)))
            @test BEAT_COMPILED_BUNDLE_NAME === :BeatEngineCompiledMetalBundle
            @test BEAT_COMPILED_BUNDLE !== nothing
            @test DRIVER === BeatEngineCompiledMetalBundle
            @test !isdefined(Main, :BeatEngineCore)
            mumps = DRIVER.BeatEngineCoupledCondensed.BeatEngineMumps
            @test mumps.LIBRARY[] === nothing
            @test isempty(mumps.LIVE_SOLVERS)
            @test !mumps.ATEXIT_REGISTERED[]
            # The first use in this fresh worker must load/self-test the JLL
            # and restore LP64 forwarding, rather than reuse image pointers.
            library = mumps.mumps_library()
            @test library.available
            @test library.version == mumps.MUMPS_LAYOUT_VERSION
            @test any(lib -> lib.interface == :lp64, DRIVER.BLAS.get_config().loaded_libs)
            @test mumps.ATEXIT_REGISTERED[]
            mumps.reset_precompile_state!()
            """)
        project = dirname(Base.active_project())
        command = addenv(`$(Base.julia_cmd()) --threads=2 --startup-file=no --project=$project $wrapper --worker`,
            "BLAB_BEAT_ENGINE_GPU_BACKEND" => "metal", "BLAB_BEAT_ENGINE_BUNDLE" => "1")
        output = read(pipeline(command; stdin=devnull), String)
        ready = JSON.parse(first(split(output, '\n')))
        @test ready["type"] == "ready"
        @test ready["compiled_worker"]["loaded_bundle"] == "BeatEngineCompiledMetalBundle"
        @test ready["compiled_worker"]["fallback_reason"] === nothing
        @test ready["contracts"]["compiled_system"] == [1, 2, 3]
        @test ready["runtime"]["julia_threads"] == 2
        @test ready["runtime"]["project_file"] == Base.active_project()
    end
end

@testset "Metal bundle's coupled workload reaches MUMPS (strict)" begin
    # The precompile wrapper catches and logs failures so installation never breaks; this calls
    # the inner solve and check directly so a fallback away from MUMPS fails the test instead.
    bundle = BeatEngineCompiledMetalBundle
    request = bundle.JSON.parse(bundle.JSON.json(bundle.coupled_workload_request(; tiny=true)))
    withenv(bundle.coupled_workload_environment(; mumps=true)...) do
        run = bundle.solve_coupled_workload(request)
        bundle.check_coupled_workload(run; mumps=true)
        @test all(result["diagnostics"]["fem_condensation_backend"] == "mumps_seq" for result in run.results)
    end
    bundle.reset_compiled_workload_state!()
    mumps = bundle.BeatEngineCoupledCondensed.BeatEngineMumps
    @test mumps.LIBRARY[] === nothing
    @test isempty(mumps.LIVE_SOLVERS)
    @test !bundle.BeatEngineCoupledCondensed._ACCELERATE_ZGEMM_LOOKED_UP[]
    @test !mumps.ATEXIT_REGISTERED[]
end
