using Test, JSON
import BeatEngineCompiledMetalBundle

@testset "Metal runtime inventory follows Channel's task wrapper" begin
    wrapper = BeatEngineCompiledMetalBundle.metal_channel_task_wrapper_type()
    @test fieldnames(Base.unwrap_unionall(wrapper)) == (:func, :chnl)
    taskref = Ref{Task}()
    Channel{Tuple{Int64,Any}}(_ -> nothing; taskref)
    @test Base.typename(typeof(taskref[].code)).wrapper === wrapper
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
            """)
        project = dirname(Base.active_project())
        command = addenv(`$(Base.julia_cmd()) --threads=2 --startup-file=no --project=$project $wrapper --worker`,
            "BLAB_BEAT_ENGINE_GPU_BACKEND" => "metal", "BLAB_BEAT_ENGINE_BUNDLE" => "1")
        output = read(pipeline(command; stdin=devnull), String)
        ready = JSON.parse(first(split(output, '\n')))
        @test ready["type"] == "ready"
        @test ready["contracts"]["compiled_system"] == [1]
        @test ready["runtime"]["julia_threads"] == 2
        @test ready["runtime"]["project_file"] == Base.active_project()
    end
end
