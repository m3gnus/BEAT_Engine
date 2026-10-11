using Test, JSON

include(joinpath(@__DIR__, "..", "src", "BeatEngineContract.jl"))
using .BeatEngineContract

const CONTRACT_CORPUS = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "conformance.json"))

@testset "BEAT compiled-system wire conformance" begin
    for case in CONTRACT_CORPUS["cases"]
        @testset "$(case["name"])" begin
            request = deepcopy(CONTRACT_CORPUS["base_request"])
            for change in case["changes"]
                parent = request
                for key in change["path"][1:end-1]
                    parent = parent[key isa Integer ? key + 1 : key]
                end
                key = last(change["path"])
                key = key isa Integer ? key + 1 : key
                if get(change, "remove", false)
                    delete!(parent, key)
                else
                    parent[key] = change["value"]
                end
            end
            if case["valid"]
                @test validate_system_request(request) === nothing
            else
                if haskey(case, "expected_message")
                    @test_throws case["expected_message"] validate_system_request(request)
                else
                    @test_throws ErrorException validate_system_request(request)
                end
            end
        end
    end
    for value in (NaN, Inf)
        request = deepcopy(CONTRACT_CORPUS["base_request"])
        request["solver_options"]["custom"] = value
        @test_throws ErrorException validate_system_request(request)
    end
end

@testset "compiled v2 axial source and retained v1 normal source" begin
    request = deepcopy(CONTRACT_CORPUS["base_request"])
    @test validate_system_request(request) === nothing
    request["compiled_system"]["contract_version"] = 2
    @test validate_system_request(request) === nothing
    source = request["compiled_system"]["components"][1]
    source["parameters"] = Dict("motion_profile" => "rigid_translation", "motion_axis" => [0, 0, 1])
    @test validate_system_request(request) === nothing
    request["compiled_system"]["contract_version"] = 1
    @test_throws ErrorException validate_system_request(request)
    source["parameters"] = Dict("motion_axis" => [0, 0, 1])
    @test_throws ErrorException validate_system_request(request)
end

@testset "BEAT identity without Git" begin
    mktempdir() do directory
        source = joinpath(@__DIR__, "..", "src", "BeatEngineProvenance.jl")
        # Keep the copied engine and contract together in this isolated fixture.
        fixture_root = joinpath(directory, "engine")
        mkpath(joinpath(fixture_root, "src"))
        cp(source, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        mkpath(joinpath(directory, "beat_contract"))
        first_module = Module(:FirstProvenanceFixture)
        Base.include(first_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        first = withenv("PATH" => "") do
            Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())
        end
        @test first["repository_revision"] === nothing
        @test first["repository_dirty"] === nothing
        @test length(first["source_sha256"]) == 64
        write(joinpath(fixture_root, "solver.jl"), "# changed engine source\n")
        @test Base.invokelatest(() -> first_module.BeatEngineProvenance.engine_identity())["source_sha256"] == first["source_sha256"]
        second_module = Module(:SecondProvenanceFixture)
        Base.include(second_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
        second = Base.invokelatest(() -> second_module.BeatEngineProvenance.engine_identity())
        @test second["source_sha256"] != first["source_sha256"]
        # Outside any checkout the identity never asks Git.
        @test !Base.invokelatest(() -> second_module.BeatEngineProvenance.inside_git_checkout())
        if !Sys.iswindows()
            fake_bin = joinpath(directory, "fake-bin")
            marker = joinpath(directory, "git-was-called")
            mkpath(fake_bin)
            write(joinpath(fake_bin, "git"), "#!/bin/sh\ntouch '$marker'\nexit 1\n")
            chmod(joinpath(fake_bin, "git"), 0o755)
            third_module = Module(:ThirdProvenanceFixture)
            Base.include(third_module, joinpath(fixture_root, "src", "BeatEngineProvenance.jl"))
            third = withenv("PATH" => fake_bin) do
                Base.invokelatest(() -> third_module.BeatEngineProvenance.engine_identity())
            end
            @test third["repository_revision"] === nothing
            @test !isfile(marker)
        end
        checkout = joinpath(directory, "checkout")
        mkpath(checkout)
        write(joinpath(checkout, ".git"), "gitdir: elsewhere\n")  # worktree/submodule form
        @test Base.invokelatest(() -> second_module.BeatEngineProvenance.inside_git_checkout(checkout))
        if !Sys.iswindows()
            link = joinpath(mktempdir(), "engine-link")
            symlink(checkout, link)
            @test Base.invokelatest(() -> second_module.BeatEngineProvenance.inside_git_checkout(link))
        end
        mkpath(joinpath(directory, ".git"))
        @test Base.invokelatest(() -> second_module.BeatEngineProvenance.inside_git_checkout())
    end
end

@testset "BEAT worker negotiation" begin
    info = worker_ready(Dict("cpu" => Dict("available" => true, "reason" => "")))
    @test info["protocol"]["version"] == 1
    @test info["contracts"]["system_request"] == [1]
    @test info["contracts"]["compiled_system"] == [1, 2, 3]
    @test info["exterior_component_kinds"] == ["ideal_velocity_source", "electrodynamic_transducer"]
    @test info["contracts"]["system_result"] == [2]
    @test "radiation_impedance_matrix" in info["optional_output_quantities"]
    @test info["runtime"]["julia_version"] == string(VERSION)
    @test length(info["engine"]["source_sha256"]) == 64
    @test haskey(info["engine"]["source_files_sha256"], "julia_local/coupled_solver.jl")
    @test haskey(info["engine"], "repository_revision")
    @test haskey(info["engine"], "repository_dirty")
    @test length(info["runtime"]["project_sha256"]) == 64
    @test info["runtime"]["julia_threads"] >= 1
    @test info["runtime"]["blas_threads"] >= 1
    empty!(info["engine"]["source_files_sha256"])
    @test !isempty(worker_ready(Dict())["engine"]["source_files_sha256"])
    command = Dict{String,Any}("protocol_version" => 1, "operation" => "solve",
        "request" => "does-not-exist.json", "result_schema_version" => 2)
    @test validate_worker_submission(command) === nothing
    for value in (nothing, true, 1.0, 2, "1")
        bad = merge(command, Dict("protocol_version" => value))
        @test_throws ErrorException validate_worker_submission(bad)
    end
    for value in (nothing, true, 1, 3)
        @test_throws ErrorException validate_worker_submission(merge(command, Dict("result_schema_version" => value)))
    end
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("operation" => "unknown")))
    @test_throws ErrorException validate_worker_submission(merge(command, Dict("request" => "")))
    field = Dict("protocol_version" => 1, "operation" => "bem_field", "request" => "field.json", "field_array_schema_version" => 1)
    @test validate_worker_submission(field) === nothing
    @test_throws ErrorException validate_worker_submission(merge(field, Dict("field_array_schema_version" => 2)))
end
