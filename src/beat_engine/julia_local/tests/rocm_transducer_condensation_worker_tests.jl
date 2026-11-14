module RocmTransducerCondensationWorkerTests
using Test, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "..", "julia_engine", "CompiledCoupledWorkload.jl"))

function capture(request, enabled)
    mode = isnothing(enabled) ? nothing : (enabled ? "on" : "off")
    return withenv("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => mode) do
        mktemp() do _, output
            redirect_stdout(output) do
                solve_request(request; event_mode=true)
            end
            flush(output)
            seekstart(output)
            events = [JSON.parse(line) for line in eachline(output)]
            @test last(events)["type"] == "result"
            only(event["result"] for event in events if event["type"] == "result")
        end
    end
end

function decoded_outputs(result)
    Dict(item["id"] => (shape=item["values"]["shape"], data=collect(reinterpret(
        item["values"]["dtype"] == "complex128" ? ComplexF64 : ComplexF32,
        base64decode(item["values"]["content_base64"])))) for item in result["quantities"])
end

try
    @testset "Compiled ROCm transducer condensation cache selection and CPU parity" begin
        for precision in ("float32", "float64"), cached in (false, true),
            convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
            request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true, bem_backend="rocm")))
            request["frequencies_hz"] = [500.0]
            options = request["solver_options"]
            options["precision"] = precision
            options["cache_frequency_invariant"] = cached
            options["phasor_convention"] = convention
            reference_request = deepcopy(request)
            reference_request["solver_options"]["bem_backend"] = "cpu"
            reference_request["solver_options"]["precision"] = "float64"
            reference = decoded_outputs(capture(reference_request, false))
            original = capture(request, false)
            candidate = capture(request, true)
            automatic = capture(request, nothing)
            @test automatic["diagnostics"]["rocm_transducer_condensation"] == true
            @test decoded_outputs(automatic) == decoded_outputs(candidate)
            @test original["diagnostics"]["rocm_transducer_condensation"] == false
            diagnostics = candidate["diagnostics"]
            @test diagnostics["rocm_transducer_condensation"] == true
            @test diagnostics["bem_backend"] == "rocm"
            @test diagnostics["precision"] == precision
            @test diagnostics["rocm_dense_factorization_precision"] == "float64"
            @test diagnostics["linear_solver"] == "rocm_hybrid_cpu_sparse_schur_plus_rocsolver_dense_lu"
            @test diagnostics["solved_system_order"] < original["diagnostics"]["solved_system_order"]
            actual, retained = decoded_outputs(candidate), decoded_outputs(original)
            @test keys(actual) == keys(reference) == keys(retained)
            for key in keys(reference)
                @test actual[key].shape == reference[key].shape == retained[key].shape
                @test all(isfinite, actual[key].data)
                @test isapprox(actual[key].data, reference[key].data; rtol=precision == "float32" ? 5e-4 : 1e-8)
                @test isapprox(actual[key].data, retained[key].data; rtol=precision == "float32" ? 5e-4 : 1e-8)
            end
            @test phasor_convention() == NEGATIVE_TIME_PHASOR
        end
    end
    @testset "ROM export retains transducer surfaces when condensation is requested" begin
        for precision in ("float32", "float64")
            request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true, bem_backend="rocm")))
            request["frequencies_hz"] = [500.0]
            request["solver_options"]["precision"] = precision
            request["solver_options"]["speaker_rom"] = Dict(
                "rank_per_sector" => 4, "training_count_per_sector" => 12,
                "validation_count_per_sector" => 4, "symmetry" => "off")
            push!(request["outputs"], Dict("id" => "test:rom-k", "quantity" => "speaker_rom_k",
                "target_ids" => [], "options" => Dict()))
            reference = capture(request, false)
            requested = capture(request, true)
            automatic = capture(request, nothing)
            @test automatic["diagnostics"]["rocm_transducer_condensation"] == false
            @test decoded_outputs(automatic) == decoded_outputs(requested)
            @test requested["diagnostics"]["rocm_transducer_condensation"] == false
            @test requested["diagnostics"]["solved_system_order"] == reference["diagnostics"]["solved_system_order"]
            expected, actual = decoded_outputs(reference), decoded_outputs(requested)
            @test haskey(actual, "test:rom-k")
            @test keys(actual) == keys(expected)
            for key in keys(expected)
                @test actual[key].shape == expected[key].shape
                @test all(isfinite, actual[key].data)
                @test actual[key].data == expected[key].data
            end
        end
    end
finally
    accelerator = BeatEngineCore.amdgpu_module()
    accelerator.synchronize()
    GC.gc(true)
    accelerator.reclaim()
end
end
