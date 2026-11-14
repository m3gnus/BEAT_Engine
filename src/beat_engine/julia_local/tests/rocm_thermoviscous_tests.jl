# Explicit hardware gate: unavailable ROCm must fail rather than skip.
module RocmThermoviscousTests
using Test, LinearAlgebra
include(joinpath(@__DIR__, "thermoviscous_driver_tests.jl"))
const Driver = ThermoviscousDriverTests
Driver.BeatEngineCore._require_rocm!(rocsolver=true)

try
    @testset "ROCm thermoviscous coupled production parity" begin
        for precision in ("float32", "float64"), convention in ("exp(-i omega t)", "exp(+i omega t)"), condensed in (false, true), cached in (false, true)
            @testset "$precision $convention condensed=$condensed cached=$cached" begin
                request = Driver.tv_request()
                request["frequencies_hz"] = [250.0, 700.0]
                merge!(request["solver_options"], Dict(
                    "precision" => precision, "phasor_convention" => convention,
                    "static_condensation" => condensed,
                    "cache_frequency_invariant" => cached,
                    "validation_diagnostics" => false, "bem_backend" => "cpu"))
                cpu = Driver.solve_coupled_workload(request)
                request["solver_options"]["bem_backend"] = "rocm"
                gpu = Driver.solve_coupled_workload(request)
                @test cpu.outcome.solved_count == gpu.outcome.solved_count == 2
                @test length(cpu.results) == length(gpu.results) == 2
                tolerance = precision == "float32" ? 5e-4 : 1e-8
                for (reference, actual) in zip(cpu.results, gpu.results)
                    diagnostics = actual["diagnostics"]
                    @test diagnostics["bem_backend"] == "rocm"
                    @test diagnostics["linear_solver"] == (condensed ?
                        "rocm_hybrid_cpu_sparse_schur_plus_rocsolver_dense_lu" : "rocm_rocsolver_dense_lu")
                    @test diagnostics["static_condensation_active"] == condensed
                    losses = diagnostics["thermoviscous_wall_losses"]
                    @test losses["boundary_ids"] == ["boundary:wall"]
                    @test losses["treated_face_count"] == 8
                    @test losses["treated_area_m2"] > 0
                    @test losses["viscous_boundary_layer_m"] > 0
                    @test length(reference["quantities"]) == length(actual["quantities"]) == 4
                    for (a, b) in zip(reference["quantities"], actual["quantities"])
                        @test a["quantity"] == b["quantity"]
                        @test a["values"]["shape"] == b["values"]["shape"]
                        expected, observed = Driver.decode(a), Driver.decode(b)
                        @test all(isfinite, observed)
                        @test norm(observed - expected) <= tolerance * norm(expected) + eps(Float64)
                    end
                end
            end
        end
    end
finally
    amdgpu = Driver.BeatEngineCore.amdgpu_module()
    amdgpu.synchronize()
    GC.gc(true)
    amdgpu.reclaim()
    println("ROCM_THERMOVISCOUS_TEARDOWN_COMPLETE")
    flush(stdout)
end
end
