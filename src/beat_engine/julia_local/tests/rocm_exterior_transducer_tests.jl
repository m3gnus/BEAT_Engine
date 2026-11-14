# Reuse the independent physical and symmetry gates on actual ROCm hardware.
using AMDGPU
AMDGPU.functional() && AMDGPU.functional(:rocblas) && AMDGPU.functional(:rocsolver) ||
    error("ROCm exterior transducer qualification requires functional hardware and solver libraries.")
withenv("BLAB_TEST_EXTERIOR_TRANSDUCER_BACKEND" => "rocm") do
    include(joinpath(@__DIR__, "exterior_transducer_tests.jl"))
    include(joinpath(@__DIR__, "exterior_transducer_symmetry_tests.jl"))
end

module RocmExteriorTransducerPrecisionTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
include(joinpath(@__DIR__, "exterior_transducer_helpers.jl"))

@testset "ROCm exterior transducers preserve CPU precision and phase" begin
    for mode in (:off, :ground)
    mktemp() do path, io
        centre = mode == :ground ? SVector(0.,.3,0.) : SVector(0.,0.,0.)
        mesh_file(io, sphere(.1, centre, 2; refinements=2))
        for precision in ("float32", "float64"), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
            request = driver_request(path; backend="rocm")
            request["frequencies_hz"] = [50., 58., 100., 1000.]
            request["solver_options"]["precision"] = precision
            request["solver_options"]["symmetry"] = String(mode)
            request["solver_options"]["phasor_convention"] = convention
            actual = captured(request)
            reference_request = deepcopy(request)
            reference_request["solver_options"]["bem_backend"] = "cpu"
            expected = captured(reference_request)
            @test length(actual) == length(expected) == 4
            tolerance = precision == "float32" ? 1e-4 : 1e-10
            for (a, b) in zip(actual, expected)
                @test a["diagnostics"]["precision"] == precision
                @test a["diagnostics"]["phasor_convention"] == convention
                qa, qb = quantities(a), quantities(b)
                for key in keys(qa)
                    va, vb = decoded(qa[key]), decoded(qb[key])
                    @test size(va) == size(vb)
                    @test all(isfinite, va)
                    @test norm(va-vb) <= tolerance * norm(vb) + eps(Float64)
                end
                @test qa["diaphragm_velocity"]["values"]["dtype"] == "complex128"
                @test qa["voice_coil_current"]["values"]["dtype"] == "complex128"
            end
        end
    end
    end
end
end
AMDGPU.synchronize()
GC.gc(true)
AMDGPU.reclaim()
