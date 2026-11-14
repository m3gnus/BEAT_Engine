# Hardware gate for changing and revisiting quadrature orders within one sweep.
module RocmWavelengthTests
using Test, StaticArrays, LinearAlgebra, Base64, Statistics
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
include(joinpath(@__DIR__, "exterior_transducer_helpers.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)

function wavelength_request!(request, mesh, precision, convention)
    h = sqrt(quantile(Float64.(mesh.areas), .9))
    request["frequencies_hz"] = [.25, 1., 3., .25] .* (343 / (2pi*h))
    merge!(request["solver_options"], Dict("bem_backend"=>"rocm", "precision"=>precision,
        "phasor_convention"=>convention, "quadrature_order"=>4, "singular_order"=>2,
        "regular_quadrature_mode"=>"wavelength", "wavelength_kh_q1_max"=>.5,
        "wavelength_kh_q2_max"=>2.))
    return request
end

function check_sweep(request)
    results = captured(request)
    @test length(results) == 4
    orders = [1,2,4,1]
    tolerance = request["solver_options"]["precision"] == "float32" ? 5e-4 : 1e-8
    for (index, result) in enumerate(results)
        diagnostics = result["diagnostics"]
        @test diagnostics["bem_backend"] == "rocm"
        @test startswith(diagnostics["linear_solver"], "rocm_")
        @test diagnostics["regular_quadrature_mode"] == "wavelength"
        @test diagnostics["regular_quadrature_order"] == orders[index]
        reference = deepcopy(request)
        reference["frequencies_hz"] = [request["frequencies_hz"][index]]
        merge!(reference["solver_options"], Dict("bem_backend"=>"cpu",
            "regular_quadrature_mode"=>"fixed", "quadrature_order"=>orders[index]))
        actual, expected = quantities(result), quantities(only(captured(reference)))
        for key in keys(actual)
            a, b = decoded(actual[key]), decoded(expected[key])
            @test size(a) == size(b)
            @test all(isfinite, a)
            @test norm(a-b) <= tolerance * norm(b) + eps(Float64)
        end
    end
    first_values, last_values = quantities(first(results)), quantities(last(results))
    for key in keys(first_values)
        @test isapprox(decoded(first_values[key]), decoded(last_values[key]); rtol=tolerance)
    end
end

@testset "ROCm compiled exterior wavelength cache transitions" begin
    for precision in ("float32", "float64"), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR), mode in (:off, :ground)
        mktemp() do path, io
            centre = mode == :ground ? SVector(0.,.3,0.) : SVector(0.,0.,0.)
            mesh = sphere(.1, centre, 2; refinements=1)
            mesh_file(io, mesh)
            request = JSON.parsefile(joinpath(ENGINE_DIR, "..", "beat_contract", "example-exterior-request.json"))
            request["compiled_system"]["meshes"][1]["file"] = path
            request["compiled_system"]["meshes"][1]["scale_to_m"] = 1.
            request["solver_options"] = Dict{String,Any}("symmetry"=>String(mode))
            request["outputs"] = [Dict("id"=>q,"quantity"=>q,"target_ids"=>[],
                "options"=>q == "exterior_pressure" ? Dict("points_m"=>[[2.,1.,1.],[-1.,2.,2.]]) : Dict())
                for q in ("exterior_pressure","bem_boundary_pressure","bem_boundary_neumann")]
            check_sweep(wavelength_request!(request, mesh, precision, convention))
        end
    end
end

@testset "ROCm exterior transducer wavelength cache transitions" begin
    for precision in ("float32", "float64"), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR), mode in (:off, :ground)
        mktemp() do path, io
            centre = mode == :ground ? SVector(0.,.3,0.) : SVector(0.,0.,0.)
            mesh = sphere(.1, centre, 2; refinements=1)
            mesh_file(io, mesh)
            request = driver_request(path)
            request["solver_options"]["symmetry"] = String(mode)
            check_sweep(wavelength_request!(request, mesh, precision, convention))
        end
    end
end

@testset "ROCm coupled wavelength cached and uncached sweeps" begin
    for precision in ("float32", "float64"), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR),
        condensed in (false,true), cached in (false,true)
        request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true, bem_backend="rocm")))
        resource = only(filter(m -> m["purpose"] == "bem_surface", request["compiled_system"]["meshes"]))
        mesh = translated_boundary_mesh(resource, Float64)
        request["solver_options"]["static_condensation"] = condensed
        request["solver_options"]["cache_frequency_invariant"] = cached
        check_sweep(wavelength_request!(request, mesh, precision, convention))
    end
end
BeatEngineCore.amdgpu_module().synchronize()
GC.gc(true)
BeatEngineCore.amdgpu_module().reclaim()
end
