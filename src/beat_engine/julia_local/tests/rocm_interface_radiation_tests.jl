# Explicit hardware qualification: no CPU fallback or unavailable-device skip.
module RocmInterfaceRadiationTests
using Test, LinearAlgebra, StaticArrays, SparseArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineInterfaceRadiation.jl"))
using .BeatEngineInterfaceRadiation
BeatEngineCore._require_rocm!(rocsolver=true)

function qualify(::Type{T}, condensed, convention) where {T}
    fixtures = joinpath(@__DIR__, "fixtures")
    fem = load_gmsh41_volume(joinpath(fixtures, "femvolume.msh"), T(.001))
    bem = load_gmsh22_with_tags(joinpath(fixtures, "exterior_conforming.msh"), T(.001))
    mapping = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    faces = findall(==(1), bem.physical_tags)
    prescribed = sparse(faces, ones(Int, length(faces)), ones(T, length(faces)), length(bem.faces), 1)
    tolerance = T == Float32 ? 5e-4 : 1e-8
    snapshots = []
    for backend in (:cpu, :rocm)
        system = build_coupled_system(fem, bem, mapping, T(500), T(343), T(1.21);
            bem_backend=backend, static_condensation=backend == :rocm && condensed, validation_diagnostics=false,
            quadrature_order=1, singular_order=1, bulk_loss_factor=T(.01),
            prescribed_bem_normal_velocity=prescribed, retain_interface_radiation=true)
        try
            @test system.linear_backend == backend
            amplitude = Complex{T}(.3, .7)
            convention == POSITIVE_TIME_PHASOR && (amplitude = conj(amplitude))
            excitations = [
                (kind=:normal_velocity, fem_boundary_tags=[radiator], fem_boundary_weights=T[1],
                 bem_source_index=0, transducer_index=0, amplitude),
                (kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=T[],
                 bem_source_index=1, transducer_index=0, amplitude=amplitude * T(-.4)),
            ]
            solutions = solve_coupled_excitations(system, excitations)
            traces_by_excitation = []
            for (index, solution) in enumerate(solutions)
                n = length(solution.interface_flux)
                # Nonuniform, interleaved partitions catch accidental flux averaging.
                ranges = [1:2:n, 2:2:n]
                traces = interface_radiation_traces(system, solution, ranges; include_other=index == 2)
                @test all(isfinite, traces.pressure)
                @test traces.reconstruction_error < tolerance
                @test isapprox(vec(sum(traces.pressure; dims=2)), solution.bem_pressure; rtol=tolerance)
                @test isapprox(vec(sum(traces.normal_derivative; dims=2)), solution.bem_neumann; rtol=tolerance)
                reversed = interface_radiation_traces(system, solution, reverse(ranges); include_other=index == 2)
                @test isapprox(reversed.pressure[:, 1:2], traces.pressure[:, 2:-1:1]; rtol=tolerance)
                zero_solution = merge(solution, (interface_flux=zero.(solution.interface_flux),
                    bem_pressure=zero.(solution.bem_pressure), bem_neumann=zero.(solution.bem_neumann)))
                zero_traces = interface_radiation_traces(system, zero_solution, ranges; include_other=index == 2)
                @test iszero(norm(zero_traces.pressure))
                @test iszero(norm(zero_traces.normal_derivative))
                points = SVector{3,T}[(2, 1, 1), (-1, 2, 2)]
                evaluate(p, q) = backend == :rocm ? evaluate_galerkin_field_rocm(points, bem, p, q,
                    system.wavenumber, system.field_cache) : evaluate_galerkin_field_cpu(points, bem, p, q,
                    system.wavenumber, system.field_cache)
                fields = hcat([evaluate(traces.pressure[:, j], traces.normal_derivative[:, j])
                    for j in axes(traces.pressure, 2)]...)
                total = evaluate(solution.bem_pressure, solution.bem_neumann)
                @test isapprox(vec(sum(fields; dims=2)), total; rtol=tolerance)
                index == 2 && (@test norm(traces.pressure[:, end]) > 0)
                push!(traces_by_excitation, (; pressure=traces.pressure, flux=traces.normal_derivative, fields))
            end
            push!(snapshots, traces_by_excitation)
        finally
            release_coupled_system!(system)
        end
    end
    for excitation in 1:2, key in (:pressure, :flux, :fields)
        @test isapprox(getproperty(snapshots[2][excitation], key),
            getproperty(snapshots[1][excitation], key); rtol=tolerance)
    end
    return snapshots[2]
end

try
    @testset "ROCm interface radiation reconstruction and CPU parity" begin
        for T in (Float32, Float64), condensed in (false, true)
            @testset "$T condensed=$condensed" begin
                snapshots = []
                for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                    push!(snapshots, with_phasor_convention(convention) do
                        qualify(T, condensed, convention)
                    end)
                end
                for excitation in 1:2, key in (:pressure, :flux, :fields)
                    @test isapprox(getproperty(snapshots[2][excitation], key),
                        conj.(getproperty(snapshots[1][excitation], key)); rtol=T == Float32 ? 5e-4 : 1e-8)
                end
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
