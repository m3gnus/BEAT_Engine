# Explicit hardware qualification: missing ROCm/rocSOLVER is a failure.
module RocmTransducerCondensationTests
using Test, LinearAlgebra, StaticArrays, SparseArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
BeatEngineCore._require_rocm!(rocsolver=true)

const OUTPUTS = (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux,
                 :diaphragm_velocity, :voice_coil_current)

function snapshots(::Type{T}, backend, eliminate, cached, convention;
    frequencies=(250, 700), loss_factor=0, automatic=false) where {T}
    fixtures = joinpath(@__DIR__, "fixtures")
    fem = load_gmsh41_volume(joinpath(fixtures, "femvolume.msh"), T(.001))
    bem = load_gmsh22_with_tags(joinpath(fixtures, "exterior_conforming.msh"), T(.001))
    mapping = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    tag = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("condensed-driver", [tag], T[1], [1], T[-1],
        SVector{3,T}(0, 0, 1), T(2), 1, T(6), T(.0005), T(7), T(.015), T(.0005), T(1))
    transducers = [transducer]
    operators = assemble_transducer_operators(fem, bem, transducers)
    retained = sort(unique(vcat(mapping.fem_vertex_indices,
        eliminate ? Int[] : findnz(operators.fem_surface)[1])))
    cache = cached ? prepare_coupled_cache(fem, bem, mapping;
        bem_backend=backend, quadrature_order=1, singular_order=1,
        retained_fem_vertices=retained,
        bulk_loss_factor_by_vertex=fill(T(loss_factor), length(fem.vertices))) : nothing
    result = []
    selection = automatic ? (;) : (; transducer_condensation=eliminate)
    try
        for frequency in frequencies
            system = build_coupled_system(fem, bem, mapping, T(frequency), T(343), T(1.21);
                bem_backend=backend, static_condensation=backend == :rocm,
                selection..., validation_diagnostics=false,
                quadrature_order=1, singular_order=1, transducers=transducers,
                transducer_operators=operators, cache=cache, bulk_loss_factor=T(loss_factor))
            try
                @test system.linear_backend == backend
                @test system.transducer_condensation == eliminate
                if eliminate
                    @test system.condensation.transducer_condensed
                    @test system.retained_fem_vertices == sort(unique(mapping.fem_vertex_indices))
                    @test length(system.retained_fem_vertices) <
                        length(union(mapping.fem_vertex_indices, findnz(operators.fem_surface)[1]))
                    @test eltype(system.factorization) == ComplexF64
                end
                amplitudes = Complex{T}[.3+.7im, -.2+.4im, .6-.1im]
                convention == POSITIVE_TIME_PHASOR && (amplitudes = conj.(amplitudes))
                excitations = [
                    (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=amplitudes[1]),
                    # Nonzero FEM forcing exercises the mechanical RHS correction,
                    # which voltage-only tests cannot observe.
                    (kind=:normal_velocity, fem_boundary_tags=[tag], fem_boundary_weights=T[1],
                     bem_source_index=0, transducer_index=0, amplitude=amplitudes[2]),
                    (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=amplitudes[3]),
                ]
                solutions = solve_coupled_excitations(system, excitations)
                points = SVector{3,T}[(2, 1, 1), (-1, 2, 2)]
                fields = [backend == :rocm ? evaluate_galerkin_field_rocm(points, bem,
                    s.bem_pressure, s.bem_neumann, system.wavenumber, system.field_cache) :
                    evaluate_galerkin_field_cpu(points, bem, s.bem_pressure, s.bem_neumann,
                        system.wavenumber, system.field_cache) for s in solutions]
                for solution in solutions, key in OUTPUTS
                    @test all(isfinite, getproperty(solution, key))
                    @test eltype(getproperty(solution, key)) == Complex{T}
                end
                push!(result, (; solutions, fields, solved_order=system.solved_system_order))
            finally
                release_coupled_system!(system)
            end
        end
    finally
        isnothing(cache) || release_coupled_cache!(cache)
    end
    return result
end

function compare(actual, reference, tolerance; conjugate=false)
    for frequency in eachindex(reference), excitation in 1:3
        for key in OUTPUTS
            expected = getproperty(reference[frequency].solutions[excitation], key)
            conjugate && (expected = conj.(expected))
            @test isapprox(getproperty(actual[frequency].solutions[excitation], key), expected; rtol=tolerance)
        end
        expected = reference[frequency].fields[excitation]
        conjugate && (expected = conj.(expected))
        @test isapprox(actual[frequency].fields[excitation], expected; rtol=tolerance)
    end
end

try
    @testset "ROCm transducer condensation: forcing, reconstruction and CPU parity" begin
        references = Dict()
        for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
            references[convention] = with_phasor_convention(convention) do
                snapshots(Float64, :cpu, false, false, convention)
            end
        end
        for T in (Float32, Float64), cached in (false, true)
            tolerance = T == Float32 ? 5e-4 : 1e-8
            candidates = Dict()
            for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                with_phasor_convention(convention) do
                    retained = snapshots(T, :rocm, false, cached, convention)
                    candidate = snapshots(T, :rocm, true, cached, convention)
                    compare(candidate, references[convention], tolerance)
                    compare(candidate, retained, tolerance)
                    for frequency in eachindex(candidate)
                        @test candidate[frequency].solved_order < retained[frequency].solved_order
                    end
                    candidates[convention] = candidate
                end
            end
            compare(candidates[POSITIVE_TIME_PHASOR], candidates[NEGATIVE_TIME_PHASOR], tolerance;
                conjugate=true)
        end
    end
    @testset "ROCm automatic selection supports new and retained-surface caches" begin
        withenv("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => nothing) do
            for T in (Float32, Float64), (eliminate, cached) in ((true, false), (true, true), (false, true))
                convention = NEGATIVE_TIME_PHASOR
                expected = snapshots(T, :rocm, eliminate, cached, convention)
                actual = snapshots(T, :rocm, eliminate, cached, convention; automatic=true)
                compare(actual, expected, T == Float32 ? 5e-4 : 1e-8)
                @test [r.solved_order for r in actual] == [r.solved_order for r in expected]
            end
        end
    end
    # Match the existing phasor gate's detuning and damping, adding independent
    # CPU and retained-surface comparisons for every complex output and RHS.
    pole_fem = load_gmsh41_volume(joinpath(@__DIR__, "fixtures", "femvolume.msh"), .001)
    pole_bem = load_gmsh22_with_tags(joinpath(@__DIR__, "fixtures", "exterior_conforming.msh"), .001)
    pole_mapping = build_conforming_interface_map(pole_fem, pole_bem,
        physical_tag(pole_fem, 2, "Interface"), 2)
    stiffness, mass = assemble_p1_fem_matrices(pole_fem)
    interior = setdiff(1:length(pole_fem.vertices), unique(pole_mapping.fem_vertex_indices))
    eigenvalues = eigen(Symmetric(Matrix(stiffness[interior, interior])),
                        Symmetric(Matrix(mass[interior, interior]))).values
    pole = 343 * sqrt(first(filter(x -> x > 1e-8 * maximum(abs, eigenvalues), sort(eigenvalues)))) / (2pi)
    near_pole_frequencies = (pole * .999, pole * 1.001)
    @info "Transducer condensation CPU parity frequencies" pole near_pole_frequencies
    @testset "ROCm transducer condensation near a cavity pole: monolithic CPU parity" begin
        references = Dict()
        for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
            references[convention] = with_phasor_convention(convention) do
                snapshots(Float64, :cpu, false, false, convention; frequencies=near_pole_frequencies, loss_factor=1e-4)
            end
        end
        for T in (Float32, Float64), cached in (false, true)
            tolerance = T == Float32 ? 5e-4 : 1e-8
            candidates = Dict()
            for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                with_phasor_convention(convention) do
                    retained = snapshots(T, :rocm, false, cached, convention; frequencies=near_pole_frequencies, loss_factor=1e-4)
                    candidate = snapshots(T, :rocm, true, cached, convention; frequencies=near_pole_frequencies, loss_factor=1e-4)
                    compare(candidate, references[convention], tolerance)
                    compare(candidate, retained, tolerance)
                    for frequency in eachindex(candidate)
                        @test candidate[frequency].solved_order < retained[frequency].solved_order
                    end
                    candidates[convention] = candidate
                end
            end
            compare(candidates[POSITIVE_TIME_PHASOR], candidates[NEGATIVE_TIME_PHASOR], tolerance;
                conjugate=true)
        end
    end
finally
    accelerator = BeatEngineCore.amdgpu_module()
    accelerator.synchronize()
    GC.gc(true)
    accelerator.reclaim()
end
end
