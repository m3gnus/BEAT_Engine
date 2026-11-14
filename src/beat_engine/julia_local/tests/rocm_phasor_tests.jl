# Hardware qualification, deliberately fails when ROCm is unavailable.
# Run with --project=src/beat_engine/julia_rocm. This does not bypass worker guards.
module RocmPhasorTests
using Test, LinearAlgebra, StaticArrays, JSON
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
using .BeatEngineCoupled

BeatEngineCore._require_rocm!(rocsolver=true)
const CONVENTIONS = (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
relative_error(actual, reference) = norm(actual - reference) / max(norm(reference), eps(real(eltype(reference))))
function check_error(label, actual, reference, tolerance; context...)
    value = relative_error(actual, reference)
    println("ROCM_PHASOR_METRIC ", JSON.json((; label, error=value, tolerance, context...)))
    flush(stdout)
    @test isfinite(value) && value <= tolerance
end

function exterior_case(::Type{T}, symmetry) where {T}
    filename = symmetry == :x ? "sample_half.msh" : symmetry == :xy ? "sample_quarter.msh" : "sample.msh"
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", filename), T(0.001))
    # Ground reflection requires clearance, while keeping the source geometry intact.
    if symmetry == :ground
        mesh = BoundaryMesh([v + SVector{3,T}(0, 2, 0) for v in mesh.vertices], mesh.faces, mesh.physical_tags)
    end
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule, singular_order = triangle_rule(T, 2), 2
    k = T(2pi * 500 / 343)
    singular = build_singular_correction_cache(mesh, singular_order)
    host_cache = build_beat_cpu_assembly_cache(mesh, p1, dp0, rule; singular_order, symmetry_mode=symmetry)
    device_cache = build_rocm_regular_assembly_cache(mesh, p1, dp0, rule; singular_order, symmetry_mode=symmetry)
    ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    ipq = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    identity = build_rocm_burton_miller_identity_cache(ipp, ipq, T)
    host_field = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry)
    device_field = build_rocm_field_evaluation_cache(host_field)
    points = SVector{3,T}[(2, 3, 1), (-2, 3, 2), (1, 4, -2)]
    # Independent non-real columns catch lost phase and inadvertent RHS mixing.
    qminus = hcat([Complex{T}(sin(i * .17), cos(i * .13)) for i in eachindex(mesh.faces)],
                  [Complex{T}(cos(i * .11), sin(i * .19)) for i in eachindex(mesh.faces)])
    tolerance = T == Float32 ? 5e-4 : 1e-8
    context = (; precision=string(T), symmetry=string(symmetry))
    snapshots = []
    try
        for convention in CONVENTIONS
            with_phasor_convention(convention) do
                q = convention == NEGATIVE_TIME_PHASOR ? qminus : conj.(qminus)
                cpu = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                    backend=:cpu, cpu_cache=host_cache, singular_cache=singular,
                    skip_singular=false, singular_order, symmetry_mode=symmetry)
                gpu = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                    backend=:rocm, device_cache, singular_cache=singular,
                    rocm_assembly_mode=:native, skip_singular=false, singular_order, symmetry_mode=symmetry)
                try
                    @test gpu.on_gpu && gpu.gpu_backend == :rocm
                    operators = map((:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)) do key
                        host = Array(getproperty(gpu, key))
                        check_error("operator_$(key)", host, getproperty(cpu, key), tolerance; context..., convention)
                        host
                    end
                    pressure = hcat([solve_burton_miller_neumann(gpu, identity, column, k) for column in eachcol(q)]...)
                    reference = hcat([solve_burton_miller_neumann(cpu, ipp, ipq, column, k) for column in eachcol(q)]...)
                    check_error("pressure_cpu", pressure, reference, tolerance; context..., convention)
                    field = hcat([evaluate_galerkin_field_rocm(points, mesh, pressure[:, i], q[:, i], k, device_field) for i in axes(q, 2)]...)
                    cpu_field = hcat([evaluate_galerkin_field_cpu(points, mesh, reference[:, i], q[:, i], k, host_field) for i in axes(q, 2)]...)
                    check_error("field_cpu", field, cpu_field, tolerance; context..., convention)
                    push!(snapshots, (; operators, pressure, field))
                finally
                    release_operator_storage!(gpu)
                end
            end
        end
        for i in eachindex(snapshots[1].operators)
            check_error("operator_conjugation_$i", snapshots[2].operators[i], conj.(snapshots[1].operators[i]), tolerance; context...)
        end
        for key in (:pressure, :field)
            check_error("$(key)_conjugation", getproperty(snapshots[2], key), conj.(getproperty(snapshots[1], key)), tolerance; context...)
        end
        @test phasor_convention() == NEGATIVE_TIME_PHASOR
    finally
        release_rocm_field_evaluation_cache!(device_field)
        release_rocm_burton_miller_identity_cache!(identity)
        release_rocm_regular_assembly_cache!(device_cache)
    end
end

function coupled_case(::Type{T}) where {T}
    fixtures = joinpath(@__DIR__, "fixtures")
    fem = load_gmsh41_volume(joinpath(fixtures, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(fixtures, "exterior_conforming.msh"), T(0.001))
    mapping = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    tag = physical_tag(fem, 2, "Radiator")
    stiffness, mass = assemble_p1_fem_matrices(fem)
    interior = setdiff(1:length(fem.vertices), unique(mapping.fem_vertex_indices))
    eigenvalues = eigen(Symmetric(Float64.(Matrix(stiffness[interior, interior]))),
                        Symmetric(Float64.(Matrix(mass[interior, interior])))).values
    pole = 343 * sqrt(first(filter(x -> x > 1e-8 * maximum(abs, eigenvalues), sort(eigenvalues)))) / (2pi)
    transducer = ElectrodynamicTransducer{T}("rocm-phasor", [tag], T[1], [1], T[-1],
        SVector{3,T}(0, 0, 1), T(2), 1, T(6), T(.0005), T(7), T(.015), T(.0005), T(1))
    tolerance = T == Float32 ? 5e-4 : 1e-8
    for frequency in (500.0, pole * .999, pole * 1.001), condensed in (false, true)
        snapshots = []
        for convention in CONVENTIONS
            with_phasor_convention(convention) do
                system = build_coupled_system(fem, bem, mapping, T(frequency), T(343), T(1.21);
                    bem_backend=:rocm, static_condensation=condensed, validation_diagnostics=false, quadrature_order=1,
                    singular_order=1, bulk_loss_factor=T(1e-4), transducers=[transducer])
                try
                    @test system.linear_backend == :rocm
                    amplitudes = Complex{T}[.3+.7im, -.2+.4im]
                    convention == POSITIVE_TIME_PHASOR && (amplitudes = conj.(amplitudes))
                    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude) for amplitude in amplitudes]
                    solutions = solve_coupled_excitations(system, excitations)
                    points = SVector{3,T}[(2, 1, 1), (-1, 2, 2)]
                    fields = [evaluate_galerkin_field_rocm(points, bem, s.bem_pressure, s.bem_neumann, system.wavenumber, system.field_cache) for s in solutions]
                    push!(snapshots, (; solutions, fields))
                finally
                    release_coupled_system!(system)
                end
            end
        end
        context = (; precision=string(T), frequency, condensed)
        for column in 1:2
            for key in (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
                check_error("coupled_$(key)", getproperty(snapshots[2].solutions[column], key),
                    conj.(getproperty(snapshots[1].solutions[column], key)), tolerance; context..., column)
            end
            check_error("coupled_field", snapshots[2].fields[column], conj.(snapshots[1].fields[column]), tolerance; context..., column)
        end
    end
end

try
    @testset "ROCm native phasor hardware qualification" begin
        for T in (Float32, Float64), symmetry in (:off, :x, :xy, :ground)
            @testset "$T $symmetry" begin
                exterior_case(T, symmetry)
            end
        end
    end
    @testset "ROCm coupled phasors near a cavity pole" begin
        for T in (Float32, Float64)
            @testset "$T" begin
                coupled_case(T)
            end
        end
    end
finally
    # Finish pending work and release pooled allocations while Julia and HIP are
    # still fully initialized. Keep teardown failures visible to qualification.
    accelerator = BeatEngineCore.amdgpu_module()
    accelerator.synchronize()
    GC.gc(true)
    accelerator.reclaim()
    println("ROCM_PHASOR_TEARDOWN_COMPLETE")
    flush(stdout)
end
end
