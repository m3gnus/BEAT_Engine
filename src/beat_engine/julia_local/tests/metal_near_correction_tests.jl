# Run explicitly on Metal hardware; unavailable hardware must not certify this gate.
using Test, StaticArrays, LinearAlgebra, Metal
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

@test Metal.functional()
@testset "Metal host near corrections preserve CPU operators" begin
    T = Float32
    mesh = BoundaryMesh([
        SVector{3,T}(0, 0, 0), SVector{3,T}(0.04, 0, 0), SVector{3,T}(0, 0.04, 0),
        SVector{3,T}(0, 0, 0.01), SVector{3,T}(0.04, 0, 0.01), SVector{3,T}(0, 0.04, 0.01),
    ], [(1, 2, 3), (4, 5, 6)], [1, 1])
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    near = build_near_correction_cache(mesh, [(1, 2, 4), (2, 1, 6)], 6)
    image = build_near_correction_cache(mesh, [(1, 2)], 6; trial_transform=rigid_ground_transform())
    for storage in ("shared", "private"), mode in (:off, :ground), order in (1, 2), phasor in
        ("exp(-i omega t)", "exp(+i omega t)")
        withenv("BLAB_METAL_OPERATOR_STORAGE"=>storage) do
            with_phasor_convention(phasor) do
                rule = triangle_rule(T, order)
                kwargs = (skip_singular=false, singular_order=3, near_correction_cache=near,
                    image_near_correction_cache=mode == :ground ? image : nothing, symmetry_mode=mode)
                k = T(2pi * 100 / 343)
                reference = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule; backend=:cpu, kwargs...)
                cache = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=3, symmetry_mode=mode)
                operators = nothing
                try
                    operators = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                        backend=:metal, return_device=true, accelerator_quadrature=true, device_cache=cache, kwargs...)
                    operators = metal_host_operators(operators)
                    @test operators.near_pair_count == (mode == :ground ? 3 : 2)
                    for key in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                        actual, expected = getfield(operators, key), getfield(reference, key)
                        @test all(isfinite, actual)
                        @test norm(actual - expected) <= 5f-4 * norm(expected) + 1f-7
                    end
                finally
                    operators === nothing || release_operator_storage!(operators)
                    release_metal_regular_assembly_cache!(cache)
                end
            end
        end
    end
end
