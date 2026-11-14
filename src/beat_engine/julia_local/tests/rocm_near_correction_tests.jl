# Explicit hardware gate: unavailable ROCm must not certify this feature.
module RocmNearCorrectionTests
using Test, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
BeatEngineCore._require_rocm!()

@testset "ROCm near corrections preserve CPU operators" begin
    for T in (Float32, Float64)
        mesh = BoundaryMesh([
            SVector{3,T}(.01,.01,0), SVector{3,T}(.05,.01,0), SVector{3,T}(.01,.05,0),
            SVector{3,T}(.01,.01,.01), SVector{3,T}(.05,.01,.01), SVector{3,T}(.01,.05,.01),
        ], [(1,2,3),(4,5,6)], [1,1])
        p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
        near = build_near_correction_cache(mesh, [(1,2,4),(2,1,6)], 6)
        tolerance = T == Float32 ? 5e-4 : 1e-10
        for mode in (:off, :x, :xy, :ground)
            images = [build_near_correction_cache(mesh, [(1,2),(2,1)], 6; trial_transform=transform)
                      for transform in symmetry_image_transforms(mode)]
            for order in (1,2), phasor in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR), assembly_mode in (:native, :host_staged)
                @testset "$T $mode q$order $phasor $assembly_mode" begin
                    with_phasor_convention(phasor) do
                        rule = triangle_rule(T, order)
                        kwargs = (skip_singular=false, singular_order=3, near_correction_cache=near,
                                  image_near_correction_cache=images, symmetry_mode=mode)
                        k = T(.7)
                        cpu = assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule; backend=:cpu, kwargs...)
                        cache = build_rocm_regular_assembly_cache(mesh,p1,dp0,rule; singular_order=3,symmetry_mode=mode,assembly_mode)
                        gpu = nothing
                        try
                            timing = Dict{String,Float64}()
                            gpu = assemble_regular_galerkin_operators(mesh,p1,dp0,k,rule;
                                backend=:rocm, device_cache=cache, rocm_assembly_mode=assembly_mode, timing, kwargs...)
                            @test gpu.on_gpu && gpu.host_staged_assembly == (assembly_mode == :host_staged)
                            @test gpu.near_pair_count == 2 * (length(images)+1)
                            @test gpu.near_pair_quadrature_order == 6
                            @test haskey(timing, assembly_mode == :native ? "rocm_native_near_corrections" : "rocm_host_assembly")
                            for key in (:single_layer,:double_layer,:adjoint_double_layer,:hypersingular)
                                actual, expected = Array(getproperty(gpu,key)), getproperty(cpu,key)
                                @test all(isfinite,actual)
                                @test norm(actual-expected) <= tolerance * norm(expected) + eps(T)
                            end
                        finally
                            gpu === nothing || release_operator_storage!(gpu)
                            release_rocm_regular_assembly_cache!(cache)
                        end
                    end
                end
            end
        end
    end
end
BeatEngineCore.amdgpu_module().synchronize()
GC.gc(true)
BeatEngineCore.amdgpu_module().reclaim()
end
