module RocmFieldGradientTests
using Test, LinearAlgebra, StaticArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
BeatEngineCore._require_rocm!()
const ROCM_GRADIENT_QUALIFICATION = true
try
    include(joinpath(@__DIR__, "field_gradient_tests.jl"))
    @testset "ROCm gradient empty fields and cleanup after rejected boundary samples" begin
        for T in (Float32, Float64)
            mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0)], [(1,2,3)], [1])
            cache = build_rocm_field_evaluation_cache(mesh, triangle_rule(T,4))
            p, q = ones(Complex{T}, 3), ones(Complex{T}, 1)
            evaluate = BeatEngineCore.evaluate_galerkin_field_gradient_rocm
            try
                empty_result = evaluate(SVector{3,T}[], mesh, p, q, one(T), cache)
                @test size(empty_result) == (0, 4)
                @test eltype(empty_result) == Complex{T}
                @test_throws "BEM gradient sample is too close to a boundary" evaluate([SVector{3,T}(0.2,0.2,0)], mesh, p, q, one(T), cache)
                valid = evaluate([SVector{3,T}(0.2,0.2,1)], mesh, p, q, one(T), cache)
                @test all(isfinite, valid)
            finally
                release_rocm_field_evaluation_cache!(cache)
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
