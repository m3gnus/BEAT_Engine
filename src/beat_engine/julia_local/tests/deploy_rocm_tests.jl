module DeployRocmTests
using Test, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "deploy_rocm.jl"))
include(joinpath(@__DIR__, "..", "deploy_cpu.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)
const accelerator = BeatEngineCore.amdgpu_module()

try
    @testset "Deploy identity uploads preserve host complex conversion" begin
        for T in (Float32, Float64)
            # Include signed zero, extreme finite values and a rectangular block.
            real_block = reshape(T[0.0, -0.0, 1, -1, floatmin(T), floatmax(T)], 2, 3)
            for input in (real_block, Complex{T}.(real_block, one(T)),
                          view(real_block, :, :), Float64.(real_block))
                saved = copy(input)
                uploaded = upload_deploy_rocm_identity(input, T)
                try
                    actual = Array(uploaded)
                    @test eltype(actual) == Complex{T}
                    @test size(actual) == size(input)
                    @test isequal(actual, Complex{T}.(input))
                    @test isequal(input, saved)
                finally
                    accelerator.synchronize()
                    accelerator.unsafe_free!(uploaded)
                end
            end
        end
    end
    @testset "Deploy ROCm reusable factors and ROM feedback match full complex systems" begin
        rng = MersenneTwister(39125)
        for T in (Float32, Float64), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
            with_phasor_convention(convention) do
                n, m = 9, 13
                operators = (single_layer=randn(rng, Complex{T}, n, m),
                    double_layer=randn(rng, Complex{T}, n, n),
                    adjoint_double_layer=randn(rng, Complex{T}, n, m),
                    hypersingular=randn(rng, Complex{T}, n, n))
                identity_pp = Matrix{T}(I, n, n) .* T(30)
                identity_pq = randn(rng, T, n, m)
                k = T(2.3)
                lhs, rhs_operator = BeatEngineCore.burton_miller_neumann_matrices(operators, identity_pp, identity_pq, k)
                device_operators = merge(map(accelerator.ROCArray, operators), (on_gpu=true, gpu_backend=:rocm))
                system = nothing
                try
                    system = build_deploy_rocm_system(device_operators, identity_pp, identity_pq, k)
                    @test eltype(system.factorization) == ComplexF64
                    tolerance = T == Float32 ? 2e-5 : 1e-11
                    for _ in 1:3
                        q = randn(rng, Complex{T}, m)
                        actual = solve_deploy_rocm_system(system, q, T)
                        expected = ComplexF64.(lhs) \ ComplexF64.(rhs_operator*q)
                        @test eltype(actual) == Complex{T}
                        @test isapprox(actual, expected; rtol=tolerance)
                    end
                    # Low-rank acoustic feedback is checked against an independently
                    # assembled full system, not against another iterative solve.
                    feedback = T(.01) .* (randn(rng, Complex{T}, m, 2) * randn(rng, Complex{T}, 2, n))
                    drive = randn(rng, Complex{T}, m)
                    initial = solve_deploy_rocm_system(system, drive, T)
                    action = p -> p - solve_deploy_rocm_system(system, feedback*p, T)
                    solution = deploy_cpu_gmres(action, initial; tolerance=tolerance/10, max_iterations=n)
                    expected = (ComplexF64.(lhs) - ComplexF64.(rhs_operator*feedback)) \ ComplexF64.(rhs_operator*drive)
                    @test isapprox(solution[1], expected; rtol=tolerance)
                    @test solution[3] < tolerance
                    warm = deploy_cpu_gmres(action, initial; tolerance=tolerance/10, max_iterations=n, initial_guess=solution[1])
                    @test isapprox(warm[1], expected; rtol=tolerance)
                    @test warm[2] == 0
                finally
                    release_deploy_rocm_system!(system)
                    release_operator_storage!(device_operators)
                end
            end
        end
    end
finally
    accelerator.synchronize()
    GC.gc(true)
    accelerator.reclaim()
end
end
