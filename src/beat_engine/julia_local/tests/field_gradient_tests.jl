using Test, LinearAlgebra, StaticArrays
if !isdefined(@__MODULE__, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore

function check_rocm_gradient(points, mesh, pressure, neumann, k, cpu_cache, expected; symmetry_mode=:off, rtol=1e-10, atol=0)
    isdefined(@__MODULE__, :ROCM_GRADIENT_QUALIFICATION) || return
    cache = build_rocm_field_evaluation_cache(cpu_cache)
    try
        actual = BeatEngineCore.evaluate_galerkin_field_gradient_rocm(points, mesh, pressure, neumann, k, cache; symmetry_mode)
        @test isapprox(actual, expected; rtol, atol)
    finally
        release_rocm_field_evaluation_cache!(cache)
    end
end

# A closed radiating surface checks the physical sign and full vector, rather
# than comparing two evaluations of the same integral implementation.
@testset "radiating monopole velocity and intensity" begin
    vertices = SVector{3,Float64}[(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    faces = [(1,3,5),(3,2,5),(2,4,5),(4,1,5),(3,1,6),(2,3,6),(4,2,6),(1,4,6)]
    for _ in 1:4
        next = NTuple{3,Int}[]
        for (a,b,c) in faces
            ab,bc,ca = length(vertices)+1,length(vertices)+2,length(vertices)+3
            append!(vertices, [normalize(vertices[a]+vertices[b]),normalize(vertices[b]+vertices[c]),normalize(vertices[c]+vertices[a])])
            append!(next, [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)])
        end
        faces = next
    end
    mesh = BoundaryMesh(0.1 .* vertices,faces,fill(1,length(faces)))
    points = [SVector(0.21,0.08,0.06),SVector(-0.3,0.4,0.2),SVector(1.,2.,3.)]
    omega, rho, k = 2pi*500, 1.2041, 2pi*500/343
    with_phasor_convention(POSITIVE_TIME_PHASOR) do
        p(x) = exp(-im*k*norm(x))/norm(x)
        grad(x) = p(x)*(-im*k-1/norm(x))*x/norm(x)
        pressure = p.(mesh.vertices)
        q = [sum(grad(mesh.centroids[j]) .* mesh.normals[j]) for j in eachindex(faces)]
        cache = build_field_evaluation_cache(mesh,triangle_rule(Float64,4))
        result = BeatEngineCore.evaluate_galerkin_field_gradient_cpu(points,mesh,pressure,q,k,cache)
        check_rocm_gradient(points,mesh,pressure,q,k,cache,result)
        for (i,x) in enumerate(points)
            @test result[i,1] ≈ p(x) rtol=0.01
            @test result[i,2:4] ≈ grad(x) rtol=0.01
            velocity = result[i,2:4]/(-im*omega*rho)
            intensity = real.(result[i,1] .* conj.(velocity))
            @test dot(intensity,x) > 0
            @test norm(cross(intensity,x))/norm(intensity)/norm(x) < 0.01
        end
        if BeatEngineCore.CUDA_MODULE !== nothing && BeatEngineCore.CUDA_MODULE.functional()
            dc = build_cuda_field_evaluation_cache(cache)
            try
                gpu = BeatEngineCore.evaluate_galerkin_field_gradient_cuda(points,mesh,pressure,q,k,dc)
                @test gpu ≈ result rtol=1e-10
            finally
                release_cuda_field_evaluation_cache!(dc)
            end
        end
    end
end

@testset "analytic BEM pressure gradients" begin
    for T in (Float32, Float64)
        mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0)],[(1,2,3)],[1])
        rule = triangle_rule(T,4)
        cache = build_field_evaluation_cache(mesh,rule)
        points = [SVector{3,T}(0.2,0.2,z) for z in (0.2,0.02,0.002)]
        p, q = zeros(Complex{T},3), ones(Complex{T},1)
        values = BeatEngineCore.evaluate_galerkin_field_gradient_cpu(points,mesh,p,q,zero(T),cache)
        check_rocm_gradient(points,mesh,p,q,zero(T),cache,values; rtol=3e-4, atol=1e-6)
        for (i,x) in enumerate(points)
            a,b,c = (SVector{3,Float64}(v-x) for v in mesh.vertices)
            angle = abs(2 * atan(dot(a,cross(b,c)),norm(a)*norm(b)*norm(c)+dot(a,b)*norm(c)+dot(b,c)*norm(a)+dot(c,a)*norm(b)))
            @test real(values[i,4]) ≈ angle/(4pi) rtol=2e-4
            @test all(isfinite,values[i,:])
        end
        if BeatEngineCore.CUDA_MODULE !== nothing && BeatEngineCore.CUDA_MODULE.functional()
            gpu_cache = build_cuda_field_evaluation_cache(cache)
            try
                gpu = BeatEngineCore.evaluate_galerkin_field_gradient_cuda(points,mesh,p,q,zero(T),gpu_cache)
                @test gpu ≈ values rtol=3e-4 atol=1e-6
            finally
                release_cuda_field_evaluation_cache!(gpu_cache)
            end
        end
    end
    T = Float64
    mesh = BoundaryMesh(SVector{3,T}[(0,0,0),(1,0,0),(0,1,0)],[(1,2,3)],[1])
    p, q, k = ComplexF64[1+im,2-im,0.5im], ComplexF64[0.2+0.3im], 1.7
    x = SVector(4.0,3.0,2.0)
    for symmetry in (:off,:x,:xy), convention in (NEGATIVE_TIME_PHASOR,POSITIVE_TIME_PHASOR)
        with_phasor_convention(convention) do
            cache = build_field_evaluation_cache(mesh,triangle_rule(T,4);symmetry_mode=symmetry)
            values = BeatEngineCore.evaluate_galerkin_field_gradient_cpu([x],mesh,p,q,k,cache;symmetry_mode=symmetry)
            check_rocm_gradient([x],mesh,p,q,k,cache,values;symmetry_mode=symmetry)
            @test values[1,1] ≈ evaluate_galerkin_field_cpu([x],mesh,p,q,k,cache)[1] rtol=1e-12
            for d in 1:3
                delta = SVector{3,T}(ntuple(j -> j == d ? 1e-5 : 0.0,3))
                pm = evaluate_galerkin_field_cpu([x+delta,x-delta],mesh,p,q,k,cache)
                @test values[1,d+1] ≈ (pm[1]-pm[2])/2e-5 rtol=1e-8
            end
            if BeatEngineCore.CUDA_MODULE !== nothing && BeatEngineCore.CUDA_MODULE.functional()
                dc = build_cuda_field_evaluation_cache(cache)
                try
                    gpu = BeatEngineCore.evaluate_galerkin_field_gradient_cuda([x],mesh,p,q,k,dc;symmetry_mode=symmetry)
                    @test gpu ≈ values rtol=1e-10
                finally
                    release_cuda_field_evaluation_cache!(dc)
                end
            end
        end
    end
end
