using Test, LinearAlgebra, Random, StaticArrays, JSON
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "deploy_solver.jl"))

@testset "ROM range projection preserves complex feedback" begin
    Random.seed!(892)
    cuda = BeatEngineCore.CUDA_MODULE
    @test cuda !== nothing && cuda.functional()
    for images in (1, 2, 4)
        rank, sectors = 3, images
        signs = images == 1 ? [(1,)] : images == 2 ? [(1,1),(1,-1)] :
            [(1,1,1,1),(1,-1,1,-1),(1,1,-1,-1),(1,-1,-1,1)]
        orbits = [collect(1:images), collect(images+1:2images)]
        arrays = Dict("d"=>randn(ComplexF32,sectors,2,rank),
            "c"=>randn(ComplexF32,sectors,rank,2), "b"=>randn(ComplexF32,sectors,rank,1),
            "e"=>randn(ComplexF32,sectors,2,1), "velocity"=>randn(ComplexF32,sectors,1,rank),
            "current"=>randn(ComplexF32,sectors,1,rank), "velocity_drive"=>zeros(ComplexF32,sectors,1,1),
            "current_drive"=>zeros(ComplexF32,sectors,1,1))
        instances = [(node_offset=0,face_offset=0,input=ComplexF32[1]),
                     (node_offset=2images,face_offset=2images,input=ComplexF32[2])]
        model = (rank=rank,sector_count=sectors,image_count=images,image_signs=signs,
            node_orbits=orbits,face_orbits=orbits,arrays=arrays,instances=instances,
            factors=[lu(randn(ComplexF32,rank,rank)+5I) for _ in 1:sectors],
            package_node_count=2images,package_face_count=2images,node_count=4images,face_count=4images)
        # Mixed-model wrapper deliberately reverses package keys/instance ordering.
        mixed = (models=Dict("z"=>merge(model,(instances=instances[1:1],)),
                             "a"=>merge(model,(instances=instances[2:2],))),
                 order=[("z",1),("a",1)],node_count=4images,face_count=4images)
        for input_model in (model, mixed)
            pressure = randn(ComplexF32,4images)
            exact_q = deploy_speaker_rom_response(input_model,pressure;include_drive=false).q
            groups = deploy_projection_groups(input_model)
            basis = zeros(ComplexF32,4images,2rank*sectors)
            cursor = 0
            for g in groups
                local_basis = deploy_projection_basis(g)
                width = size(local_basis,2)
                basis[g.instance.face_offset .+ (1:2images),cursor+1:cursor+width] .= local_basis
                cursor += width
            end
            request = Dict{String,Any}("frequency_hz"=>32.0,"quadrature_order"=>2,
                "proximity"=>Dict("close_face_pairs"=>[[1,2,4]]))
            mesh = (vertices=[SVector(0f0,0f0,0f0)],faces=[(1,1,1)])
            signature = deploy_factor_signature(request,mesh,input_model)
            changed = deepcopy(input_model)
            for g in deploy_projection_groups(changed)
                g.instance.input .*= ComplexF32(0.3,0.7)
            end
            @test deploy_factor_signature(request,mesh,changed) == signature
            @test deploy_factor_signature(merge(request,Dict("frequency_hz"=>33.0)),mesh,changed) != signature
            @test deploy_factor_signature(merge(request,Dict("quadrature_order"=>3)),mesh,changed) != signature
            @test deploy_factor_signature(merge(request,Dict("proximity"=>Dict("close_face_pairs"=>[]))),mesh,changed) != signature
            explicit = merge(request,Dict("close_pair_quadrature_order"=>4))
            implicit = merge(explicit,Dict("proximity"=>Dict("close_face_pairs"=>[[1,2]])))
            @test deploy_factor_signature(explicit,mesh,changed) == deploy_factor_signature(implicit,mesh,changed)
            opposite = BeatEngineCore.phasor_convention() == NEGATIVE_TIME_PHASOR ? POSITIVE_TIME_PHASOR : NEGATIVE_TIME_PHASOR
            @test with_phasor_convention(opposite) do
                deploy_factor_signature(request,mesh,changed) != signature
            end
            @test deploy_factor_signature(request,merge(mesh,(vertices=[SVector(1f0,0f0,0f0)],)),changed) != signature
            first(deploy_projection_groups(changed)).model.arrays["d"][1] += 1
            @test deploy_factor_signature(request,mesh,changed) != signature
            coordinates = deploy_projection_coordinates(groups,pressure)
            @test basis*coordinates ≈ exact_q rtol=2f-5
            exterior = randn(ComplexF32,4images,4images)
            projected = build_deploy_projected_rhs(input_model, columns->cuda.CuArray(exterior[:,columns]))
            @test Array(projected.operator)*coordinates ≈ exterior*exact_q rtol=2f-5
            @test projected.report["rank"] == 2rank*sectors
            cuda.unsafe_free!(projected.operator)
        end
    end
end
