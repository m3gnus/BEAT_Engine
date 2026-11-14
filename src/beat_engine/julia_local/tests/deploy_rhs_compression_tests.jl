using Test, LinearAlgebra, Random, StaticArrays, JSON
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "deploy_rhs_compression.jl"))

@testset "checked complex low-rank approximation" begin
    Random.seed!(711)
    for T in (ComplexF32, ComplexF64)
        a = randn(T, 70, 3) * randn(T, 3, 80)
        c = rhs_compress_block(a, 1e-4)
        @test c.dense === nothing
        @test c.rank <= 3
        @test norm(c.u*c.v-a)/norm(a) < 1e-4
        @test rhs_compress_block(randn(T, 70, 80), 1e-5; max_rank=2).dense !== nothing
        c = rhs_compress_block(zeros(T, 15, 20), 1e-4)
        @test c.rank == 0
    end
    @test_throws ErrorException rhs_compress_block(ones(ComplexF32, 5, 5), 0)
    @test_throws ErrorException rhs_compress_block(fill(ComplexF32(NaN), 5, 5), 1e-4)
    @test !rhs_far(([0.,0,0],[1.,1,1]), ([0.,0,0],[1.,1,1]))
    @test rhs_far(([0.,0,0],[1.,1,1]), ([0.,5,0],[1.,6,1]))
    leaves = rhs_spatial_leaves([SVector(Float32(i), 0f0, 0f0) for i in 1:81], 16)
    @test sort(vcat(leaves...)) == collect(1:81)
    @test maximum(length, leaves) <= 16
    points = [SVector(Float32(i), 0f0, 0f0) for i in 1:81]
    boxes = [rhs_box(points[leaf]) for leaf in leaves]
    nodes, root = rhs_row_tree(leaves, boxes)
    for box in (([0.,0,0],[1.,1,1]), ([100.,0,0],[101.,1,1]))
        selected = rhs_admissible_rows(nodes, root, box)
        @test sort(vcat([nodes[i].indices for i in selected]...)) == collect(1:81)
    end
end

@testset "CUDA exact selected RHS columns" begin
    cuda = BeatEngineCore.CUDA_MODULE
    @test cuda !== nothing && cuda.functional()
    mesh = BoundaryMesh(SVector{3,Float32}[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
        [(1,3,2),(1,2,4),(1,4,3),(2,3,4)], ones(Int,4))
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(Float32, 3)
    dc = build_cuda_regular_assembly_cache(mesh, rule)
    sc = build_singular_correction_cache(mesh, 3)
    dsc = BeatEngineCore.build_cuda_singular_correction_cache(sc, p1, dp0)
    units = cuda.ones(ComplexF32, 4)
    image_cache = build_cuda_image_singular_correction_cache(mesh,p1,dp0,3,eachindex(mesh.faces),:ground)
    near = build_near_correction_cache(mesh, [(4,4,4)], 4; trial_transform=rigid_ground_transform())
    device_near = build_cuda_near_correction_cache(near,p1,dp0)
    for symmetry in (:off, :ground), convention in (POSITIVE_TIME_PHASOR, NEGATIVE_TIME_PHASOR)
        options = symmetry == :off ? (;) : (symmetry_mode=:ground,
            device_image_singular_cache=image_cache, image_near_correction_cache=near,
            device_image_near_correction_cache=device_near)
        with_phasor_convention(convention) do
            full = assemble_burton_miller_rhs_cuda(mesh,p1,dp0,units,0.7f0,rule;
                device_cache=dc,singular_cache=sc,device_singular_cache=dsc,assemble_operator=true,options...)
            correction_blocks = Dict{Any,Any}()
            for columns in ([4,2], [1], [3,1,4,2])
                tile = assemble_burton_miller_rhs_cuda(mesh,p1,dp0,units,0.7f0,rule;
                    device_cache=dc,singular_cache=sc,device_singular_cache=dsc,
                    assemble_operator=true,operator_columns=columns,block_cache=correction_blocks,options...)
                @test Array(tile) ≈ Array(full)[:,columns] rtol=2f-5 atol=2f-6
                cuda.unsafe_free!(tile)
            end
            for blocks in values(correction_blocks)
                BeatEngineCore._release_cuda_bm_block_arrays!(blocks)
            end
            cuda.unsafe_free!(full)
            # Partitioned shared regular assembly must reconstruct the full L.
            system = assemble_burton_miller_neumann_system_cuda(mesh,p1,dp0,units,0.7f0,rule;
                device_cache=dc,singular_cache=sc,device_singular_cache=dsc,options...)
            matrix = cuda.zeros(ComplexF32,4,4)
            blocks_cache = Dict{Any,Any}()
            for columns in ([1,3], [2,4])
                tile = assemble_burton_miller_rhs_cuda(mesh,p1,dp0,units,0.7f0,rule;
                    device_cache=dc,singular_cache=sc,device_singular_cache=dsc,
                    assemble_operator=true,operator_columns=columns,regular_system=matrix,
                    block_cache=blocks_cache,options...)
                cuda.unsafe_free!(tile)
            end
            finalize_options = symmetry == :off ? (symmetry_mode=:off,) :
                (symmetry_mode=:ground,device_image_singular_cache=image_cache,
                 device_image_near_correction_cache=device_near)
            BeatEngineCore.complete_cuda_bm_matrix!(matrix,mesh,0.7f0,rule;
                device_cache=dc,singular_cache=sc,device_singular_cache=dsc,
                block_cache=blocks_cache,finalize_options...)
            @test isapprox(Array(matrix), Array(system.matrix); rtol=2f-5, atol=2f-6)
            cuda.unsafe_free!(matrix)
            release_burton_miller_system_cuda!(system)
            for b in values(blocks_cache)
                BeatEngineCore._release_cuda_bm_block_arrays!(b)
            end
        end
    end
    # Exercise the GPU apply including noncontiguous gather/scatter and complex factors.
    a = randn(ComplexF32, 7, 3); b = randn(ComplexF32, 3, 9)
    rows = [7,2,4,1,6,3,5]; columns = [9,1,8,2,7,3,6,4,5]
    op = DeployCompressedRhs([(row=1,column=1,dense=nothing,u=cuda.CuArray(a),v=cuda.CuArray(b))],
        [cuda.CuArray(Int32.(rows))], [cuda.CuArray(Int32.(columns))], 7, Dict{String,Any}())
    q = randn(ComplexF32, 9); dq = cuda.CuArray(q)
    y = apply_deploy_compressed_rhs(op, dq)
    @test Array(y)[rows] ≈ a*b*q[columns] rtol=2f-5
    dense = fill(ComplexF32(0.1,0.2),7,9)
    push!(op.blocks, (row=1,column=1,dense=cuda.CuArray(dense),u=nothing,v=nothing))
    push!(op.blocks, (row=1,column=1,dense=nothing,u=cuda.zeros(ComplexF32,7,0),v=cuda.zeros(ComplexF32,0,9)))
    pack_deploy_rhs!(op)
    packed_y = apply_deploy_compressed_rhs(op, dq)
    @test Array(packed_y)[rows] ≈ (a*b+dense)*q[columns] rtol=2f-5
    cuda.unsafe_free!(packed_y)
    release_deploy_compressed_rhs!(op)
    cuda.unsafe_free!(y); cuda.unsafe_free!(dq); cuda.unsafe_free!(units)
end
