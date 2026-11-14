# Regression: ROM state retains interface variables eliminated from the solve.
module SpeakerRomRetainedLayoutTests
using Test
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
# Four reflected copies of the tiny coupled workload's full-rank interface.
# Separate cavities preserve exact X/Y symmetry without forcing a rank-deficient
# regular interface triangulation. Reflection also reverses face/tet orientation.
function symmetric_meshes(::Type{T}) where {T}
    points = [(0,0,0),(80,0,0),(80,80,0),(0,80,0),(42,56,0),(42,23,0),
              (0,0,80),(80,0,80),(80,80,80),(0,80,80),(40,40,40)]
    triangles = [(3,2,5),(4,3,5),(5,1,4),(2,1,6),(6,1,5),(5,2,6),
                 (7,8,9),(7,9,10),(1,2,8),(1,8,7),(2,3,9),(2,9,8),
                 (3,4,10),(3,10,9),(4,1,7),(4,7,10)]
    fv, bv = SVector{3,T}[], SVector{3,T}[]
    ff, bf, tetrahedra = NTuple{3,Int}[], NTuple{3,Int}[], NTuple{4,Int}[]
    for (sx, sy) in ((1,1),(-1,1),(1,-1),(-1,-1))
        fo, bo = length(fv), length(bv)
        vertices = [SVector{3,T}(sx*(T(p[1]) / 1000 + T(.1)),
            sy*(T(p[2]) / 1000 + T(.1)), T(p[3]) / 1000) for p in points]
        append!(fv, vertices); append!(bv, vertices[1:10])
        for face in triangles
            oriented = sx*sy == 1 ? face : (face[2],face[1],face[3])
            push!(ff, ntuple(i -> oriented[i]+fo, 3))
            push!(bf, ntuple(i -> oriented[i]+bo, 3))
            push!(tetrahedra, (oriented[1]+fo,oriented[2]+fo,oriented[3]+fo,11+fo))
        end
    end
    fem = VolumeMesh{T}(fv,tetrahedra,ones(Int,length(tetrahedra)),ff,
        repeat(vcat(fill(3,6),fill(2,2),fill(4,8)),4),
        Dict((3,1)=>"Volume",(2,2)=>"Radiator",(2,3)=>"Interface",(2,4)=>"Wall"))
    bem = BoundaryMesh(bv,bf,repeat(vcat(fill(2,6),fill(1,10)),4))
    return fem,bem,build_conforming_interface_map(fem,bem,3,2)
end

@testset "speaker ROM retains physical state across solve elimination" begin
    for T in (Float32,Float64), convention in (NEGATIVE_TIME_PHASOR,POSITIVE_TIME_PHASOR)
        @testset "$T $convention" begin
            with_phasor_convention(convention) do
                fem,bem,mapping = symmetric_meshes(T)
                baseline = nothing
                baseline_curves = nothing
                for mode in (:none,:pressure,:flux)
                    withenv("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => (mode == :flux ? "on" : "off"),
                            "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION" => (mode == :pressure ? "on" : "off")) do
                        system = build_condensed_coupled_system(fem,bem,mapping,T(500),T(343),T(1.21);
                            bem_backend=:cpu,quadrature_order=2,singular_order=2,bulk_loss_factor=T(.01))
                        try
                            @test system.interface_elimination == mode
                            matrix,layout = speaker_interior_state_matrix(system)
                            @test length(layout.flux_range) == length(mapping.fem_vertex_indices)
                            @test length(layout.gamma_range) == length(system.retained_fem_vertices)
                            @test size(matrix) == (layout.state_count,layout.state_count)
                            @test all(isfinite,matrix)
                            result = speaker_rom_rank_experiment(system,Dict("train_per_sector"=>12,"test_per_sector"=>4,"ranks"=>[1,2,4]))
                            curves = [sector["curve"] for sector in result["sectors"]]
                            if mode == :none
                                baseline = matrix
                                baseline_curves = curves
                            else
                                @test matrix == baseline
                                @test curves == baseline_curves
                            end
                        finally
                            release_condensed_coupled_system!(system)
                        end
                    end
                end
            end
        end
    end
end
end
