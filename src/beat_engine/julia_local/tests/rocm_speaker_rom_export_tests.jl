module RocmSpeakerRomExportTests
using Test
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)
const Rom = BeatEngineSpeakerRom
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

function evaluate_export(rom, pressure, drive, points)
    metadata = rom["metadata"]
    node_orbits = [Int.(orbit) .+ 1 for orbit in metadata["node_orbits"]]
    face_orbits = [Int.(orbit) .+ 1 for orbit in metadata["face_orbits"]]
    q = zeros(eltype(pressure),sum(length,face_orbits),size(pressure,2))
    velocity = zeros(eltype(pressure),size(rom["speaker_rom_velocity"],2),size(pressure,2))
    current = zero(velocity)
    for (i,sector) in enumerate(Rom._parity_sectors(:xy))
        maps = Rom._image_maps(points, :xy)
        projected = Rom._parity_project(pressure,maps,sector.image_signs)
        compact = Rom._compact_parity_values(projected,node_orbits,sector.image_signs)
        state = rom["speaker_rom_k"][i,:,:] \ (rom["speaker_rom_b"][i,:,:]*drive-rom["speaker_rom_c"][i,:,:]*compact)
        output = rom["speaker_rom_d"][i,:,:]*state+rom["speaker_rom_e"][i,:,:]*drive
        q .+= Rom._reconstruct_parity_values(output,face_orbits,sector.image_signs,size(q,1))
        velocity .+= rom["speaker_rom_velocity"][i,:,:]*state+rom["speaker_rom_velocity_drive"][i,:,:]*drive
        current .+= rom["speaker_rom_current"][i,:,:]*state+rom["speaker_rom_current_drive"][i,:,:]*drive
    end
    return (;q,velocity,current)
end
try
    @testset "ROCm exported ROM physical response parity" begin
        for T in (Float32,Float64), convention in (NEGATIVE_TIME_PHASOR,POSITIVE_TIME_PHASOR)
            @testset "$T $convention" begin
                withenv("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION"=>"off", "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION"=>"off") do
                  with_phasor_convention(convention) do
                    fem,bem,mapping = symmetric_meshes(T)
                    transducer = ElectrodynamicTransducer{T}("rom-test",[2],T[1],[1],T[-1],SVector{3,T}(0,0,1),T(1),1,T(6),T(.0005),T(7),T(.015),T(.0005),T(1))
                    faces=findall(==(1),bem.physical_tags)
                    prescribed=sparse(faces,ones(Int,length(faces)),ones(T,length(faces)),length(bem.faces),1)
                    excitations=[(kind=:voltage,transducer_index=1,bem_source_index=0,fem_boundary_tags=Int[]),
                                 (kind=:normal_velocity,transducer_index=0,bem_source_index=1,fem_boundary_tags=Int[])]
                    snapshots=[]
                    for backend in (:cpu,:rocm)
                        options=(;bem_backend=backend,quadrature_order=2,singular_order=2,bulk_loss_factor=T(.01),transducers=[transducer],prescribed_bem_normal_velocity=prescribed)
                        system=backend == :cpu ? build_condensed_coupled_system(fem,bem,mapping,T(500),T(343),T(1.21);options...,allow_transducer_condensation=false) :
                            build_coupled_system(fem,bem,mapping,T(500),T(343),T(1.21);options...,static_condensation=true,transducer_condensation=false,validation_diagnostics=false)
                        try
                            @test system.linear_backend == backend
                            matrix,layout=speaker_interior_state_matrix(system)
                            rom=build_parity_petrov_galerkin_rom(system,matrix,layout,excitations;rank=12,training_count=32,validation_count=8,symmetry=:xy)
                            @test rom["metadata"]["input_port_count"] == 2
                            @test rom["metadata"]["transducer_count"] == 1
                            @test length(rom["metadata"]["effective_rank_per_sector"]) == 4
                            for (key,values) in rom
                                key == "metadata" && continue
                                @test eltype(values) == Complex{T}
                                @test all(isfinite,values)
                            end
                            pressure=Rom._sample_patterns(bem.vertices,system.wavenumber,3,200009,T)
                            drive=Complex{T}[.3+.7im 0 -.2+.4im; 0 .5-.1im .1+.2im]
                            convention == POSITIVE_TIME_PHASOR && (drive=conj.(drive))
                            b,e=Rom._input_matrices(system,layout,excitations)
                            exact_state=matrix \ (Rom._right_hand_side(system,layout,pressure)+b*drive)
                            exact=(q=Rom._boundary_output(system,layout,exact_state)+e*drive,
                                velocity=Matrix(exact_state[layout.mechanical_range,:]),current=Matrix(exact_state[layout.electrical_range,:]))
                            actual=evaluate_export(rom,pressure,drive,bem.vertices)
                            for key in (:q,:velocity,:current)
                                @test isapprox(getproperty(actual,key),getproperty(exact,key);rtol=T == Float32 ? 5e-4 : 1e-8)
                            end
                            push!(snapshots,actual)
                        finally
                            backend == :cpu ? release_condensed_coupled_system!(system) : release_coupled_system!(system)
                        end
                    end
                    for key in (:q,:velocity,:current)
                        @test isapprox(getproperty(snapshots[2],key),getproperty(snapshots[1],key);rtol=T == Float32 ? 5e-4 : 1e-8)
                    end
                  end
                end
            end
        end
    end
finally
    accelerator=BeatEngineCore.amdgpu_module()
    accelerator.synchronize()
    GC.gc(true)
    accelerator.reclaim()
end
end
