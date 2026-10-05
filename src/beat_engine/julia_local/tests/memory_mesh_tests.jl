# Standalone loader parity gate, including meshio/VTK quadratic ordering.
using Test, Base64
include(normpath(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl")))

function packed(values, dtype)
    shape = collect(size(values))
    flat = ndims(values) == 1 ? vec(values) : vec(permutedims(values))
    bits = htol.(reinterpret(UInt64, flat))
    Dict("dtype" => dtype, "shape" => shape, "data" => base64encode(reinterpret(UInt8, bits)))
end
function payload(vertices, cells, names)
    Dict("schema_version" => 1,
         "points" => packed(reduce(vcat, [permutedims(collect(v)) for v in vertices]), "<f8"),
         "physical_names" => names,
         "cells" => [Dict("type" => kind, "connectivity" => packed(indices, "<i8"),
                          "physical_tags" => packed(tags, "<i8")) for (kind, indices, tags) in cells])
end
indices(rows) = reduce(vcat, [permutedims(collect(row) .- 1) for row in rows])

@testset "file and memory loaders preserve P1 topology and transforms" begin
    root = joinpath(@__DIR__, "fixtures")
    file_bem = load_gmsh22_with_tags(joinpath(root, "exterior_conforming.msh"), 1.0)
    tags = unique(file_bem.physical_tags)
    raw = payload(file_bem.vertices, [("triangle", indices(file_bem.faces), file_bem.physical_tags)],
                  Dict("surface$tag" => [tag, 2] for tag in tags))
    memory_bem = boundary_mesh_from_data(raw, 0.001)
    @test memory_bem.vertices == [v * 0.001 for v in file_bem.vertices]
    @test memory_bem.faces == file_bem.faces
    @test memory_bem.physical_tags == file_bem.physical_tags
    fem = load_gmsh41_volume(joinpath(root, "femvolume.msh"), 1.0)
    names = Dict(name => [tag, dim] for ((dim, tag), name) in fem.physical_names)
    raw = payload(fem.vertices, [("triangle", indices(fem.boundary_faces), fem.boundary_physical_tags),
                                 ("tetra", indices(fem.tetrahedra), fem.tetra_physical_tags)], names)
    memory_fem = volume_mesh_from_data(raw, 0.001)
    @test memory_fem.vertices == [v * 0.001 for v in fem.vertices]
    @test memory_fem.tetrahedra == fem.tetrahedra
    @test memory_fem.boundary_faces == fem.boundary_faces
    @test memory_fem.tetra_physical_tags == fem.tetra_physical_tags
    @test memory_fem.physical_names == fem.physical_names
    resource = Dict("mesh_data" => raw, "scale_to_m" => 0.001, "translation_m" => [1., 2., 3.])
    @test translated_volume_mesh(resource, Float64).vertices == [v + SVector(1., 2., 3.) for v in memory_fem.vertices]
end

@testset "quadratic memory ordering matches Gmsh kernel ordering" begin
    vertices = [SVector{3,Float64}(0,0,0), SVector{3,Float64}(1,0,0),
                SVector{3,Float64}(0,1,0), SVector{3,Float64}(0,0,1),
                SVector{3,Float64}(.5,0,0), SVector{3,Float64}(.5,.5,0),
                SVector{3,Float64}(0,.5,0), SVector{3,Float64}(0,0,.5),
                SVector{3,Float64}(.5,0,.5), SVector{3,Float64}(0,.5,.5)]
    raw = payload(vertices, [("triangle6", reshape(Int64[0,1,2,4,5,6], 1,6), Int64[2]),
                             ("tetra10", reshape(collect(Int64, 0:9), 1,10), Int64[3])],
                  Dict("surface" => [2,2], "air" => [3,3]))
    mesh = volume_mesh_from_data(raw, 1.0)
    @test mesh.quadratic_tetrahedra == [(1,2,3,4,5,6,7,8,10,9)]
    @test mesh.quadratic_boundary_faces == [(1,2,3,5,6,7)]
    @test mesh.tetrahedra == [(1,2,3,4)]
    bad = deepcopy(raw)
    bad["points"]["shape"][1] += 1
    @test_throws Exception volume_mesh_from_data(bad, 1.0)
end
