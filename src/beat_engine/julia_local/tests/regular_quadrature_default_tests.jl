# Standalone regression for both production request parsers; no solve or fixtures.
module RegularQuadratureDefaultTests
using Test, StaticArrays

module CompiledDriver
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
end

module SourceDriver
using ..CompiledDriver.BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))
end

using .CompiledDriver.BeatEngineCore

@testset "CPU fixed regular quadrature default on a graded mesh" begin
    for T in (Float32, Float64)
        # Twenty tiny triangles dominate p90, hiding a coarse enclosure triangle.
        vertices = SVector{3,T}[]
        faces = NTuple{3,Int}[]
        for index in 1:21
            edge = index == 21 ? T(0.1) : T(0.001)
            offset = T(index)
            first_vertex = length(vertices) + 1
            append!(vertices, [SVector{3,T}(offset, 0, 0),
                SVector{3,T}(offset + edge, 0, 0), SVector{3,T}(offset, edge, 0)])
            push!(faces, (first_vertex, first_vertex + 1, first_vertex + 2))
        end
        mesh = BoundaryMesh(vertices, faces, fill(1, length(faces)))
        frequency, sound_speed, base_order = T(8000), T(343), 4
        options = Dict{String,Any}()

        selection = CompiledDriver.exterior_quadrature_selection(
            options, mesh, frequency, sound_speed, base_order, :cpu,
        )
        @test selection.mode == "fixed"
        @test selection.order == base_order
        @test length(triangle_rule(T, selection.order).weights) == 6
        @test selection.kh === nothing

        mode = SourceDriver.regular_quadrature_mode_from_config(options, :cpu)
        source_selection = SourceDriver.regular_quadrature_selection(
            options, mesh, frequency, sound_speed, base_order, mode,
        )
        @test mode == "fixed"
        @test source_selection.order == base_order
        @test length(triangle_rule(T, source_selection.order).weights) == 6
        @test source_selection.kh === nothing

        # Explicit opt-in reproduces the old default's three-point selection.
        wavelength_options = Dict("regular_quadrature_mode" => "wavelength")
        wavelength = CompiledDriver.exterior_quadrature_selection(
            wavelength_options, mesh, frequency, sound_speed, base_order, :cpu,
        )
        wavelength_mode = SourceDriver.regular_quadrature_mode_from_config(wavelength_options, :cpu)
        source_wavelength = SourceDriver.regular_quadrature_selection(
            wavelength_options, mesh, frequency, sound_speed, base_order, wavelength_mode,
        )
        @test wavelength.mode == wavelength_mode == "wavelength"
        @test wavelength.mesh_stat == source_wavelength.mesh_stat == "p90"
        @test 0 < wavelength.kh <= 2.0
        @test wavelength.kh ≈ source_wavelength.kh
        @test wavelength.order == source_wavelength.order == 2
        @test length(triangle_rule(T, wavelength.order).weights) == 3
        @test Float64(2pi * frequency / sound_speed) * sqrt(maximum(mesh.areas)) > 2.0
        @test SourceDriver.regular_quadrature_mode_from_config(
            Dict("quadrature_mode" => "wavelength"), :cpu,
        ) == "wavelength"

        # Fixed means the supplied base order, even if a caller overrides it.
        @test CompiledDriver.exterior_quadrature_selection(
            options, mesh, frequency, sound_speed, 2, :cpu,
        ).order == 2
        @test SourceDriver.regular_quadrature_selection(
            options, mesh, frequency, sound_speed, 2, mode,
        ).order == 2
        for backend in (:cuda, :rocm, :metal)
            @test SourceDriver.regular_quadrature_mode_from_config(options, backend) == "fixed"
            @test CompiledDriver.exterior_quadrature_selection(
                options, mesh, frequency, sound_speed, base_order, backend,
            ).mode == "fixed"
            if backend == :rocm
                @test SourceDriver.regular_quadrature_mode_from_config(wavelength_options, backend) == "wavelength"
                @test CompiledDriver.exterior_quadrature_selection(
                    wavelength_options, mesh, frequency, sound_speed, base_order, backend,
                ) == wavelength
            else
                @test_throws ErrorException SourceDriver.regular_quadrature_mode_from_config(wavelength_options, backend)
                @test_throws ErrorException CompiledDriver.exterior_quadrature_selection(
                    wavelength_options, mesh, frequency, sound_speed, base_order, backend,
                )
            end
        end
    end
end
end # module
