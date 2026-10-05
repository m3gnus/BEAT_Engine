# Exercise the production source entry, including the shared compiled rules.
module SourceGroundContractTests
using Test, StaticArrays, LinearAlgebra
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))

@testset "source ground parser and physical impedance count" begin
    @test symmetry_mode_from_config(Dict()) == "off"
    for mode in ("off", "x", "xy", "ground")
        @test symmetry_mode_from_config(Dict("symmetry" => " $(uppercase(mode)) ")) == mode
    end
    for mode in ("y", "z", "x+ground", "xy+ground")
        @test_throws ErrorException symmetry_mode_from_config(Dict("symmetry" => mode))
    end
    mesh = BoundaryMesh(SVector{3,Float32}[(0,1,0), (1,1,0), (0,1,1)], [(1,2,3)], [2])
    radiators = [Dict("tag" => 2, "mesh_id" => 1)]
    pressure = fill(ComplexF32(2,3), 3)
    drives = ComplexF32[1]
    impedance(mode) = only(impedance_for_radiators(
        mesh, [1], pressure, radiators, drives, Float32; symmetry_mode=mode,
    ))
    local_impedance = impedance(:off)
    @test local_impedance == Float32[5,-7.5]
    @test impedance(:ground) == local_impedance
    @test impedance(:x) == 2local_impedance
    @test impedance(:xy) == 4local_impedance
    @test physical_radiator_count(:ground) == 1
    @test symmetry_reduction_factor(:ground) == 2 # the image is still assembled
end

const TETRAHEDRON = raw"""$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
2 2 "radiator"
$EndPhysicalNames
$Nodes
4
1 0.0 0.0 0.0
2 0.08 0.0 0.0
3 0.0 0.08 0.0
4 0.0 0.0 0.08
$EndNodes
$Elements
4
1 2 2 2 2 1 3 2
2 2 2 2 2 1 2 4
3 2 2 2 2 2 3 4
4 2 2 2 2 3 1 4
$EndElements
"""

function source_request(path; mode="ground", height=1.0, clearance=nothing)
    config = Dict{String,Any}(
        "meshes" => [Dict("file" => path, "scale_factor" => 1.0,
            "name" => "radiator", "translation_m" => [0.0, height, 0.0])],
        "tag_throat" => 2, "symmetry" => mode,
        "min_angle" => 0.0, "max_angle" => 90.0, "step_size" => 30.0,
        "distance" => 2.0, "quadrature_order" => 3, "singular_order" => 3,
        "regular_quadrature_mode" => "fixed", "rho" => 1.2041, "sound_speed" => 343.0,
    )
    clearance === nothing || (config["ground_plane_min_clearance_m"] = clearance)
    return Dict("config" => config, "frequencies_hz" => [300.0, 500.0],
        "beat_engine_backend" => "cpu")
end

function captured_results(request)
    mktemp() do _, output
        redirect_stdout(output) do
            solve_request(request)
        end
        flush(output)
        seekstart(output)
        events = [JSON.parse(line) for line in eachline(output)]
        @test last(events)["type"] == "completed"
        results = [event["result"] for event in events if event["type"] == "result"]
        @test length(results) == length(request["frequencies_hz"])
        return results
    end
end

wire_impedance(result) = complex(result["impedance"][1]...)
wire_pressure(result) = complex.(result["horizontal_pressure"]["real"][1],
    result["horizontal_pressure"]["imag"][1])

@testset "source rigid-ground requests and clearance" begin
    mktempdir() do directory
        path = joinpath(directory, "tetrahedron.msh")
        write(path, TETRAHEDRON)
        for (height, clearance, message) in (
            (-0.02, nothing, "Y >= 0"),
            (-1.0, nothing, "Y >= 0"),
            (0.0, nothing, "flat on Y=0"),
            (0.01, 0.05, "clearance"),
            (1.0, -0.01, "finite and non-negative"),
            (1.0, NaN, "finite and non-negative"),
            (1.0, Inf, "finite and non-negative"),
        )
            @test_throws message captured_results(source_request(path; height=height, clearance=clearance))
        end
        previous_fused = get(ENV, "BLAB_BEAT_FUSED_BM", nothing)
        try
            # Protect both fused assembly and the four-operator source path.
            for fused in ("0", "1")
                ENV["BLAB_BEAT_FUSED_BM"] = fused
                free = captured_results(source_request(path; mode="off", height=0.0))
                lifted_free = captured_results(source_request(path; mode="off"))
                ground = captured_results(source_request(path; clearance=0.5))
                move_db = [20log10(abs(wire_impedance(g)) / abs(wire_impedance(f)))
                    for (g, f) in zip(ground, free)]
                @test all(isfinite, move_db)
                @test maximum(abs.(move_db)) < 1.0 # a fictitious copy would add 6.02 dB
                delta_db = vcat([20log10.(abs.(wire_pressure(g)) ./ abs.(wire_pressure(f)))
                    for (g, f) in zip(ground, lifted_free)]...)
                @test all(isfinite, delta_db)
                @test maximum(abs.(delta_db)) > 1.0 # an inert image must fail
                @test all(g["diagnostics"]["symmetry"] == "ground" for g in ground)
                edge_path = joinpath(directory, "contact_edge.msh")
                write(edge_path, replace(TETRAHEDRON, "4 0.0 0.0 0.08" => "4 0.0 0.08 0.08"))
                resting = captured_results(source_request(edge_path; height=0.0))
                @test all(isfinite(wire_impedance(r)) for r in resting)
                @test all(all(isfinite, wire_pressure(r)) for r in resting)
            end
        finally
            previous_fused === nothing ? delete!(ENV, "BLAB_BEAT_FUSED_BM") :
                (ENV["BLAB_BEAT_FUSED_BM"] = previous_fused)
        end
    end
end
end
