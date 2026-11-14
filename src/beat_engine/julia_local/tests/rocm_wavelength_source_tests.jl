module RocmWavelengthSourceTests
using Test, StaticArrays, LinearAlgebra, Statistics
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))
include(joinpath(@__DIR__, "exterior_test_helpers.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)

function source_results(request)
    mktemp() do _, io
        redirect_stdout(io) do
            solve_request(request)
        end
        flush(io)
        seekstart(io)
        events = [JSON.parse(line) for line in eachline(io)]
        @test last(events)["type"] == "completed"
        [event["result"] for event in events if event["type"] == "result"]
    end
end

function write_mesh(io, mesh)
    println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n", length(mesh.vertices))
    for (i, v) in enumerate(mesh.vertices)
        println(io, i, " ", join(v, " "))
    end
    println(io, "\$EndNodes\n\$Elements\n", length(mesh.faces))
    for (i, face) in enumerate(mesh.faces)
        println(io, i, " 2 2 2 2 ", join(face, " "))
    end
    println(io, "\$EndElements")
    flush(io)
end

@testset "ROCm source wavelength sweeps and near corrections" begin
    for mode in (:off, :xy, :ground), near in (false,true), convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        centre = mode == :ground ? SVector(0.,.3,0.) : SVector(0.,0.,0.)
        mesh = sphere(.1, centre, 2; refinements=1)
        if mode == :xy
            selected = filter(f -> all(mesh.vertices[v][a] >= 0 for v in f for a in (1,2)), mesh.faces)
            used = sort!(unique(collect(Iterators.flatten(selected))))
            renumber = Dict(v=>i for (i,v) in enumerate(used))
            mesh = BoundaryMesh(mesh.vertices[used], [ntuple(i->renumber[f[i]],3) for f in selected], fill(2,length(selected)))
        end
        frequencies = [.25,1.,3.,.25] .* (343 / (2pi * sqrt(quantile(mesh.areas,.9))))
        mktemp() do path, io
            write_mesh(io, mesh)
            config = Dict{String,Any}("mesh_file"=>path,"scale_factor"=>1.,"tag_throat"=>2,
                "symmetry"=>String(mode),"min_angle"=>0.,"max_angle"=>10.,"step_size"=>10.,"distance"=>3.,
                "rho"=>1.21,"sound_speed"=>343.,"quadrature_order"=>4,"singular_order"=>2,
                "regular_quadrature_mode"=>"wavelength","wavelength_kh_q1_max"=>.5,"wavelength_kh_q2_max"=>2.,
                "near_correction_enabled"=>near,"near_correction_order"=>6)
            request = Dict("config"=>config,"frequencies_hz"=>frequencies,
                "beat_engine_backend"=>"rocm","phasor_convention"=>convention)
            actual = source_results(request)
            @test length(actual) == 4
            for (i, order) in enumerate((1,2,4,1))
                @test actual[i]["diagnostics"]["backend"] == "rocm"
                @test actual[i]["diagnostics"]["regular_quadrature_order"] == order
                @test actual[i]["diagnostics"]["regular_quadrature_mode"] == "wavelength"
                reference = deepcopy(request)
                reference["beat_engine_backend"] = "cpu"
                reference["frequencies_hz"] = [frequencies[i]]
                reference["config"]["regular_quadrature_mode"] = "fixed"
                reference["config"]["quadrature_order"] = order
                expected = only(source_results(reference))
                for axis in ("horizontal_pressure", "vertical_pressure")
                    a = ComplexF32.(actual[i][axis]["real"][1],actual[i][axis]["imag"][1])
                    b = ComplexF32.(expected[axis]["real"][1],expected[axis]["imag"][1])
                    @test all(isfinite,a)
                    @test isapprox(a,b;rtol=5e-4)
                end
            end
        end
    end
end
BeatEngineCore.amdgpu_module().synchronize()
GC.gc(true)
BeatEngineCore.amdgpu_module().reclaim()
end
