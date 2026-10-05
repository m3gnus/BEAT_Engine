using Test
import BeatEngineCompiledCpuBundle

@testset "compiled exterior workload covers production requests" begin
    bundle = BeatEngineCompiledCpuBundle
    core = bundle.BeatEngineCore
    mktempdir() do directory
        path = joinpath(directory, "plate.msh")
        write(path, bundle.workload_plate_mesh())
        mesh = core.load_gmsh22_with_tags(path, 1f0)
        @test length(mesh.vertices) == 9
        @test length(mesh.faces) == 8
        @test all(==(2), mesh.physical_tags)
        @test any(isempty(intersect(a, b)) for a in mesh.faces for b in mesh.faces)
        @test any(!isempty(core.image_singular_candidates(mesh, eachindex(mesh.faces), image))
                  for image in core.symmetry_image_transforms(:xy))
        request = bundle.JSON.parse(bundle.JSON.json(bundle.representative_workload_request(path)))
        @test request isa bundle.JSON.Object{String,Any}
        @test request["compiled_system"] isa bundle.JSON.Object{String,Any}
        @test request["frequencies_hz"] == [1000.0, 20000.0]
        @test request["solver_options"]["symmetry"] == "xy"
        @test request["solver_options"]["singular_order"] == 4
        @test request["solver_options"]["quadrature_order"] == 4
        outputs = Dict(output["id"] => output for output in request["outputs"])
        @test length(outputs["sphere"]["options"]["points_m"]) == 37 * 72
        @test length(outputs["diagonal"]["options"]["points_m"]) == 37
        @test outputs["surface:p"]["quantity"] == "bem_boundary_pressure"
        @test outputs["surface:q"]["quantity"] == "bem_boundary_neumann"
        outcome = redirect_stdout(devnull) do
            bundle.solve_request(request; event_mode=true)
        end
        @test !outcome.cancelled
        @test outcome.solved_count == 2
    end
end
