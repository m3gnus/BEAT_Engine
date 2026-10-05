using Test
# Hardware suites use the Metal project, whose compiled bundle also contains
# the CPU host workload. Do not require an undeclared CPU package there.
if basename(dirname(Base.active_project())) == "julia_metal"
    import BeatEngineCompiledMetalBundle
    const CompiledWorkloadBundle = BeatEngineCompiledMetalBundle
else
    import BeatEngineCompiledCpuBundle
    const CompiledWorkloadBundle = BeatEngineCompiledCpuBundle
end

@testset "compiled exterior workload covers production requests" begin
    bundle = CompiledWorkloadBundle
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
        # Capture the real JSON-decoded driver result, including effective dispatch.
        result_path = joinpath(directory, "events.jsonl")
        outcome = open(result_path, "w") do io
            redirect_stdout(io) do
                bundle.solve_request(request; event_mode=true)
            end
        end
        events = bundle.JSON.parse.(readlines(result_path))
        results = [event["result"] for event in events if event["type"] == "result"]
        @test length(results) == 2
        for result in results
            diagnostics = result["diagnostics"]
            @test diagnostics["burton_miller_assembly"] == "direct_system"
            @test diagnostics["cpu_regular_kernel"] == "simd"
            @test diagnostics["cpu_singular_kernel"] == "simd"
            @test diagnostics["cpu_field_kernel"] == "simd"
            @test diagnostics["linear_solver"] == "cpu_dense_lu"
            @test diagnostics["factorization_count"] == 1
        end
        @test !outcome.cancelled
        @test outcome.solved_count == 2
    end
end
