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
            # These SIMD assertions assume the default BLAB_BEAT_CPU_*_KERNEL
            # settings: REGULAR, SINGULAR and FIELD are unset and select simd.
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

@testset "compiled exterior transducer host workload" begin
    bundle = CompiledWorkloadBundle
    mktempdir() do directory
        for symmetry in ("off", "xy"), precision in ("float64", "float32")
            path = joinpath(directory, "driver-$symmetry.msh")
            write(path, bundle.workload_transducer_mesh(symmetry))
            request = bundle.JSON.parse(bundle.JSON.json(bundle.transducer_workload_request(path, symmetry; precision=precision)))
            @test bundle.BeatEngineContract.validate_system_request(request) === nothing
            events = joinpath(directory, "events.jsonl")
            outcome = open(events, "w") do io
                redirect_stdout(io) do
                    bundle.solve_request(request; event_mode=true)
                end
            end
            results = [e["result"] for e in bundle.JSON.parse.(readlines(events)) if haskey(e,"result")]
            @test outcome.solved_count == length(results) == 2
            for result in results
                @test length(result["quantities"]) == 4
                @test result["diagnostics"]["exterior_lumped_network"]["termination"] == "shorted"
                @test result["quantities"][3]["metadata"]["row_weights"] == [1.]
            end
        end
    end
end

@testset "compiled coupled workload contract and condensed host coverage" begin
    bundle = CompiledWorkloadBundle
    # Validate the actual packaged fixture request without solving the large
    # reference geometry in the ordinary CPU CI gate.
    fixture = bundle.JSON.parse(bundle.JSON.json(bundle.coupled_workload_request()))
    @test bundle.BeatEngineContract.validate_system_request(fixture) === nothing
    @test fixture isa bundle.JSON.Object{String,Any}
    @test fixture["compiled_system"] isa bundle.JSON.Object{String,Any}
    @test fixture["frequencies_hz"] == [500.0, 1000.0]
    @test all(isfile(mesh["file"]) for mesh in fixture["compiled_system"]["meshes"])
    topology = only(fixture["compiled_system"]["interfaces"])["topology"]
    @test !isempty(topology["fem_vertex_indices"])
    @test topology["max_coordinate_error"] <= 1e-6

    request = bundle.JSON.parse(bundle.JSON.json(bundle.coupled_workload_request(; tiny=true)))
    @test bundle.BeatEngineContract.validate_system_request(request) === nothing
    fem = bundle.translated_volume_mesh(request["compiled_system"]["meshes"][2], Float64)
    @test length(fem.vertices) == 11 && length(fem.tetrahedra) == 16
    topology = only(request["compiled_system"]["interfaces"])["topology"]
    @test length(topology["fem_vertex_indices"]) == 6
    @test length(topology["fem_face_indices"]) == 6
    @test length(request["outputs"]) == 4
    settings = bundle.coupled_workload_environment(; mumps=false)
    withenv(settings...) do
        # Strict gate: the precompile wrapper intentionally catches failures,
        # so tests call the inner solve/check and let any failure reach Test.
        run = bundle.solve_coupled_workload(request)
        bundle.check_coupled_workload(run; mumps=false)
        @test run.outcome.solved_count == 2
        @test length(run.results) == 2
        @test all(length(result["quantities"]) == 4 for result in run.results)
    end
    bundle.reset_compiled_workload_state!()
    mumps = bundle.BeatEngineCoupledCondensed.BeatEngineMumps
    @test mumps.LIBRARY[] === nothing
    @test isempty(mumps.LIVE_SOLVERS)
    @test !mumps.ATEXIT_REGISTERED[]
    @test isempty(bundle.BEM_FIELD_EVALUATION_CACHES)
    @test bundle.BeatEngineContract.BeatEngineProvenance.RUNTIME[] === nothing
end

# The CPU compiled entry is only declared in the CPU project.
if @isdefined(BeatEngineCompiledCpuBundle)
@testset "tiny coupled request solves through compiled CPU entry and fallback" begin
    bundle = BeatEngineCompiledCpuBundle
    request = bundle.coupled_workload_request(; tiny=true)
    request["frequencies_hz"] = [1000.0]
    settings = bundle.coupled_workload_environment(; mumps=false)
    mktempdir() do directory
        entry = normpath(joinpath(@__DIR__, "..", "coupled_solver.jl"))
        wrapper = joinpath(directory, "coupled_entry_test.jl")
        write(wrapper, """
            using Test
            include($(repr(entry)))
            @test BEAT_COMPILED_BUNDLE_NAME === :BeatEngineCompiledCpuBundle
            if ENV["BLAB_BEAT_ENGINE_BUNDLE"] == "1"
                @test BEAT_COMPILED_BUNDLE !== nothing
                @test DRIVER === BeatEngineCompiledCpuBundle
                @test !isdefined(Main, :BeatEngineCore)
                mumps = DRIVER.BeatEngineCoupledCondensed.BeatEngineMumps
                @test mumps.LIBRARY[] === nothing
                @test isempty(mumps.LIVE_SOLVERS)
            else
                @test BEAT_COMPILED_BUNDLE === nothing
                @test DRIVER === Main
            end
            """)
        project = dirname(Base.active_project())
        results = []
        for enabled in ("1", "0")
            command = addenv(`$(Base.julia_cmd()) --threads=1 --startup-file=no --project=$project $wrapper`,
                settings..., "BLAB_BEAT_ENGINE_GPU_BACKEND" => "cpu", "BLAB_BEAT_ENGINE_BUNDLE" => enabled,
                "OPENBLAS_NUM_THREADS" => "1")
            text = read(pipeline(command; stdin=IOBuffer(bundle.JSON.json(request))), String)
            result = bundle.JSON.parse(only(filter(!isempty, split(text, '\n'))))
            @test result["freq_hz"] == 1000.0
            @test result["diagnostics"]["formulation"] == "fem_interface_condensed"
            @test result["diagnostics"]["interface_mass_solver"] == "cholmod"
            @test length(result["quantities"]) == 4
            @test result["excitation_port_ids"] == ["port:voltage"]
            for quantity in result["quantities"]
                values = quantity["values"]
                @test values["shape"][1] == 1
                @test values["dtype"] == "complex64"
                decoded = reinterpret(ComplexF32, bundle.base64decode(values["content_base64"]))
                @test all(isfinite, decoded)
                @test any(!iszero, decoded)
            end
            push!(results, result)
        end
        # Native cached code and source fallback must preserve every complex
        # output. Allow Float32 code-generation roundoff, never replace a baseline.
        for (cached, fallback) in zip(results[1]["quantities"], results[2]["quantities"])
            @test cached["id"] == fallback["id"]
            @test cached["values"]["shape"] == fallback["values"]["shape"]
            decode(q) = reinterpret(ComplexF32, bundle.base64decode(q["values"]["content_base64"]))
            @test decode(cached) ≈ decode(fallback) rtol=5e-5
        end
    end
end
end

@testset "coupled workload environment clears inherited overrides (strict)" begin
    bundle = CompiledWorkloadBundle
    inherited = ("BLAB_COUPLED_DENSE_REFINEMENT" => "0", "BLAB_COUPLED_FEM_SOLVER" => "umfpack",
                 "BLAB_MUMPS_THREADS" => "2", "BLAB_COUPLED_WORKLOAD_TEST_EXTRA" => "x")
    withenv(inherited...) do
        for mumps in (false, true)
            settings = bundle.coupled_workload_environment(; mumps)
            @test allunique(first.(settings))
            effective = Dict(settings)
            @test effective["BLAB_COUPLED_FEM_SOLVER"] == (mumps ? "mumps" : "umfpack")
            @test effective["BLAB_COUPLED_DENSE_REFINEMENT"] == "auto"
            @test effective["BLAB_MUMPS_THREADS"] === nothing
            @test effective["BLAB_COUPLED_WORKLOAD_TEST_EXTRA"] === nothing
            @test effective["BLAB_COUPLED_STAGE_OVERLAP"] == "off"
            withenv(settings...) do
                @test ENV["BLAB_COUPLED_FEM_SOLVER"] == (mumps ? "mumps" : "umfpack")
                @test ENV["BLAB_COUPLED_DENSE_REFINEMENT"] == "auto"
                @test !haskey(ENV, "BLAB_MUMPS_THREADS") && !haskey(ENV, "BLAB_COUPLED_WORKLOAD_TEST_EXTRA")
            end
            # The caller's environment comes back unchanged.
            @test all(ENV[name] == value for (name, value) in inherited)
        end
        # The beat_cpu configuration: the same reductions with Float32 FEM and UMFPACK.
        cpu = Dict(bundle.coupled_workload_environment(; mumps=false, defaults=:cpu))
        @test cpu["BLAB_COUPLED_FEM_FLOAT64"] == "off" && cpu["BLAB_COUPLED_FEM_SOLVER"] == "umfpack"
        @test cpu["BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION"] == "auto" && cpu["BLAB_COUPLED_INTERFACE_MASS_SOLVER"] == "cholmod"
        @test cpu["BLAB_COUPLED_DENSE_REFINEMENT"] == "auto" && cpu["BLAB_MUMPS_THREADS"] === nothing
        @test all(ENV[name] == value for (name, value) in inherited)
    end
    @test_throws "defaults" bundle.coupled_workload_environment(; mumps=false, defaults=:cuda)
end

@testset "coupled workload pins exactly what an unset environment selects" begin
    bundle = CompiledWorkloadBundle
    cc = bundle.BeatEngineCoupledCondensed
    probes = (cc._transducer_condensation_enabled, cc._dense_float64_enabled, cc._dense_refinement_enabled,
              cc._fem_float64_enabled, cc._interface_flux_elimination_enabled, cc._interface_mass_overlap_enabled,
              cc._interface_blocks_enabled, cc._demand_reconstruction_enabled, cc._fem_solver_selection,
              cc._interface_mass_solver_selection)
    cleared = [name => nothing for name in keys(ENV) if startswith(name, "BLAB_COUPLED_")]
    for defaults in (:metal, :cpu)
        unset = withenv(() -> [probe(defaults) for probe in probes], cleared...)
        # mumps=true keeps the resolved FEM solver instead of the host workload's UMFPACK override.
        pinned = withenv(() -> [probe(defaults) for probe in probes],
                         bundle.coupled_workload_environment(; mumps=true, defaults)...)
        @test pinned == unset
    end
end

@testset "tiny coupled workload with beat_cpu defaults solves (strict)" begin
    bundle = CompiledWorkloadBundle
    request = bundle.JSON.parse(bundle.JSON.json(bundle.coupled_workload_request(; tiny=true)))
    withenv(bundle.coupled_workload_environment(; mumps=false, defaults=:cpu)...) do
        run = bundle.solve_coupled_workload(request)
        bundle.check_coupled_workload(run; mumps=false, defaults=:cpu)
        @test all(result["diagnostics"]["fem_matrix_precision"] == "float32" for result in run.results)
    end
    bundle.reset_compiled_workload_state!()
end
