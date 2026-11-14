module ThermoviscousDriverTests
using Test
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

function tv_request()
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    wall = only(filter(b -> b["id"] == "boundary:wall", request["compiled_system"]["boundaries"]))
    wall["parameters"] = Dict{String,Any}("thermoviscous_wall_losses" => "thin_boundary_layer")
    request["solver_options"]["phasor_convention"] = "exp(+i omega t)"
    return request
end

decode(q) = copy(reinterpret(q["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64,
                            base64decode(q["values"]["content_base64"])))

@testset "thermoviscous compiled contract and boundary selection" begin
    request = tv_request()
    @test validate_system_request(request) === nothing
    system = request["compiled_system"]
    regions = filter(r -> r["kind"] == "bounded_air", system["regions"])
    aggregate() = aggregate_fem_domains(system["meshes"], regions, system["boundaries"], Float64;
        moving_boundary_ids=Set(String(id) for c in system["components"] for id in c["boundary_ids"]))
    domains = aggregate()
    @test [w.boundary_id for w in domains.thermoviscous_walls] == ["boundary:wall"]
    wall = only(filter(b -> b["id"] == "boundary:wall", system["boundaries"]))
    wall["parameters"]["wall_impedance"] = Dict("model" => "miki")
    @test_throws ErrorException validate_system_request(request)
    @test_throws ErrorException aggregate()
    delete!(wall["parameters"], "wall_impedance")
    push!(system["components"][1]["boundary_ids"], "boundary:wall")
    @test_throws ErrorException validate_system_request(request)
    @test_throws ErrorException aggregate()
    pop!(system["components"][1]["boundary_ids"])
    for model in ("unknown", true)
        wall["parameters"]["thermoviscous_wall_losses"] = model
        @test_throws ErrorException validate_system_request(request)
    end

end

@testset "thermoviscous independent walls within one region" begin
    request = tv_request()
    system = request["compiled_system"]
    filter!(r -> r["kind"] == "bounded_air", system["regions"])
    filter!(m -> m["purpose"] == "fem_volume", system["meshes"])
    filter!(b -> b["region_id"] == "region:interior", system["boundaries"])
    empty!(system["interfaces"])
    second = only(filter(b -> b["kind"] == "interface", system["boundaries"]))
    second["kind"] = "rigid"
    wall = only(filter(b -> b["id"] == "boundary:wall", system["boundaries"]))
    aggregate() = aggregate_fem_domains(system["meshes"], system["regions"], system["boundaries"], Float64)
    operator(d) = prepare_thermoviscous_operator(d.mesh, d.thermoviscous_walls)
    first = aggregate()
    @test [w.boundary_id for w in first.thermoviscous_walls] == [wall["id"]]
    first_op = operator(first)
    second["parameters"]["thermoviscous_wall_losses"] = "thin_boundary_layer"
    both = aggregate()
    @test Set(w.boundary_id for w in both.thermoviscous_walls) == Set([wall["id"], second["id"]])
    both_op = operator(both)
    wall["parameters"]["thermoviscous_wall_losses"] = "off"
    second_only = aggregate()
    @test [w.boundary_id for w in second_only.thermoviscous_walls] == [second["id"]]
    second_op = operator(second_only)
    @test isempty(intersect(first_op.face_indices, second_op.face_indices))
    @test isapprox(both_op.stiffness, first_op.stiffness + second_op.stiffness)
    @test isapprox(both_op.mass, first_op.mass + second_op.mass)
    @test isapprox(both_op.area_m2, first_op.area_m2 + second_op.area_m2)
    empty!(second["parameters"])
    @test isempty(aggregate().thermoviscous_walls)
end

@testset "thermoviscous coupled production sweep and Float64 reassembly" begin
    request = tv_request()
    withenv(coupled_workload_environment(; mumps=false, defaults=:cpu)...) do
        enabled = solve_coupled_workload(request)
        @test enabled.outcome.solved_count == 2
        for result in enabled.results
            diagnostics = result["diagnostics"]["thermoviscous_wall_losses"]
            @test diagnostics["boundary_ids"] == ["boundary:wall"]
            @test diagnostics["treated_face_count"] == 8
            @test diagnostics["treated_area_m2"] > 0
            @test diagnostics["viscous_boundary_layer_m"] > 0
            @test all(q -> all(isfinite, decode(q)), result["quantities"])
        end
        double = withenv("BLAB_COUPLED_FEM_FLOAT64" => "on") do
            solve_coupled_workload(request)
        end
        @test all(r["diagnostics"]["fem_matrix_precision"] == "float64" for r in double.results)
        for (a,b) in zip(enabled.results, double.results), (qa,qb) in zip(a["quantities"], b["quantities"])
            @test decode(qa) ≈ decode(qb) rtol=5e-4
        end
        # Explicit Off must agree exactly with omission, including repeated worker use.
        loss = only(filter(b -> b["id"] == "boundary:wall", request["compiled_system"]["boundaries"]))["parameters"]
        loss["thermoviscous_wall_losses"] = "off"
        disabled = solve_coupled_workload(request)
        empty!(loss)
        omitted = solve_coupled_workload(request)
        for (a,b,c) in zip(enabled.results, disabled.results, omitted.results)
            @test !haskey(b["diagnostics"], "thermoviscous_wall_losses")
            @test any(norm(decode(qa)-decode(qb)) > 1e-6 * norm(decode(qb))
                      for (qa,qb) in zip(a["quantities"], b["quantities"]))
            for (qb,qc) in zip(b["quantities"], c["quantities"])
                @test decode(qb) == decode(qc)
            end
        end
    end
end

@testset "thermoviscous monolithic and condensed FEM-BEM agree" begin
    request = tv_request()
    system = request["compiled_system"]
    regions = filter(r -> r["kind"] == "bounded_air", system["regions"])
    domains = aggregate_fem_domains(system["meshes"], regions, system["boundaries"], Float64)
    fem = domains.mesh
    bem = translated_boundary_mesh(system["meshes"][1], Float64)
    mapping = build_conforming_interface_map(fem, bem,
        domains.fem_boundary_tag_by_id["boundary:fem-interface"], 2)
    tag = domains.fem_boundary_tag_by_id["boundary:radiator"]
    with_phasor_convention("exp(+i omega t)") do
        direct = build_coupled_system(fem,bem,mapping,700.0,343.0,1.21;
            quadrature_order=1, singular_order=1, thermoviscous_walls=domains.thermoviscous_walls)
        condensed = build_condensed_coupled_system(fem,bem,mapping,700.0,343.0,1.21;
            quadrature_order=1, singular_order=1, thermoviscous_walls=domains.thermoviscous_walls)
        try
            a = solve_coupled_system(direct,tag; radiator_velocity=1.0+0.3im)
            b = solve_condensed_coupled_system(condensed,tag; radiator_velocity=1.0+0.3im)
            for field in (:fem_pressure,:bem_pressure,:interface_flux,:bem_neumann)
                @test getproperty(a,field) ≈ getproperty(b,field) rtol=1e-9
            end
        finally
            release_coupled_system!(direct)
            release_condensed_coupled_system!(condensed)
        end
    end
end

@testset "thermoviscous standalone interior contract" begin
    request = tv_request()
    system = request["compiled_system"]
    filter!(r -> r["kind"] == "bounded_air", system["regions"])
    filter!(m -> m["purpose"] == "fem_volume", system["meshes"])
    filter!(b -> b["region_id"] == "region:interior", system["boundaries"])
    empty!(system["interfaces"])
    only(filter(b -> b["kind"] == "interface", system["boundaries"]))["kind"] = "rigid"
    request["solver_options"]["precision"] = "float64"
    request["outputs"] = [Dict("id" => "pressure", "quantity" => "fem_nodal_pressure",
        "target_ids" => [], "options" => Dict())]
    @test validate_system_request(request) === nothing
    positive = solve_coupled_workload(request)
    request["solver_options"]["phasor_convention"] = "exp(-i omega t)"
    negative = solve_coupled_workload(request)
    for (a,b) in zip(positive.results, negative.results)
        @test a["diagnostics"]["thermoviscous_wall_losses"]["treated_face_count"] == 8
        @test decode(only(a["quantities"])) ≈ conj.(decode(only(b["quantities"]))) rtol=1e-10
    end
end
if get(ENV, "BEAT_RUN_THERMOVISCOUS_CUDA", "0") == "1"
    @testset "thermoviscous CUDA monolithic and condensed parity" begin
        @test BeatEngineCore.cuda_module().functional()
        request = tv_request()
        request["frequencies_hz"] = [700.0]
        for condensed in (false, true)
            request["solver_options"]["static_condensation"] = condensed
            request["solver_options"]["bem_backend"] = "cpu"
            cpu = only(solve_coupled_workload(request).results)
            request["solver_options"]["bem_backend"] = "cuda"
            gpu = only(solve_coupled_workload(request).results)
            @test gpu["diagnostics"]["static_condensation_active"] == condensed
            @test gpu["diagnostics"]["thermoviscous_wall_losses"]["treated_face_count"] == 8
            for (a,b) in zip(cpu["quantities"], gpu["quantities"])
                @test all(isfinite, decode(b))
                @test decode(a) ≈ decode(b) rtol=5e-4
            end
        end
    end
end
end # module
