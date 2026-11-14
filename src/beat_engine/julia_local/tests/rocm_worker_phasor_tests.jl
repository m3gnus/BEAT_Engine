# Public request and retained-field qualification; requires functional hardware.
module RocmWorkerPhasorTests
using Test, LinearAlgebra, StaticArrays, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)
include(joinpath(@__DIR__, "..", "worker_backend_availability.jl"))
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

function capture_result(request)
    mktemp() do _, output
        redirect_stdout(output) do
            solve_request(request; event_mode=true)
        end
        flush(output)
        seekstart(output)
        events = [JSON.parse(line) for line in eachline(output)]
        @test last(events)["type"] == "result"
        only([event["result"] for event in events if event["type"] == "result"])
    end
end

decode(result) = Dict(item["quantity"] => collect(reinterpret(
    item["values"]["dtype"] == "complex128" ? ComplexF64 : ComplexF32,
    base64decode(item["values"]["content_base64"]))) for item in result["quantities"])

function write_mesh(path, mesh)
    open(path, "w") do io
        println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n", length(mesh.vertices))
        for (i, vertex) in enumerate(mesh.vertices)
            println(io, i, " ", join(vertex, " "))
        end
        println(io, "\$EndNodes\n\$Elements\n", length(mesh.faces))
        for (i, face) in enumerate(mesh.faces)
            println(io, i, " 2 2 2 2 ", join(face, " "))
        end
        println(io, "\$EndElements")
    end
end

function binary_array(directory, name, values, shape)
    filename = name * ".bin"
    open(joinpath(directory, filename), "w") do io
        write(io, values)
    end
    Dict("file" => filename, "offset" => 0, "nbytes" => sizeof(values),
        "dtype" => binary_dtype_name(eltype(values)), "shape" => shape,
        "order" => "C", "byte_order" => "little")
end

@testset "ROCm public compiled and retained fields conjugate" begin
    @test worker_backend_availability()["rocm"]["phasor_conventions"] ==
          [NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR]
    for T in (Float32, Float64), symmetry in (:off, :ground)
        @testset "$T $symmetry" begin
            mktempdir() do directory
                # Dyadic coordinates also survive the retained Float32 geometry wire
                # format exactly when the solve itself uses Float64.
                offset = symmetry == :ground ? one(T) : zero(T)
                vertices = [SVector{3,T}(v) + SVector{3,T}(0, offset, 0) for v in
                    ((.125,0,0),(-.125,0,0),(0,.125,0),(0,-.125,0),(0,0,.125),(0,0,-.125))]
                faces = [(1,3,5),(3,2,5),(2,4,5),(4,1,5),(3,1,6),(2,3,6),(4,2,6),(1,4,6)]
                mesh = BoundaryMesh(vertices, faces, fill(2, length(faces)))
                mesh_path = joinpath(directory, "surface.msh")
                write_mesh(mesh_path, mesh)
                points = [[2., 3., 1.], [-2., 3., 2.]]
                request = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "example-exterior-request.json"))
                request["compiled_system"]["meshes"][1]["file"] = mesh_path
                request["compiled_system"]["meshes"][1]["scale_to_m"] = 1.0
                request["frequencies_hz"] = [500.0]
                precision = T == Float32 ? "float32" : "float64"
                request["solver_options"] = Dict("precision"=>precision, "bem_backend"=>"rocm",
                    "symmetry"=>String(symmetry), "quadrature_order"=>3, "singular_order"=>3,
                    "burton_miller_assembly"=>"operator_matrices", "regular_quadrature_mode"=>"fixed")
                request["outputs"] = [Dict("id"=>q, "quantity"=>q, "target_ids"=>[],
                    "options"=>q == "exterior_pressure" ? Dict("points_m"=>points) : Dict())
                    for q in ("exterior_pressure", "bem_boundary_pressure", "bem_boundary_neumann")]
                snapshots = []
                first_cache = nothing
                tolerance = T == Float32 ? 5e-4 : 1e-8
                try
                    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR, NEGATIVE_TIME_PHASOR)
                        request["solver_options"]["phasor_convention"] = convention
                        result = capture_result(request)
                        @test result["diagnostics"]["bem_backend"] == "rocm"
                        @test result["diagnostics"]["phasor_convention"] == convention
                        quantities = decode(result)
                        @test norm(quantities["exterior_pressure"]) > 0
                        reference_request = deepcopy(request)
                        reference_request["solver_options"]["bem_backend"] = "cpu"
                        reference = decode(capture_result(reference_request))
                        for key in keys(quantities)
                            @test isapprox(quantities[key], reference[key]; rtol=tolerance)
                        end
                        arrays = Dict(
                            "boundary_points_m" => binary_array(directory, "vertices", Float32.(reduce(vcat, collect.(vertices))), [length(vertices), 3]),
                            "boundary_triangles" => binary_array(directory, "triangles", Int64.(reduce(vcat, collect.(faces))) .- 1, [length(faces), 3]),
                            "points_m" => binary_array(directory, "points", Float32.(reduce(vcat, points)), [length(points), 3]),
                            "boundary_pressure" => binary_array(directory, "pressure", quantities["bem_boundary_pressure"], [length(vertices)]),
                            "boundary_normal_derivative" => binary_array(directory, "neumann", quantities["bem_boundary_neumann"], [length(faces)]),
                        )
                        field_request = Dict("binary_array_schema_version"=>1, "binary_arrays"=>arrays,
                            "domain_key"=>"phasor-reuse", "precision"=>precision, "bem_backend"=>"rocm",
                            "symmetry"=>String(symmetry), "quadrature_order"=>3, "frequency_hz"=>500.,
                            "sound_speed_m_per_s"=>343., "phasor_convention"=>convention)
                        retained = evaluate_bem_field_request(field_request, joinpath(directory, "request.json"))
                        @test isapprox(retained, quantities["exterior_pressure"]; rtol=tolerance)
                        cache = only(values(BEM_FIELD_EVALUATION_CACHES))
                        first_cache === nothing ? (first_cache = cache.field_cache) : (@test cache.field_cache === first_cache)
                        @test phasor_convention() == NEGATIVE_TIME_PHASOR
                        push!(snapshots, quantities)
                    end
                    for key in keys(snapshots[1])
                        @test isapprox(snapshots[2][key], conj.(snapshots[1][key]); rtol=tolerance)
                        @test isapprox(snapshots[3][key], snapshots[1][key]; rtol=tolerance)
                    end
                finally
                    release_all_bem_field_evaluation_caches!()
                end
            end
        end
    end
end

@testset "ROCm public coupled requests conjugate" begin
    for precision in ("float32", "float64"), condensed in (false, true)
        @testset "$precision condensed=$condensed" begin
            request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true, bem_backend="rocm")))
            request["frequencies_hz"] = [500.0]
            request["solver_options"]["precision"] = precision
            request["solver_options"]["static_condensation"] = condensed
            tolerance = precision == "float32" ? 5e-4 : 1e-8
            snapshots = []
            for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
                request["solver_options"]["phasor_convention"] = convention
                result = capture_result(request)
                @test result["diagnostics"]["bem_backend"] == "rocm"
                @test startswith(result["diagnostics"]["linear_solver"], "rocm_")
                @test result["diagnostics"]["phasor_convention"] == convention
                quantities = decode(result)
                reference_request = deepcopy(request)
                reference_request["solver_options"]["bem_backend"] = "cpu"
                reference = decode(capture_result(reference_request))
                for key in keys(quantities)
                    @test norm(quantities[key]) > 0
                    @test isapprox(quantities[key], reference[key]; rtol=tolerance)
                end
                push!(snapshots, quantities)
                @test phasor_convention() == NEGATIVE_TIME_PHASOR
            end
            for key in keys(snapshots[1])
                @test isapprox(snapshots[2][key], conj.(snapshots[1][key]); rtol=tolerance)
            end
        end
    end
end

# Reuse the public source-request conjugation gate with an explicit GPU backend.
old_backend = get(ENV, "BLAB_TEST_SOURCE_BACKEND", nothing)
try
    ENV["BLAB_TEST_SOURCE_BACKEND"] = "rocm"
    include(joinpath(@__DIR__, "source_driver_phasor_tests.jl"))
finally
    old_backend === nothing ? delete!(ENV, "BLAB_TEST_SOURCE_BACKEND") : (ENV["BLAB_TEST_SOURCE_BACKEND"] = old_backend)
end
end
