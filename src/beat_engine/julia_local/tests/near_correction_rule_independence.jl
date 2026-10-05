# Run standalone or from runtests.jl. Every disjoint pair is corrected, so
# changing the regular rule must leave all four operators unchanged. Legal
# fundamental domains ensure that reflected singular pairs use Duffy rules.
if !isdefined(@__MODULE__, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end

module NearCorrectionRuleIndependenceTests
using Test, LinearAlgebra, Printf, StaticArrays
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))
const T = Float64

"""An octahedron refined twice: 128 faces, plenty of non-adjacent pairs.

`domain` selects the seed faces: `:full` the whole octahedron, `:half` the
octants with x >= 0, `:quarter` those with x >= 0 and y >= 0 -- legal
fundamental domains for `:off`, `:x` and `:xy`. Mirroring a *whole* sphere about x = 0 maps the mesh
onto itself, so every image pair would be coincident and the assembly would be
integrating a 1/r kernel at zero separation -- a fixture that fails for reasons
that have nothing to do with the correction under test.
"""
function unit_sphere_mesh(domain::Symbol=:full)
    vertices = SVector{3,T}[
        SVector{3,T}(1, 0, 0), SVector{3,T}(-1, 0, 0), SVector{3,T}(0, 1, 0),
        SVector{3,T}(0, -1, 0), SVector{3,T}(0, 0, 1), SVector{3,T}(0, 0, -1),
    ]
    faces =
        domain == :quarter ? NTuple{3,Int}[(1, 3, 5), (3, 1, 6)] :
        domain == :half ? NTuple{3,Int}[(1, 3, 5), (4, 1, 5), (3, 1, 6), (1, 4, 6)] :
        NTuple{3,Int}[
            (1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5),
            (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6),
        ]
    for _ in 1:2
        midpoints = Dict{Tuple{Int,Int},Int}()
        refined = NTuple{3,Int}[]
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        for (a, b, c) in faces
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            append!(refined, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        end
        faces = refined
    end
    # Drop vertices no face references, or the -x octahedron pole would keep a
    # half domain from validating as one.
    used = sort(unique(Iterators.flatten(faces)))
    renumbered = Dict(old_index => new_index for (new_index, old_index) in enumerate(used))
    return BoundaryMesh(
        vertices[used],
        [(renumbered[a], renumbered[b], renumbered[c]) for (a, b, c) in faces],
        fill(1, length(faces)),
    )
end

const MESHES = Dict(
    :off => unit_sphere_mesh(),
    :x => snap_symmetry_planes(unit_sphere_mesh(:half), :x),
    :xy => snap_symmetry_planes(unit_sphere_mesh(:quarter), :xy),
)
const K = T(3.0)

function corrections(mesh, symmetry_mode::Symbol)
    selection = near_correction_selection(
        Dict{String,Any}(
            "near_correction_enabled" => true,
            "near_correction_cutoff" => 1.0e9,
            "near_correction_order" => 8,
        ),
        mesh,
        symmetry_mode,
    )
    selection === nothing && error("full-mesh selection returned no pairs")
    identity_cache = build_near_correction_cache(mesh, selection.identity_pairs, selection.top_order)
    image_caches = [
        build_near_correction_cache(mesh, pairs, selection.top_order; trial_transform=transform)
        for (transform, pairs) in selection.image_selections
    ]
    return identity_cache, image_caches, selection
end

function assemble(mesh, order::Int, identity_cache, image_caches, symmetry_mode::Symbol)
    return assemble_regular_galerkin_operators(
        mesh, build_p1_space(mesh), build_dp0_space(mesh), K, triangle_rule(T, order);
        skip_singular=false, singular_order=4, backend=:cpu,
        singular_cache=build_singular_correction_cache(mesh, 4),
        near_correction_cache=identity_cache, image_near_correction_cache=image_caches,
        symmetry_mode=symmetry_mode,
    )
end

relative(x, y) = norm(x - y) / norm(y)
const OPERATORS = (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)

@testset "near-correction rule independence" begin
    for symmetry_mode in (:off, :x, :xy)
        @testset "$symmetry_mode" begin
            mesh = MESHES[symmetry_mode]
            validate_symmetry_fundamental_domain!(mesh, symmetry_mode)
            identity_cache, image_caches, selection = corrections(mesh, symmetry_mode)
            expected_images = symmetry_mode == :off ? 0 : symmetry_mode == :x ? 1 : 3
            @test length(image_caches) == expected_images
            @test [cache.trial_transform.signs for cache in image_caches] ==
                [transform.signs for transform in symmetry_image_transforms(symmetry_mode)]
            @test all(cache -> cache.pair_count > 0, image_caches)
            @test identity_cache.pair_count > 0
            coarse = assemble(mesh, 1, identity_cache, image_caches, symmetry_mode)
            fine = assemble(mesh, 4, identity_cache, Tuple(image_caches), symmetry_mode)
            raw_coarse = assemble(mesh, 1, nothing, nothing, symmetry_mode)
            raw_fine = assemble(mesh, 4, nothing, nothing, symmetry_mode)
            @test coarse.near_pair_count == identity_cache.pair_count +
                sum(cache.pair_count for cache in image_caches; init=0)
            @test coarse.near_pair_quadrature_order == maximum(
                cache.correction_order for cache in (identity_cache, image_caches...))
            for name in OPERATORS
                corrected = relative(getfield(coarse, name), getfield(fine, name))
                uncorrected = relative(getfield(raw_coarse, name), getfield(raw_fine, name))
                @printf("near %-3s %-22s corrected %.3e uncorrected %.3e\n",
                    symmetry_mode, name, corrected, uncorrected)
                @test corrected < 1.0e-12
                @test uncorrected > 1.0e-6
            end
            # Preserve the existing single-cache call, and allow collections in
            # either keyword, including empty tuple/vector arguments.
            combined = assemble(mesh, 1, (identity_cache, image_caches...), (), symmetry_mode)
            for name in OPERATORS
                @test isequal(getfield(combined, name), getfield(coarse, name))
            end
            if symmetry_mode == :x
                single = assemble(mesh, 1, identity_cache, only(image_caches), symmetry_mode)
                for name in OPERATORS
                    @test isequal(getfield(single, name), getfield(coarse, name))
                end
            end
            empty = assemble(mesh, 1, (), [], symmetry_mode)
            for name in OPERATORS
                @test isequal(getfield(empty, name), getfield(raw_coarse, name))
            end
        end
    end
end

@testset "near-pair selection and backend refusal" begin
    mesh = MESHES[:xy]
    @test near_correction_selection(Dict(), mesh, :xy) === nothing
    for cutoff in (0.0, -1.0, Inf, NaN)
        @test_throws ErrorException near_correction_selection(
            Dict("near_correction_enabled"=>true, "near_correction_cutoff"=>cutoff), mesh, :xy)
    end
    @test_throws ErrorException near_correction_selection(
        Dict("near_correction_enabled"=>true, "near_correction_order"=>3), mesh, :xy)
    @test near_correction_order_for_ratio(0.1, 8) == 8
    @test near_correction_order_for_ratio(2.0, 8) == 4
    radii = [maximum(norm(v - mesh.centroids[i]) for v in mesh.face_vertices[i])
        for i in eachindex(mesh.faces)]
    for transform in (nothing, symmetry_image_transforms(:xy)...)
        selected = near_correction_pairs(mesh, radii, 1.2, 8, transform)
        exhaustive = Set{Tuple{Int,Int,Int}}()
        for i in eachindex(mesh.faces), j in eachindex(mesh.faces)
            transform === nothing && elements_are_adjacent(mesh.faces[i], mesh.faces[j]) && continue
            centroid = transform === nothing ? mesh.centroids[j] : reflect_point(transform, mesh.centroids[j])
            ratio = norm(mesh.centroids[i] - centroid) / (radii[i] + radii[j])
            ratio <= 1.2 && push!(exhaustive, (i, j, near_correction_order_for_ratio(ratio, 8)))
        end
        @test Set(selected) == exhaustive
        @test length(selected) == length(exhaustive)
    end
    identity_cache, image_caches, _ = corrections(mesh, :xy)
    p1, dp0, rule = build_p1_space(mesh), build_dp0_space(mesh), triangle_rule(T, 1)
    for backend in (:metal, :rocm)
        for keyword in (:near_correction_cache, :image_near_correction_cache,
                        :device_near_correction_cache, :device_image_near_correction_cache)
            @test_throws r"Near-singular correction is not implemented" assemble_regular_galerkin_operators(
                mesh, p1, dp0, K, rule; backend=backend, (keyword=>identity_cache,)...)
        end
    end
    @test_throws r"cache collections are supported only on CPU" assemble_regular_galerkin_operators(
        mesh, p1, dp0, K, rule; backend=:cuda, image_near_correction_cache=image_caches)
    # Refuse before mesh loading/device initialization, even on a machine
    # without the accelerator package and even when there would be no pairs.
    for backend in ("cuda", "metal", "rocm")
        request = Dict("config"=>Dict("near_correction_enabled"=>true), "beat_engine_backend"=>backend)
        @test_throws r"supported only on the CPU" solve_request_impl(request)
    end
end

function write_test_mesh(io, mesh)
    println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes\n", length(mesh.vertices))
    for (i, v) in enumerate(mesh.vertices)
        println(io, i, " ", join(v, " "))
    end
    println(io, "\$EndNodes\n\$Elements\n", length(mesh.faces))
    for (i, face) in enumerate(mesh.faces)
        println(io, i, " 2 2 1 1 ", join(face, " "))
    end
    println(io, "\$EndElements")
    flush(io)
end

@testset "source requests apply near correction" begin
    previous_fused = get(ENV, "BLAB_BEAT_FUSED_BM", nothing)
    try
        ENV["BLAB_BEAT_FUSED_BM"] = "1"
        for (mode, cutoff) in ((:off, 1.0e9), (:x, 1.0e9), (:xy, 1.0e9), (:off, 1.0e-9))
            mktemp() do path, io
                write_test_mesh(io, MESHES[mode])
                config = Dict{String,Any}(
                    "mesh_file"=>path, "scale_factor"=>1.0, "tag_throat"=>1,
                    "symmetry"=>String(mode), "min_angle"=>0.0, "max_angle"=>10.0,
                    "step_size"=>10.0, "distance"=>3.0, "quadrature_order"=>1,
                    "singular_order"=>4, "regular_quadrature_mode"=>"fixed",
                    "near_correction_enabled"=>true, "near_correction_cutoff"=>cutoff,
                    "near_correction_order"=>8,
                )
                request = Dict("config"=>config, "frequencies_hz"=>[100.0], "beat_engine_backend"=>"cpu")
                mktemp() do _, output
                    redirect_stdout(output) do
                        solve_request(request)
                    end
                    flush(output)
                    seekstart(output)
                    events = [JSON.parse(line) for line in eachline(output)]
                    @test last(events)["type"] == "completed"
                    result = only([event["result"] for event in events if event["type"] == "result"])
                    diagnostics = result["diagnostics"]
                    if cutoff > 1.0
                        identity_cache, images, _ = corrections(MESHES[mode], mode)
                        @test diagnostics["near_pair_count"] == identity_cache.pair_count +
                            sum(cache.pair_count for cache in images; init=0)
                        @test diagnostics["near_pair_quadrature_order"] == maximum(
                            cache.correction_order for cache in (identity_cache, images...))
                        @test diagnostics["near_image_transform_count"] == length(symmetry_image_transforms(mode))
                        @test any(event -> event["type"] == "status" &&
                            occursin("Near-singular correction", event["message"]), events)
                    else
                        @test diagnostics["near_pair_count"] == 0
                        @test diagnostics["near_pair_quadrature_order"] == 0
                        @test diagnostics["near_image_transform_count"] == 0
                    end
                    @test diagnostics["regular_assembly_mode"] in ("cpu_colored_threads", "cpu_serial")
                    for component in ("real", "imag")
                        @test all(isfinite, Iterators.flatten(result["horizontal_pressure"][component]))
                    end
                    @test any(!iszero, Iterators.flatten(result["horizontal_pressure"]["real"]))
                end
            end
        end
    finally
        previous_fused === nothing ? delete!(ENV, "BLAB_BEAT_FUSED_BM") :
            (ENV["BLAB_BEAT_FUSED_BM"] = previous_fused)
    end
end
end
