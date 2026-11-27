# Fixtures for the large-panel gates. Sizes are chosen by k*h, never by frequency.

function _large_panel_pair_mesh(::Type{T}; aspect::Real=1.0) where {T<:AbstractFloat}
    h = 0.0526
    a = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(h, 0, 0), SVector{3,Float64}(h / 2, sqrt(3) * h / (2aspect), 0)]
    rotate = [1.0 0.0 0.0; 0.0 cos(0.7) -sin(0.7); 0.0 sin(0.7) cos(0.7)]
    b = [SVector{3,Float64}(rotate * p + SVector(2h, h, 1.5h)) for p in a]
    return BoundaryMesh(SVector{3,T}.([a; b]), [(1, 2, 3), (4, 5, 6)], [1, 1]), h
end

function _large_panel_coincident_mesh(::Type{T}) where {T<:AbstractFloat}
    h = 0.05
    vertices = SVector{3,T}[SVector{3,T}(0, 0, 0), SVector{3,T}(h, 0, 0), SVector{3,T}(h / 2, sqrt(3) * h / 2, 0)]
    return BoundaryMesh(vertices, [(1, 2, 3)], [1]), Float64(h)
end

function _large_panel_edge_mesh(::Type{T}) where {T<:AbstractFloat}
    h = 0.05
    vertices = SVector{3,T}[
        SVector{3,T}(0, 0, 0), SVector{3,T}(h, 0, 0), SVector{3,T}(h / 2, sqrt(3) * h / 2, 0),
        SVector{3,T}(h / 2, -sqrt(3) * h / 4, sqrt(3) * h / 4),
    ]
    return BoundaryMesh(vertices, [(1, 2, 3), (2, 1, 4)], [1, 1]), Float64(h)
end

# Blocks of the first panel's test functions against the second panel.
_large_panel_block(name::Symbol) = name in (:single_layer, :adjoint_double_layer) ? (1:3, 2) : (1:3, 4:6)

# One level of midpoint subdivision, with the map from the original P1 and DP0
# coefficients: the original basis functions on the partitioned surface.
function _large_panel_subdivide(mesh, P, Q)
    vertices = copy(mesh.vertices)
    rows = [collect(P[i, :]) for i in axes(P, 1)]
    midpoint = Dict{Tuple{Int,Int},Int}()
    function mid(a, b)
        return get!(midpoint, minmax(a, b)) do
            push!(vertices, (vertices[a] + vertices[b]) / 2)
            push!(rows, (rows[a] + rows[b]) / 2)
            length(vertices)
        end
    end
    faces = NTuple{3,Int}[]
    q_rows = Vector{Float64}[]
    for (face_index, (a, b, c)) in enumerate(mesh.faces)
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        append!(faces, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        append!(q_rows, [collect(Q[face_index, :]) for _ in 1:4])
    end
    return BoundaryMesh(vertices, faces, ones(Int, length(faces))), reduce(vcat, transpose.(rows)), reduce(vcat, transpose.(q_rows))
end

function _large_panel_partition(mesh, levels)
    P = Matrix{Float64}(I, length(mesh.vertices), length(mesh.vertices))
    Q = Matrix{Float64}(I, length(mesh.faces), length(mesh.faces))
    for _ in 1:levels
        mesh, P, Q = _large_panel_subdivide(mesh, P, Q)
    end
    return mesh, P, Q
end

function _large_panel_partitioned_blocks(mesh, k, levels)
    fine, P, Q = _large_panel_partition(mesh, levels)
    operators = assemble_regular_galerkin_operators_cpu(
        fine, build_p1_space(fine), build_dp0_space(fine), k, triangle_rule(Float64, 4);
        skip_singular=true, singular_order=4,
    )
    return (
        single_layer=(P' * operators.single_layer * Q)[_large_panel_block(:single_layer)...],
        double_layer=(P' * operators.double_layer * P)[_large_panel_block(:double_layer)...],
        adjoint_double_layer=(P' * operators.adjoint_double_layer * Q)[_large_panel_block(:adjoint_double_layer)...],
        hypersingular=(P' * operators.hypersingular * P)[_large_panel_block(:hypersingular)...],
    )
end

function _large_panel_partitioned_field(mesh, pressure, neumann, k, points, levels)
    fine, P, Q = _large_panel_partition(mesh, levels)
    return evaluate_galerkin_field_cpu(
        points, fine, ComplexF64.(P * pressure), ComplexF64.(Q * neumann), k,
        build_field_evaluation_cache(fine, triangle_rule(Float64, 4)),
    )
end

# A closed octahedral sphere stretched along +z, so the top panels are about
# three times the bottom ones, cut along x = 0 and y = 0 for the reduced forms.
function _large_panel_sphere(::Type{T}; symmetry::Symbol=:off, level::Int=2) where {T<:AbstractFloat}
    if symmetry == :ground
        mesh = _large_panel_sphere(T; level=level)
        return BoundaryMesh([v + SVector{3,T}(0, 2, 0) for v in mesh.vertices], mesh.faces, mesh.physical_tags)
    end
    vertices = SVector{3,Float64}[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = NTuple{3,Int}[
        (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 1, 5),
        (2, 1, 6), (3, 2, 6), (4, 3, 6), (1, 4, 6),
    ]
    for _ in 1:level
        midpoint = Dict{Tuple{Int,Int},Int}()
        mid(a, b) = get!(midpoint, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3,Int}[]
        for (a, b, c) in faces
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            append!(refined, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        end
        faces = refined
    end
    # Exact plane coordinates, so the symmetry validation sees no drift.
    stretched = [SVector{3,T}(T(p[1]), T(p[2]), T(p[3] > 0 ? 3p[3] : p[3])) for p in vertices]
    keep(face) = all(
        (symmetry == :off) ||
        (symmetry == :x && vertices[v][1] >= -1e-12) ||
        (symmetry == :xy && vertices[v][1] >= -1e-12 && vertices[v][2] >= -1e-12)
        for v in face
    )
    kept = filter(keep, faces)
    used = sort!(unique(v for face in kept for v in face))
    position = Dict(v => i for (i, v) in enumerate(used))
    return BoundaryMesh(stretched[used], [map(v -> position[v], face) for face in kept], ones(Int, length(kept)))
end

function _large_panel_write_mesh(path, mesh)
    open(path, "w") do io
        println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes")
        println(io, length(mesh.vertices))
        for (index, vertex) in enumerate(mesh.vertices)
            println(io, index, " ", join(vertex, " "))
        end
        println(io, "\$EndNodes\n\$Elements\n", length(mesh.faces))
        for (index, face) in enumerate(mesh.faces)
            tag = mesh.physical_tags[index]
            println(io, index, " 2 2 ", tag, " ", tag, " ", join(face, " "))
        end
        println(io, "\$EndElements")
    end
end

function _large_panel_capture(f, json)
    mktemp() do _, output
        outcome = redirect_stdout(f, output)
        flush(output)
        seekstart(output)
        return outcome, json.parse.(readlines(output))
    end
end

function _large_panel_binary_array(directory, name, values, driver)
    # The retained-field protocol carries matrices in row-major order.
    flat = ndims(values) == 1 ? vec(values) : vec(permutedims(values))
    filename = "$name.bin"
    write(joinpath(directory, filename), flat)
    return Dict("file" => filename, "dtype" => driver.binary_dtype_name(eltype(values)),
        "shape" => collect(size(values)), "order" => "C", "byte_order" => "little",
        "offset" => 0, "nbytes" => sizeof(eltype(values)) * length(values))
end

function _large_panel_mixed_k(mesh)
    edges = LargePanelRules(mesh).max_edges
    limit = BeatEngineCore.beat_large_panel_kh_limit()
    return 0.5 * (limit / minimum(edges) + limit / maximum(edges))
end

function _large_panel_field(mesh, symmetry, k::T, points; singular_order::Int=4) where {T<:AbstractFloat}
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(T, 4)
    plan = large_panel_plan(LargePanelRules(mesh, singular_order), k)
    q = Complex{T}[Complex{T}(1 + mesh.centroids[face][3]) for face in eachindex(mesh.faces), _ in 1:1]
    system = assemble_burton_miller_neumann_system_cpu(
        mesh, p1, dp0, q, k, rule; singular_order=singular_order, symmetry_mode=symmetry, large_panels=plan,
        identity_p1_p1=assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry),
        identity_p1_dp0=assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry),
    )
    pressure = system.matrix \ system.rhs
    cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry, large_panels=plan)
    return evaluate_galerkin_field_cpu(points, mesh, pressure[:, 1], q[:, 1], k, cache)
end
