# Experimental, opt-in block-low-rank feedback operator. Dense column slabs are
# assembled by the existing quadrature implementation, never the full R matrix.
# This first spike measures compressibility; it is not a fast H-matrix builder.

function rhs_spatial_leaves(points, leaf_size)
    leaf_size >= 8 || error("Compression leaf_size must be at least 8")
    leaves = Vector{Vector{Int}}()
    function split(indices)
        if length(indices) <= leaf_size
            push!(leaves, indices)
        else
            lo = [minimum(points[i][d] for i in indices) for d in 1:3]
            hi = [maximum(points[i][d] for i in indices) for d in 1:3]
            axis = argmax(hi - lo)
            sort!(indices; by=i -> (points[i][axis], i))
            mid = length(indices) ÷ 2
            split(indices[1:mid]); split(indices[mid+1:end])
        end
    end
    split(collect(eachindex(points)))
    return leaves
end

rhs_box(points) = ([minimum(p[d] for p in points) for d in 1:3],
                   [maximum(p[d] for p in points) for d in 1:3])

function rhs_row_tree(leaves, boxes)
    nodes = Any[]
    function merge_tree(first, last)
        if first == last
            push!(nodes, (indices=leaves[first], box=boxes[first], children=Int[]))
        else
            mid = (first+last) ÷ 2
            left = merge_tree(first, mid); right = merge_tree(mid+1, last)
            a, b = nodes[left], nodes[right]
            box = (min.(a.box[1], b.box[1]), max.(a.box[2], b.box[2]))
            push!(nodes, (indices=vcat(a.indices, b.indices), box=box, children=[left,right]))
        end
        return length(nodes)
    end
    root = merge_tree(1, length(leaves))
    return nodes, root
end

function rhs_admissible_rows(nodes, root, box, eta=1.0)
    selected = Int[]
    function visit(i)
        if isempty(nodes[i].children) || rhs_far(nodes[i].box, box, eta)
            push!(selected, i)
        else
            foreach(visit, nodes[i].children)
        end
    end
    visit(root)
    return selected
end

function rhs_far(row_box, col_box, eta=1.0)
    a, b = row_box; c, d = col_box
    distance = norm(max.(0, max.(a - d, c - b)))
    # Strong admissibility, including the reflected source support at Y=0.
    ci = [c[1], -d[2], c[3]]; di = [d[1], -c[2], d[3]]
    image_distance = norm(max.(0, max.(a - di, ci - b)))
    return max(norm(b-a), norm(d-c)) <= eta * min(distance, image_distance)
end

function rhs_compress_block(matrix, tolerance; max_rank=64)
    0 < tolerance < 1 || error("Compression tolerance must be between zero and one")
    all(isfinite, matrix) || error("Nonfinite RHS block")
    m, n = size(matrix)
    residual = copy(matrix)
    original_norm = norm(matrix)
    us = Vector{Vector{eltype(matrix)}}(); vs = Vector{Vector{eltype(matrix)}}()
    limit = min(max_rank, min(m, n), floor(Int, (m*n-1)/(m+n)))
    relative = original_norm == 0 ? 0.0 : 1.0
    for _ in 1:limit
        relative <= tolerance && break
        _, pivot = findmax(abs2, residual)
        i, j = Tuple(pivot)
        u = copy(residual[:, j]); v = residual[i, :] / residual[i, j]
        push!(us, u); push!(vs, v)
        # Complex outer product without conjugation.
        BLAS.geru!(-one(eltype(matrix)), u, v, residual)
        relative = norm(residual) / original_norm
    end
    if relative <= tolerance
        U = isempty(us) ? zeros(eltype(matrix), m, 0) : reduce(hcat, us)
        V = isempty(vs) ? zeros(eltype(matrix), 0, n) : Matrix(transpose(reduce(hcat, vs)))
        # Check the stored factors as well as the incrementally updated residual.
        relative = original_norm == 0 ? 0.0 : norm(matrix - U*V) / original_norm
        relative <= tolerance && return (dense=nothing, u=U, v=V, rank=size(U, 2), error=relative)
    end
    return (dense=matrix, u=nothing, v=nothing, rank=min(m,n), error=0.0)
end

mutable struct DeployCompressedRhs
    blocks::Vector{Any}
    rows::Vector{Any}
    columns::Vector{Any}
    row_count::Int
    report::Dict{String,Any}
    packed::Any
end
DeployCompressedRhs(blocks, rows, columns, count, report) = DeployCompressedRhs(blocks, rows, columns, count, report, nothing)

function pack_deploy_rhs!(op)
    cuda = BeatEngineCore.CUDA_MODULE
    arrays = Dict(name => Int32[] for name in (:columns,:column_starts,:ms,:ns,:ranks,
        :offsets,:v_offsets,:rank_offsets,:rank_blocks,:rank_local,:task_blocks,:task_local,:task_rows))
    column_starts = Int[]
    for c in op.columns
        push!(column_starts, length(arrays[:columns]))
        append!(arrays[:columns], Array(c))
    end
    entries = sum(b.dense === nothing ? length(b.u)+length(b.v) : length(b.dense) for b in op.blocks)
    entries < typemax(Int32) || error("Experimental packed RHS exceeds Int32 storage indexing")
    data = Vector{ComplexF32}(undef, entries)
    position = 0
    for (index,b) in enumerate(op.blocks)
        rows = Array(op.rows[b.row]); m = length(rows); n = length(op.columns[b.column])
        rank = b.dense === nothing ? size(b.u,2) : -1
        push!(arrays[:column_starts], column_starts[b.column])
        push!(arrays[:ms],m); push!(arrays[:ns],n); push!(arrays[:ranks],rank)
        push!(arrays[:offsets],position)
        left = b.dense === nothing ? b.u : b.dense
        copyto!(data,position+1,vec(Array(left)),1,length(left)); position += length(left)
        push!(arrays[:v_offsets],position)
        if b.v !== nothing
            copyto!(data,position+1,vec(Array(b.v)),1,length(b.v)); position += length(b.v)
        end
        push!(arrays[:rank_offsets],length(arrays[:rank_blocks]))
        append!(arrays[:rank_blocks],fill(Int32(index),max(rank,0)))
        append!(arrays[:rank_local],1:max(rank,0))
        append!(arrays[:task_blocks],fill(Int32(index),m)); append!(arrays[:task_local],1:m)
        append!(arrays[:task_rows],rows)
    end
    device = Dict{Symbol,Any}()
    try
        required = sizeof(data) + sum(sizeof(a) for a in values(arrays))
        # Slab and quadrature buffers have been freed but may still occupy the
        # CUDA allocator pool. Reclaim those before judging the live footprint.
        cuda.reclaim()
        available = cuda.free_memory()
        op.report["packed_bytes"] = required
        op.report["available_before_upload_bytes"] = available
        required + 256*1024^2 < available || error("Packed compressed RHS exceeds available VRAM ($required bytes required, $available available)")
        device[:data] = cuda.CuArray(data)
        for (name,array) in arrays
            device[name] = cuda.CuArray(array)
        end
        op.packed = (; device...)
        op.report["packed_bytes"] = required
    catch
        foreach(cuda.unsafe_free!, values(device))
        rethrow()
    end
    return op
end

function release_deploy_compressed_rhs!(op)
    cuda = BeatEngineCore.CUDA_MODULE
    if op.packed !== nothing
        foreach(cuda.unsafe_free!, values(op.packed))
        op.packed = nothing
    end
    for b in op.blocks
        for a in (b.dense, b.u, b.v)
            a === nothing || a isa Array || cuda.unsafe_free!(a)
        end
    end
    for a in (op.rows..., op.columns...)
        a isa Array || cuda.unsafe_free!(a)
    end
    empty!(op.blocks); empty!(op.rows); empty!(op.columns)
end

function apply_deploy_compressed_rhs(op, q)
    cuda = BeatEngineCore.CUDA_MODULE
    op.packed === nothing || return BeatEngineCore.apply_cuda_packed_rhs(op.packed, q, op.row_count)
    result = cuda.zeros(eltype(q), op.row_count)
    try
        for (j, columns) in enumerate(op.columns)
            x = q[columns]
            try
                for b in op.blocks
                    b.column == j || continue
                    t = nothing
                    y = nothing
                    try
                        if b.dense !== nothing
                            y = b.dense * x
                        elseif size(b.u, 2) == 0
                            continue
                        else
                            t = b.v * x
                            y = b.u * t
                        end
                        view(result, op.rows[b.row]) .+= y
                    finally
                        t === nothing || cuda.unsafe_free!(t)
                        y === nothing || cuda.unsafe_free!(y)
                    end
                end
            finally
                cuda.unsafe_free!(x)
            end
        end
        cuda.synchronize()
        return result
    catch
        cuda.unsafe_free!(result)
        rethrow()
    end
end

function build_deploy_compressed_rhs(mesh, assemble_columns, options; progress=println)
    cuda = BeatEngineCore.CUDA_MODULE
    tolerance = Float64(get(options, "tolerance", 1e-4))
    0 < tolerance < 1 || error("Invalid compression tolerance")
    leaf = Int(get(options, "leaf_size", 256))
    eta = Float64(get(options, "admissibility", 1.0))
    isfinite(eta) && eta > 0 || error("Admissibility must be finite and positive")
    survey = Bool(get(options, "survey_only", false))
    rows = rhs_spatial_leaves(mesh.vertices, leaf)
    centers = [(mesh.vertices[f[1]] + mesh.vertices[f[2]] + mesh.vertices[f[3]]) / 3 for f in mesh.faces]
    columns = rhs_spatial_leaves(centers, leaf)
    # Row support includes every incident triangle, not just the node positions.
    support = [Int[] for _ in mesh.vertices]
    for f in mesh.faces, i in f
        append!(support[i], f)
    end
    row_boxes = [rhs_box(mesh.vertices[unique(vcat(support[r]...))]) for r in rows]
    col_boxes = [rhs_box(mesh.vertices[unique(vcat([collect(mesh.faces[i]) for i in c]...))]) for c in columns]
    nodes, root = rhs_row_tree(rows, row_boxes)
    if !survey
        rows = [node.indices for node in nodes]
        row_boxes = [node.box for node in nodes]
    end
    report = Dict{String,Any}("experimental"=>true, "survey_only"=>survey,
        "tolerance"=>tolerance, "leaf_size"=>leaf, "admissibility"=>eta, "blocks"=>Any[],
        "dense_bytes"=>8*length(mesh.vertices)*length(mesh.faces), "stored_bytes"=>0,
        "construction"=>"exact CUDA column slabs, host complete-pivot ACA with checked Frobenius error")
    op = DeployCompressedRhs(Any[], Any[], Any[], length(mesh.vertices), report)
    start = time()
    try
        if !survey
            append!(op.rows, [Int32.(r) for r in rows])
            append!(op.columns, [Int32.(c) for c in columns])
        end
        selected_columns = survey ? unique(round.(Int, range(1, length(columns); length=min(4, length(columns))))) : eachindex(columns)
        for j in selected_columns
            (survey || j == 1 || j % 16 == 0 || j == length(columns)) && progress("Experimental RHS slab $j/$(length(columns))")
            slab_device = assemble_columns(columns[j])
            slab = try Array(slab_device) finally cuda.unsafe_free!(slab_device) end
            selected_rows = survey ? unique(round.(Int, range(1, length(rows); length=min(8, length(rows))))) :
                rhs_admissible_rows(nodes, root, col_boxes[j], eta)
            for i in selected_rows
                matrix = slab[rows[i], :]
                far = rhs_far(row_boxes[i], col_boxes[j], eta)
                compressed = far ? rhs_compress_block(matrix, tolerance) :
                    (dense=matrix, u=nothing, v=nothing, rank=min(size(matrix)...), error=0.0)
                bytes = compressed.dense === nothing ? 8*(length(compressed.u)+length(compressed.v)) : 8*length(matrix)
                entry = Dict("row_leaf"=>i, "column_leaf"=>j, "shape"=>collect(size(matrix)),
                    "far"=>far, "rank"=>compressed.rank, "relative_frobenius_error"=>compressed.error,
                    "compressed"=>compressed.dense === nothing, "bytes"=>bytes)
                if survey
                    sigma = svdvals(matrix)
                    energy = reverse(cumsum(reverse(abs2.(Float64.(sigma)))))
                    entry["svd_rank"] = sum(energy .> tolerance^2 * sum(abs2, sigma))
                    entry["singular_values"] = sigma
                end
                push!(report["blocks"], entry)
                report["stored_bytes"] += bytes
                if !survey
                    push!(op.blocks, (row=i, column=j,
                        dense=compressed.dense, u=compressed.u, v=compressed.v))
                end
            end
        end
        if !survey
            pack_deploy_rhs!(op)
            empty!(op.blocks); empty!(op.rows); empty!(op.columns)
        end
        report["block_count"] = length(report["blocks"])
        report["compressed_block_count"] = count(b -> b["compressed"], report["blocks"])
        report["build_s"] = time() - start
        report_path = String(get(options, "report_path", ""))
        isempty(report_path) || open(io -> JSON.print(io, report, 2), report_path, "w")
        return op
    catch err
        report["build_s"] = time() - start
        report["failure"] = sprint(showerror, err)
        report_path = String(get(options, "report_path", ""))
        isempty(report_path) || open(io -> JSON.print(io, report, 2), report_path, "w")
        release_deploy_compressed_rhs!(op)
        rethrow()
    end
end
