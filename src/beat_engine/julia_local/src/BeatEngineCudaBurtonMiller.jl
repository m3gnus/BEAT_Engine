# A scatter destination for an exact subset of DP0 columns. The regular,
# singular, near and image kernels all use the same global scatter indices.
# Device fields are converted explicitly; the owning arrays remain alive until
# the synchronized assembly has completed.
struct CudaRhsColumnTile{A,M}
    values::A
    column_map::M
    row_count::Int
end
Base.ndims(::CudaRhsColumnTile) = 2
Base.size(a::CudaRhsColumnTile, d::Int) = d == 1 ? a.row_count : length(a.column_map)
@inline function _cuda_atomic_add!(a::CudaRhsColumnTile, index, value)
    row = mod1(index, a.row_count)
    column = a.column_map[(index - 1) ÷ a.row_count + 1]
    if column > 0
        _cuda_atomic_add!(a.values, row + (column - 1) * a.row_count, value)
    end
    return nothing
end

function _cuda_bm_identity_kernel!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    areas,
    faces,
    q_neumann,
    inverse_k,
    p1_dof_count,
    face_count,
    rhs_only,
    identity_p1_p1_block=nothing,
)
    face_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    face_index > face_count && return nothing
    row1 = faces[face_index]
    row2 = faces[face_index + face_count]
    row3 = faces[face_index + 2 * face_count]
    area = areas[face_index]
    diagonal = area / typeof(area)(6)
    off_diagonal = area / typeof(area)(12)

    if !rhs_only
        for (entry, (row, col, value)) in enumerate((
            (row1, row1, diagonal), (row1, row2, off_diagonal), (row1, row3, off_diagonal),
            (row2, row1, off_diagonal), (row2, row2, diagonal), (row2, row3, off_diagonal),
            (row3, row1, off_diagonal), (row3, row2, off_diagonal), (row3, row3, diagonal),
        ))
            # Optional quadrature-consistent mass block for the main exterior
            # solver, including its deliberately underintegrated order-1 rule.
            # The P1 mass block is symmetric, so row/column tuple order agrees.
            if identity_p1_p1_block !== nothing
                value = area * identity_p1_p1_block[entry]
            end
            _cuda_atomic_add!(matrix_re, row + (col - 1) * p1_dof_count, typeof(area)(0.5) * value)
        end
    end

    # -0.5 * (i/k) * M_P1,DP0 * q; every local P1/DP0 entry is area/3.
    rhs_scale = area * inverse_k / typeof(area)(6)
    q = q_neumann[face_index]
    for row in (row1, row2, row3)
        row = ndims(rhs_re) == 2 ? row + (face_index - 1) * size(rhs_re, 1) : row
        _cuda_atomic_add!(rhs_re, row, rhs_scale * imag(q))
        _cuda_atomic_add!(rhs_im, row, -rhs_scale * real(q))
    end
    return nothing
end

function _cuda_bm_scale_rhs_kernel!(rhs_re, rhs_im, row_weights, dof_count)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index > length(rhs_re) && return nothing
    weight = row_weights[mod1(index, dof_count)]
    rhs_re[index] *= weight
    rhs_im[index] *= weight
    return nothing
end

function _cuda_bm_scale_rows_kernel!(matrix_re, matrix_im, rhs_re, rhs_im, row_weights, dof_count)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = dof_count * dof_count
    index > total && return nothing
    row = ((index - 1) % dof_count) + 1
    weight = row_weights[row]
    matrix_re[index] *= weight
    matrix_im[index] *= weight
    # A 2-D right-hand side (the DP0-to-P1 mapping) is scaled separately by
    # `_cuda_bm_scale_rhs_kernel!`; its first p1 entries are only column 1.
    if ndims(rhs_re) == 1 && index <= dof_count
        rhs_re[index] *= weight
        rhs_im[index] *= weight
    end
    return nothing
end

function _cuda_bm_correction_scatter_kernel!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    p1_rows,
    p1_cols,
    dp0_cols,
    slp_values,
    adjoint_values,
    dlp_values,
    hypersingular_values,
    inverse_k,
    p1_dof_count,
    pair_count,
    rhs_only,
)
    pair_index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while pair_index <= pair_count
        q = q_neumann[dp0_cols[pair_index]]
        for i in 1:3
            row = p1_rows[pair_index + (i - 1) * pair_count]
            slp = slp_values[pair_index + (i - 1) * pair_count]
            adjoint = adjoint_values[pair_index + (i - 1) * pair_count]
            _cuda_bm_add_rhs!(
                rhs_re,
                rhs_im,
                row,
                real(slp),
                imag(slp),
                real(adjoint),
                imag(adjoint),
                q,
                inverse_k,
                dp0_cols[pair_index],
            )
        end

        if !rhs_only
            value_index = 1
            for j in 1:3
                col = p1_cols[pair_index + (j - 1) * pair_count]
                for i in 1:3
                    row = p1_rows[pair_index + (i - 1) * pair_count]
                    dlp = dlp_values[pair_index + (value_index - 1) * pair_count]
                    hypersingular = hypersingular_values[pair_index + (value_index - 1) * pair_count]
                    _cuda_bm_add_lhs!(
                        matrix_re,
                        matrix_im,
                        row + (col - 1) * p1_dof_count,
                        real(dlp),
                        imag(dlp),
                        real(hypersingular),
                        imag(hypersingular),
                        inverse_k,
                    )
                    value_index += 1
                end
            end
        end
        pair_index += stride
    end
    return nothing
end

function _launch_cuda_bm_regular_transform!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    cache::CudaRegularAssemblyCache{T},
    k::T,
    transform::SymmetryTransform;
    skip_adjacent::Bool,
    rhs_only::Bool=false,
    trial_indices=cache.trial_indices,
) where {T<:AbstractFloat}
    total_pairs = length(cache.test_indices) * length(trial_indices)
    threads = 128
    blocks = min(cld(total_pairs, threads), 65_535)
    placeholder = CUDA.zeros(T, 1)
    try
        signs = T.(transform.signs)
        curl_signs = T.(transform.determinant .* transform.signs)
        CUDA.@cuda threads=threads blocks=blocks _cuda_regular_kernel!(
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            placeholder,
            cache.face_vertices,
            cache.normals,
            cache.areas,
            cache.faces,
            cache.curls,
            cache.test_indices,
            trial_indices,
            cache.rule_points,
            cache.rule_weights,
            k,
            size(matrix_re, 1),
            length(q_neumann),
            cache.face_count,
            cache.rule_count,
            total_pairs,
            skip_adjacent,
            signs[1],
            signs[2],
            signs[3],
            curl_signs[1],
            curl_signs[2],
            curl_signs[3],
            true,
            rhs_only,
            matrix_re,
            matrix_im,
            rhs_re,
            rhs_im,
            q_neumann,
        )
        CUDA.synchronize()
    finally
        CUDA.unsafe_free!(placeholder)
    end
    return total_pairs
end

function _cuda_bm_block_arrays(::Type{T}, pair_count::Int) where {T<:AbstractFloat}
    return (
        slp=CUDA.zeros(Complex{T}, pair_count, 3),
        adjoint=CUDA.zeros(Complex{T}, pair_count, 3),
        dlp=CUDA.zeros(Complex{T}, pair_count, 9),
        hypersingular=CUDA.zeros(Complex{T}, pair_count, 9),
    )
end

function _release_cuda_bm_block_arrays!(blocks)
    CUDA.unsafe_free!(blocks.slp)
    CUDA.unsafe_free!(blocks.adjoint)
    CUDA.unsafe_free!(blocks.dlp)
    CUDA.unsafe_free!(blocks.hypersingular)
    return nothing
end

function _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cache, k; rhs_only::Bool=false)
    cache.pair_count == 0 && return 0
    threads = 128
    blocks_per_grid = min(cld(cache.pair_count, threads), 65_535)
    CUDA.@cuda threads=threads blocks=blocks_per_grid _cuda_bm_correction_scatter_kernel!(
        matrix_re,
        matrix_im,
        rhs_re,
        rhs_im,
        q_neumann,
        cache.p1_rows,
        cache.p1_cols,
        cache.dp0_cols,
        blocks.slp,
        blocks.adjoint,
        blocks.dlp,
        blocks.hypersingular,
        inv(k),
        size(matrix_re, 1),
        cache.pair_count,
        rhs_only,
    )
    CUDA.synchronize()
    return cache.pair_count
end

function add_cuda_bm_singular_corrections!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    mesh::BoundaryMesh{T},
    k::T,
    host_cache,
    cuda_cache,
    regular_cache;
    timing=nothing,
    rhs_only::Bool=false,
    block_cache=nothing,
) where {T<:AbstractFloat}
    host_cache.pair_count == 0 && return 0
    cache_key = (objectid(cuda_cache), k)
    if block_cache !== nothing && haskey(block_cache, cache_key)
        return _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            block_cache[cache_key], cuda_cache, k; rhs_only=rhs_only)
    end
    blocks = _cuda_bm_block_arrays(T, host_cache.pair_count)
    try
        _cuda_timed_stage!(timing, "direct_system_singular_compute") do
            threads = 128
            block_count = min(cld(host_cache.pair_count, threads), 65_535)
            CUDA.@cuda threads=threads blocks=block_count _cuda_duffy_blocks_kernel!(
                blocks.slp,
                blocks.adjoint,
                blocks.dlp,
                blocks.hypersingular,
                cuda_cache.test_indices,
                cuda_cache.trial_indices,
                cuda_cache.rule_indices,
                cuda_cache.jac_scales,
                cuda_cache.normal_products,
                cuda_cache.rule_offsets,
                cuda_cache.rule_test_points,
                cuda_cache.rule_trial_points,
                cuda_cache.rule_weights,
                regular_cache.face_vertices,
                regular_cache.normals,
                regular_cache.curls,
                k,
                length(mesh.faces),
                host_cache.pair_count,
            )
            CUDA.synchronize()
        end
        block_cache === nothing || (block_cache[cache_key] = blocks)
        return _cuda_timed_stage!(timing, "direct_system_singular_scatter") do
            _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cuda_cache, k; rhs_only=rhs_only)
        end
    finally
        (block_cache === nothing || !haskey(block_cache, cache_key)) && _release_cuda_bm_block_arrays!(blocks)
    end
end

function add_cuda_bm_image_corrections!(
    matrix_re,
    matrix_im,
    rhs_re,
    rhs_im,
    q_neumann,
    mesh::BoundaryMesh{T},
    k::T,
    regular_rule::TriangleRule{T},
    cuda_cache,
    regular_cache;
    timing=nothing,
    timing_prefix="direct_system_image",
    rhs_only::Bool=false,
    block_cache=nothing,
) where {T<:AbstractFloat}
    cuda_cache === nothing && return 0
    cuda_cache.pair_count == 0 && return 0
    cache_key = (objectid(cuda_cache), k)
    if block_cache !== nothing && haskey(block_cache, cache_key)
        return _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            block_cache[cache_key], cuda_cache, k; rhs_only=rhs_only)
    end
    blocks = _cuda_bm_block_arrays(T, cuda_cache.pair_count)
    try
        _cuda_timed_stage!(timing, "$(timing_prefix)_compute") do
            threads = 128
            block_count = min(cld(cuda_cache.pair_count, threads), 65_535)
            CUDA.@cuda threads=threads blocks=block_count _cuda_image_singular_delta_blocks_kernel!(
                blocks.slp,
                blocks.adjoint,
                blocks.dlp,
                blocks.hypersingular,
                cuda_cache.test_indices,
                cuda_cache.trial_indices,
                cuda_cache.rule_indices,
                cuda_cache.jac_scales,
                cuda_cache.normal_products,
                cuda_cache.rule_offsets,
                cuda_cache.rule_test_points,
                cuda_cache.rule_trial_points,
                cuda_cache.rule_weights,
                regular_cache.rule_points,
                regular_cache.rule_weights,
                cuda_cache.transform_signs,
                cuda_cache.curl_signs,
                regular_cache.face_vertices,
                regular_cache.normals,
                regular_cache.curls,
                k,
                length(mesh.faces),
                length(regular_rule.weights),
                cuda_cache.pair_count,
            )
            CUDA.synchronize()
        end
        block_cache === nothing || (block_cache[cache_key] = blocks)
        return _cuda_timed_stage!(timing, "$(timing_prefix)_scatter") do
            _scatter_cuda_bm_blocks!(matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, blocks, cuda_cache, k; rhs_only=rhs_only)
        end
    finally
        (block_cache === nothing || !haskey(block_cache, cache_key)) && _release_cuda_bm_block_arrays!(blocks)
    end
end

function assemble_burton_miller_neumann_system_cuda(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann::CuArray,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache,
    device_singular_cache,
    device_image_singular_cache=nothing,
    near_correction_cache=nothing,
    device_near_correction_cache=nothing,
    image_near_correction_cache=nothing,
    device_image_near_correction_cache=nothing,
    symmetry_mode::Symbol=:off,
    timing=nothing,
    block_cache=nothing,
    identity_p1_p1_block=nothing,
    assemble_operator::Bool=false,
) where {T<:AbstractFloat}
    # With `assemble_operator`, the right-hand side is stored as a P1 x DP0
    # matrix whose column j receives only trial element j's contributions, as in
    # `assemble_burton_miller_rhs_cuda`. With unit `q_neumann` that is the
    # DP0-to-P1 right-hand-side mapping, formed in the same pair launches as the
    # system matrix.
    k = outgoing_wavenumber(k)
    CUDA.functional() || error("Direct Burton-Miller CUDA assembly requested, but CUDA.functional() is false.")
    length(q_neumann) == dp0_space.global_dof_count || error("Direct Burton-Miller Neumann vector size mismatch.")
    device_cache === nothing && error("Direct Burton-Miller CUDA assembly requires a regular device cache.")
    singular_cache === nothing && error("Direct Burton-Miller CUDA assembly requires a singular correction cache.")
    device_singular_cache === nothing && error(
        "Direct Burton-Miller CUDA assembly requires a singular device cache.",
    )
    if !isempty(symmetry_image_transforms(symmetry_mode)) && device_image_singular_cache === nothing
        error("Direct Burton-Miller symmetry assembly requires an image-singular device cache.")
    end
    if near_correction_cache !== nothing && near_correction_cache.pair_count > 0 &&
       device_near_correction_cache === nothing
        error("Direct Burton-Miller near correction requires a matching device cache.")
    end
    if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0 &&
       device_image_near_correction_cache === nothing
        error("Direct Burton-Miller image-near correction requires a matching device cache.")
    end
    p1_count = p1_space.global_dof_count
    # Store real and imaginary lanes interleaved so the assembled storage is
    # already Complex{T} without allocating a second dense matrix. This removes
    # the 3x dense-matrix peak (real + imaginary + complex) that otherwise
    # exhausts an 11 GiB GPU for eight moderate speaker boundaries.
    #
    # The complex arrays own their storage and the kernels write through a
    # derived Float32 view that is released before returning, so one
    # `unsafe_free!` of the returned matrix (or mapping) frees the memory at
    # once. A complex view derived from Float32 storage instead would keep the
    # storage alive until its unreachable parents are garbage collected.
    matrix = rhs = matrix_lanes = rhs_lanes = rhs_re = rhs_im = nothing
    succeeded = false
    try
        matrix = CUDA.zeros(Complex{T}, p1_count, p1_count)
        matrix_lanes = reinterpret(reshape, T, matrix)
        matrix_re = view(matrix_lanes, 1, :, :)
        matrix_im = view(matrix_lanes, 2, :, :)
        if assemble_operator
            rhs = CUDA.zeros(Complex{T}, p1_count, dp0_space.global_dof_count)
            rhs_lanes = reinterpret(reshape, T, rhs)
            rhs_re = view(rhs_lanes, 1, :, :)
            rhs_im = view(rhs_lanes, 2, :, :)
        else
            rhs_re = CUDA.zeros(T, p1_count)
            rhs_im = CUDA.zeros(T, p1_count)
        end
        identity_transform = symmetry_transforms(:off; include_identity=true)[1]
        _cuda_timed_stage!(timing, "direct_system_regular") do
            _launch_cuda_bm_regular_transform!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, identity_transform;
                skip_adjacent=true,
            )
        end
        for transform in symmetry_image_transforms(symmetry_mode)
            _cuda_timed_stage!(timing, "direct_system_regular_image") do
                _launch_cuda_bm_regular_transform!(
                    matrix_re, matrix_im, rhs_re, rhs_im, q_neumann, device_cache, k, transform;
                    skip_adjacent=false,
                )
            end
        end

        singular_pairs = add_cuda_bm_singular_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, singular_cache, device_singular_cache, device_cache;
            timing=timing, block_cache=block_cache,
        )
        image_singular_pairs = add_cuda_bm_image_corrections!(
            matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
            mesh, k, rule, device_image_singular_cache, device_cache;
            timing=timing, block_cache=block_cache,
        )
        near_pair_count = 0
        if near_correction_cache !== nothing && near_correction_cache.pair_count > 0
            near_pair_count += add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_near_correction_cache, device_cache;
                timing=timing, block_cache=block_cache,
                timing_prefix="direct_system_near",
            )
        end
        if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0
            near_pair_count += add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, rhs_re, rhs_im, q_neumann,
                mesh, k, rule, device_image_near_correction_cache, device_cache;
                timing=timing, block_cache=block_cache,
                timing_prefix="direct_system_ground_near",
            )
        end

        _cuda_timed_stage!(timing, "direct_system_identity") do
            threads = 256
            blocks = cld(length(mesh.faces), threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_bm_identity_kernel!(
                matrix_re,
                matrix_im,
                rhs_re,
                rhs_im,
                device_cache.areas,
                device_cache.faces,
                q_neumann,
                inv(k),
                p1_count,
                length(mesh.faces),
                false,
                identity_p1_p1_block,
            )
            CUDA.synchronize()
        end

        _cuda_timed_stage!(timing, "direct_system_row_weights") do
            row_weights = CuArray(p1_symmetry_orbit_weights(mesh, symmetry_mode))
            try
                threads = 256
                blocks = cld(p1_count * p1_count, threads)
                CUDA.@cuda threads=threads blocks=blocks _cuda_bm_scale_rows_kernel!(
                    matrix_re, matrix_im, rhs_re, rhs_im, row_weights, p1_count,
                )
                if assemble_operator
                    blocks = cld(length(rhs_re), threads)
                    CUDA.@cuda threads=threads blocks=blocks _cuda_bm_scale_rhs_kernel!(
                        rhs_re, rhs_im, row_weights, p1_count,
                    )
                end
                CUDA.synchronize()
            finally
                CUDA.unsafe_free!(row_weights)
            end
        end

        _cuda_timed_stage!(timing, "direct_system_complex_materialize") do
            assemble_operator || (rhs = complex.(rhs_re, rhs_im))
            CUDA.synchronize()
        end
        succeeded = true
        return (
            matrix=matrix,
            rhs=rhs,
            regular_pairs=length(device_cache.element_indices)^2 * symmetry_reduction_factor(symmetry_mode) - singular_cache.pair_count,
            singular_pairs=singular_pairs,
            image_singular_pairs=image_singular_pairs,
            near_pair_count=near_pair_count,
            on_gpu=true,
            assembly_mode=:direct_burton_miller,
        )
    finally
        matrix_lanes === nothing || CUDA.unsafe_free!(matrix_lanes)
        if assemble_operator
            rhs_lanes === nothing || CUDA.unsafe_free!(rhs_lanes)
        else
            rhs_re === nothing || CUDA.unsafe_free!(rhs_re)
            rhs_im === nothing || CUDA.unsafe_free!(rhs_im)
        end
        if !succeeded
            matrix === nothing || CUDA.unsafe_free!(matrix)
            rhs === nothing || CUDA.unsafe_free!(rhs)
        end
    end
end

function assemble_burton_miller_rhs_cuda(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_neumann::CuArray,
    k::T,
    rule::TriangleRule{T};
    device_cache,
    singular_cache,
    device_singular_cache,
    device_image_singular_cache=nothing,
    near_correction_cache=nothing,
    device_near_correction_cache=nothing,
    image_near_correction_cache=nothing,
    device_image_near_correction_cache=nothing,
    symmetry_mode::Symbol=:off,
    timing=nothing,
    block_cache=nothing,
    assemble_operator::Bool=false,
    operator_columns=nothing,
    regular_system=nothing,
) where {T<:AbstractFloat}
    k = outgoing_wavenumber(k)
    CUDA.functional() || error("Burton-Miller CUDA RHS assembly requested, but CUDA.functional() is false.")
    length(q_neumann) == dp0_space.global_dof_count || error("Burton-Miller Neumann vector size mismatch.")
    device_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a regular device cache.")
    singular_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a singular correction cache.")
    device_singular_cache === nothing && error("Burton-Miller CUDA RHS assembly requires a singular device cache.")
    if !isempty(symmetry_image_transforms(symmetry_mode)) && device_image_singular_cache === nothing
        error("Burton-Miller symmetry RHS assembly requires an image-singular device cache.")
    end
    if near_correction_cache !== nothing && near_correction_cache.pair_count > 0 &&
       device_near_correction_cache === nothing
        error("Burton-Miller near RHS correction requires a matching device cache.")
    end
    if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0 &&
       device_image_near_correction_cache === nothing
        error("Burton-Miller image-near RHS correction requires a matching device cache.")
    end

    p1_count = p1_space.global_dof_count
    operator_columns === nothing || assemble_operator || error("operator_columns requires assemble_operator")
    columns = operator_columns === nothing ? nothing : Int.(operator_columns)
    if columns !== nothing
        !isempty(columns) && length(unique(columns)) == length(columns) &&
            all(i -> 1 <= i <= dp0_space.global_dof_count, columns) || error("Invalid RHS column subset")
    end
    column_count = columns === nothing ? dp0_space.global_dof_count : length(columns)
    matrix_lanes = regular_system === nothing ? nothing : reinterpret(reshape, T, regular_system)
    matrix_re = regular_system === nothing ? CUDA.zeros(T, 1) : view(matrix_lanes, 1, :, :)
    matrix_im = regular_system === nothing ? CUDA.zeros(T, 1) : view(matrix_lanes, 2, :, :)
    column_map = device_columns = nothing
    storage = rhs_re = rhs_im = rhs = nothing
    succeeded = false
    try
        # Interleaved storage exposes the matrix without a second full copy.
        # With unit q, each column is the existing DP0-to-P1 RHS mapping.
        storage = assemble_operator ? CUDA.zeros(T, 2, p1_count, column_count) : nothing
        rhs_re = assemble_operator ? view(storage, 1, :, :) : CUDA.zeros(T, p1_count)
        rhs_im = assemble_operator ? view(storage, 2, :, :) : CUDA.zeros(T, p1_count)
        scatter_re, scatter_im = rhs_re, rhs_im
        if columns !== nothing
            mapping = zeros(Int32, dp0_space.global_dof_count)
            mapping[columns] = Int32.(1:column_count)
            column_map = CuArray(mapping)
            device_columns = CuArray(Int32.(columns))
            scatter_re = CudaRhsColumnTile(CUDA.cudaconvert(rhs_re), CUDA.cudaconvert(column_map), p1_count)
            scatter_im = CudaRhsColumnTile(CUDA.cudaconvert(rhs_im), CUDA.cudaconvert(column_map), p1_count)
        end
        identity_transform = symmetry_transforms(:off; include_identity=true)[1]
        _cuda_timed_stage!(timing, "rhs_regular") do
            _launch_cuda_bm_regular_transform!(
                matrix_re, matrix_im, scatter_re, scatter_im, q_neumann, device_cache, k, identity_transform;
                skip_adjacent=true,
                trial_indices=columns === nothing ? device_cache.trial_indices : device_columns,
                rhs_only=regular_system === nothing,
            )
        end
        for transform in symmetry_image_transforms(symmetry_mode)
            _cuda_timed_stage!(timing, "rhs_regular_image") do
                _launch_cuda_bm_regular_transform!(
                    matrix_re, matrix_im, scatter_re, scatter_im, q_neumann, device_cache, k, transform;
                    skip_adjacent=false,
                    trial_indices=columns === nothing ? device_cache.trial_indices : device_columns,
                    rhs_only=regular_system === nothing,
                )
            end
        end

        add_cuda_bm_singular_corrections!(
            matrix_re, matrix_im, scatter_re, scatter_im, q_neumann,
            mesh, k, singular_cache, device_singular_cache, device_cache;
            timing=timing, block_cache=block_cache,
            rhs_only=true,
        )
        add_cuda_bm_image_corrections!(
            matrix_re, matrix_im, scatter_re, scatter_im, q_neumann,
            mesh, k, rule, device_image_singular_cache, device_cache;
            timing=timing, block_cache=block_cache,
            rhs_only=true,
        )
        if near_correction_cache !== nothing && near_correction_cache.pair_count > 0
            add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, scatter_re, scatter_im, q_neumann,
                mesh, k, rule, device_near_correction_cache, device_cache;
                timing=timing, block_cache=block_cache,
                timing_prefix="rhs_near",
                rhs_only=true,
            )
        end
        if image_near_correction_cache !== nothing && image_near_correction_cache.pair_count > 0
            add_cuda_bm_image_corrections!(
                matrix_re, matrix_im, scatter_re, scatter_im, q_neumann,
                mesh, k, rule, device_image_near_correction_cache, device_cache;
                timing=timing, block_cache=block_cache,
                timing_prefix="rhs_ground_near",
                rhs_only=true,
            )
        end

        _cuda_timed_stage!(timing, "rhs_identity") do
            threads = 256
            blocks = cld(length(mesh.faces), threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_bm_identity_kernel!(
                matrix_re,
                matrix_im,
                scatter_re,
                scatter_im,
                device_cache.areas,
                device_cache.faces,
                q_neumann,
                inv(k),
                p1_count,
                length(mesh.faces),
                true,
            )
            CUDA.synchronize()
        end

        _cuda_timed_stage!(timing, "rhs_row_weights") do
            row_weights = CuArray(p1_symmetry_orbit_weights(mesh, symmetry_mode))
            try
                threads = 256
                blocks = cld(length(rhs_re), threads)
                CUDA.@cuda threads=threads blocks=blocks _cuda_bm_scale_rhs_kernel!(
                    rhs_re, rhs_im, row_weights, p1_count,
                )
                CUDA.synchronize()
            finally
                CUDA.unsafe_free!(row_weights)
            end
        end

        _cuda_timed_stage!(timing, "rhs_complex_materialize") do
            if assemble_operator
                # Both reinterpret and reshape create owning CuArray aliases.
                # Release intermediates explicitly: otherwise each cabinet slab
                # remains live until GC and can force WDDM device-memory paging.
                complex_storage = reinterpret(Complex{T}, storage)
                try
                    rhs = reshape(complex_storage, p1_count, column_count)
                finally
                    CUDA.unsafe_free!(complex_storage)
                end
            else
                rhs = complex.(rhs_re, rhs_im)
            end
            CUDA.synchronize()
        end
        succeeded = true
        return rhs
    finally
        column_map === nothing || CUDA.unsafe_free!(column_map)
        device_columns === nothing || CUDA.unsafe_free!(device_columns)
        if matrix_lanes === nothing
            CUDA.unsafe_free!(matrix_re)
            CUDA.unsafe_free!(matrix_im)
        else
            CUDA.unsafe_free!(matrix_lanes)
        end
        if assemble_operator
            storage === nothing || CUDA.unsafe_free!(storage)
        else
            rhs_re === nothing || CUDA.unsafe_free!(rhs_re)
            rhs_im === nothing || CUDA.unsafe_free!(rhs_im)
            (!succeeded && rhs !== nothing) && CUDA.unsafe_free!(rhs)
        end
    end
end

_cuda_out_of_memory(err) = err isa OutOfMemoryError || nameof(typeof(err)) == :OutOfGPUMemoryError

# Direct Burton-Miller system with one right-hand side per column of
# `q_columns` (DP0 dofs x excitations), for solves that share one factorization.
#
# `:cached_operator` runs the system assembly once with unit Neumann data and a
# P1 x DP0 right-hand side, which is the DP0-to-P1 mapping B, then forms every
# right-hand side as B * q_columns: one pair integration for any number of
# excitations. `:matrix_free` integrates excitation 1 with the system and
# repeats the regular, image and singular pair integration for every further
# excitation. If B cannot be allocated the assembly falls back to
# `:matrix_free`; the returned `rhs_mode` is the one that ran. `rhs` is always
# P1 dofs x excitations.
function assemble_burton_miller_neumann_system_columns_cuda(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    q_columns::CuMatrix,
    k::T,
    rule::TriangleRule{T};
    rhs_mode::Symbol,
    identity_p1_p1_block=nothing,
    timing=nothing,
    kwargs...,
) where {T<:AbstractFloat}
    rhs_mode in (:cached_operator, :matrix_free) || error(
        "Unknown Burton-Miller right-hand-side mode $(rhs_mode); expected :cached_operator or :matrix_free.",
    )
    size(q_columns, 1) == dp0_space.global_dof_count || error("Direct Burton-Miller Neumann matrix size mismatch.")
    excitation_count = size(q_columns, 2)
    excitation_count >= 1 || error("Direct Burton-Miller assembly requires at least one excitation.")

    if rhs_mode == :cached_operator
        units = system = rhs = nothing
        try
            units = CUDA.ones(Complex{T}, dp0_space.global_dof_count)
            system = assemble_burton_miller_neumann_system_cuda(
                mesh, p1_space, dp0_space, units, k, rule;
                identity_p1_p1_block=identity_p1_p1_block,
                timing=timing,
                assemble_operator=true,
                kwargs...,
            )
            _cuda_timed_stage!(timing, "direct_system_rhs_operator_apply") do
                rhs = system.rhs * q_columns
                CUDA.synchronize()
            end
            operator_bytes = sizeof(system.rhs)
            CUDA.unsafe_free!(system.rhs)
            return merge(system, (rhs=rhs, rhs_mode=:cached_operator, rhs_operator_bytes=operator_bytes))
        catch err
            system === nothing || release_burton_miller_system_cuda!(system)
            rhs === nothing || CUDA.unsafe_free!(rhs)
            _cuda_out_of_memory(err) || rethrow()
            @warn "Burton-Miller right-hand-side mapping did not fit in GPU memory; integrating each excitation instead." excitation_count
        finally
            units === nothing || CUDA.unsafe_free!(units)
        end
    end

    system = first_q = column_q = rhs = nothing
    columns = Any[]
    succeeded = false
    try
        first_q = q_columns[:, 1]
        system = assemble_burton_miller_neumann_system_cuda(
            mesh, p1_space, dp0_space, first_q, k, rule;
            identity_p1_p1_block=identity_p1_p1_block,
            timing=timing,
            kwargs...,
        )
        if excitation_count == 1
            # One column in both modes. Freeing the vector drops only its
            # reference; the reshaped matrix now owns the storage.
            rhs = reshape(system.rhs, :, 1)
            CUDA.unsafe_free!(system.rhs)
        else
            column_q = similar(first_q)
            dof_count = length(first_q)
            for column in 2:excitation_count
                # A contiguous column view would be a derived array holding a
                # reference to q_columns' storage until it is collected.
                copyto!(column_q, 1, q_columns, (column - 1) * dof_count + 1, dof_count)
                push!(columns, assemble_burton_miller_rhs_cuda(
                    mesh, p1_space, dp0_space, column_q, k, rule; timing=timing, kwargs...,
                ))
            end
            rhs = hcat(system.rhs, columns...)
            CUDA.unsafe_free!(system.rhs)
        end
        succeeded = true
        return merge(system, (rhs=rhs, rhs_mode=:matrix_free, rhs_operator_bytes=0))
    finally
        first_q === nothing || CUDA.unsafe_free!(first_q)
        column_q === nothing || CUDA.unsafe_free!(column_q)
        foreach(CUDA.unsafe_free!, columns)
        if !succeeded
            system === nothing || release_burton_miller_system_cuda!(system)
            rhs === nothing || CUDA.unsafe_free!(rhs)
        end
    end
end

function solve_burton_miller_system_cuda!(system; return_gpu::Bool=false)
    get(system, :on_gpu, false) || error("Direct Burton-Miller CUDA solve requires a GPU-resident system.")
    factorization = pressure = nothing
    try
        factorization = lu!(system.matrix)
        pressure = factorization \ system.rhs
        return return_gpu ? pressure : Array(pressure)
    finally
        if factorization === nothing
            CUDA.unsafe_free!(system.matrix)
        else
            CUDA.unsafe_free!(factorization.factors)
            CUDA.unsafe_free!(factorization.ipiv)
        end
        CUDA.unsafe_free!(system.rhs)
        (!return_gpu && pressure !== nothing) && CUDA.unsafe_free!(pressure)
    end
end

function release_burton_miller_system_cuda!(system)
    CUDA.unsafe_free!(system.matrix)
    CUDA.unsafe_free!(system.rhs)
    return nothing
end
