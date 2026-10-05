# Duffy singular corrections on the GPU: one kernel evaluates each singular
# pair's compact 3x1 / 3x3 blocks, a second gathers those blocks into the dense
# operators entry by entry, so no two threads write one address.

function build_metal_singular_correction_cache(cache::SingularCorrectionCache{T}) where {T<:AbstractFloat}
    _require_metal!()
    face_count = length(cache.pairs_by_test)
    pair_offsets = Vector{Int32}(undef, face_count + 1)
    test_indices = Int32[]
    trial_indices = Int32[]
    rule_indices = Int32[]
    jac_scales = T[]
    normal_products = T[]
    pair_offsets[1] = 1
    for test_index in 1:face_count
        for pair in cache.pairs_by_test[test_index]
            push!(test_indices, Int32(test_index))
            push!(trial_indices, Int32(pair.trial_index))
            push!(rule_indices, Int32(pair.rule_index))
            push!(jac_scales, pair.jac_scale)
            push!(normal_products, pair.normal_product)
        end
        pair_offsets[test_index + 1] = Int32(length(trial_indices) + 1)
    end

    total_rule_points = sum(length(rule.weights) for rule in cache.rules; init=0)
    rule_offsets = Vector{Int32}(undef, length(cache.rules) + 1)
    rule_test_points = Matrix{T}(undef, max(total_rule_points, 1), 2)
    rule_trial_points = Matrix{T}(undef, max(total_rule_points, 1), 2)
    rule_weights = Vector{T}(undef, max(total_rule_points, 1))
    offset = 1
    for (rule_index, rule) in enumerate(cache.rules)
        rule_offsets[rule_index] = Int32(offset)
        for q in eachindex(rule.weights)
            target = offset + q - 1
            rule_test_points[target, 1] = rule.test_points[q][1]
            rule_test_points[target, 2] = rule.test_points[q][2]
            rule_trial_points[target, 1] = rule.trial_points[q][1]
            rule_trial_points[target, 2] = rule.trial_points[q][2]
            rule_weights[target] = rule.weights[q]
        end
        offset += length(rule.weights)
    end
    rule_offsets[end] = Int32(offset)
    if total_rule_points == 0
        # Empty device arrays are legal but a zero-point rule table would give
        # the kernels a zero-length weights array to take `length` of; keep one
        # padding row (never indexed, because there are no pairs).
        rule_weights[1] = zero(T)
        rule_test_points[1, :] .= zero(T)
        rule_trial_points[1, :] .= zero(T)
    end

    return MetalSingularCorrectionCache{T}(
        MtlArray(pair_offsets),
        MtlArray(test_indices),
        MtlArray(trial_indices),
        MtlArray(rule_indices),
        MtlArray(jac_scales),
        MtlArray(normal_products),
        MtlArray(rule_offsets),
        MtlArray(rule_test_points),
        MtlArray(rule_trial_points),
        MtlArray(rule_weights),
        cache.pair_count,
        Ref{Any}(nothing),
    )
end

# `host_staged` assembly never builds this cache, so the release paths that run
# for every Metal solve may legitimately be handed `nothing`.
release_metal_singular_correction_cache!(::Nothing) = nothing

function release_metal_singular_correction_cache!(cache::MetalSingularCorrectionCache)
    Metal.unsafe_free!(cache.pair_offsets)
    Metal.unsafe_free!(cache.test_indices)
    Metal.unsafe_free!(cache.trial_indices)
    Metal.unsafe_free!(cache.rule_indices)
    Metal.unsafe_free!(cache.jac_scales)
    Metal.unsafe_free!(cache.normal_products)
    Metal.unsafe_free!(cache.rule_offsets)
    Metal.unsafe_free!(cache.rule_test_points)
    Metal.unsafe_free!(cache.rule_trial_points)
    Metal.unsafe_free!(cache.rule_weights)
    _release_metal_fused_singular_tables!(cache)
    tables = cache.gather_tables[]
    tables isa MetalSingularGatherTables && _metal_release_singular_gather_tables!(tables)
    cache.gather_tables[] = nothing
    return nothing
end

# ---------------------------------------------------------------------------
# Deterministic singular write-back
#
# The block kernels evaluate each singular pair into a compact value buffer.
# Getting those values into the operators can be done two ways:
#
#   scatter  one thread per pair, atomically adding its 3x1 and 3x3 blocks into
#            cells that other pairs also write. Cheap to launch, but the order
#            in which the atomics land is whatever the GPU chooses, so the same
#            tree gives a different float32 sum on every run.
#
#   gather   one thread per touched cell. The thread reads the list of values
#            that belong to it and writes their total once. No two threads share
#            a cell, so there are no atomics and the summation order is fixed by
#            the map. This is what makes assembly reproducible.
#
# The map is a CSR structure built on the host, once per mesh, and cached on the
# singular correction cache. Building it is pure index arithmetic; it never
# touches the kernel results.
# ---------------------------------------------------------------------------

# Collapses (entry, value) contributions into a CSR gather map: one row per
# unique destination cell, holding the run of block-value indices that sum into
# it. Sorting fixes the order inside each cell, which is what makes the device
# write-back reproducible.
function _metal_build_singular_gather_map(
    entries::Vector{Int32},
    values::Vector{Int32},
    column_of_value::F,
) where {F}
    if isempty(entries)
        return MetalSingularGatherMap(
            MtlArray(Int32[]),
            MtlArray(Int32[1]),
            MtlArray(Int32[]),
            column_of_value === nothing ? nothing : MtlArray(Int32[]),
            0,
        )
    end
    # (entry, value) is unique across contributions, so sorting the pair gives
    # one fixed order whatever algorithm Base picks. Both halves live in the
    # packed key, so the keys are sorted directly (radix sort for UInt64) and
    # split back apart: no permutation, no gathers through it. This is the
    # order the map has always had, so the device summation order is unchanged.
    keys = Vector{UInt64}(undef, length(entries))
    @inbounds for i in eachindex(entries)
        keys[i] = (UInt64(reinterpret(UInt32, entries[i])) << 32) | UInt64(reinterpret(UInt32, values[i]))
    end
    sort!(keys)
    entry_indices = Int32[]
    contrib_offsets = Int32[]
    contrib_values = Vector{Int32}(undef, length(keys))
    contrib_columns = column_of_value === nothing ? nothing : Vector{Int32}(undef, length(keys))
    previous = zero(Int32)
    @inbounds for (position, key) in enumerate(keys)
        entry = reinterpret(Int32, UInt32(key >> 32))
        value = reinterpret(Int32, UInt32(key & 0xffffffff))
        if position == 1 || entry != previous
            push!(entry_indices, entry)
            push!(contrib_offsets, Int32(position))
            previous = entry
        end
        contrib_values[position] = value
        contrib_columns === nothing || (contrib_columns[position] = column_of_value(value))
    end
    push!(contrib_offsets, Int32(length(keys) + 1))
    return MetalSingularGatherMap(
        MtlArray(entry_indices),
        MtlArray(contrib_offsets),
        MtlArray(contrib_values),
        contrib_columns === nothing ? nothing : MtlArray(contrib_columns),
        length(entry_indices),
    )
end

# Builds all three maps in one host pass over the pair list. The pair order here
# is the same order the block kernels use, because both walk `test_indices` and
# `trial_indices` by position.
function _metal_build_singular_gather_tables(
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    part_count::Int,
)
    pair_count = singular_cache.pair_count
    face_count = regular_cache.face_count
    p1_dof_count = regular_cache.p1_dof_count
    dp0_dof_count = regular_cache.dp0_dof_count
    # Destination indices are Int32 on the device. The dense operators would not
    # fit in memory long before this overflows, but fail loudly rather than wrap.
    largest_entry = p1_dof_count * max(p1_dof_count, dp0_dof_count)
    largest_entry <= typemax(Int32) || error(
        "Singular gather map needs Int32 dense indices, but p1_dof_count=$(p1_dof_count) " *
        "and dp0_dof_count=$(dp0_dof_count) reach $(largest_entry).",
    )
    value_stride = pair_count * part_count
    test_indices = Array(singular_cache.test_indices)
    trial_indices = Array(singular_cache.trial_indices)
    p1_dofs = Array(regular_cache.p1_dofs)
    element_dp0_dofs = Array(regular_cache.element_dp0_dofs)

    row_count = 3 * pair_count
    block_count = 9 * pair_count
    row_entries = Vector{Int32}(undef, row_count)
    row_values = Vector{Int32}(undef, row_count)
    rhs_entries = Vector{Int32}(undef, row_count)
    rhs_columns = Vector{Int32}(undef, row_count)
    block_entries = Vector{Int32}(undef, block_count)
    block_values = Vector{Int32}(undef, block_count)

    largest_value = pair_count + 8 * value_stride
    largest_value <= typemax(UInt32) || error(
        "Singular gather map packs value indices into 32 bits, but pair_count=$(pair_count) " *
        "and part_count=$(part_count) reach $(largest_value).",
    )

    # The pair loop runs behind a function barrier: the cache fields it reads are
    # untyped, and inline every index into them was a dynamic dispatch, about a
    # million per build at sample_detailed.msh -- most of the build's time.
    _metal_fill_singular_gather_inputs!(
        row_entries, row_values, rhs_entries, rhs_columns, block_entries, block_values,
        test_indices, trial_indices, p1_dofs, element_dp0_dofs,
        pair_count, face_count, p1_dof_count, value_stride,
    )

    # A right-hand-side value's column is its pair's trial DP0 dof, and the pair
    # is recoverable from the value index (`pair + (local_row - 1) * stride`), so
    # the column is looked up after sorting instead of being carried through it.
    rhs_column_of_value = value -> rhs_columns[3 * ((Int(value) - 1) % value_stride) + 1]
    return MetalSingularGatherTables(
        _metal_build_singular_gather_map(row_entries, row_values, nothing),
        _metal_build_singular_gather_map(block_entries, block_values, nothing),
        _metal_build_singular_gather_map(rhs_entries, row_values, rhs_column_of_value),
        part_count,
        objectid(regular_cache),
    )
end

# The host pass of `_metal_build_singular_gather_tables`, over concretely typed
# arrays. Fills the (entry, value) contributions of all three maps in pair order.
function _metal_fill_singular_gather_inputs!(
    row_entries, row_values, rhs_entries, rhs_columns, block_entries, block_values,
    test_indices::AbstractArray{<:Integer}, trial_indices::AbstractArray{<:Integer},
    p1_dofs::AbstractArray{<:Integer}, element_dp0_dofs::AbstractArray{<:Integer},
    pair_count::Int, face_count::Int, p1_dof_count::Int, value_stride::Int,
)
    at_row = 0
    at_block = 0
    for pair_position in 1:pair_count
        test_index = Int(test_indices[pair_position])
        trial_index = Int(trial_indices[pair_position])
        dp0_column = Int(element_dp0_dofs[trial_index])
        # `p1_dofs` and `element_dp0_dofs` hold 0 for faces outside the assembled
        # element set. The scatter hid such a face behind an atomic add to the
        # wrong cell; the gather would corrupt a cell silently, so refuse here.
        dp0_column >= 1 || error(
            "Singular pair $(pair_position) has trial element $(trial_index) with no DP0 " *
            "degree of freedom; the singular cache and the regular cache disagree.",
        )
        for local_row in 1:3
            row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
            row >= 1 || error(
                "Singular pair $(pair_position) has test element $(test_index) with no P1 " *
                "degree of freedom at local row $(local_row).",
            )
            at_row += 1
            # Single layer and adjoint double layer: P1 row, DP0 column.
            row_entries[at_row] = Int32(row + (dp0_column - 1) * p1_dof_count)
            row_values[at_row] = Int32(pair_position + (local_row - 1) * value_stride)
            # The fused right-hand side reuses those values but lands on the row
            # alone, weighted by the trial element's Neumann datum.
            rhs_entries[at_row] = Int32(row)
            rhs_columns[at_row] = Int32(dp0_column)
        end
        for local_column in 1:3
            column = Int(p1_dofs[trial_index + (local_column - 1) * face_count])
            column >= 1 || error(
                "Singular pair $(pair_position) has trial element $(trial_index) with no P1 " *
                "degree of freedom at local column $(local_column).",
            )
            for local_row in 1:3
                row = Int(p1_dofs[test_index + (local_row - 1) * face_count])
                component = (local_column - 1) * 3 + (local_row - 1)
                at_block += 1
                # Double layer, hypersingular, and the fused left-hand side:
                # P1 row, P1 column.
                block_entries[at_block] = Int32(row + (column - 1) * p1_dof_count)
                block_values[at_block] = Int32(pair_position + component * value_stride)
            end
        end
    end
    return nothing
end

# Lazily builds and caches the maps, following the Ref{Any} idiom the regular
# assembly cache already uses. The part split and the regular cache are part of
# the key: both change the indices the map stores.
function _metal_singular_gather_tables(
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    part_count::Int,
)
    tables = singular_cache.gather_tables[]
    if tables isa MetalSingularGatherTables &&
       tables.part_count == part_count &&
       tables.regular_id == objectid(regular_cache)
        return tables
    end
    tables isa MetalSingularGatherTables && _metal_release_singular_gather_tables!(tables)
    tables = _metal_build_singular_gather_tables(regular_cache, singular_cache, part_count)
    singular_cache.gather_tables[] = tables
    return tables
end

function _metal_release_singular_gather_map!(map::MetalSingularGatherMap)
    Metal.unsafe_free!(map.entry_indices)
    Metal.unsafe_free!(map.contrib_offsets)
    Metal.unsafe_free!(map.contrib_values)
    map.contrib_columns === nothing || Metal.unsafe_free!(map.contrib_columns)
    return nothing
end

function _metal_release_singular_gather_tables!(tables::MetalSingularGatherTables)
    _metal_release_singular_gather_map!(tables.p1_dp0)
    _metal_release_singular_gather_map!(tables.p1_p1)
    _metal_release_singular_gather_map!(tables.rhs)
    return nothing
end

# One thread per touched cell, writing two operators that share a destination
# (single layer with adjoint double layer, double layer with hypersingular, and
# so the map is walked once for both).
function _metal_singular_pair_gather_kernel!(
    first_operator,
    second_operator,
    first_values,
    second_values,
    entry_indices,
    contrib_offsets,
    contrib_values,
    entry_count,
    pair_count,
    part_count,
)
    index = _metal_global_linear_index()
    index > entry_count && return nothing
    @inbounds begin
        position = Int(contrib_offsets[index])
        position_stop = Int(contrib_offsets[index + 1]) - 1
        first_sum = zero(eltype(first_values))
        second_sum = zero(eltype(second_values))
        while position <= position_stop
            value_index = Int(contrib_values[position])
            part = 1
            while part <= part_count
                first_sum += first_values[value_index]
                second_sum += second_values[value_index]
                value_index += pair_count
                part += 1
            end
            position += 1
        end
        entry_index = Int(entry_indices[index])
        first_operator[entry_index] += first_sum
        second_operator[entry_index] += second_sum
    end
    return nothing
end

# One thread per touched cell, writing a single operator. Used by the fused
# Burton-Miller left-hand side, which has no second operator to share the walk.
function _metal_singular_entry_gather_kernel!(
    operator,
    values,
    entry_indices,
    contrib_offsets,
    contrib_values,
    entry_count,
    pair_count,
    part_count,
)
    index = _metal_global_linear_index()
    index > entry_count && return nothing
    @inbounds begin
        position = Int(contrib_offsets[index])
        position_stop = Int(contrib_offsets[index + 1]) - 1
        total = zero(eltype(values))
        while position <= position_stop
            value_index = Int(contrib_values[position])
            part = 1
            while part <= part_count
                total += values[value_index]
                value_index += pair_count
                part += 1
            end
            position += 1
        end
        operator[Int(entry_indices[index])] += total
    end
    return nothing
end

# One thread per touched right-hand-side row. Each contribution carries the DP0
# column of its trial element, because the Neumann datum differs per column and
# per drive.
function _metal_singular_rhs_gather_kernel!(
    rhs,
    rhs_values,
    q_neumann,
    entry_indices,
    contrib_offsets,
    contrib_values,
    contrib_columns,
    entry_count,
    pair_count,
    part_count,
    p1_dof_count,
    dp0_dof_count,
    drive_count,
)
    index = _metal_global_linear_index()
    index > entry_count && return nothing
    @inbounds begin
        row = Int(entry_indices[index])
        position_start = Int(contrib_offsets[index])
        position_stop = Int(contrib_offsets[index + 1]) - 1
        drive = 1
        while drive <= drive_count
            total = zero(eltype(rhs_values))
            position = position_start
            while position <= position_stop
                value_index = Int(contrib_values[position])
                dp0_column = Int(contrib_columns[position])
                coefficient = zero(eltype(rhs_values))
                part = 1
                while part <= part_count
                    coefficient += rhs_values[value_index]
                    value_index += pair_count
                    part += 1
                end
                total += coefficient * q_neumann[dp0_column + (drive - 1) * dp0_dof_count]
                position += 1
            end
            rhs[row + (drive - 1) * p1_dof_count] += total
            drive += 1
        end
    end
    return nothing
end

@inline function _metal_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
    pair_position = Int(pair_offsets[test_index])
    pair_stop = Int(pair_offsets[test_index + 1]) - 1
    while pair_position <= pair_stop
        Int(trial_indices[pair_position]) == trial_index && return pair_position
        pair_position += 1
    end
    return 0
end

function _metal_singular_slp_adjoint_blocks_kernel!(
    slp_values,
    adjoint_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    k,
    face_count,
    pair_count,
    rule_point_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
)
    pair_position = _metal_global_linear_index()
    pair_position > pair_count && return nothing
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    rule_index = Int(rule_indices[pair_position])
    q = Int(rule_offsets[rule_index])
    q_stop = Int(rule_offsets[rule_index + 1]) - 1
    jac_scale = jac_scales[pair_position]
    inv_four_pi = typeof(k)(0.07957747154594767)
    test_nx = normals[test_index]
    test_ny = normals[test_index + face_count]
    test_nz = normals[test_index + 2 * face_count]
    slp1_re = zero(k); slp1_im = zero(k)
    slp2_re = zero(k); slp2_im = zero(k)
    slp3_re = zero(k); slp3_im = zero(k)
    adj1_re = zero(k); adj1_im = zero(k)
    adj2_re = zero(k); adj2_im = zero(k)
    adj3_re = zero(k); adj3_im = zero(k)
    while q <= q_stop
        test_xi = rule_test_points[q]
        test_eta = rule_test_points[q + rule_point_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        trial_xi = rule_trial_points[q]
        trial_eta = rule_trial_points[q + rule_point_count]
        rb1 = one(k) - trial_xi - trial_eta
        rb2 = trial_xi
        rb3 = trial_eta
        sx, sy, sz = _metal_face_point(face_vertices, trial_index, face_count, rb1, rb2, rb3)
        sx *= trial_sign_x
        sy *= trial_sign_y
        sz *= trial_sign_z
        dx = sx - x
        dy = sy - y
        dz = sz - z
        radius2 = dx * dx + dy * dy + dz * dz
        if radius2 > zero(k)
            inv_radius = _metal_fast_rsqrt(radius2)
            radius = radius2 * inv_radius
            phase = k * radius
            green_scale = inv_radius * inv_four_pi
            green_re = _metal_fast_cos(phase) * green_scale
            green_im = _metal_fast_sin(phase) * green_scale
            weight = rule_weights[q] * jac_scale
            test_dot = -(dx * test_nx + dy * test_ny + dz * test_nz) * inv_radius
            grad_re = (-green_re * inv_radius - green_im * k) * test_dot
            grad_im = (green_re * k - green_im * inv_radius) * test_dot
            slp1_re += tb1 * green_re * weight; slp1_im += tb1 * green_im * weight
            slp2_re += tb2 * green_re * weight; slp2_im += tb2 * green_im * weight
            slp3_re += tb3 * green_re * weight; slp3_im += tb3 * green_im * weight
            adj1_re += tb1 * grad_re * weight; adj1_im += tb1 * grad_im * weight
            adj2_re += tb2 * grad_re * weight; adj2_im += tb2 * grad_im * weight
            adj3_re += tb3 * grad_re * weight; adj3_im += tb3 * grad_im * weight
        end
        q += 1
    end
    slp_values[pair_position] = Complex(slp1_re, slp1_im)
    slp_values[pair_position + pair_count] = Complex(slp2_re, slp2_im)
    slp_values[pair_position + 2 * pair_count] = Complex(slp3_re, slp3_im)
    adjoint_values[pair_position] = Complex(adj1_re, adj1_im)
    adjoint_values[pair_position + pair_count] = Complex(adj2_re, adj2_im)
    adjoint_values[pair_position + 2 * pair_count] = Complex(adj3_re, adj3_im)
    return nothing
end

function _metal_singular_dlp_hyp_blocks_kernel!(
    dlp_values,
    hypersingular_values,
    test_indices,
    trial_indices,
    rule_indices,
    jac_scales,
    normal_products,
    rule_offsets,
    rule_test_points,
    rule_trial_points,
    rule_weights,
    face_vertices,
    normals,
    curls,
    k,
    face_count,
    pair_count,
    rule_point_count,
    trial_sign_x,
    trial_sign_y,
    trial_sign_z,
    trial_curl_sign_x,
    trial_curl_sign_y,
    trial_curl_sign_z,
)
    linear_index = _metal_global_linear_index()
    linear_index > 3 * pair_count && return nothing
    pair_position = ((linear_index - 1) % pair_count) + 1
    local_column = div(linear_index - 1, pair_count) + 1
    test_index = Int(test_indices[pair_position])
    trial_index = Int(trial_indices[pair_position])
    rule_index = Int(rule_indices[pair_position])
    q = Int(rule_offsets[rule_index])
    q_stop = Int(rule_offsets[rule_index + 1]) - 1
    jac_scale = jac_scales[pair_position]
    normal_product = normal_products[pair_position]
    inv_four_pi = typeof(k)(0.07957747154594767)
    k2 = k * k
    trial_nx = trial_sign_x * normals[trial_index]
    trial_ny = trial_sign_y * normals[trial_index + face_count]
    trial_nz = trial_sign_z * normals[trial_index + 2 * face_count]
    trial_curl_offset = 3 * (local_column - 1)
    trial_curl_x = trial_curl_sign_x * curls[trial_index + trial_curl_offset * face_count]
    trial_curl_y = trial_curl_sign_y * curls[trial_index + (trial_curl_offset + 1) * face_count]
    trial_curl_z = trial_curl_sign_z * curls[trial_index + (trial_curl_offset + 2) * face_count]
    test_curl1_x = curls[test_index]
    test_curl1_y = curls[test_index + face_count]
    test_curl1_z = curls[test_index + 2 * face_count]
    test_curl2_x = curls[test_index + 3 * face_count]
    test_curl2_y = curls[test_index + 4 * face_count]
    test_curl2_z = curls[test_index + 5 * face_count]
    test_curl3_x = curls[test_index + 6 * face_count]
    test_curl3_y = curls[test_index + 7 * face_count]
    test_curl3_z = curls[test_index + 8 * face_count]
    curl1 = test_curl1_x * trial_curl_x + test_curl1_y * trial_curl_y + test_curl1_z * trial_curl_z
    curl2 = test_curl2_x * trial_curl_x + test_curl2_y * trial_curl_y + test_curl2_z * trial_curl_z
    curl3 = test_curl3_x * trial_curl_x + test_curl3_y * trial_curl_y + test_curl3_z * trial_curl_z
    dlp1_re = zero(k); dlp1_im = zero(k)
    dlp2_re = zero(k); dlp2_im = zero(k)
    dlp3_re = zero(k); dlp3_im = zero(k)
    hyp1_re = zero(k); hyp1_im = zero(k)
    hyp2_re = zero(k); hyp2_im = zero(k)
    hyp3_re = zero(k); hyp3_im = zero(k)
    while q <= q_stop
        test_xi = rule_test_points[q]
        test_eta = rule_test_points[q + rule_point_count]
        tb1 = one(k) - test_xi - test_eta
        tb2 = test_xi
        tb3 = test_eta
        x, y, z = _metal_face_point(face_vertices, test_index, face_count, tb1, tb2, tb3)
        trial_xi = rule_trial_points[q]
        trial_eta = rule_trial_points[q + rule_point_count]
        rb1 = one(k) - trial_xi - trial_eta
        rb2 = trial_xi
        rb3 = trial_eta
        trial_basis = _metal_basis_value(local_column, rb1, rb2, rb3)
        sx, sy, sz = _metal_face_point(face_vertices, trial_index, face_count, rb1, rb2, rb3)
        sx *= trial_sign_x
        sy *= trial_sign_y
        sz *= trial_sign_z
        dx = sx - x
        dy = sy - y
        dz = sz - z
        radius2 = dx * dx + dy * dy + dz * dz
        if radius2 > zero(k)
            inv_radius = _metal_fast_rsqrt(radius2)
            radius = radius2 * inv_radius
            phase = k * radius
            green_scale = inv_radius * inv_four_pi
            green_re = _metal_fast_cos(phase) * green_scale
            green_im = _metal_fast_sin(phase) * green_scale
            weight = rule_weights[q] * jac_scale
            trial_dot = (dx * trial_nx + dy * trial_ny + dz * trial_nz) * inv_radius
            grad_re = (-green_re * inv_radius - green_im * k) * trial_dot
            grad_im = (green_re * k - green_im * inv_radius) * trial_dot
            basis1 = tb1 * trial_basis
            basis2 = tb2 * trial_basis
            basis3 = tb3 * trial_basis
            dlp1_re += basis1 * grad_re * weight; dlp1_im += basis1 * grad_im * weight
            dlp2_re += basis2 * grad_re * weight; dlp2_im += basis2 * grad_im * weight
            dlp3_re += basis3 * grad_re * weight; dlp3_im += basis3 * grad_im * weight
            factor1 = curl1 - k2 * basis1 * normal_product
            factor2 = curl2 - k2 * basis2 * normal_product
            factor3 = curl3 - k2 * basis3 * normal_product
            hyp1_re += factor1 * green_re * weight; hyp1_im += factor1 * green_im * weight
            hyp2_re += factor2 * green_re * weight; hyp2_im += factor2 * green_im * weight
            hyp3_re += factor3 * green_re * weight; hyp3_im += factor3 * green_im * weight
        end
        q += 1
    end
    value_offset = (local_column - 1) * 3
    dlp_values[pair_position + value_offset * pair_count] = Complex(dlp1_re, dlp1_im)
    dlp_values[pair_position + (value_offset + 1) * pair_count] = Complex(dlp2_re, dlp2_im)
    dlp_values[pair_position + (value_offset + 2) * pair_count] = Complex(dlp3_re, dlp3_im)
    hypersingular_values[pair_position + value_offset * pair_count] = Complex(hyp1_re, hyp1_im)
    hypersingular_values[pair_position + (value_offset + 1) * pair_count] = Complex(hyp2_re, hyp2_im)
    hypersingular_values[pair_position + (value_offset + 2) * pair_count] = Complex(hyp3_re, hyp3_im)
    return nothing
end

function _metal_singular_slp_adjoint_gather_kernel!(
    single_layer,
    adjoint_double_layer,
    slp_values,
    adjoint_values,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    dp0_elements,
    pair_offsets,
    trial_indices,
    pair_count,
    p1_dof_count,
    dp0_dof_count,
)
    linear_index = _metal_global_linear_index()
    linear_index > p1_dof_count * dp0_dof_count && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    trial_index = Int(dp0_elements[column])
    trial_index == 0 && return nothing
    slp = zero(eltype(single_layer))
    adj = zero(eltype(adjoint_double_layer))
    position = Int(vertex_offsets[row])
    position_stop = Int(vertex_offsets[row + 1]) - 1
    while position <= position_stop
        test_index = Int(incident_elements[position])
        local_row = Int(incident_local_indices[position])
        pair_position = _metal_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
        if pair_position != 0
            value_index = pair_position + (local_row - 1) * pair_count
            slp += slp_values[value_index]
            adj += adjoint_values[value_index]
        end
        position += 1
    end
    single_layer[linear_index] += slp
    adjoint_double_layer[linear_index] += adj
    return nothing
end

function _metal_singular_dlp_hyp_gather_kernel!(
    double_layer,
    hypersingular,
    dlp_values,
    hypersingular_values,
    vertex_offsets,
    incident_elements,
    incident_local_indices,
    pair_offsets,
    trial_indices,
    pair_count,
    p1_dof_count,
)
    linear_index = _metal_global_linear_index()
    linear_index > p1_dof_count * p1_dof_count && return nothing
    row = ((linear_index - 1) % p1_dof_count) + 1
    column = div(linear_index - 1, p1_dof_count) + 1
    dlp = zero(eltype(double_layer))
    hyp = zero(eltype(hypersingular))
    test_position = Int(vertex_offsets[row])
    test_stop = Int(vertex_offsets[row + 1]) - 1
    while test_position <= test_stop
        test_index = Int(incident_elements[test_position])
        local_row = Int(incident_local_indices[test_position])
        trial_position = Int(vertex_offsets[column])
        trial_stop = Int(vertex_offsets[column + 1]) - 1
        while trial_position <= trial_stop
            trial_index = Int(incident_elements[trial_position])
            local_column = Int(incident_local_indices[trial_position])
            pair_position = _metal_find_singular_pair(pair_offsets, trial_indices, test_index, trial_index)
            if pair_position != 0
                value_index = pair_position + ((local_column - 1) * 3 + local_row - 1) * pair_count
                dlp += dlp_values[value_index]
                hyp += hypersingular_values[value_index]
            end
            trial_position += 1
        end
        test_position += 1
    end
    double_layer[linear_index] += dlp
    hypersingular[linear_index] += hyp
    return nothing
end

function _launch_metal_singular_block_gather_kernels!(
    operators,
    regular_cache::MetalRegularAssemblyCache,
    singular_cache::MetalSingularCorrectionCache,
    k,
    transform::SymmetryTransform=SymmetryTransform(:identity, SVector{3,Int}(1, 1, 1), 1),
)
    slp_entries = regular_cache.p1_dof_count * regular_cache.dp0_dof_count
    p1_entries = regular_cache.p1_dof_count * regular_cache.p1_dof_count
    sx = typeof(k)(transform.signs[1])
    sy = typeof(k)(transform.signs[2])
    sz = typeof(k)(transform.signs[3])
    csx = typeof(k)(transform.determinant * transform.signs[1])
    csy = typeof(k)(transform.determinant * transform.signs[2])
    csz = typeof(k)(transform.determinant * transform.signs[3])
    pair_count = singular_cache.pair_count
    pair_count == 0 && return nothing
    rule_point_count = length(singular_cache.rule_weights)
    slp_values = Metal.zeros(eltype(operators.single_layer), pair_count, 3)
    adjoint_values = Metal.zeros(eltype(operators.adjoint_double_layer), pair_count, 3)
    dlp_values = Metal.zeros(eltype(operators.double_layer), pair_count, 9)
    hypersingular_values = Metal.zeros(eltype(operators.hypersingular), pair_count, 9)
    _metal_launch(
        _metal_singular_slp_adjoint_blocks_kernel!,
        pair_count,
        slp_values,
        adjoint_values,
        singular_cache.test_indices,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        regular_cache.face_vertices,
        regular_cache.normals,
        k,
        regular_cache.face_count,
        pair_count,
        rule_point_count,
        sx,
        sy,
        sz,
    )
    _metal_launch(
        _metal_singular_dlp_hyp_blocks_kernel!,
        3 * pair_count,
        dlp_values,
        hypersingular_values,
        singular_cache.test_indices,
        singular_cache.trial_indices,
        singular_cache.rule_indices,
        singular_cache.jac_scales,
        singular_cache.normal_products,
        singular_cache.rule_offsets,
        singular_cache.rule_test_points,
        singular_cache.rule_trial_points,
        singular_cache.rule_weights,
        regular_cache.face_vertices,
        regular_cache.normals,
        regular_cache.curls,
        k,
        regular_cache.face_count,
        pair_count,
        rule_point_count,
        sx,
        sy,
        sz,
        csx,
        csy,
        csz,
    )
    _metal_launch(
        _metal_singular_slp_adjoint_gather_kernel!,
        slp_entries,
        operators.single_layer,
        operators.adjoint_double_layer,
        slp_values,
        adjoint_values,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        regular_cache.dp0_elements,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        pair_count,
        regular_cache.p1_dof_count,
        regular_cache.dp0_dof_count,
    )
    _metal_launch(
        _metal_singular_dlp_hyp_gather_kernel!,
        p1_entries,
        operators.double_layer,
        operators.hypersingular,
        dlp_values,
        hypersingular_values,
        regular_cache.vertex_offsets,
        regular_cache.incident_elements,
        regular_cache.incident_local_indices,
        singular_cache.pair_offsets,
        singular_cache.trial_indices,
        pair_count,
        regular_cache.p1_dof_count,
    )
    Metal.synchronize()
    Metal.unsafe_free!(slp_values)
    Metal.unsafe_free!(adjoint_values)
    Metal.unsafe_free!(dlp_values)
    Metal.unsafe_free!(hypersingular_values)
    return nothing
end

# Host fallback: the same singular corrections computed with the CPU Duffy code
# and added to the device operators. Used when BLAB_METAL_SINGULAR_MODE=host,
# and by the validation script to separate kernel defects from rule defects.
function _metal_add_host_singular_corrections!(
    operators,
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    correction_cache::SingularCorrectionCache{T},
    rule::TriangleRule{T},
    singular_order::Int,
    element_indices,
    image_transforms::Vector{SymmetryTransform},
) where {T<:AbstractFloat}
    p1_count = p1_space.global_dof_count
    dp0_count = dp0_space.global_dof_count
    single_layer = zeros(Complex{T}, p1_count, dp0_count)
    double_layer = zeros(Complex{T}, p1_count, p1_count)
    adjoint_double_layer = zeros(Complex{T}, p1_count, dp0_count)
    hypersingular = zeros(Complex{T}, p1_count, p1_count)
    elements = _beat_cpu_element_data(mesh, p1_space, dp0_space)
    indices = collect(element_indices)
    for test_index in indices
        _beat_cpu_accumulate_singular_test!(
            single_layer,
            double_layer,
            adjoint_double_layer,
            hypersingular,
            elements,
            correction_cache.pairs_by_test[test_index],
            correction_cache.rules,
            k,
        )
    end
    image_singular_pairs = 0
    if !isempty(image_transforms)
        regular_quadrature = _beat_cpu_regular_quadrature_data(mesh, rule)
        for transform in image_transforms
            image_cache = _beat_cpu_image_singular_cache(mesh, singular_order, indices, transform)
            image_singular_pairs += image_cache.pair_count
            image_cache.pair_count == 0 && continue
            image_elements = _beat_cpu_reflect_element_data(elements, transform)
            image_quadrature = _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform)
            for test_index in indices
                _beat_cpu_accumulate_image_singular_delta_test!(
                    single_layer,
                    double_layer,
                    adjoint_double_layer,
                    hypersingular,
                    elements,
                    image_elements,
                    image_cache.pairs_by_test[test_index],
                    image_cache.rules,
                    k,
                    regular_quadrature,
                    image_quadrature,
                )
            end
        end
    end
    for (device, host) in (
        (operators.single_layer, single_layer),
        (operators.double_layer, double_layer),
        (operators.adjoint_double_layer, adjoint_double_layer),
        (operators.hypersingular, hypersingular),
    )
        d_host = MtlArray(host)
        device .+= d_host
        Metal.synchronize()
        Metal.unsafe_free!(d_host)
    end
    return image_singular_pairs
end
