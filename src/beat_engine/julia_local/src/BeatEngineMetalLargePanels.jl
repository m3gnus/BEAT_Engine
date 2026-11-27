# The regular half of the large-panel correction on Metal (BeatEngineLargePanels.jl).
#
# The fused pair kernels fix one rule per launch at compile time, and a large
# panel's rule is a per-frequency choice, so this kernel walks point tables at
# run time instead. It adds I(fine, fine) - I(base, base) into the same pair
# blocks the fused kernels fill, after them and before the gathers, so it runs
# once per chunk and needs no gather of its own. A pair of small panels returns
# straight away and those blocks are untouched.
#
# Image-touching pairs are not skipped here, unlike in the fused kernels: the
# host's singular correction takes the same form as on the CPU, which assumes
# the regular correction reached them.

# One launch stays below this many point pairs. A chunk can be all large panels.
const _METAL_LARGE_PANEL_LAUNCH_POINT_PAIRS = 1.5e9

struct MetalLargePanelTables
    points4     # (x, y, z, rule weight), each face's fine points then its base points
    coords4     # (xi, eta, 0, 0), same order
    offsets     # Int32 per face: fine start, fine stop, base start, base stop
    point_counts::Vector{Int}   # fine points per element position
    total_points::Int
end

function _metal_large_panel_tables(plan::LargePanelPlan, base_rule::TriangleRule, element_indices)
    mesh_vertices = plan.rules.face_vertices
    face_count = length(mesh_vertices)
    offsets = Vector{Int32}(undef, 4face_count)
    points = _MetalFloat4[]
    coords = _MetalFloat4[]
    function append_rule!(face, rule)
        first_point = length(points) + 1
        for (point, weight) in zip(rule.points, rule.weights)
            position = local_to_global(mesh_vertices[face], point)
            push!(points, _metal_float4(Float32(position[1]), Float32(position[2]), Float32(position[3]), Float32(weight)))
            push!(coords, _metal_float4(Float32(point[1]), Float32(point[2]), 0.0f0, 0.0f0))
        end
        return Int32(first_point), Int32(length(points))
    end
    for face in 1:face_count
        base_start, base_stop = append_rule!(face, base_rule)
        fine_start, fine_stop = plan.panel_orders[face] == 0 ? (base_start, base_stop) :
            append_rule!(face, plan.panel_rules[plan.panel_orders[face]])
        offsets[4(face - 1) + 1:4face] = Int32[fine_start, fine_stop, base_start, base_stop]
    end
    point_counts = [Int(offsets[4(face - 1) + 2] - offsets[4(face - 1) + 1]) + 1 for face in element_indices]
    device_points = device_coords = nothing
    try
        device_points = MtlArray(points)
        device_coords = MtlArray(coords)
        return MetalLargePanelTables(device_points, device_coords, MtlArray(offsets), point_counts, sum(point_counts))
    catch
        device_points === nothing || Metal.unsafe_free!(device_points)
        device_coords === nothing || Metal.unsafe_free!(device_coords)
        rethrow()
    end
end

function _release_metal_large_panel_tables!(tables::MetalLargePanelTables)
    Metal.unsafe_free!(tables.points4)
    Metal.unsafe_free!(tables.coords4)
    Metal.unsafe_free!(tables.offsets)
    return nothing
end

# Trial positions per launch. Split the test positions by point count when a
# trial panel alone exceeds the budget; runs of small panels pack together.
function _metal_large_panel_spans(tables::MetalLargePanelTables, chunk_start::Int, chunk_count::Int)
    trial_spans = Tuple{Int,Int}[]
    first_position = chunk_start
    point_pairs = 0.0
    for position in chunk_start:(chunk_start + chunk_count - 1)
        cost = Float64(tables.point_counts[position]) * tables.total_points
        if position > first_position && point_pairs + cost > _METAL_LARGE_PANEL_LAUNCH_POINT_PAIRS
            push!(trial_spans, (first_position, position - first_position))
            first_position = position
            point_pairs = 0.0
        end
        point_pairs += cost
    end
    push!(trial_spans, (first_position, chunk_start + chunk_count - first_position))
    spans = NTuple{4,Int}[]
    for (trial_start, trial_count) in trial_spans
        trial_points = sum(@view tables.point_counts[trial_start:(trial_start + trial_count - 1)])
        test_span_count = ceil(Int, Float64(trial_points) * tables.total_points / _METAL_LARGE_PANEL_LAUNCH_POINT_PAIRS)
        if test_span_count == 1
            push!(spans, (trial_start, trial_count, 1, length(tables.point_counts)))
            continue
        end
        test_point_limit = floor(Int, _METAL_LARGE_PANEL_LAUNCH_POINT_PAIRS / trial_points)
        test_point_target = cld(tables.total_points, test_span_count)
        test_start = 1
        test_points = 0
        for position in eachindex(tables.point_counts)
            cost = tables.point_counts[position]
            if position > test_start && (test_points >= test_point_target || test_points + cost > test_point_limit)
                push!(spans, (trial_start, trial_count, test_start, position - test_start))
                test_start = position
                test_points = 0
            end
            test_points += cost
        end
        push!(spans, (trial_start, trial_count, test_start, length(tables.point_counts) + 1 - test_start))
    end
    return spans
end

function _launch_metal_large_panel_kernels!(blocks, packed, cache, gather_tables, tables::MetalLargePanelTables, k, transforms,
    chunk_start, chunk_count, pair_stride, tile_x, tile_y)
    element_count = length(cache.element_indices)
    spans = _metal_large_panel_spans(tables, chunk_start, chunk_count)
    for transform in transforms
        (pair_offsets, singular_trial_indices, skip_mode, trial_sign_x, trial_sign_y, trial_sign_z,
         trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z) = transform
        panel_skip_mode = skip_mode == Int32(1) ? Int32(2) : skip_mode
        for (span_start, span_count, test_start, test_count) in spans
            Metal.@metal threads=(tile_x, tile_y) groups=(cld(test_count, tile_x), cld(span_count, tile_y)) _metal_large_panel_blocks_kernel!(
                blocks, tables.points4, tables.coords4, tables.offsets, packed.normals4, cache.areas, packed.curls4,
                cache.faces, gather_tables.elements,
                Int32(element_count), Int32(test_start - 1), Int32(test_count),
                Int32(chunk_start), Int32(span_start - chunk_start), Int32(span_count), pair_stride,
                k, inv(k), Int32(cache.face_count), pair_offsets, singular_trial_indices, panel_skip_mode,
                trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
            )
        end
    end
    return nothing
end

# Adds `sign` times the sum over one pair of point ranges to the block accumulators.
@inline function _metal_large_panel_range_sum(acc, context, points4, coords4, test_start, test_stop, trial_start, trial_stop, sign)
    k, inv_four_pi, jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs,
        inverse_k, curl_scale = context
    z3 = zero(SVector{3,typeof(k)})
    test_q = test_start
    while test_q <= test_stop
        @inbounds p = points4[test_q]
        @inbounds test_coords = coords4[test_q]
        test_xi = test_coords[1].value
        test_eta = test_coords[2].value
        test_basis = SVector(one(k) - test_xi - test_eta, test_xi, test_eta)
        x, y, z = p[1].value, p[2].value, p[3].value
        test_weight_scale = sign * p[4].value * jac_scale
        folded = (zero(k), zero(k), zero(k), zero(k), z3, z3, z3, z3)
        trial_q = trial_start
        while trial_q <= trial_stop
            @inbounds s = points4[trial_q]
            @inbounds trial_coords = coords4[trial_q]
            xi = trial_coords[1].value
            eta = trial_coords[2].value
            folded = _metal_packed_point_term(folded, s, xi, eta, one(k) - xi - eta, s[4].value, x, y, z,
                test_weight_scale, k, inv_four_pi, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz, trial_signs)
            trial_q += Int32(1)
        end
        acc = _metal_fused_test_point_update(acc, folded, test_basis, inverse_k, curl_scale)
        test_q += Int32(1)
    end
    return acc
end

function _metal_large_panel_blocks_kernel!(
    blocks, points4, coords4, offsets, normals4, areas, curls4, faces, elements,
    element_count::Int32, test_offset::Int32, test_count::Int32,
    chunk_start::Int32, span_offset::Int32, span_count::Int32, pair_stride::Int32,
    k, inverse_k, face_count::Int32, pair_offsets, singular_trial_indices, skip_mode,
    trial_sign_x, trial_sign_y, trial_sign_z, trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z,
)
    position = thread_position_in_grid_2d()
    test_position = Int32(position.x) + test_offset
    trial_local = Int32(position.y) + span_offset
    (Int32(position.x) > test_count || Int32(position.y) > span_count) && return nothing
    @inbounds test_index = Int32(elements[test_position])
    @inbounds trial_index = Int32(elements[chunk_start + trial_local - Int32(1)])
    @inbounds test_fine_start = Int32(offsets[(test_index - Int32(1)) * Int32(4) + Int32(1)])
    @inbounds test_fine_stop = Int32(offsets[(test_index - Int32(1)) * Int32(4) + Int32(2)])
    @inbounds test_base_start = Int32(offsets[(test_index - Int32(1)) * Int32(4) + Int32(3)])
    @inbounds test_base_stop = Int32(offsets[(test_index - Int32(1)) * Int32(4) + Int32(4)])
    @inbounds trial_fine_start = Int32(offsets[(trial_index - Int32(1)) * Int32(4) + Int32(1)])
    @inbounds trial_fine_stop = Int32(offsets[(trial_index - Int32(1)) * Int32(4) + Int32(2)])
    @inbounds trial_base_start = Int32(offsets[(trial_index - Int32(1)) * Int32(4) + Int32(3)])
    @inbounds trial_base_stop = Int32(offsets[(trial_index - Int32(1)) * Int32(4) + Int32(4)])
    # A small panel's fine range is its base range.
    test_fine_start == test_base_start && trial_fine_start == trial_base_start && return nothing
    _metal_pair_is_skipped(faces, face_count, test_index, trial_index, pair_offsets, singular_trial_indices, skip_mode) && return nothing
    T = typeof(k)
    @inbounds tn = normals4[test_index]
    @inbounds rn = normals4[trial_index]
    test_nx = tn[1].value
    test_ny = tn[2].value
    test_nz = tn[3].value
    trial_nx = trial_sign_x * rn[1].value
    trial_ny = trial_sign_y * rn[2].value
    trial_nz = trial_sign_z * rn[3].value
    normal_product = test_nx * trial_nx + test_ny * trial_ny + test_nz * trial_nz
    @inbounds jac_scale = T(4) * areas[test_index] * areas[trial_index]
    context = (k, T(0.07957747154594767), jac_scale, test_nx, test_ny, test_nz, trial_nx, trial_ny, trial_nz,
        SVector(trial_sign_x, trial_sign_y, trial_sign_z), inverse_k, inverse_k * k * k * normal_product)
    acc = (zero(SVector{9,T}), zero(SVector{9,T}), zero(SVector{3,T}), zero(SVector{3,T}), zero(k), zero(k))
    acc = _metal_large_panel_range_sum(acc, context, points4, coords4,
        test_fine_start, test_fine_stop, trial_fine_start, trial_fine_stop, one(T))
    acc = _metal_large_panel_range_sum(acc, context, points4, coords4,
        test_base_start, test_base_stop, trial_base_start, trial_base_stop, -one(T))
    lhs_re, lhs_im, rhs_re, rhs_im, g_total_re, g_total_im = acc
    curl_products = _metal_pair_curl_products4(curls4, test_index, trial_index,
        trial_curl_sign_x, trial_curl_sign_y, trial_curl_sign_z)
    lhs_re -= curl_products * (inverse_k * g_total_im)
    lhs_im += curl_products * (inverse_k * g_total_re)
    base = test_position + element_count * (trial_local - Int32(1))
    _metal_add_block!(blocks, base, pair_stride, Int32(0), lhs_re)
    _metal_add_block!(blocks, base, pair_stride, Int32(9), lhs_im)
    _metal_add_block!(blocks, base, pair_stride, Int32(18), rhs_re)
    _metal_add_block!(blocks, base, pair_stride, Int32(21), rhs_im)
    return nothing
end
