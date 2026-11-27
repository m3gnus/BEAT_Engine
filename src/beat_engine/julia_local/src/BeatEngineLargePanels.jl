# Integration of electrically large panels.
#
# The fixed regular rule has six points per triangle. The kernel's phase moves
# by up to k*h across a panel of largest edge h, so the rule stops resolving it
# once a panel spans a wavelength or so: at k*h = 16 a far-apart pair's blocks
# are off by several hundred percent, on equilateral panels as much as on
# slender ones. Field evaluation sums over the same six points and has the same
# problem. Nothing about a large panel is singular, it just needs more points.
#
# So each panel above the k*h limit gets its own rule, chosen from its
# own electrical size. A pair is integrated with its test panel's rule on the
# test side and its trial panel's rule on the trial side, which is what the
# kernel needs: the error along one side depends on that side's size, so a
# large panel paired with a small one pays for one fine rule, not two. Touching
# pairs take the same decision through the Duffy order.
#
# The result is added as a correction after the existing passes,
#
#   operator += I(fine, fine) - I(base, base)
#
# so a frequency with no large panel has no plan at all and runs the same code
# as before, bit for bit, and so does every pair of small panels at a frequency
# that has one. The correction reuses the assembly's own kernels; the subtracted
# term is the same kernel with negated weights.
#
# `LargePanelRules` lives as long as a request and owns what does not depend on
# frequency. `large_panel_plan` returns this frequency's decision, or `nothing`.

const BEAT_LARGE_PANEL_MAX_ORDER = 24
const BEAT_LARGE_PANEL_MAX_SINGULAR_ORDER = 16

function _large_panel_env(name::String, default::Float64)
    value = strip(get(ENV, name, ""))
    isempty(value) && return default
    parsed = tryparse(Float64, value)
    parsed === nothing && error("$(name) must be a number; got $(repr(value)).")
    return parsed
end

beat_large_panel_kh_limit() = _large_panel_env("BLAB_BEAT_LARGE_PANEL_KH", 6.0)

# Gauss needs about one point per two radians of phase range. One more keeps every
# block within 4e-4 of partitioned panels up to k*h = 40; see docs/Large Panels.md.
large_panel_regular_order(kh::Float64) = min(BEAT_LARGE_PANEL_MAX_ORDER, ceil(Int, kh / 2) + 1)
# One more order holds touching pairs to the same 1e-3; cap the Duffy rule's growth.
large_panel_singular_order(kh::Float64, base_order::Int) =
    max(base_order, min(BEAT_LARGE_PANEL_MAX_SINGULAR_ORDER, large_panel_regular_order(kh) + 1))

# A collapsed Gauss rule, thinned towards the collapsed vertex: the lines of
# constant u shorten with 1 - u, so the phase range along them does too. This
# keeps the tensor rule's accuracy on oscillating kernels with 60% of its points
# (52 against 81 at order 9), which matters squared for a large-large pair.
large_panel_triangle_rule(::Type{T}, order::Int) where {T<:AbstractFloat} =
    collapsed_triangle_rule(T, order, jacobian -> clamp(ceil(Int, jacobian * order) + 1, 2, order))

struct LargePanelRules{T<:AbstractFloat}
    face_vertices::Vector{NTuple{3,SVector{3,T}}}
    max_edges::Vector{Float64}
    singular_order::Int
    panel_rules::Dict{Int,TriangleRule{T}}
    singular_caches::Dict{Tuple{Int,Vector{Int}},SingularCorrectionCache{T}}
    image_singular_caches::Dict{Tuple{Int,SVector{3,Int},Vector{Int}},BeatCpuImageSingularCache{T}}
    lock::ReentrantLock
end

# Field evaluation reads panel rules only, so it can leave the Duffy order out.
function LargePanelRules(mesh::BoundaryMesh{T}, singular_order::Int=4) where {T<:AbstractFloat}
    max_edges = [
        maximum(norm(SVector{3,Float64}(vertices[a]) - SVector{3,Float64}(vertices[b])) for (a, b) in ((1, 2), (2, 3), (3, 1)))
        for vertices in mesh.face_vertices
    ]
    return LargePanelRules{T}(
        mesh.face_vertices, max_edges, singular_order,
        Dict{Int,TriangleRule{T}}(), Dict{Tuple{Int,Vector{Int}},SingularCorrectionCache{T}}(),
        Dict{Tuple{Int,SVector{3,Int},Vector{Int}},BeatCpuImageSingularCache{T}}(), ReentrantLock(),
    )
end

struct LargePanelPlan{T<:AbstractFloat}
    rules::LargePanelRules{T}
    k::Float64
    # Per face: rule order, 0 for the base rule.
    panel_orders::Vector{Int}
    # Per face: Duffy order of a touching pair is the larger of its two panels'.
    singular_orders::Vector{Int}
    panel_rules::Dict{Int,TriangleRule{T}}
end

"""
    large_panel_plan(rules, k)

This frequency's rule per panel, or `nothing` when no panel is above the limit:
callers then run exactly the unplanned code.
"""
large_panel_plan(::Nothing, k::Real) = nothing
function large_panel_plan(rules::LargePanelRules{T}, k::Real) where {T<:AbstractFloat}
    wavenumber = abs(Float64(k))
    limit = beat_large_panel_kh_limit()
    limit > 0 || return nothing
    any(edge -> wavenumber * edge > limit, rules.max_edges) || return nothing
    panel_orders = [wavenumber * edge > limit ? large_panel_regular_order(wavenumber * edge) : 0 for edge in rules.max_edges]
    singular_orders = [
        wavenumber * edge > limit ? large_panel_singular_order(wavenumber * edge, rules.singular_order) : rules.singular_order
        for edge in rules.max_edges
    ]
    panel_rules = lock(rules.lock) do
        Dict(order => get!(() -> large_panel_triangle_rule(T, order), rules.panel_rules, order)
             for order in unique(panel_orders) if order != 0)
    end
    return LargePanelPlan{T}(rules, wavenumber, panel_orders, singular_orders, panel_rules)
end

"""Panels above the limit, and the mesh's largest k*h, for diagnostics."""
function large_panel_summary(plan)
    plan === nothing && return (panels=0, max_kh=nothing, max_order=0)
    return (
        panels=count(!=(0), plan.panel_orders),
        max_kh=plan.k * maximum(plan.rules.max_edges),
        max_order=maximum(plan.panel_orders),
    )
end

# Why a solve path left the correction out, as `large_panel_report` reports it.
const LARGE_PANEL_UNSUPPORTED_BACKEND = "unsupported_backend"
const LARGE_PANEL_NEAR_CORRECTION = "near_correction"

"""
    large_panel_report(rules, k, plan, unapplied)

One frequency's large-panel diagnostics, whether or not the correction ran.
`rules` is built for every solve path. `plan` is what the assembly used, or
`nothing`; `unapplied` names why a path without the correction leaves it out
(`LARGE_PANEL_UNSUPPORTED_BACKEND` or `LARGE_PANEL_NEAR_CORRECTION`), and is
ignored where the correction runs.

`correction` is `applied`, `not_needed` (no panel above the limit), `disabled`
(limit set to 0), or `unapplied`. `panels` counts panels above the limit,
corrected or not; `max_order` is the largest rule order used, 0 when none was.
"""
function large_panel_report(rules::LargePanelRules, k::Real, plan, unapplied::String)
    wavenumber = abs(Float64(k))
    limit = beat_large_panel_kh_limit()
    max_kh = wavenumber * maximum(rules.max_edges; init=0.0)
    panels = limit > 0 ? count(edge -> wavenumber * edge > limit, rules.max_edges) : 0
    correction = plan !== nothing ? "applied" :
                 limit <= 0 ? "disabled" :
                 panels == 0 ? "not_needed" : unapplied
    return (
        panels=panels,
        max_kh=max_kh,
        max_order=plan === nothing ? 0 : maximum(plan.panel_orders),
        correction=correction,
        limit=limit,
    )
end

"""The result diagnostics for one `large_panel_report`."""
large_panel_diagnostics(report) = Dict{String,Any}(
    "large_panel_count" => report.panels,
    "large_panel_max_order" => report.max_order,
    "large_panel_max_kh" => report.max_kh,
    "large_panel_correction" => report.correction,
)

"""
A warning when panels above the limit ran on the fixed rules, else `nothing`.
Drivers emit it once per request, at the first such frequency.
"""
function large_panel_warning(report, frequency_hz::Real)
    report.correction in (LARGE_PANEL_UNSUPPORTED_BACKEND, LARGE_PANEL_NEAR_CORRECTION) || return nothing
    reason = report.correction == LARGE_PANEL_NEAR_CORRECTION ?
        "near corrections are enabled and replace the same pairs" :
        "this backend or solve path does not have it; CPU and Metal exterior solves do"
    return string(
        "Large-panel integration not applied: ", reason, ". At ", round(Float64(frequency_hz); digits=1), " Hz, ",
        report.panels, " panels exceed k*h = ", report.limit, " (largest ", round(report.max_kh; digits=2),
        "), so results at this and higher frequencies may be inaccurate. See docs/Large Panels.md.",
    )
end

# The correction subtracts the base pass's Duffy rule by the order the plan was
# built with, and floors each panel's Duffy order there: both must be the
# assembly's own `singular_order`.
_large_panel_require_singular_order(::Nothing, singular_order::Int) = nothing
function _large_panel_require_singular_order(plan::LargePanelPlan, singular_order::Int)
    plan.rules.singular_order == singular_order || throw(ArgumentError(
        "LargePanelRules was built for singular_order $(plan.rules.singular_order), but the assembly uses " *
        "$(singular_order); build it with LargePanelRules(mesh, $(singular_order)).",
    ))
    return nothing
end

_large_panel_rule(::Nothing, base_rule, element_index) = base_rule
function _large_panel_rule(plan::LargePanelPlan, base_rule, element_index)
    order = plan.panel_orders[element_index]
    return order == 0 ? base_rule : plan.panel_rules[order]
end

# --------------------------------------------------------------------------
# CPU quadrature of one plan.

struct BeatCpuPanelClass{T<:AbstractFloat}
    order::Int
    indices::Vector{Int}
    elements::Vector{BeatCpuElementData{T}}
    # The class's elements under their own rule, and under the base rule with
    # negated weights: the subtracted half of the correction.
    fine::BeatCpuRegularSoA{T}
    negated_base::BeatCpuRegularSoA{T}
    image_elements::Vector{Vector{BeatCpuElementData{T}}}
    image_fine::Vector{BeatCpuRegularSoA{T}}
    image_negated_base::Vector{BeatCpuRegularSoA{T}}
end

_beat_cpu_negated(quad::BeatCpuRegularQuadratureData) =
    BeatCpuRegularQuadratureData(quad.points, quad.basis, -quad.weights)

function _beat_cpu_negated(soa::BeatCpuRegularSoA{T}) where {T<:AbstractFloat}
    return BeatCpuRegularSoA{T}(soa.indices, soa.n, soa.px, soa.py, soa.pz, soa.nx, soa.ny, soa.nz,
        soa.area, soa.curl, soa.p1_dofs, soa.dp0_dofs, soa.basis, -soa.weights)
end

function _beat_cpu_panel_quadrature(mesh::BoundaryMesh{T}, plan::LargePanelPlan{T}, base_quadrature) where {T<:AbstractFloat}
    templates = Dict(order => (basis=[p1_values(point) for point in rule.points], weights=rule.weights)
                     for (order, rule) in plan.panel_rules)
    quadrature = copy(base_quadrature)
    for element_index in eachindex(quadrature)
        order = plan.panel_orders[element_index]
        order == 0 && continue
        template = templates[order]
        quadrature[element_index] = BeatCpuRegularQuadratureData(
            [local_to_global(mesh.face_vertices[element_index], point) for point in plan.panel_rules[order].points],
            template.basis,
            template.weights,
        )
    end
    return quadrature
end

function _beat_cpu_panel_classes(elements, passes, plan::LargePanelPlan{T}, indices, image_transforms) where {T<:AbstractFloat}
    (; base_quadrature, quadrature) = passes
    reflected_base = passes.image_base_quadrature
    reflected_fine = passes.image_quadrature
    orders = sort!(unique(plan.panel_orders[element_index] for element_index in indices))
    classes = BeatCpuPanelClass{T}[]
    for order in orders
        class_indices = [element_index for element_index in indices if plan.panel_orders[element_index] == order]
        class_elements = elements[class_indices]
        positions = eachindex(class_indices)
        soa_of(class_elements, source) = BeatCpuRegularSoA(class_elements, source[class_indices], positions)
        image_elements = [_beat_cpu_reflect_element_data(class_elements, transform) for transform in image_transforms]
        push!(classes, BeatCpuPanelClass{T}(
            order, class_indices, class_elements,
            soa_of(class_elements, quadrature),
            _beat_cpu_negated(soa_of(class_elements, base_quadrature)),
            image_elements,
            [soa_of(image_elements[i], reflected_fine[i]) for i in eachindex(image_transforms)],
            [_beat_cpu_negated(soa_of(image_elements[i], reflected_base[i])) for i in eachindex(image_transforms)],
        ))
    end
    return classes
end

function _beat_cpu_panel_regular_costs(passes, plan, indices)
    point_counts = [length(quad.points) for quad in passes.quadrature]
    total_points = sum(point_counts[index] for index in indices)
    large_indices = [index for index in indices if plan.panel_orders[index] != 0]
    large_points = sum((point_counts[index] for index in large_indices); init=0)
    return [
        plan.panel_orders[index] != 0 ? point_counts[index] * total_points :
            length(passes.base_quadrature[index].points) * large_points +
            length(passes.base_quadrature[index].points)^2 * length(large_indices)
        for index in eachindex(point_counts)
    ]
end

# Large panels can be contiguous in element order. Deal the expensive tests
# across scratch-owning chunks, keeping each colour's disjoint rows together.
function _beat_cpu_panel_regular_groups!(apply_test!, groups, scratch, threaded_enabled::Bool, costs)
    if !threaded_enabled
        return _beat_cpu_regular_simd_groups!(apply_test!, groups, scratch, false)
    end
    for group in groups
        ordered = sort(group; by=index -> costs[index], rev=true)
        chunk_count = min(length(group), length(scratch))
        Threads.@threads for chunk in 1:chunk_count
            for group_index in chunk:chunk_count:length(ordered)
                apply_test!(ordered[group_index], scratch[chunk])
            end
        end
    end
    return nothing
end

# The correction's regular half. Test ownership is that of the regular pass, so
# the colouring keeps the writes race-free. `apply_test!` is the regular pass's
# per-test kernel call.
function _beat_cpu_panel_regular_pass!(
    apply_test!, elements, passes, plan, indices, image_transforms, color_groups, threaded_enabled::Bool, scratch,
)
    (; base_quadrature, quadrature) = passes
    classes = _beat_cpu_panel_classes(elements, passes, plan, indices, image_transforms)
    base_class = isempty(classes) ? 0 : (first(classes).order == 0 ? 1 : 0)
    costs = _beat_cpu_panel_regular_costs(passes, plan, indices)
    _beat_cpu_panel_regular_groups!(color_groups, scratch, threaded_enabled, costs) do test_index, work
        test_data = elements[test_index]
        test_large = plan.panel_orders[test_index] != 0
        for (class_index, class) in enumerate(classes)
            !test_large && class_index == base_class && continue
            apply_test!(test_data, quadrature[test_index], class.elements, class.fine, work, true)
            apply_test!(test_data, base_quadrature[test_index], class.elements, class.negated_base, work, true)
            for image in eachindex(class.image_elements)
                apply_test!(test_data, quadrature[test_index], class.image_elements[image], class.image_fine[image], work, false)
                apply_test!(test_data, base_quadrature[test_index], class.image_elements[image], class.image_negated_base[image], work, false)
            end
        end
    end
    return nothing
end

# Touching pairs by Duffy order, for pairs with at least one large panel. The
# pairs of an order's cache are picked with the same test as the base cache's,
# so the two lists describe the same pairs.
_large_panel_pair_order(plan, pair) = max(plan.singular_orders[pair.test_index], plan.singular_orders[pair.trial_index])
_large_panel_pair_is_large(plan, pair) = plan.panel_orders[pair.test_index] != 0 || plan.panel_orders[pair.trial_index] != 0

function _large_panel_pairs(plan, pairs_by_test, order; negate::Bool=false)
    return [
        [
            negate ? SingularCorrectionPair(pair.test_index, pair.trial_index, pair.rule_index, -pair.jac_scale, pair.normal_product) : pair
            for pair in pairs
            if _large_panel_pair_is_large(plan, pair) && _large_panel_pair_order(plan, pair) == order
        ]
        for pairs in pairs_by_test
    ]
end

function _large_panel_singular_cache(mesh, plan::LargePanelPlan, order::Int, indices)
    rules = plan.rules
    return lock(rules.lock) do
        get!(() -> build_singular_correction_cache(mesh, order, indices), rules.singular_caches, (order, collect(Int, indices)))
    end
end

function _large_panel_image_singular_cache(mesh, plan::LargePanelPlan, order::Int, indices, transform)
    rules = plan.rules
    return lock(rules.lock) do
        get!(() -> _beat_cpu_image_singular_cache(mesh, order, indices, transform), rules.image_singular_caches, (order, transform.signs, collect(Int, indices)))
    end
end

# Every order a pair with a large panel uses, the base order included: an
# image-touching pair still needs its regular correction taken back out when its
# Duffy order does not rise.
_large_panel_singular_order_set(plan) = sort!(unique(
    plan.singular_orders[face] for face in eachindex(plan.panel_orders) if plan.panel_orders[face] != 0
))

function _beat_cpu_panel_groups!(apply!, indices, color_groups, threaded_enabled::Bool)
    if threaded_enabled
        for group in color_groups
            Threads.@threads for group_index in eachindex(group)
                apply!(group[group_index])
            end
        end
    else
        for test_index in indices
            apply!(test_index)
        end
    end
    return nothing
end

# Everything a CPU correction pass needs from one plan.
struct BeatCpuPanelPasses{T<:AbstractFloat}
    elements::Vector{BeatCpuElementData{T}}
    base_quadrature::Vector{BeatCpuRegularQuadratureData{T}}
    quadrature::Vector{BeatCpuRegularQuadratureData{T}}
    negated_base_quadrature::Vector{BeatCpuRegularQuadratureData{T}}
    image_elements::Vector{Vector{BeatCpuElementData{T}}}
    image_base_quadrature::Vector{Vector{BeatCpuRegularQuadratureData{T}}}
    image_quadrature::Vector{Vector{BeatCpuRegularQuadratureData{T}}}
end

function BeatCpuPanelPasses(mesh::BoundaryMesh{T}, plan::LargePanelPlan{T}, elements, base_quadrature, image_transforms) where {T<:AbstractFloat}
    quadrature = _beat_cpu_panel_quadrature(mesh, plan, base_quadrature)
    image_elements = [_beat_cpu_reflect_element_data(elements, transform) for transform in image_transforms]
    image_base = [_beat_cpu_reflect_regular_quadrature_data(base_quadrature, transform) for transform in image_transforms]
    image_fine = [_beat_cpu_reflect_regular_quadrature_data(quadrature, transform) for transform in image_transforms]
    return BeatCpuPanelPasses{T}(
        elements, base_quadrature, quadrature, _beat_cpu_negated.(base_quadrature), image_elements, image_base, image_fine,
    )
end

"""
Add the large-panel correction to a fused Burton-Miller system, before the
orbit weights and the identity terms.
"""
function _beat_cpu_bm_apply_large_panels!(
    lhs, rhs, q_complex, mesh::BoundaryMesh{T}, plan::LargePanelPlan{T}, elements, base_quadrature, base_singular_cache,
    indices, color_groups, threaded_enabled::Bool, image_transforms, k::T, coupling::Complex{T}, singular_rule_form,
) where {T<:AbstractFloat}
    passes = BeatCpuPanelPasses(mesh, plan, elements, base_quadrature, image_transforms)
    groups = threaded_enabled ? color_groups : _beat_cpu_element_color_groups(mesh, indices)
    scratch = [BeatCpuRegularScratch{T}(_BEAT_CPU_REGULAR_BLOCK_SIZE)
               for _ in 1:(threaded_enabled ? Threads.nthreads() : 1)]
    _beat_cpu_transpose_square!(lhs)
    _beat_cpu_panel_regular_pass!(
        elements, passes, plan, indices, image_transforms, groups, threaded_enabled, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q_complex, test_data, test_quad, trial_elements, soa, k, coupling, work, skip_adjacent,
        )
    end
    _beat_cpu_transpose_square!(lhs)
    _beat_cpu_bm_apply_large_panel_singular!(
        lhs, rhs, q_complex, mesh, plan, passes, base_singular_cache, indices, color_groups, threaded_enabled,
        image_transforms, k, coupling, singular_rule_form,
    )
    return nothing
end

# The touching pairs. Host arrays only: Metal hands in its shared buffers.
function _beat_cpu_bm_apply_large_panel_singular!(
    lhs, rhs, q_complex, mesh::BoundaryMesh{T}, plan::LargePanelPlan{T}, passes, base_singular_cache,
    indices, color_groups, threaded_enabled::Bool, image_transforms, k::T, coupling::Complex{T}, singular_rule_form,
) where {T<:AbstractFloat}
    elements = passes.elements
    for order in _large_panel_singular_order_set(plan)
        cache = _large_panel_singular_cache(mesh, plan, order, indices)
        fine_pairs = _large_panel_pairs(plan, cache.pairs_by_test, order)
        base_pairs = _large_panel_pairs(plan, base_singular_cache.pairs_by_test, order; negate=true)
        fine_rules = singular_rule_form(cache.rules)
        base_rules = singular_rule_form(base_singular_cache.rules)
        _beat_cpu_panel_groups!(indices, color_groups, threaded_enabled) do test_index
            _beat_cpu_bm_singular_test!(lhs, rhs, q_complex, elements, fine_pairs[test_index], fine_rules, k, coupling)
            _beat_cpu_bm_singular_test!(lhs, rhs, q_complex, elements, base_pairs[test_index], base_rules, k, coupling)
        end
        for (transform_index, transform) in enumerate(image_transforms)
            image_cache = _large_panel_image_singular_cache(mesh, plan, order, indices, transform)
            image_base = _large_panel_image_singular_cache(mesh, plan, plan.rules.singular_order, indices, transform)
            fine_image_pairs = _large_panel_pairs(plan, image_cache.pairs_by_test, order)
            base_image_pairs = _large_panel_pairs(plan, image_base.pairs_by_test, order; negate=true)
            image_fine_rules = singular_rule_form(image_cache.rules)
            image_base_rules = singular_rule_form(image_base.rules)
            image_elements = passes.image_elements[transform_index]
            # The regular correction already moved these pairs from the base
            # rule to the fine one, so what is left is
            # (Duffy fine - regular fine) - (Duffy base - regular base).
            _beat_cpu_panel_groups!(indices, color_groups, threaded_enabled) do test_index
                _beat_cpu_bm_image_singular_delta_test!(
                    lhs, rhs, q_complex, elements, image_elements, fine_image_pairs[test_index], image_fine_rules,
                    k, passes.quadrature, passes.image_quadrature[transform_index], coupling,
                )
                _beat_cpu_bm_image_singular_delta_test!(
                    lhs, rhs, q_complex, elements, image_elements, base_image_pairs[test_index], image_base_rules,
                    k, passes.negated_base_quadrature, passes.image_base_quadrature[transform_index], coupling,
                )
            end
        end
    end
    return nothing
end

"""The same correction for the four operators, in place, before the orbit weights."""
function _beat_cpu_apply_large_panels!(
    operators, mesh::BoundaryMesh{T}, plan::LargePanelPlan{T}, elements, base_quadrature, base_singular_cache,
    indices, color_groups, threaded_enabled::Bool, image_transforms, k::T,
) where {T<:AbstractFloat}
    passes = BeatCpuPanelPasses(mesh, plan, elements, base_quadrature, image_transforms)
    groups = threaded_enabled ? color_groups : _beat_cpu_element_color_groups(mesh, indices)
    scratch = [BeatCpuOperatorRegularScratch{T}(_BEAT_CPU_REGULAR_BLOCK_SIZE)
               for _ in 1:(threaded_enabled ? Threads.nthreads() : 1)]
    (; single_layer, double_layer, adjoint_double_layer, hypersingular) = operators
    _beat_cpu_transpose_square!(double_layer)
    _beat_cpu_transpose_square!(hypersingular)
    _beat_cpu_panel_regular_pass!(
        elements, passes, plan, indices, image_transforms, groups, threaded_enabled, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_accumulate_regular_test_simd!(
            single_layer, double_layer, adjoint_double_layer, hypersingular,
            test_data, test_quad, trial_elements, soa, k, work, skip_adjacent,
        )
    end
    _beat_cpu_transpose_square!(double_layer)
    _beat_cpu_transpose_square!(hypersingular)

    for order in _large_panel_singular_order_set(plan)
        cache = _large_panel_singular_cache(mesh, plan, order, indices)
        fine_pairs = _large_panel_pairs(plan, cache.pairs_by_test, order)
        base_pairs = _large_panel_pairs(plan, base_singular_cache.pairs_by_test, order; negate=true)
        _beat_cpu_panel_groups!(indices, color_groups, threaded_enabled) do test_index
            _beat_cpu_accumulate_singular_test!(single_layer, double_layer, adjoint_double_layer, hypersingular,
                elements, fine_pairs[test_index], cache.rules, k)
            _beat_cpu_accumulate_singular_test!(single_layer, double_layer, adjoint_double_layer, hypersingular,
                elements, base_pairs[test_index], base_singular_cache.rules, k)
        end
        for (transform_index, transform) in enumerate(image_transforms)
            image_cache = _large_panel_image_singular_cache(mesh, plan, order, indices, transform)
            image_base = _large_panel_image_singular_cache(mesh, plan, plan.rules.singular_order, indices, transform)
            fine_image_pairs = _large_panel_pairs(plan, image_cache.pairs_by_test, order)
            base_image_pairs = _large_panel_pairs(plan, image_base.pairs_by_test, order; negate=true)
            image_elements = passes.image_elements[transform_index]
            _beat_cpu_panel_groups!(indices, color_groups, threaded_enabled) do test_index
                _beat_cpu_accumulate_image_singular_delta_test!(
                    single_layer, double_layer, adjoint_double_layer, hypersingular, elements, image_elements,
                    fine_image_pairs[test_index], image_cache.rules, k, passes.quadrature, passes.image_quadrature[transform_index],
                )
                _beat_cpu_accumulate_image_singular_delta_test!(
                    single_layer, double_layer, adjoint_double_layer, hypersingular, elements, image_elements,
                    base_image_pairs[test_index], image_base.rules, k, passes.negated_base_quadrature, passes.image_base_quadrature[transform_index],
                )
            end
        end
    end
    return nothing
end
