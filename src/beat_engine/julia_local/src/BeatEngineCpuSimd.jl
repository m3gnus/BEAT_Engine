# Vectorised regular-pair kernels for the CPU backend.
#
# The regular pass is an all-pairs loop. The scalar kernels visit one element
# pair at a time and, inside it, 9 (order 2) or 36 (order 4) quadrature-point
# pairs, each with a sincos, a reciprocal and a dozen complex accumulations.
# Here the test element is fixed and the TRIAL ELEMENTS are batched along a
# structure-of-arrays layout, so the innermost loop is a plain `@simd ivdep`
# loop over trial elements and LLVM picks the host's vector width. No SIMD
# package is involved.
#
# Three things differ from the scalar arithmetic, all of them so that the loop
# has no branch and no call:
#
# - sincos is a branch-free polynomial (three-part Cody-Waite reduction by
#   pi/2, Cephes coefficients). Measured max absolute error on [0, 2500] rad:
#   0.78 eps(T), against Base's 0.25 (Float32) and 0.39 (Float64).
# - A coincident point pair is masked (inv_radius = 0, which zeroes every term
#   it contributes) instead of skipped, and an adjacent trial element gets a
#   zero Jacobian and is not scattered.
# - The basis expansion is summed over trial points first and the outer
#   product with the test basis is taken afterwards.
#
# So the two kernels agree to rounding, not bitwise. The phase is also formed
# as k * (r^2 / r) rather than k * norm(r), so at large k*r they can differ by
# one ulp of phase, i.e. by about k*r*eps(T) in sin and cos -- the resolution
# Float32 phase has in either kernel.
#
# Nothing selects these kernels by default. Both CPU assembly functions and the
# backend dispatcher default to :scalar, so every caller that does not ask --
# the Metal and ROCm host-staged paths, the accelerator validators' CPU
# references, the coupled solvers -- keeps the scalar arithmetic bit for bit.
# The CPU solver entry points opt in by passing beat_cpu_regular_kernel().

const _BEAT_CPU_REGULAR_BLOCK_SIZE = 256

function beat_cpu_regular_kernel()
    value = lowercase(strip(get(ENV, "BLAB_BEAT_CPU_REGULAR_KERNEL", "simd")))
    value in ("simd", "scalar") || error(
        "BLAB_BEAT_CPU_REGULAR_KERNEL must be simd or scalar; got $(repr(value)).",
    )
    return Symbol(value)
end

function _beat_cpu_validated_regular_kernel(kernel::Symbol)
    kernel in (:simd, :scalar) || error(
        "CPU regular kernel must be :simd or :scalar; got $(repr(kernel)).",
    )
    return kernel
end

# Three-part Cody-Waite reduction and branch-free quadrant selection.
@inline function _beat_cpu_fast_sincos(x::Float32)
    n = round(x * 0.63661975f0)
    r = muladd(n, -1.5703125f0, x)
    r = muladd(n, -4.837512969970703125f-4, r)
    r = muladd(n, -7.54978995489188216f-8, r)
    z = r * r
    s = muladd(r * z, muladd(z, muladd(z, -1.9515295891f-4, 8.3321608736f-3), -1.6666654611f-1), r)
    c = muladd(z * z, muladd(z, muladd(z, 2.443315711809948f-5, -1.388731625493765f-3), 4.166664568298827f-2),
        muladd(z, -0.5f0, 1.0f0))
    q = unsafe_trunc(Int32, n)
    swap = (q & Int32(1)) != Int32(0)
    ss = ifelse(swap, c, s)
    cc = ifelse(swap, s, c)
    return ifelse((q & Int32(2)) != Int32(0), -ss, ss), ifelse(((q + Int32(1)) & Int32(2)) != Int32(0), -cc, cc)
end

@inline function _beat_cpu_fast_sincos(x::Float64)
    n = round(x * 0.6366197723675814)
    r = muladd(n, -1.5707962512969971, x)
    r = muladd(n, -7.549789415861596e-8, r)
    r = muladd(n, -5.390302858158119e-15, r)
    z = r * r
    ps = muladd(z, muladd(z, muladd(z, muladd(z, muladd(z, 1.58962301576546568060e-10, -2.50507477628578072866e-8),
        2.75573136213857245213e-6), -1.98412698295895385996e-4), 8.33333333332211858878e-3), -1.66666666666666307295e-1)
    pc = muladd(z, muladd(z, muladd(z, muladd(z, muladd(z, -1.13585365213876817300e-11, 2.08757008419747316778e-9),
        -2.75573141792967388112e-7), 2.48015872888517045348e-5), -1.38888888888730564116e-3), 4.16666666666665929218e-2)
    s = muladd(r * z, ps, r)
    c = muladd(z * z, pc, muladd(z, -0.5, 1.0))
    q = unsafe_trunc(Int64, n)
    swap = (q & 1) != 0
    ss = ifelse(swap, c, s)
    cc = ifelse(swap, s, c)
    return ifelse((q & 2) != 0, -ss, ss), ifelse(((q + 1) & 2) != 0, -cc, cc)
end

struct BeatCpuRegularSoA{T<:AbstractFloat}
    indices::Vector{Int}
    n::Int
    px::Matrix{T}
    py::Matrix{T}
    pz::Matrix{T}
    nx::Vector{T}
    ny::Vector{T}
    nz::Vector{T}
    area::Vector{T}
    curl::Matrix{T}
    basis::Vector{SVector{3,T}}
    weights::Vector{T}
end

function BeatCpuRegularSoA(
    elements::Vector{BeatCpuElementData{T}}, regular_quadrature, indices,
) where {T<:AbstractFloat}
    trial_indices = collect(Int, indices)
    n = length(trial_indices)
    first_quad = isempty(regular_quadrature) ? nothing : first(regular_quadrature)
    basis = first_quad === nothing ? SVector{3,T}[] : first_quad.basis
    weights = first_quad === nothing ? T[] : first_quad.weights
    nq = length(weights)
    px = zeros(T, n, nq)
    py = zeros(T, n, nq)
    pz = zeros(T, n, nq)
    nx = zeros(T, n)
    ny = zeros(T, n)
    nz = zeros(T, n)
    area = zeros(T, n)
    curl = zeros(T, n, 9)
    for (j, element_index) in enumerate(trial_indices)
        element = elements[element_index]
        for q in 1:nq
            px[j, q], py[j, q], pz[j, q] = regular_quadrature[element_index].points[q]
        end
        nx[j], ny[j], nz[j] = element.normal
        area[j] = element.area
        for col in 1:3, d in 1:3
            curl[j, 3 * (col - 1) + d] = element.curls[col][d]
        end
    end
    return BeatCpuRegularSoA{T}(trial_indices, n, px, py, pz, nx, ny, nz, area, curl, basis, weights)
end

struct BeatCpuRegularScratch{T}
    jac::Vector{T}
    kn::Vector{T}
    are::Matrix{T}
    aim::Matrix{T}
    rsr::Vector{T}
    rsi::Vector{T}
    cre::Vector{T}
    cim::Vector{T}
    lre::Matrix{T}
    lim::Matrix{T}
    rre::Matrix{T}
    rim::Matrix{T}
end
BeatCpuRegularScratch{T}(b::Int) where {T} = BeatCpuRegularScratch{T}(zeros(T, b), zeros(T, b), zeros(T, b, 3), zeros(T, b, 3),
    zeros(T, b), zeros(T, b), zeros(T, b), zeros(T, b), zeros(T, b, 9), zeros(T, b, 9), zeros(T, b, 3), zeros(T, b, 3))

function _beat_cpu_bm_regular_test_simd!(
    lhs, rhs, q_neumann, test_data::BeatCpuElementData{T}, test_quad::BeatCpuRegularQuadratureData{T},
    trial_elements, ssoa::BeatCpuRegularSoA{T},
    k::T, coupling::Complex{T}, scratch::BeatCpuRegularScratch{T}, skip_adjacent::Bool,
) where {T<:AbstractFloat}
    block = _BEAT_CPU_REGULAR_BLOCK_SIZE
    n = ssoa.n
    nq = length(ssoa.weights)
    tnx, tny, tnz = test_data.normal
    cr, ci = reim(coupling)
    k2 = k * k
    inv4pi = inv(T(4) * T(pi))
    area4 = T(4) * test_data.area
    (; jac, kn, are, aim, rsr, rsi, cre, cim, lre, lim, rre, rim) = scratch
    spx, spy, spz, snx, sny, snz = ssoa.px, ssoa.py, ssoa.pz, ssoa.nx, ssoa.ny, ssoa.nz
    @inbounds for j0 in 1:block:n
        m = min(block, n - j0 + 1)
        off = j0 - 1
        for j in 1:m
            adjacent = skip_adjacent && elements_are_adjacent(test_data.face, trial_elements[ssoa.indices[off + j]].face)
            jac[j] = adjacent ? zero(T) : area4 * ssoa.area[off + j]
            kn[j] = k2 * (tnx * snx[off + j] + tny * sny[off + j] + tnz * snz[off + j])
        end
        fill!(lre, zero(T)); fill!(lim, zero(T)); fill!(rre, zero(T)); fill!(rim, zero(T))
        fill!(cre, zero(T)); fill!(cim, zero(T))
        for tq in eachindex(test_quad.weights)
            x1, x2, x3 = test_quad.points[tq]
            tw = test_quad.weights[tq]
            fill!(are, zero(T)); fill!(aim, zero(T)); fill!(rsr, zero(T)); fill!(rsi, zero(T))
            for sq in 1:nq
                w0 = tw * ssoa.weights[sq]
                b1, b2, b3 = ssoa.basis[sq]
                @simd ivdep for j in 1:m
                    jj = off + j
                    rx = spx[jj, sq] - x1
                    ry = spy[jj, sq] - x2
                    rz = spz[jj, sq] - x3
                    r2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
                    inv_r = ifelse(r2 > zero(T), inv(sqrt(r2)), zero(T))
                    sine, cosine = _beat_cpu_fast_sincos(k * (r2 * inv_r))
                    w = w0 * jac[j]
                    sc = inv_r * inv4pi * w
                    wgr = cosine * sc
                    wgi = sine * sc
                    # weighted grad_scale = weighted green * (-1/r + i k)
                    hr = -(wgr * inv_r) - wgi * k
                    hi = wgr * k - wgi * inv_r
                    td = muladd(rx, snx[jj], muladd(ry, sny[jj], rz * snz[jj])) * inv_r
                    sd = -muladd(rx, tnx, muladd(ry, tny, rz * tnz)) * inv_r
                    knj = kn[j]
                    lsr = -(hr * td) - knj * (cr * wgr - ci * wgi)
                    lsi = -(hi * td) - knj * (cr * wgi + ci * wgr)
                    er = hr * sd
                    ei = hi * sd
                    rsr[j] += -wgr - (cr * er - ci * ei)
                    rsi[j] += -wgi - (cr * ei + ci * er)
                    cre[j] += wgr
                    cim[j] += wgi
                    are[j, 1] = muladd(b1, lsr, are[j, 1])
                    are[j, 2] = muladd(b2, lsr, are[j, 2])
                    are[j, 3] = muladd(b3, lsr, are[j, 3])
                    aim[j, 1] = muladd(b1, lsi, aim[j, 1])
                    aim[j, 2] = muladd(b2, lsi, aim[j, 2])
                    aim[j, 3] = muladd(b3, lsi, aim[j, 3])
                end
            end
            tb = test_quad.basis[tq]
            for col in 1:3, row in 1:3
                t = tb[row]
                idx = row + 3 * (col - 1)
                @simd ivdep for j in 1:m
                    lre[j, idx] = muladd(t, are[j, col], lre[j, idx])
                    lim[j, idx] = muladd(t, aim[j, col], lim[j, idx])
                end
            end
            for row in 1:3
                t = tb[row]
                @simd ivdep for j in 1:m
                    rre[j, row] = muladd(t, rsr[j], rre[j, row])
                    rim[j, row] = muladd(t, rsi[j], rim[j, row])
                end
            end
        end
        drive_count = size(q_neumann, 2)
        scurl = ssoa.curl
        for col in 1:3, row in 1:3
            c1, c2, c3 = test_data.curls[row]
            idx = row + 3 * (col - 1)
            o = 3 * (col - 1)
            @simd ivdep for j in 1:m
                jj = off + j
                cp = muladd(c1, scurl[jj, o + 1], muladd(c2, scurl[jj, o + 2], c3 * scurl[jj, o + 3]))
                lre[j, idx] = muladd(cp, cr * cre[j] - ci * cim[j], lre[j, idx])
                lim[j, idx] = muladd(cp, cr * cim[j] + ci * cre[j], lim[j, idx])
            end
        end
        for j in 1:m
            jac[j] == zero(T) && continue
            trial_data = trial_elements[ssoa.indices[off + j]]
            for row in 1:3
                grow = test_data.p1_dofs[row]
                coefficient = Complex{T}(rre[j, row], rim[j, row])
                for drive in 1:drive_count
                    rhs[grow, drive] += coefficient * q_neumann[trial_data.dp0_dof, drive]
                end
                for col in 1:3
                    idx = row + 3 * (col - 1)
                    lhs[grow, trial_data.p1_dofs[col]] += Complex{T}(lre[j, idx], lim[j, idx])
                end
            end
        end
    end
    return nothing
end

struct BeatCpuOperatorRegularScratch{T<:AbstractFloat}
    common::BeatCpuRegularScratch{T}
    agre::Matrix{T}
    agim::Matrix{T}
    ar::Vector{T}
    ai::Vector{T}
    adjre::Matrix{T}
    adjim::Matrix{T}
    hbre::Matrix{T}
    hbim::Matrix{T}
end

function BeatCpuOperatorRegularScratch{T}(block::Int) where {T<:AbstractFloat}
    return BeatCpuOperatorRegularScratch{T}(
        BeatCpuRegularScratch{T}(block), zeros(T, block, 3), zeros(T, block, 3),
        zeros(T, block), zeros(T, block), zeros(T, block, 3), zeros(T, block, 3),
        zeros(T, block, 9), zeros(T, block, 9),
    )
end

function _beat_cpu_accumulate_regular_test_simd!(
    single_layer, double_layer, adjoint_double_layer, hypersingular,
    test_data::BeatCpuElementData{T}, test_quad::BeatCpuRegularQuadratureData{T},
    trial_elements, ssoa::BeatCpuRegularSoA{T}, k::T,
    scratch::BeatCpuOperatorRegularScratch{T}, skip_adjacent::Bool,
) where {T<:AbstractFloat}
    block = _BEAT_CPU_REGULAR_BLOCK_SIZE
    tnx, tny, tnz = test_data.normal
    k2 = k * k
    inv4pi = inv(T(4) * T(pi))
    area4 = T(4) * test_data.area
    (; jac, kn, are, aim, rsr, rsi, cre, cim, lre, lim, rre, rim) = scratch.common
    (; agre, agim, ar, ai, adjre, adjim, hbre, hbim) = scratch
    spx, spy, spz, snx, sny, snz = ssoa.px, ssoa.py, ssoa.pz, ssoa.nx, ssoa.ny, ssoa.nz
    @inbounds for j0 in 1:block:ssoa.n
        m = min(block, ssoa.n - j0 + 1)
        off = j0 - 1
        for j in 1:m
            trial_data = trial_elements[ssoa.indices[off + j]]
            adjacent = skip_adjacent && elements_are_adjacent(test_data.face, trial_data.face)
            jac[j] = adjacent ? zero(T) : area4 * ssoa.area[off + j]
            kn[j] = k2 * (tnx * snx[off + j] + tny * sny[off + j] + tnz * snz[off + j])
        end
        fill!(lre, zero(T)); fill!(lim, zero(T)); fill!(rre, zero(T)); fill!(rim, zero(T))
        fill!(adjre, zero(T)); fill!(adjim, zero(T)); fill!(hbre, zero(T)); fill!(hbim, zero(T))
        fill!(cre, zero(T)); fill!(cim, zero(T))
        for tq in eachindex(test_quad.weights)
            x1, x2, x3 = test_quad.points[tq]
            tw = test_quad.weights[tq]
            fill!(are, zero(T)); fill!(aim, zero(T)); fill!(rsr, zero(T)); fill!(rsi, zero(T))
            fill!(agre, zero(T)); fill!(agim, zero(T)); fill!(ar, zero(T)); fill!(ai, zero(T))
            for sq in eachindex(ssoa.weights)
                w0 = tw * ssoa.weights[sq]
                b1, b2, b3 = ssoa.basis[sq]
                @simd ivdep for j in 1:m
                    jj = off + j
                    rx = spx[jj, sq] - x1
                    ry = spy[jj, sq] - x2
                    rz = spz[jj, sq] - x3
                    r2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
                    inv_r = ifelse(r2 > zero(T), inv(sqrt(r2)), zero(T))
                    sine, cosine = _beat_cpu_fast_sincos(k * (r2 * inv_r))
                    sc = inv_r * inv4pi * (w0 * jac[j])
                    wgr = cosine * sc
                    wgi = sine * sc
                    hr = -(wgr * inv_r) - wgi * k
                    hi = wgr * k - wgi * inv_r
                    td = muladd(rx, snx[jj], muladd(ry, sny[jj], rz * snz[jj])) * inv_r
                    sd = -muladd(rx, tnx, muladd(ry, tny, rz * tnz)) * inv_r
                    dr = hr * td
                    di = hi * td
                    rsr[j] += wgr
                    rsi[j] += wgi
                    ar[j] += hr * sd
                    ai[j] += hi * sd
                    cre[j] += wgr
                    cim[j] += wgi
                    are[j, 1] = muladd(b1, dr, are[j, 1])
                    are[j, 2] = muladd(b2, dr, are[j, 2])
                    are[j, 3] = muladd(b3, dr, are[j, 3])
                    aim[j, 1] = muladd(b1, di, aim[j, 1])
                    aim[j, 2] = muladd(b2, di, aim[j, 2])
                    aim[j, 3] = muladd(b3, di, aim[j, 3])
                    agre[j, 1] = muladd(b1, wgr, agre[j, 1])
                    agre[j, 2] = muladd(b2, wgr, agre[j, 2])
                    agre[j, 3] = muladd(b3, wgr, agre[j, 3])
                    agim[j, 1] = muladd(b1, wgi, agim[j, 1])
                    agim[j, 2] = muladd(b2, wgi, agim[j, 2])
                    agim[j, 3] = muladd(b3, wgi, agim[j, 3])
                end
            end
            tb = test_quad.basis[tq]
            for col in 1:3, row in 1:3
                t = tb[row]
                idx = row + 3 * (col - 1)
                @simd ivdep for j in 1:m
                    lre[j, idx] = muladd(t, are[j, col], lre[j, idx])
                    lim[j, idx] = muladd(t, aim[j, col], lim[j, idx])
                    hbre[j, idx] = muladd(t, agre[j, col], hbre[j, idx])
                    hbim[j, idx] = muladd(t, agim[j, col], hbim[j, idx])
                end
            end
            for row in 1:3
                t = tb[row]
                @simd ivdep for j in 1:m
                    rre[j, row] = muladd(t, rsr[j], rre[j, row])
                    rim[j, row] = muladd(t, rsi[j], rim[j, row])
                    adjre[j, row] = muladd(t, ar[j], adjre[j, row])
                    adjim[j, row] = muladd(t, ai[j], adjim[j, row])
                end
            end
        end
        scurl = ssoa.curl
        for col in 1:3, row in 1:3
            c1, c2, c3 = test_data.curls[row]
            idx = row + 3 * (col - 1)
            o = 3 * (col - 1)
            @simd ivdep for j in 1:m
                jj = off + j
                cp = muladd(c1, scurl[jj, o + 1], muladd(c2, scurl[jj, o + 2], c3 * scurl[jj, o + 3]))
                hbre[j, idx] = cp * cre[j] - kn[j] * hbre[j, idx]
                hbim[j, idx] = cp * cim[j] - kn[j] * hbim[j, idx]
            end
        end
        for j in 1:m
            jac[j] == zero(T) && continue
            trial_data = trial_elements[ssoa.indices[off + j]]
            for row in 1:3
                grow = test_data.p1_dofs[row]
                single_layer[grow, trial_data.dp0_dof] += Complex{T}(rre[j, row], rim[j, row])
                adjoint_double_layer[grow, trial_data.dp0_dof] += Complex{T}(adjre[j, row], adjim[j, row])
                for col in 1:3
                    idx = row + 3 * (col - 1)
                    gcol = trial_data.p1_dofs[col]
                    double_layer[grow, gcol] += Complex{T}(lre[j, idx], lim[j, idx])
                    hypersingular[grow, gcol] += Complex{T}(hbre[j, idx], hbim[j, idx])
                end
            end
        end
    end
    return nothing
end

# Chunks own scratch, independent of task migration or the thread pool's IDs.
function _beat_cpu_regular_simd_groups!(apply_test!, groups, scratch, threaded_enabled::Bool)
    for group in groups
        chunk_count = min(length(group), length(scratch))
        if threaded_enabled
            Threads.@threads for chunk in 1:chunk_count
                first_index = fld((chunk - 1) * length(group), chunk_count) + 1
                last_index = fld(chunk * length(group), chunk_count)
                for group_index in first_index:last_index
                    apply_test!(group[group_index], scratch[chunk])
                end
            end
        else
            for test_index in group
                apply_test!(test_index, scratch[1])
            end
        end
    end
    return nothing
end

function _beat_cpu_regular_simd_pass!(
    apply_test!, elements::Vector{BeatCpuElementData{T}}, regular_quadrature,
    indices, color_groups, threaded_enabled::Bool, image_transforms, cpu_cache, scratch,
) where {T<:AbstractFloat}
    soa = BeatCpuRegularSoA(elements, regular_quadrature, indices)
    _beat_cpu_regular_simd_groups!(color_groups, scratch, threaded_enabled) do test_index, work
        apply_test!(elements[test_index], regular_quadrature[test_index], elements, soa, work, true)
    end
    for (transform_index, transform) in enumerate(image_transforms)
        image_elements = cpu_cache === nothing ?
            _beat_cpu_reflect_element_data(elements, transform) : cpu_cache.image_elements[transform_index]
        image_quadrature = cpu_cache === nothing ?
            _beat_cpu_reflect_regular_quadrature_data(regular_quadrature, transform) : cpu_cache.image_quadrature[transform_index]
        image_soa = BeatCpuRegularSoA(image_elements, image_quadrature, indices)
        _beat_cpu_regular_simd_groups!(color_groups, scratch, threaded_enabled) do test_index, work
            apply_test!(elements[test_index], regular_quadrature[test_index], image_elements, image_soa, work, false)
        end
    end
    return nothing
end

function _beat_cpu_bm_regular_simd_pass!(
    lhs, rhs, q_neumann, mesh, elements::Vector{BeatCpuElementData{T}}, regular_quadrature,
    indices, color_groups, threaded_enabled::Bool, image_transforms, cpu_cache,
    k::T, coupling::Complex{T},
) where {T<:AbstractFloat}
    # Keep regular scatter order identical when threading is disabled.
    groups = threaded_enabled ? color_groups : _beat_cpu_element_color_groups(mesh, indices)
    scratch = [BeatCpuRegularScratch{T}(_BEAT_CPU_REGULAR_BLOCK_SIZE)
               for _ in 1:(threaded_enabled ? Threads.nthreads() : 1)]
    _beat_cpu_regular_simd_pass!(
        elements, regular_quadrature, indices, groups, threaded_enabled, image_transforms, cpu_cache, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q_neumann, test_data, test_quad, trial_elements, soa, k, coupling, work, skip_adjacent,
        )
    end
    return nothing
end

function _beat_cpu_accumulate_regular_simd_pass!(
    single_layer, double_layer, adjoint_double_layer, hypersingular,
    mesh, elements::Vector{BeatCpuElementData{T}}, regular_quadrature,
    indices, color_groups, threaded_enabled::Bool, image_transforms, cpu_cache, k::T,
) where {T<:AbstractFloat}
    # Keep regular scatter order identical when threading is disabled.
    groups = threaded_enabled ? color_groups : _beat_cpu_element_color_groups(mesh, indices)
    scratch = [BeatCpuOperatorRegularScratch{T}(_BEAT_CPU_REGULAR_BLOCK_SIZE)
               for _ in 1:(threaded_enabled ? Threads.nthreads() : 1)]
    _beat_cpu_regular_simd_pass!(
        elements, regular_quadrature, indices, groups, threaded_enabled, image_transforms, cpu_cache, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_accumulate_regular_test_simd!(
            single_layer, double_layer, adjoint_double_layer, hypersingular,
            test_data, test_quad, trial_elements, soa, k, work, skip_adjacent,
        )
    end
    return nothing
end
