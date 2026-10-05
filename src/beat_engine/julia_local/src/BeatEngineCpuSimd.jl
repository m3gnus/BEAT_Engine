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
    p1_dofs::Matrix{Int}
    dp0_dofs::Vector{Int}
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
    p1_dofs = zeros(Int, n, 3)
    dp0_dofs = zeros(Int, n)
    for (j, element_index) in enumerate(trial_indices)
        element = elements[element_index]
        for q in 1:nq
            px[j, q], py[j, q], pz[j, q] = regular_quadrature[element_index].points[q]
        end
        nx[j], ny[j], nz[j] = element.normal
        area[j] = element.area
        p1_dofs[j, 1], p1_dofs[j, 2], p1_dofs[j, 3] = element.p1_dofs
        dp0_dofs[j] = element.dp0_dof
        for col in 1:3, d in 1:3
            curl[j, 3 * (col - 1) + d] = element.curls[col][d]
        end
    end
    return BeatCpuRegularSoA{T}(trial_indices, n, px, py, pz, nx, ny, nz, area, curl, p1_dofs, dp0_dofs, basis, weights)
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
        # One test row at a time: its right-hand entries are a sum over the
        # block, written once, and its matrix entries all land in one column
        # of the transposed storage (see _beat_cpu_transpose_square!).
        trial_p1 = ssoa.p1_dofs
        trial_dp0 = ssoa.dp0_dofs
        for row in 1:3
            grow = test_data.p1_dofs[row]
            for drive in 1:drive_count
                total = zero(Complex{T})
                for j in 1:m
                    jac[j] == zero(T) && continue
                    total += Complex{T}(rre[j, row], rim[j, row]) * q_neumann[trial_dp0[off + j], drive]
                end
                rhs[grow, drive] += total
            end
            for j in 1:m
                jac[j] == zero(T) && continue
                for col in 1:3
                    idx = row + 3 * (col - 1)
                    lhs[trial_p1[off + j, col], grow] += Complex{T}(lre[j, idx], lim[j, idx])
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
        trial_p1 = ssoa.p1_dofs
        trial_dp0 = ssoa.dp0_dofs
        for row in 1:3
            grow = test_data.p1_dofs[row]
            for j in 1:m
                jac[j] == zero(T) && continue
                dp0 = trial_dp0[off + j]
                single_layer[grow, dp0] += Complex{T}(rre[j, row], rim[j, row])
                adjoint_double_layer[grow, dp0] += Complex{T}(adjre[j, row], adjim[j, row])
                for col in 1:3
                    idx = row + 3 * (col - 1)
                    gcol = trial_p1[off + j, col]
                    # Transposed storage; see _beat_cpu_transpose_square!.
                    double_layer[gcol, grow] += Complex{T}(lre[j, idx], lim[j, idx])
                    hypersingular[gcol, grow] += Complex{T}(hbre[j, idx], hbim[j, idx])
                end
            end
        end
    end
    return nothing
end

# The regular kernels write a test element's rows. In Julia's column-major
# storage a row is strided, so for each trial element the nine block entries
# land in nine different cache lines of a matrix far larger than the cache.
# Writing the transpose instead puts them in three columns, consecutive in the
# trial element's dof order, and measured about half the regular pass's time
# at quadrature order 2. The passes therefore transpose the square operators
# in place, accumulate into the transposed storage, and transpose back; an
# involution, so whatever the matrix held before is preserved.
function _beat_cpu_transpose_square!(matrix::AbstractMatrix)
    n = size(matrix, 1)
    size(matrix, 2) == n || error("In-place transpose needs a square matrix.")
    block = 64
    @inbounds for jb in 1:block:n, ib in jb:block:n
        for j in jb:min(jb + block - 1, n)
            for i in max(ib, j + 1):min(ib + block - 1, n)
                matrix[i, j], matrix[j, i] = matrix[j, i], matrix[i, j]
            end
        end
    end
    return matrix
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
    _beat_cpu_transpose_square!(lhs)
    _beat_cpu_regular_simd_pass!(
        elements, regular_quadrature, indices, groups, threaded_enabled, image_transforms, cpu_cache, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q_neumann, test_data, test_quad, trial_elements, soa, k, coupling, work, skip_adjacent,
        )
    end
    _beat_cpu_transpose_square!(lhs)
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
    _beat_cpu_transpose_square!(double_layer)
    _beat_cpu_transpose_square!(hypersingular)
    _beat_cpu_regular_simd_pass!(
        elements, regular_quadrature, indices, groups, threaded_enabled, image_transforms, cpu_cache, scratch,
    ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
        _beat_cpu_accumulate_regular_test_simd!(
            single_layer, double_layer, adjoint_double_layer, hypersingular,
            test_data, test_quad, trial_elements, soa, k, work, skip_adjacent,
        )
    end
    _beat_cpu_transpose_square!(double_layer)
    _beat_cpu_transpose_square!(hypersingular)
    return nothing
end

# --------------------------------------------------------------------------
# Field evaluation.
#
# The scalar field loop recomputes, for every observation point and every
# source quadrature point, the source's pressure density
# (basis . p[face] * weight) and Neumann density (q[element] * weight). Neither
# depends on the observation point, so here they are formed once per call and
# stored with the source coordinates and normals in contiguous arrays. What is
# left per (point, source) is a plain @simd reduction: one distance, one
# polynomial sincos and a handful of FMAs, with the same zero-radius mask as the
# assembly kernels.

function beat_cpu_field_kernel()
    value = lowercase(strip(get(ENV, "BLAB_BEAT_CPU_FIELD_KERNEL", "simd")))
    value in ("simd", "scalar") || error(
        "BLAB_BEAT_CPU_FIELD_KERNEL must be simd or scalar; got $(repr(value)).",
    )
    return Symbol(value)
end

struct BeatCpuFieldSources{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    nx::Vector{T}
    ny::Vector{T}
    nz::Vector{T}
    pr::Vector{T}
    pim::Vector{T}
    qr::Vector{T}
    qim::Vector{T}
end

function BeatCpuFieldSources(cache::FieldEvaluationCache{T}, pressure, q_neumann) where {T<:AbstractFloat}
    n = length(cache.source_points)
    x = Vector{T}(undef, n); y = Vector{T}(undef, n); z = Vector{T}(undef, n)
    nx = Vector{T}(undef, n); ny = Vector{T}(undef, n); nz = Vector{T}(undef, n)
    pr = Vector{T}(undef, n); pim = Vector{T}(undef, n); qr = Vector{T}(undef, n); qim = Vector{T}(undef, n)
    @inbounds for s in 1:n
        x[s], y[s], z[s] = cache.source_points[s]
        nx[s], ny[s], nz[s] = cache.source_normals[s]
        face = cache.source_faces[s]
        basis = cache.basis_values[s]
        weight = cache.source_weights[s]
        p_source = Complex{T}(
            basis[1] * pressure[face[1]] + basis[2] * pressure[face[2]] + basis[3] * pressure[face[3]],
        ) * weight
        q_source = Complex{T}(q_neumann[cache.source_elements[s]]) * weight
        pr[s], pim[s] = reim(p_source)
        qr[s], qim[s] = reim(q_source)
    end
    return BeatCpuFieldSources{T}(x, y, z, nx, ny, nz, pr, pim, qr, qim)
end

@inline function _beat_cpu_field_point_simd(sources::BeatCpuFieldSources{T}, x1::T, x2::T, x3::T, k::T) where {T<:AbstractFloat}
    inv4pi = inv(T(4) * T(pi))
    (; x, y, z, nx, ny, nz, pr, pim, qr, qim) = sources
    acc_re = zero(T)
    acc_im = zero(T)
    @inbounds @simd for s in eachindex(x)
        rx = x[s] - x1
        ry = y[s] - x2
        rz = z[s] - x3
        r2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
        inv_r = ifelse(r2 > zero(T), inv(sqrt(r2)), zero(T))
        sine, cosine = _beat_cpu_fast_sincos(k * (r2 * inv_r))
        scale = inv_r * inv4pi
        gr = cosine * scale
        gi = sine * scale
        projection = muladd(rx, nx[s], muladd(ry, ny[s], rz * nz[s])) * inv_r
        # double layer = green * (i k - 1/r) * projection
        dr = (-(gr * inv_r) - gi * k) * projection
        di = (gr * k - gi * inv_r) * projection
        acc_re += (dr * pr[s] - di * pim[s]) - (gr * qr[s] - gi * qim[s])
        acc_im += (dr * pim[s] + di * pr[s]) - (gr * qim[s] + gi * qr[s])
    end
    return Complex{T}(acc_re, acc_im)
end

function _beat_cpu_field_simd(eval_points, pressure, q_neumann, k::T, cache::FieldEvaluationCache{T}) where {T<:AbstractFloat}
    point_count = length(eval_points)
    potentials = Vector{Complex{T}}(undef, point_count)
    point_count == 0 && return potentials
    sources = BeatCpuFieldSources(cache, pressure, q_neumann)
    Threads.@threads for point_index in 1:point_count
        point = eval_points[point_index]
        potentials[point_index] = _beat_cpu_field_point_simd(sources, T(point[1]), T(point[2]), T(point[3]), k)
    end
    return potentials
end

# --------------------------------------------------------------------------
# Singular (touching-pair) corrections, fused Burton-Miller.
#
# A touching pair is integrated with a Duffy rule of 32 to 1,536 point pairs
# (singular_order 2 to 4). The scalar kernel walks them one at a time, calling
# p1_values, local_to_global and sincos for each. Here the rule is stored as
# contiguous arrays of reference coordinates and weights, the basis values and
# global points are formed inline (both are affine in the reference
# coordinates), and the loop over the rule's points is a single @simd
# reduction into 26 real accumulators: the 3x3 complex left-hand block, the 3
# complex right-hand entries and the complex curl sum. Same mathematics as
# _beat_cpu_bm_pair_blocks, same zero-radius mask as the regular kernel.

function beat_cpu_singular_kernel()
    value = lowercase(strip(get(ENV, "BLAB_BEAT_CPU_SINGULAR_KERNEL", "simd")))
    value in ("simd", "scalar") || error(
        "BLAB_BEAT_CPU_SINGULAR_KERNEL must be simd or scalar; got $(repr(value)).",
    )
    return Symbol(value)
end

# The rule's reference coordinates, split into components. Wrapped in its own
# type so that the singular drivers need no change: they pass
# `rule.test_points, rule.trial_points, rule.weights` to
# _beat_cpu_bm_pair_blocks, and a rule built here dispatches that call to the
# vectorised method below.
struct BeatCpuDuffyCoords{T<:AbstractFloat}
    xi::Vector{T}
    eta::Vector{T}
end

struct BeatCpuDuffyRuleSoA{T<:AbstractFloat}
    test_points::BeatCpuDuffyCoords{T}
    trial_points::BeatCpuDuffyCoords{T}
    weights::Vector{T}
end

function BeatCpuDuffyRuleSoA(rule::DuffyRule{T}) where {T<:AbstractFloat}
    return BeatCpuDuffyRuleSoA{T}(
        BeatCpuDuffyCoords{T}(T[p[1] for p in rule.test_points], T[p[2] for p in rule.test_points]),
        BeatCpuDuffyCoords{T}(T[p[1] for p in rule.trial_points], T[p[2] for p in rule.trial_points]),
        copy(rule.weights),
    )
end

_beat_cpu_duffy_rule_soas(rules) = [BeatCpuDuffyRuleSoA(rule) for rule in rules]

function _beat_cpu_bm_pair_blocks(
    test_vertices::NTuple{3,SVector{3,T}},
    trial_vertices::NTuple{3,SVector{3,T}},
    test_normal::SVector{3,T},
    trial_normal::SVector{3,T},
    test_curls::NTuple{3,SVector{3,T}},
    trial_curls::NTuple{3,SVector{3,T}},
    normal_product::T,
    jac_scale::T,
    k::T,
    coupling::Complex{T},
    test_points::BeatCpuDuffyCoords{T},
    trial_points::BeatCpuDuffyCoords{T},
    w::Vector{T},
) where {T<:AbstractFloat}
    cr, ci = reim(coupling)
    k2n = coupling * (k * k * normal_product)
    knr, kni = reim(k2n)
    inv4pi = inv(T(4) * T(pi))
    a0 = test_vertices[1]
    ae1 = test_vertices[2] - a0
    ae2 = test_vertices[3] - a0
    b0 = trial_vertices[1]
    be1 = trial_vertices[2] - b0
    be2 = trial_vertices[3] - b0
    d0 = b0 - a0
    tnx, tny, tnz = test_normal
    snx, sny, snz = trial_normal
    u1, u2 = test_points.xi, test_points.eta
    v1, v2 = trial_points.xi, trial_points.eta

    l11r = l21r = l31r = l12r = l22r = l32r = l13r = l23r = l33r = zero(T)
    l11i = l21i = l31i = l12i = l22i = l32i = l13i = l23i = l33i = zero(T)
    r1r = r2r = r3r = r1i = r2i = r3i = cre = cim = zero(T)
    @inbounds @simd for q in eachindex(w)
        a1 = u1[q]
        a2 = u2[q]
        b1 = v1[q]
        b2 = v2[q]
        t1 = one(T) - a1 - a2
        s1 = one(T) - b1 - b2
        # r = y - x, with x and y affine in the reference coordinates
        rx = d0[1] + b1 * be1[1] + b2 * be2[1] - a1 * ae1[1] - a2 * ae2[1]
        ry = d0[2] + b1 * be1[2] + b2 * be2[2] - a1 * ae1[2] - a2 * ae2[2]
        rz = d0[3] + b1 * be1[3] + b2 * be2[3] - a1 * ae1[3] - a2 * ae2[3]
        r2 = muladd(rx, rx, muladd(ry, ry, rz * rz))
        inv_r = ifelse(r2 > zero(T), inv(sqrt(r2)), zero(T))
        sine, cosine = _beat_cpu_fast_sincos(k * (r2 * inv_r))
        scale = inv_r * inv4pi * (w[q] * jac_scale)
        wgr = cosine * scale
        wgi = sine * scale
        hr = -(wgr * inv_r) - wgi * k
        hi = wgr * k - wgi * inv_r
        td = muladd(rx, snx, muladd(ry, sny, rz * snz)) * inv_r
        sd = -muladd(rx, tnx, muladd(ry, tny, rz * tnz)) * inv_r
        lsr = -(hr * td) - (knr * wgr - kni * wgi)
        lsi = -(hi * td) - (knr * wgi + kni * wgr)
        er = hr * sd
        ei = hi * sd
        rsr = -wgr - (cr * er - ci * ei)
        rsi = -wgi - (cr * ei + ci * er)
        cre += wgr
        cim += wgi
        r1r = muladd(t1, rsr, r1r)
        r1i = muladd(t1, rsi, r1i)
        r2r = muladd(a1, rsr, r2r)
        r2i = muladd(a1, rsi, r2i)
        r3r = muladd(a2, rsr, r3r)
        r3i = muladd(a2, rsi, r3i)
        p11 = t1 * s1
        l11r = muladd(p11, lsr, l11r)
        l11i = muladd(p11, lsi, l11i)
        p21 = a1 * s1
        l21r = muladd(p21, lsr, l21r)
        l21i = muladd(p21, lsi, l21i)
        p31 = a2 * s1
        l31r = muladd(p31, lsr, l31r)
        l31i = muladd(p31, lsi, l31i)
        p12 = t1 * b1
        l12r = muladd(p12, lsr, l12r)
        l12i = muladd(p12, lsi, l12i)
        p22 = a1 * b1
        l22r = muladd(p22, lsr, l22r)
        l22i = muladd(p22, lsi, l22i)
        p32 = a2 * b1
        l32r = muladd(p32, lsr, l32r)
        l32i = muladd(p32, lsi, l32i)
        p13 = t1 * b2
        l13r = muladd(p13, lsr, l13r)
        l13i = muladd(p13, lsi, l13i)
        p23 = a1 * b2
        l23r = muladd(p23, lsr, l23r)
        l23i = muladd(p23, lsi, l23i)
        p33 = a2 * b2
        l33r = muladd(p33, lsr, l33r)
        l33i = muladd(p33, lsi, l33i)
    end
    curl_total = coupling * Complex{T}(cre, cim)
    lhs_block = MMatrix{3,3,Complex{T},9}(Complex{T}(l11r, l11i), Complex{T}(l21r, l21i), Complex{T}(l31r, l31i), Complex{T}(l12r, l12i), Complex{T}(l22r, l22i), Complex{T}(l32r, l32i), Complex{T}(l13r, l13i), Complex{T}(l23r, l23i), Complex{T}(l33r, l33i))
    @inbounds for local_row in 1:3, local_col in 1:3
        lhs_block[local_row, local_col] += dot(test_curls[local_row], trial_curls[local_col]) * curl_total
    end
    rhs_block = MVector{3,Complex{T}}(Complex{T}(r1r, r1i), Complex{T}(r2r, r2i), Complex{T}(r3r, r3i))
    return lhs_block, rhs_block
end
