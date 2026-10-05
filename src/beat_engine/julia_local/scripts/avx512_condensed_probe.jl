# Measure separately compiled condensed/shared CPU assembly differences at symmetry :off.
# Run with the julia_local project, alongside avx512_codegen_assay.jl, both with
# the native CPU target and with AVX-512 masked out on the same AVX-512 host.
# Reports entry counts, component ULP gaps, absolute/relative errors and operator
# scales for uncached, cached and fork-self comparisons. This is a diagnostic,
# not a pass/fail gate; acceptance is checked in coupled_condensed_tests.jl.

# Include order mirrors `runtests.jl`: the core module, then `BeatEngineCoupled`
# (which `coupled_solver_tests.jl` brings in one file earlier and which
# `BeatEngineCoupledCondensed.jl` expects to already exist), then the fork.
const SOURCE = joinpath(@__DIR__, "..", "src")

# `norm` is used below and is not re-exported by the engine modules. It sits on
# a branch only reached when the operators actually differ, so leaving it out
# cost nothing on every host that agreed and threw on the first host that did
# not -- which is the only host this script exists for.
using LinearAlgebra

include(joinpath(SOURCE, "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(SOURCE, "BeatEngineCoupled.jl"))
using .BeatEngineCoupled
include(joinpath(SOURCE, "BeatEngineCoupledCondensed.jl"))
using .BeatEngineCoupledCondensed

const TEST_MESHES = joinpath(@__DIR__, "..", "test_meshes")

"""Map a Float32 onto the integer line that counts representable values.

Adjacent floats differ by 1, so `|key(a) - key(b)|` is the ULP gap, and -0.0f0
and 0.0f0 share a key rather than reporting a spurious gap of 2^31.
"""
function float_key(value::Float32)
    bits = Int64(reinterpret(Int32, value))
    return bits < 0 ? Int64(-2147483648) - bits : bits
end

ulp_gap(a::Float32, b::Float32) = abs(float_key(a) - float_key(b))

"""Compare two operator matrices entry by entry, in ULPs of their parts.

Also reports the two relative measures the contract question turns on. `norm_rel`
is what Julia's `≈` actually tests on arrays -- `norm(a - b) <= rtol * max(norm(a),
norm(b))` -- so it says directly whether the sibling branches' `rtol = 1.0f-5`
would hold here and with how much room. `entry_rel` is the worst *entrywise*
relative gap, which is much larger, because the entries that disagree are tiny
ones where a few ULPs are a big fraction of a small number and a negligible
fraction of the operator.
"""
function compare(forked::AbstractMatrix, shared::AbstractMatrix)
    differing = 0
    worst = 0
    entry_rel = 0.0
    max_abs = 0.0
    first_index = nothing
    for index in eachindex(forked, shared)
        left = forked[index]
        right = shared[index]
        left == right && continue
        differing += 1
        gap = max(ulp_gap(real(left), real(right)), ulp_gap(imag(left), imag(right)))
        if gap > worst
            worst = gap
        end
        max_abs = max(max_abs, Float64(abs(left - right)))
        scale = max(abs(left), abs(right))
        if scale > 0
            entry_rel = max(entry_rel, abs(left - right) / scale)
        end
        if first_index === nothing
            first_index = index
        end
    end
    norm_rel = differing == 0 ? 0.0 :
               norm(forked - shared) / max(norm(forked), norm(shared))
    # `operator_scale` is what an absolute entrywise floor would be set against:
    # a tolerance tied to the operator's own largest entry, rather than to each
    # differing entry's magnitude, is the one measure that neither a norm test nor
    # a per-entry relative test gives.
    operator_scale = Float64(maximum(abs, shared))
    return (; differing, worst, entry_rel, norm_rel, max_abs, operator_scale,
              first_index, total=length(forked))
end

function report(label, forked, shared)
    for operator in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
        result = compare(getproperty(forked, operator), getproperty(shared, operator))
        location = result.first_index === nothing ? "-" :
                   string(Tuple(CartesianIndices(getproperty(forked, operator))[result.first_index]))
        println("probe $label $operator differing=$(result.differing)/$(result.total) " *
                "max_ulp=$(result.worst) norm_rel=$(result.norm_rel) " *
                "entry_rel=$(result.entry_rel) max_abs=$(result.max_abs) " *
                "operator_scale=$(result.operator_scale) first=$location")
    end
end

const RULE = triangle_rule(Float32, 2)
const WAVENUMBER = Float32(2pi * 1000.0 / 343.0)

mesh = load_gmsh22_with_tags(joinpath(TEST_MESHES, "sample.msh"), Float32(0.001))
p1 = build_p1_space(mesh)
dp0 = build_dp0_space(mesh)
element_indices = 1:min(24, length(mesh.faces))
singular_cache = build_singular_correction_cache(mesh, 2, element_indices)

shared = assemble_regular_galerkin_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, element_indices=element_indices,
    backend=:cpu, singular_cache=singular_cache, symmetry_mode=:off,
)
forked = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, element_indices=element_indices,
    singular_cache=singular_cache, symmetry_mode=:off,
)

cache = build_beat_cpu_assembly_cache(
    mesh, p1, dp0, RULE;
    singular_order=2, element_indices=element_indices, symmetry_mode=:off,
)
shared_cached = assemble_regular_galerkin_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2, backend=:cpu,
    singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=:off,
)
forked_cached = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
    mesh, p1, dp0, WAVENUMBER, RULE;
    skip_singular=false, singular_order=2,
    singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=:off,
)

# The fork-self comparison remains exact in the test suite. It distinguishes
# repeated calls to one compiled body from the separately compiled shared path.
report("uncached", forked, shared)
report("cached", forked_cached, shared_cached)
report("fork-self", forked_cached, forked)

function tally_differences(pairs)
    total = 0
    for (left, right) in pairs
        for operator in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            total += compare(getproperty(left, operator), getproperty(right, operator)).differing
        end
    end
    return total
end

println("probe_verdict differing_entries=",
        tally_differences(((forked, shared), (forked_cached, shared_cached))))
