# Report whether this Julia process JITs AVX-512, and what CPU target produced it.
#
# The masked arm of the AVX-512 experiment is worthless without this. A green
# suite under `-C native,-avx512f` proves nothing unless the mask demonstrably
# changed code generation: "no failure" and "no AVX-512" have to be established
# separately, or the experiment is measuring a flag it never applied.
#
# The assay compiles a Float32 fused-multiply-add reduction -- the shape the
# condensed quadrature loop reduces to -- and counts AVX-512 signatures in the
# emitted assembly.
#
# Counting `zmm` alone is not enough, and the first run of this experiment showed
# why: on `znver4` the unmasked arm emitted 43 zmm registers, but on Intel's
# `graniterapids` and `sapphirerapids` it emitted none at all -- while the
# numerics still changed when AVX-512 was masked off. LLVM sets
# `prefer-vector-width=256` for those Intel parts to avoid frequency throttling,
# so AVX-512 reaches the code as EVEX-encoded *256-bit* operations rather than as
# 512-bit registers. Two further signatures catch that: the upper vector
# registers `ymm16`-`ymm31`, which have no VEX encoding and so exist only under
# AVX-512, and the `k1`-`k7` mask registers.
#
# Prints machine-readable `assay_*` lines and exits 0 always: the caller decides
# what an unexpected count means, because on a host without AVX-512 every count
# is zero in both arms and that is a correct null result rather than a broken mask.

using InteractiveUtils

function assay_kernel(x::Vector{Float32}, y::Vector{Float32})
    total = zero(Float32)
    @inbounds @simd for i in eachindex(x)
        total = muladd(x[i], y[i], total)
    end
    return total
end

buffer = IOBuffer()
code_native(buffer, assay_kernel, (Vector{Float32}, Vector{Float32}); debuginfo=:none)
assembly = String(take!(buffer))

count_matches(pattern) = count(_ -> true, eachmatch(pattern, assembly))

# `Base.JLOptions().cpu_target` is the string this process was actually started
# with, not the string the workflow believes it passed.
requested = let pointer = Base.JLOptions().cpu_target
    pointer == C_NULL ? "(default)" : unsafe_string(pointer)
end

zmm = count_matches(r"zmm[0-9]+")
high_vector = count_matches(r"[xyz]mm(1[6-9]|2[0-9]|3[01])\b")
# Match Intel's k1 and AT&T's %k1 spelling; code_native defaults to Intel syntax.
mask_register = count_matches(r"\bk[1-7]\b")

println("assay_requested_target ", requested)
println("assay_cpu_name         ", Sys.CPU_NAME)
println("assay_zmm              ", zmm)
println("assay_high_vector      ", high_vector)
println("assay_mask_register    ", mask_register)
println("assay_avx512_total     ", zmm + high_vector + mask_register)
println("assay_ymm              ", count_matches(r"ymm[0-9]+"))
println("assay_xmm              ", count_matches(r"xmm[0-9]+"))
