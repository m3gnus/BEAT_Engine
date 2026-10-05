# Scalar against vectorised CPU regular-stage benchmark, with a host LLVM vector-width probe.
# Example: julia -t auto --project=src/beat_engine/julia_local <this script> --repetitions 5
using LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore

const _BEAT_CPU_BENCH_INTERACTIVE_UTILS = try
    @eval import InteractiveUtils
    InteractiveUtils
catch
    nothing
end

function _beat_cpu_benchmark_options(args)
    options = Dict(
        "mesh" => joinpath(@__DIR__, "..", "test_meshes", "sample.msh"),
        "scale" => "0.001",
        "symmetry" => "off",
        "precision" => "float32",
        "order" => "2",
        "freq" => "1500",
        "repetitions" => "5",
    )
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("--help", "-h")
            println("Benchmark CPU Burton-Miller regular assembly (base and symmetry images).")
            println("Options: --mesh FILE --scale NUMBER --symmetry off|x|xy|ground")
            println("         --precision float32|float64 --order INTEGER --freq HZ --repetitions INTEGER")
            println("Defaults: bundled sample.msh, scale 0.001, off, float32, order 2, 1500 Hz, 5 repetitions.")
            println("Use julia -t auto to measure coloured assembly on multiple threads.")
            return nothing
        end
        startswith(argument, "--") || error("Expected a --option, got $(repr(argument)).")
        parts = split(argument[3:end], '='; limit=2)
        key = parts[1]
        haskey(options, key) || error("Unknown option --$key.")
        if length(parts) == 2
            options[key] = parts[2]
        else
            index < length(args) || error("Missing value for --$key.")
            index += 1
            options[key] = args[index]
        end
        index += 1
    end
    return options
end

function _beat_cpu_benchmark_relative_difference(actual, scalar)
    scale = maximum(abs, scalar)
    difference = maximum(abs, actual .- scalar)
    return iszero(scale) ? (iszero(difference) ? 0.0 : Inf) : difference / scale
end

function _beat_cpu_benchmark_llvm(elements, quadrature, cache, q, k::T) where {T<:AbstractFloat}
    _BEAT_CPU_BENCH_INTERACTIVE_UTILS === nothing && return "unknown (InteractiveUtils unavailable)"
    isempty(cache.indices) && return "unknown (no test elements)"
    test_index = first(cache.indices)
    n = maximum(maximum(element.p1_dofs) for element in elements)
    lhs = zeros(Complex{T}, n, n)
    rhs = zeros(Complex{T}, n, size(q, 2))
    soa = BeatEngineCore.BeatCpuRegularSoA(elements, quadrature, cache.indices)
    scratch = BeatEngineCore.BeatCpuRegularScratch{T}(BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE)
    arguments = (lhs, rhs, q, elements[test_index], quadrature[test_index],
                 elements, soa, k, Complex{T}(0, 1) / k, scratch, true)
    try
        buffer = IOBuffer()
        _BEAT_CPU_BENCH_INTERACTIVE_UTILS.code_llvm(
            buffer, BeatEngineCore._beat_cpu_bm_regular_test_simd!,
            Tuple{map(typeof, arguments)...}; optimize=true, debuginfo=:none,
        )
        llvm = String(take!(buffer))
        scalar_name = T === Float32 ? "float" : "double"
        pattern = Regex("<(\\d+) x $scalar_name>")
        widths = [parse(Int, matched.captures[1]) for matched in eachmatch(pattern, llvm)]
        width = maximum(widths; init=1)
        return width > 1 ? "<$width x $scalar_name> ($(width * sizeof(T) * 8) bits)" : "not vectorised"
    catch exception
        return "unknown (LLVM inspection failed: $(typeof(exception)))"
    end
end

function _beat_cpu_benchmark_regular(options, ::Type{T}) where {T<:AbstractFloat}
    scale = parse(T, options["scale"])
    frequency = parse(T, options["freq"])
    order = parse(Int, options["order"])
    repetitions = parse(Int, options["repetitions"])
    symmetry = Symbol(lowercase(strip(options["symmetry"])))
    isfinite(scale) && scale > 0 || error("--scale must be finite and positive.")
    isfinite(frequency) && frequency > 0 || error("--freq must be finite and positive.")
    order > 0 || error("--order must be positive.")
    repetitions > 0 || error("--repetitions must be positive.")
    symmetry in (:off, :x, :xy, :ground) || error("--symmetry must be off, x, xy or ground.")
    mesh = load_gmsh22_with_tags(options["mesh"], scale)
    mesh = snap_symmetry_planes(mesh, symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry)
    isempty(mesh.faces) && error("The mesh must contain triangles.")
    p1 = build_p1_space(mesh)
    dp0 = build_dp0_space(mesh)
    rule = triangle_rule(T, order)
    cpu_cache = build_beat_cpu_assembly_cache(mesh, p1, dp0, rule; symmetry_mode=symmetry)
    singular_cache = build_singular_correction_cache(mesh, 2)
    identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    k = T(2pi) * frequency / T(343)
    q = zeros(Complex{T}, dp0.global_dof_count, 3)
    driven = findall(==(2), mesh.physical_tags)
    isempty(driven) && (driven = collect(eachindex(mesh.faces)))
    for element_index in driven, drive in 1:3
        q[dp0.local_to_global[element_index], drive] = Complex{T}(T(drive), T(0.5 * drive))
    end
    assemble = function(kernel; timing=nothing)
        assemble_burton_miller_neumann_system_cpu(
            mesh, p1, dp0, q, k, rule;
            identity_p1_p1, identity_p1_dp0, cpu_cache, singular_cache,
            skip_singular=false, symmetry_mode=symmetry, regular_kernel=kernel, timing,
        )
    end
    println("CPU regular-kernel benchmark: $T, order $order, $frequency Hz, symmetry $symmetry")
    println("$(length(mesh.faces)) elements, $(p1.global_dof_count) P1 dofs, 3 RHS columns, $(Threads.nthreads()) Julia threads")
    println("Warm both kernels before measurement; report minimum of $repetitions passes.")
    println("Times include regular-stage SoA/scratch setup and all symmetry images; exclude singular/identity stages.")
    println("Use an otherwise idle host; shared load can invalidate speedups.")
    assemble(:scalar)
    assemble(:simd)
    times = Dict(:scalar => Inf, :simd => Inf)
    scalar = simd = nothing
    for repetition in 1:repetitions
        # Alternate order so one kernel does not always inherit the same load.
        kernels = isodd(repetition) ? (:scalar, :simd) : (:simd, :scalar)
        for kernel in kernels
            timing = Dict{String,Float64}()
            system = assemble(kernel; timing)
            times[kernel] = min(times[kernel], timing["fused_regular_cpu_scatter"])
            if kernel === :scalar
                scalar = system
            else
                simd = system
            end
        end
    end
    println("Scalar regular stage: $(round(times[:scalar]; sigdigits=6)) s")
    println("SIMD regular stage:   $(round(times[:simd]; sigdigits=6)) s")
    println("Regular-stage speedup: $(round(times[:scalar] / times[:simd]; sigdigits=5))x")
    println("Max entrywise matrix difference / scalar maximum: ",
            _beat_cpu_benchmark_relative_difference(simd.matrix, scalar.matrix))
    println("Max entrywise RHS difference / scalar maximum: ",
            _beat_cpu_benchmark_relative_difference(simd.rhs, scalar.rhs))
    println("Widest floating-point LLVM vector in the concrete SIMD regular kernel: ",
            _beat_cpu_benchmark_llvm(cpu_cache.elements, cpu_cache.regular_quadrature, cpu_cache, q, k))
    println("LLVM vector width is a code-generation diagnostic; stage timing measures the speedup on this host.")
    return nothing
end

function _beat_cpu_benchmark_main(args)
    options = _beat_cpu_benchmark_options(args)
    options === nothing && return
    precision = lowercase(strip(options["precision"]))
    precision in ("float32", "float64") || error("--precision must be float32 or float64.")
    _beat_cpu_benchmark_regular(options, precision == "float32" ? Float32 : Float64)
end

_beat_cpu_benchmark_main(ARGS)
