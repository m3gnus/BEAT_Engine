using Metal: MtlDeviceArray, MtlDeviceMatrix, MtlDeviceVector

# Keep signature evaluation inside the workload: internal GPUArrays function
# names can change with dependency upgrades. A stale entry must warn visibly,
# rather than preventing the worker from loading its host-code bundle.
metal_kernel_signatures() = include(joinpath(@__DIR__, "MetalKernelSignatures.jl"))

# Host launches of the generated device signatures, for every inventory that compiles
# them. Device arrays erase their storage mode, so each kernel records the argument
# positions production passes as shared arrays (all other device arrays are private),
# its launch size type (`Tuple{Int,Int}` for the 2-D pair grids, which production
# launches with `@metal`; every other kernel goes through `_metal_launch`), and the
# inventories that use it: `:exterior` (MetalHostPrecompile.jl, the full-argument
# HostKernel call) and `:coupled` (MetalCoupledPrecompile.jl, the traced call with
# erased trailing arguments plus its autorelease closure). One table, so the storage
# modes cannot differ between inventories or from the tests.
function metal_launch_rules()
    core = BeatEngineCore
    rule(shared, paths...; grid=Int) = (; shared=Tuple(shared), grid, paths=Tuple(paths))
    pair_grid = Tuple{Int,Int}
    return IdDict{Any,Any}(
        # Four-operator coupled path.
        core._metal_regular_pair_blocks_kernel! => rule((), :coupled; grid=pair_grid),
        core._metal_gather_slp_adjoint_kernel! => rule((1, 2), :coupled),
        core._metal_gather_dlp_hyp_kernel! => rule((1, 2), :coupled),
        core._metal_singular_fused_blocks_kernel! => rule((), :coupled),
        core._metal_singular_pair_gather_kernel! => rule((1, 2), :coupled),
        # Fused exterior assembly; the combined A/C coupled path (the Metal coupled
        # default) reuses its pair, left-hand-side and singular kernels.
        core._metal_fused_pair_blocks_kernel! => rule((), :exterior, :coupled; grid=pair_grid),
        core._metal_fused_lhs_gather_kernel! => rule((1,), :exterior, :coupled),
        core._metal_fused_singular_packed_kernel! => rule((), :exterior, :coupled),
        core._metal_singular_entry_gather_kernel! => rule((1,), :exterior, :coupled),
        core._metal_coupled_flux_gather_kernel! => rule((1,), :coupled),
        core._metal_fused_rhs_gather_kernel! => rule((), :exterior),
        core._metal_fused_rhs_reduce_kernel! => rule((1,), :exterior),
        core._metal_singular_rhs_gather_kernel! => rule((1,), :exterior),
        core._metal_sparse_scatter_kernel! => rule((1,), :exterior),
        # Exterior field evaluation.
        core._metal_weighted_field_sources_kernel! => rule((), :exterior),
        core._metal_field_eval_entries_kernel! => rule((), :exterior),
        core._metal_field_eval_chunked_kernel! => rule((), :exterior),
        core._metal_field_reduce_partials_kernel! => rule((), :exterior),
        core._metal_fast_field_kernel! => rule((), :exterior),
        core._metal_fast_field_sources_kernel! => rule((), :exterior),
        core._metal_fast_field_reduce_kernel! => rule((), :exterior),
        core._metal_fast_field_outputs_kernel! => rule((), :exterior),
        core._metal_fast_field_outputs_reduce_kernel! => rule((), :exterior),
    )
end

# (kernel, device signature, host argument types, launch size type) for every generated
# signature `path` launches, with the storage modes of `metal_launch_rules`.
function metal_launch_types(path::Symbol)
    rules = metal_launch_rules()
    launches = Tuple{Any,Type,Tuple,Type}[]
    for (f, tt) in metal_kernel_signatures()
        rule = get(rules, f, nothing)
        rule !== nothing && path in rule.paths || continue
        # fieldtypes expands fixed Vararg entries in the generated inventory.
        args = Tuple(type <: Metal.MtlDeviceArray ?
            Metal.MtlArray{eltype(type),ndims(type),
                index in rule.shared ? Metal.SharedStorage : Metal.PrivateStorage} : type
            for (index, type) in enumerate(fieldtypes(tt)))
        push!(launches, (f, tt, args, rule.grid))
    end
    return launches
end

# Production calls `_metal_launch` for every kernel except the `@metal` pair grids.
metal_launch_uses_helper(size) = size === Int

function clear_metal_precompile_state!()
    # Matches Metal 1.11.1's own precompile cleanup. Objective-C objects are
    # process-local; only GPUCompiler's portable code cache belongs in the image.
    Base.@lock Metal.compiler_configs_lock empty!(Metal._compiler_configs)
    empty!(Metal.kernel_instances)
    Base.@lock Metal.global_queues_lock empty!(Metal.global_queues)
    Base.@lock Metal.batched_queues_lock empty!(Metal.batched_queues)
    empty!(Metal.queue_residency_sets)
    Base.@lock Metal.memory_pressure_stats_lock empty!(Metal._memory_pressure_stats)
    empty!(Metal.device_malloc_bufs)
    empty!(Metal.MTL.last_committed_per_queue)
    empty!(Metal.MTL.submission_state_per_queue)
    empty!(Metal.device_exception_info)
    Metal.reset_binary_archives!()
end

function precompile_metal_kernel_signatures()
    # Metal.functional() is false while generating a package image, even on a
    # GPU host. Follow Metal's platform gate; device failures warn below.
    if VERSION < v"1.11"
        @info "BEAT Metal kernel workload skipped: requires Julia 1.11 or newer"
        return
    end
    if !(Sys.isapple() && Sys.ARCH === :aarch64)
        @info "BEAT Metal kernel workload skipped: requires Apple Silicon"
        return
    end
    successes = 0
    failures = 0
    try
        signatures = metal_kernel_signatures()
        isempty(signatures) && error("No production Metal kernel signatures.")
        for (index, (f, tt)) in enumerate(signatures)
            try
                # Compile and link only; never submit a GPU command buffer.
                Metal.mtlfunction(f, tt)
                successes += 1
            catch exception
                failures += 1
                @warn "BEAT Metal kernel precompile failed" index function_type=typeof(f) argument_types=tt exception=(exception, catch_backtrace())
            end
        end
    catch exception
        failures += 1
        @warn "BEAT Metal kernel inventory could not be loaded" exception=(exception, catch_backtrace())
    finally
        clear_metal_precompile_state!()
        @info "BEAT Metal kernel workload complete" compiled_signatures=successes failures
    end
end
