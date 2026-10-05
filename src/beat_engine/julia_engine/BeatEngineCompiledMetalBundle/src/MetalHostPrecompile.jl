# Compile host methods, never execute them. Cache fields holding device arrays
# are intentionally untyped in the engine, so inference from the exterior
# driver cannot reach every concrete gather, launch and wrapping specialization.
function metal_host_signatures()
    core = BeatEngineCore
    mesh = core.BoundaryMesh{Float32}
    rule = core.TriangleRule{Float32}
    regular = core.MetalRegularAssemblyCache{Float32,Nothing}
    singular = core.MetalSingularCorrectionCache{Float32}
    field = core.MetalFieldEvaluationCache{Float32}
    private(T, N) = Metal.MtlArray{T,N,Metal.PrivateStorage}
    shared(T, N) = Metal.MtlArray{T,N,Metal.SharedStorage}
    sparse_scatter = core.MetalSparseScatterCache{
        private(Int32, 1),private(Int32, 1),private(ComplexF32, 1)}
    identity = core.MetalFusedIdentityCache{sparse_scatter,SparseMatrixCSC{ComplexF32,Int}}
    system = NamedTuple{
        (:matrix, :rhs, :regular_pairs, :singular_pairs, :image_singular_pairs,
         :drive_count, :on_gpu, :gpu_backend, :assembly_mode),
        Tuple{shared(ComplexF32, 2),shared(ComplexF32, 2),Int,Int,Int,Int,Bool,Symbol,Symbol},
    }
    signatures = Tuple{Any,Tuple}[]
    function add(f, args; kwargs...)
        if isempty(kwargs)
            push!(signatures, (f, args))
        else
            kwtype = NamedTuple{keys(kwargs),Tuple{values(kwargs)...}}
            push!(signatures, (Core.kwcall, (kwtype, typeof(f), args...)))
        end
    end

    # Exact production driver argument types: Float32 geometry, host excitation
    # vectors, native cache (host_cache == nothing), and shared dense systems.
    geometry = (mesh, core.P1Space, core.DP0Space, rule)
    add(core.build_metal_regular_assembly_cache, geometry;
        singular_order=Int, symmetry_mode=Symbol)
    add(core.build_metal_singular_correction_cache, (core.SingularCorrectionCache{Float32},))
    add(core.build_metal_field_evaluation_cache, (core.FieldEvaluationCache{Float32},))
    add(core.build_metal_fused_identity_cache, (Matrix{Float32}, Matrix{Float32}, Type{Float32}))
    add(core.build_metal_sparse_scatter_cache, (SparseMatrixCSC{ComplexF32,Int},))
    fused_args = (mesh, core.P1Space, core.DP0Space, Vector{Vector{ComplexF32}}, Float32, rule)
    fused_kwargs = (device_cache=regular, singular_cache=core.SingularCorrectionCache{Float32},
                   device_singular_cache=singular, identity_cache=identity,
                   singular_order=Int, symmetry_mode=Symbol)
    add(assemble_exterior_direct_metal, fused_args; fused_kwargs...)
    add(solve_exterior_direct_metal, fused_args; fused_kwargs...)
    add(core.assemble_burton_miller_neumann_system_metal,
        (mesh, core.P1Space, core.DP0Space, Matrix{ComplexF32}, Float32, rule); fused_kwargs...)
    add(core._metal_fused_gather_tables, (regular,))
    add(core._metal_singular_gather_tables, (regular, singular, Int))
    add(core._metal_build_singular_gather_tables, (regular, singular, Int))
    add(core._metal_fill_singular_gather_inputs!,
        (ntuple(_ -> Vector{Int32}, 8)..., Matrix{Int32}, Vector{Int32},
         ntuple(_ -> Int, 4)...))
    add(core._metal_build_singular_gather_map, (Vector{Int32}, Vector{Int32}, Nothing))
    add(core._launch_metal_fused_pair_kernels!,
        (shared(ComplexF32, 2), private(ComplexF32, 3), private(ComplexF32, 2),
         regular, Float32, private(Int32, 1), private(Int32, 1), Int32,
         ntuple(_ -> Float32, 6)...))
    for args in ((shared(ComplexF32, 2), shared(ComplexF32, 2), private(ComplexF32, 2),
                  regular, singular, Float32),
                 (shared(ComplexF32, 2), shared(ComplexF32, 2), private(ComplexF32, 2),
                  regular, singular, Float32, core.SymmetryTransform))
        add(core._launch_metal_fused_singular_kernels!, args)
    end
    add(core.scatter_metal_sparse_to_dense!, (shared(ComplexF32, 2), sparse_scatter);
        alpha=ComplexF32, add=Bool)
    add(core.metal_sweep_overlap_plan, (Int, Int, Symbol); frequency_count=Int)
    add(core.metal_sweep_assembly_lookahead, (Int, Int, Int, Type{Float32}))
    add(core.take_sweep_assembly!, (core.SweepAssemblyPipeline, Int))
    add(solve_exterior_direct_metal_system, (system,))
    add(core.metal_host_burton_miller_system, (system,))
    add(core.solve_metal_burton_miller_system_with_report, (system,))
    add(core.release_metal_burton_miller_system!, (system,))
    points = Vector{SVector{3,Float32}}
    field_args = (points, mesh, Vector{ComplexF32}, Vector{ComplexF32}, Float32, field)
    add(exterior_field, (field_args..., Symbol))
    add(core.evaluate_galerkin_field_metal, field_args)
    for (f, args) in ((core.release_metal_regular_assembly_cache!, (regular,)),
                      (core.release_metal_singular_correction_cache!, (singular,)),
                      (core.release_metal_field_evaluation_cache!, (field,)),
                      (core.release_metal_fused_identity_cache!, (identity,)))
        add(f, args)
    end

    # Conversions and shared-memory wrapping reached across untyped cache fields.
    for (T, dimensions) in ((Float32, (1, 2, 3)), (Int32, (1, 2)), (ComplexF32, (1, 2)))
        for N in dimensions
            add(Metal.MtlArray, (Array{T,N},))
            # Gather tables download indices; assembly/field download complex
            # data. Float32 geometry is only uploaded on the native path.
            T === Float32 || add(Array, (private(T, N),))
            add(Metal.unsafe_free!, (private(T, N),))
        end
    end
    add(unsafe_wrap, (Type{Array}, shared(ComplexF32, 2)))
    add(Metal.unsafe_free!, (shared(ComplexF32, 2),))
    for N in 1:3
        add(Metal.zeros, (Type{ComplexF32}, ntuple(_ -> Int, N)...))
    end
    add(Metal.zeros, (Type{ComplexF32}, Int, Int); storage=Type{Metal.SharedStorage})

    # Reuse the recorded device signatures for host launches of the fused and
    # field kernels. Device arrays erase storage mode; restore the exact mode
    # used at each call site. This evaluates types only, without fake buffers.
    shared_destination = Set((core._metal_fused_lhs_gather_kernel!,
        core._metal_fused_rhs_reduce_kernel!, core._metal_singular_entry_gather_kernel!,
        core._metal_singular_rhs_gather_kernel!, core._metal_sparse_scatter_kernel!))
    launches = Set((core._metal_fused_pair_blocks_kernel!,
        core._metal_fused_lhs_gather_kernel!, core._metal_fused_rhs_gather_kernel!,
        core._metal_fused_rhs_reduce_kernel!, core._metal_singular_fused_bm_blocks_kernel!,
        core._metal_singular_entry_gather_kernel!, core._metal_singular_rhs_gather_kernel!,
        core._metal_sparse_scatter_kernel!, core._metal_weighted_field_sources_kernel!,
        core._metal_field_eval_entries_kernel!, core._metal_field_eval_chunked_kernel!,
        core._metal_field_reduce_partials_kernel!))
    for (f, tt) in metal_kernel_signatures()
        f in launches || continue
        # fieldtypes expands fixed Vararg entries in the generated inventory.
        args = Tuple(type <: Metal.MtlDeviceArray ?
            Metal.MtlArray{eltype(type),ndims(type),
                index == 1 && f in shared_destination ? Metal.SharedStorage : Metal.PrivateStorage} : type
            for (index, type) in enumerate(fieldtypes(tt)))
        if f !== core._metal_fused_pair_blocks_kernel!
            add(core._metal_launch, (typeof(f), Int, args...))
            add(core._metal_launch, (typeof(f), Int, args...); groupsize=Int)
        end
        # The pair driver uses @metal directly across untyped cache fields.
        # Compile the callable HostKernel method by type, without an instance.
        launch_size = f === core._metal_fused_pair_blocks_kernel! ? Tuple{Int,Int} : Int
        kwtype = NamedTuple{(:threads, :groups),Tuple{launch_size,launch_size}}
        push!(signatures, (Core.kwcall, (kwtype, Metal.HostKernel{typeof(f),tt}, args...)))
    end
    return signatures
end

function precompile_metal_host_signatures()
    successes = 0
    failures = 0
    for (index, (f, args)) in enumerate(metal_host_signatures())
        try
            precompile(f, args) || error("No matching host method")
            successes += 1
        catch exception
            failures += 1
            @warn "BEAT Metal host precompile failed" index function_type=typeof(f) argument_types=args exception=(exception, catch_backtrace())
        end
    end
    @info "BEAT Metal host workload complete" compiled_signatures=successes failures
end
