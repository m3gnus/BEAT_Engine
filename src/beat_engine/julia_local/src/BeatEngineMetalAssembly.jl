function _normalized_metal_assembly_mode(value)
    value === nothing && (value = get(ENV, "BLAB_METAL_ASSEMBLY_MODE", "native"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :host => :host_staged,
        :staged => :host_staged,
        :host_staged => :host_staged,
        :native => :native,
        :native_regular => :native,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error("Metal assembly mode must be host_staged or native; got $(value).")
    return normalized
end

function _assemble_regular_galerkin_operators_metal_native(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool,
    singular_order::Int,
    element_indices,
    cache,
    timing,
    singular_cache,
    metal_singular_cache,
    near_correction_cache,
    image_near_correction_cache,
    symmetry_mode::Symbol,
) where {T<:AbstractFloat}
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    native_cache = cache === nothing ? build_metal_regular_assembly_cache(
        mesh,
        p1_space,
        dp0_space,
        rule;
        singular_order=singular_order,
        element_indices=element_indices,
        symmetry_mode=normalized_mode,
    ) : cache
    native_cache isa MetalRegularAssemblyCache || error("Native Metal assembly requires a MetalRegularAssemblyCache.")
    native_cache.symmetry_mode == normalized_mode ||
        error("Metal assembly cache symmetry mode $(native_cache.symmetry_mode) does not match requested $(normalized_mode).")
    singular_mode = _normalized_metal_singular_mode()

    operators = nothing
    storage = metal_operator_storage_mode()
    allocation_elapsed = @elapsed begin
        operators = (
            single_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count; storage=storage),
            double_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count; storage=storage),
            adjoint_double_layer=Metal.zeros(Complex{T}, p1_space.global_dof_count, dp0_space.global_dof_count; storage=storage),
            hypersingular=Metal.zeros(Complex{T}, p1_space.global_dof_count, p1_space.global_dof_count; storage=storage),
        )
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_native_operator_alloc"] = allocation_elapsed)

    regular_kernel_mode = _normalized_metal_regular_kernel_mode()
    # In host singular mode the image regular kernels must integrate every
    # image pair with the regular rule (skip_mode 2), because the CPU image
    # correction is a Duffy-minus-regular delta. In native mode they skip the
    # image-singular pairs (skip_mode 1) and the gather kernel adds Duffy.
    skip_image_singular = !skip_singular && singular_mode == :native
    empty!(_metal_gather_stage_timing)
    kernel_elapsed = @elapsed begin
        if regular_kernel_mode == :pair_owned
            _launch_metal_regular_pair_kernels!(operators, native_cache, k)
        elseif regular_kernel_mode == :pair_atomic
            _launch_metal_regular_atomic_kernels!(operators, native_cache, k)
        elseif regular_kernel_mode == :pair_gather
            _launch_metal_regular_gather_kernels!(operators, native_cache, k)
        else
            _launch_metal_regular_entry_kernels!(operators, native_cache, k)
        end
        for (transform, image_cache) in zip(native_cache.image_transforms, native_cache.image_singular_caches)
            if regular_kernel_mode == :pair_owned
                _launch_metal_symmetry_regular_pair_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            elseif regular_kernel_mode == :pair_atomic
                _launch_metal_symmetry_regular_atomic_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            elseif regular_kernel_mode == :pair_gather
                _launch_metal_symmetry_regular_gather_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            else
                _launch_metal_symmetry_regular_entry_kernels!(
                    operators,
                    native_cache,
                    image_cache,
                    transform,
                    k;
                    skip_image_singular=skip_image_singular,
                )
            end
        end
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_native_regular_kernel"] = kernel_elapsed)
    if timing !== nothing
        for (stage, elapsed) in _metal_gather_stage_timing
            timing["metal_native_gather_" * stage] = elapsed
        end
    end

    indices = native_cache.element_indices
    correction_cache = singular_cache === nothing ? build_singular_correction_cache(mesh, singular_order, indices) : singular_cache
    adjacent_pairs = correction_cache.pair_count
    singular_pairs = 0
    image_singular_pairs = 0
    if !skip_singular && singular_mode == :native
        owns_device_singular_cache = metal_singular_cache === nothing
        device_singular_cache = metal_singular_cache
        cache_elapsed = @elapsed begin
            if device_singular_cache === nothing
                device_singular_cache = build_metal_singular_correction_cache(correction_cache)
            end
        end
        timing !== nothing && (timing["metal_native_singular_cache"] = cache_elapsed)
        singular_launch! = regular_kernel_mode in (:pair_atomic, :pair_gather) ?
            _launch_metal_singular_block_scatter_kernels! :
            _launch_metal_singular_block_gather_kernels!
        singular_elapsed = @elapsed begin
            singular_launch!(operators, native_cache, device_singular_cache, k)
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_native_singular_kernel"] = singular_elapsed)
        singular_pairs = correction_cache.pair_count
        owns_device_singular_cache && release_metal_singular_correction_cache!(device_singular_cache)
        image_elapsed = @elapsed begin
            for (transform, image_cache) in zip(native_cache.image_transforms, native_cache.image_singular_caches)
                image_cache.pair_count == 0 && continue
                singular_launch!(operators, native_cache, image_cache, k, transform)
            end
            Metal.synchronize()
        end
        timing !== nothing && (timing["metal_native_image_singular_kernel"] = image_elapsed)
        image_singular_pairs = native_cache.image_singular_pair_count
    elseif !skip_singular
        host_elapsed = @elapsed begin
            image_singular_pairs = _metal_add_host_singular_corrections!(
                operators,
                mesh,
                p1_space,
                dp0_space,
                k,
                correction_cache,
                rule,
                singular_order,
                indices,
                native_cache.image_transforms,
            )
        end
        timing !== nothing && (timing["metal_host_singular_corrections"] = host_elapsed)
        singular_pairs = correction_cache.pair_count
    end
    # Replace the regular-rule contribution for close pairs before applying
    # orbit weights, exactly as CPU assembly does. Shared buffers can be edited
    # on the host after synchronization; private buffers need a round trip.
    near_caches = (
        _near_correction_cache_tuple(near_correction_cache)...,
        _near_correction_cache_tuple(image_near_correction_cache)...,
    )
    near_pair_count = sum(c.pair_count for c in near_caches; init=0)
    near_elapsed = @elapsed if near_pair_count > 0
        Metal.synchronize()
        keys = (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
        host = NamedTuple{keys}(map(keys) do key
            buffer = getfield(operators, key)
            Metal.is_shared(buffer) ? unsafe_wrap(Array, buffer) : Array(buffer)
        end)
        elements = _beat_cpu_element_data(mesh, p1_space, dp0_space)
        quadrature = _beat_cpu_regular_quadrature_data(mesh, rule)
        groups = _beat_cpu_element_color_groups(mesh, indices)
        for correction in near_caches
            _beat_cpu_apply_near_cache!(host, mesh, elements, indices, groups,
                Threads.nthreads() > 1, k, quadrature, correction)
        end
        for key in keys
            buffer = getfield(operators, key)
            Metal.is_shared(buffer) || copyto!(buffer, getfield(host, key))
        end
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_host_near_corrections"] = near_elapsed)
    weight_elapsed = @elapsed _apply_metal_operator_p1_row_weights!(operators, mesh, normalized_mode)
    timing !== nothing && (timing["metal_native_symmetry_row_weights"] = weight_elapsed)
    total_pairs = length(indices) * length(indices)
    image_count = length(native_cache.image_transforms)
    kernel_groupsize = _metal_kernel_groupsize()
    color_count = length(native_cache.color_offsets) - 1
    kernel_name = regular_kernel_mode == :pair_owned ? "colored_pair_owned" :
        regular_kernel_mode == :pair_atomic ? "fused_pair_atomic" :
        regular_kernel_mode == :pair_gather ? "chunked_pair_gather" : "entry_owned"
    mode_name = "metal_native_" * kernel_name * (singular_mode == :native ? "" : "_host_singular")
    return merge(
        operators,
        (
            regular_pairs=total_pairs - adjacent_pairs + image_count * total_pairs - image_singular_pairs,
            singular_pairs=singular_pairs,
            skipped_pairs=skip_singular ? adjacent_pairs : 0,
            image_singular_pairs=image_singular_pairs,
            near_pair_count=near_pair_count,
            near_pair_quadrature_order=maximum((c.correction_order for c in near_caches); init=0),
            on_gpu=true,
            gpu_backend=:metal,
            host_staged_assembly=false,
            regular_kernel_threads=kernel_groupsize,
            regular_kernel_qpair_count=length(rule.points)^2,
            regular_kernel_total_pairs=(image_count + 1) * total_pairs,
            regular_kernel_color_count=color_count,
            regular_kernel_launches=regular_kernel_mode == :pair_owned ?
                2 * (image_count + 1) * color_count^2 :
                regular_kernel_mode == :pair_atomic ? (image_count + 1) :
                regular_kernel_mode == :pair_gather ? 3 * (image_count + 1) * _metal_gather_chunk_count(native_cache) :
                2 * (image_count + 1),
            regular_kernel_mode=mode_name,
            regular_assembly_mode=Symbol(mode_name),
            singular_mode=singular_mode,
        ),
    )
end

function assemble_regular_galerkin_operators_metal_regular(
    mesh::BoundaryMesh{T},
    p1_space::P1Space,
    dp0_space::DP0Space,
    k::T,
    rule::TriangleRule{T};
    skip_singular::Bool=true,
    singular_order::Int=2,
    element_indices=eachindex(mesh.faces),
    cache=nothing,
    return_device::Bool=true,
    accelerator_quadrature::Bool=true,
    timing=nothing,
    singular_cache=nothing,
    metal_singular_cache=nothing,
    near_correction_cache=nothing,
    image_near_correction_cache=nothing,
    assembly_mode=nothing,
    symmetry_mode::Symbol=:off,
) where {T<:AbstractFloat}
    _require_metal!()
    k = outgoing_wavenumber(k)
    return_device || error("Metal assembly requires return_device=true.")
    accelerator_quadrature || error("Metal assembly requires accelerator_quadrature=true.")
    normalized_mode = normalized_symmetry_mode(symmetry_mode)
    normalized_assembly_mode = _normalized_metal_assembly_mode(assembly_mode)
    if normalized_assembly_mode == :native
        return _assemble_regular_galerkin_operators_metal_native(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            cache=cache,
            timing=timing,
            singular_cache=singular_cache,
            metal_singular_cache=metal_singular_cache,
            near_correction_cache=near_correction_cache,
            image_near_correction_cache=image_near_correction_cache,
            symmetry_mode=symmetry_mode,
        )
    end
    metal_singular_cache === nothing ||
        error("Native Metal singular-correction caches are not supported by the host-staged backend.")

    host_cache = cache === nothing ? nothing : cache.host_cache
    host_operators = nothing
    host_elapsed = @elapsed begin
        host_operators = assemble_regular_galerkin_operators_cpu(
            mesh,
            p1_space,
            dp0_space,
            k,
            rule;
            skip_singular=skip_singular,
            singular_order=singular_order,
            element_indices=element_indices,
            threaded=true,
            singular_cache=singular_cache,
            cpu_cache=host_cache,
            near_correction_cache=near_correction_cache,
            image_near_correction_cache=image_near_correction_cache,
            symmetry_mode=normalized_mode,
        )
    end
    timing !== nothing && (timing["metal_host_assembly"] = host_elapsed)

    device_operators = nothing
    transfer_elapsed = @elapsed begin
        device_operators = (
            single_layer=MtlArray(host_operators.single_layer),
            double_layer=MtlArray(host_operators.double_layer),
            adjoint_double_layer=MtlArray(host_operators.adjoint_double_layer),
            hypersingular=MtlArray(host_operators.hypersingular),
        )
        Metal.synchronize()
    end
    timing !== nothing && (timing["metal_operator_upload"] = transfer_elapsed)

    return merge(
        host_operators,
        device_operators,
        (
            on_gpu=true,
            gpu_backend=:metal,
            host_staged_assembly=true,
            regular_kernel_mode="metal_host_staged_cpu_assembly",
            regular_assembly_mode=:metal_host_staged_cpu_assembly,
        ),
    )
end
