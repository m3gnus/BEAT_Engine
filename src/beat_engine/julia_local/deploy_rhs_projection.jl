# Exact projection onto the already-exported ROM's flux range. No new truncation.
function deploy_projection_groups(model)
    if hasproperty(model, :models)
        return vcat([deploy_projection_groups(model.models[id]) for id in sort!(collect(keys(model.models)))]...)
    end
    return [(model=model, instance=instance) for instance in model.instances]
end

function build_deploy_fused_system(mesh, p1, dp0, q, k, rule, model;
    device_cache, singular_cache, device_singular_cache, device_image_singular_cache,
    near_correction_cache, device_near_correction_cache,
    image_near_correction_cache, device_image_near_correction_cache, block_cache,
    progress=identity)
    cuda = BeatEngineCore.CUDA_MODULE
    matrix = cuda.zeros(eltype(q), length(mesh.vertices), length(mesh.vertices))
    units = cuda.ones(eltype(q), length(q))
    projected = nothing
    succeeded = false
    try
        assemble_columns = function(columns; regular_system=nothing)
            assemble_burton_miller_rhs_cuda(mesh, p1, dp0, units, k, rule;
                device_cache=device_cache, singular_cache=singular_cache,
                device_singular_cache=device_singular_cache,
                device_image_singular_cache=device_image_singular_cache,
                near_correction_cache=near_correction_cache,
                device_near_correction_cache=device_near_correction_cache,
                image_near_correction_cache=image_near_correction_cache,
                device_image_near_correction_cache=device_image_near_correction_cache,
                block_cache=block_cache, symmetry_mode=:ground, assemble_operator=true,
                operator_columns=columns, regular_system=regular_system)
        end
        projected = build_deploy_projected_rhs(model, assemble_columns;
            regular_system=matrix, drive=q, progress=progress)
        finalize_s = @elapsed BeatEngineCore.complete_cuda_bm_matrix!(matrix, mesh, k, rule;
            device_cache=device_cache, singular_cache=singular_cache,
            device_singular_cache=device_singular_cache,
            device_image_singular_cache=device_image_singular_cache,
            device_near_correction_cache=device_near_correction_cache,
            device_image_near_correction_cache=device_image_near_correction_cache,
            block_cache=block_cache)
        projected.report["shared_assembly"] = true
        projected.report["matrix_finalize_s"] = finalize_s
        system = (matrix=matrix, rhs=projected.drive_rhs,
            regular_pairs=2length(mesh.faces)^2-singular_cache.pair_count,
            singular_pairs=singular_cache.pair_count,
            image_singular_pairs=device_image_singular_cache.pair_count,
            near_pair_count=near_correction_cache.pair_count+image_near_correction_cache.pair_count,
            on_gpu=true, assembly_mode=:direct_burton_miller)
        succeeded = true
        return system, projected
    finally
        cuda.unsafe_free!(units)
        if !succeeded
            cuda.unsafe_free!(matrix)
            if projected !== nothing
                cuda.unsafe_free!(projected.operator)
                cuda.unsafe_free!(projected.drive_rhs)
            end
        end
    end
end

function deploy_projection_basis(group)
    model = group.model
    basis = zeros(eltype(model.arrays["d"]), model.package_face_count, model.rank * model.sector_count)
    for sector in 1:model.sector_count
        columns = (sector-1)*model.rank+1:sector*model.rank
        for (i, orbit) in enumerate(model.face_orbits), image in eachindex(orbit)
            basis[orbit[image], columns] .= model.image_signs[sector][image] .* view(model.arrays["d"], sector, i, :)
        end
    end
    return basis
end

function deploy_projection_coordinates(groups, pressure)
    coordinates = eltype(pressure)[]
    for group in groups
        model, instance = group.model, group.instance
        for sector in 1:model.sector_count
            compact = [sum(model.image_signs[sector][image] * pressure[instance.node_offset+orbit[image]]
                           for image in eachindex(orbit)) / model.image_count for orbit in model.node_orbits]
            state = model.factors[sector] \ (-view(model.arrays["c"], sector, :, :) * compact)
            append!(coordinates, state)
        end
    end
    return coordinates
end

function build_deploy_projected_rhs(model, assemble_columns; progress=identity, regular_system=nothing, drive=nothing)
    cuda = BeatEngineCore.CUDA_MODULE
    groups = deploy_projection_groups(model)
    rank = sum(g.model.rank * g.model.sector_count for g in groups)
    T = eltype(first(groups).model.arrays["d"])
    cuda.reclaim()
    projected = cuda.zeros(T, model.node_count, rank)
    drive_rhs = drive === nothing ? nothing : cuda.zeros(T, model.node_count)
    cursor = 0
    assembly_s = projection_s = 0.0
    succeeded = false
    try
        for (i, group) in enumerate(groups)
            local_basis = deploy_projection_basis(group)
            width = size(local_basis, 2)
            slab = basis = nothing
            try
                progress("Projecting exterior feedback for cabinet $i/$(length(groups))")
                faces = group.instance.face_offset .+ collect(1:group.model.package_face_count)
                assembly_s += @elapsed slab = regular_system === nothing ? assemble_columns(faces) : assemble_columns(faces; regular_system=regular_system)
                projection_s += @elapsed begin
                    basis = cuda.CuArray(local_basis)
                    mul!(view(projected, :, cursor+1:cursor+width), slab, basis)
                    if drive_rhs !== nothing
                        local_drive = cuda.CuArray(drive[faces])
                        try
                            mul!(drive_rhs, slab, local_drive, one(T), one(T))
                            cuda.synchronize()
                        finally
                            cuda.unsafe_free!(local_drive)
                        end
                    end
                    cuda.synchronize()
                end
            finally
                slab === nothing || cuda.unsafe_free!(slab)
                basis === nothing || cuda.unsafe_free!(basis)
            end
            cursor += width
        end
        succeeded = true
        return (operator=projected, groups=groups, drive_rhs=drive_rhs,
                report=Dict{String,Any}("mode"=>"projected", "rank"=>rank,
                    "stored_bytes"=>sizeof(T)*length(projected), "additional_truncation"=>false,
                    "column_assembly_s"=>assembly_s, "projection_s"=>projection_s))
    finally
        succeeded || cuda.unsafe_free!(projected)
        (succeeded || drive_rhs === nothing) || cuda.unsafe_free!(drive_rhs)
    end
end
