using SHA, Serialization

# One entry, explicitly opt-in. Never reuse by a caller-supplied scene key alone.
const DEPLOY_FACTOR_STATE = Ref{Any}(nothing)

function release_deploy_factor_state!()
    state = DEPLOY_FACTOR_STATE[]
    DEPLOY_FACTOR_STATE[] = nothing
    state === nothing && return nothing
    cuda = BeatEngineCore.CUDA_MODULE
    cuda.unsafe_free!(state.factorization.factors)
    cuda.unsafe_free!(state.factorization.ipiv)
    cuda.unsafe_free!(state.operator)
    return nothing
end

function deploy_canonical(value)
    value isa AbstractDict && return [(k, deploy_canonical(value[k])) for k in sort!(collect(keys(value)); by=string)]
    value isa AbstractVector && return map(deploy_canonical, value)
    return value
end

function deploy_factor_signature(request, mesh, model)
    io = IOBuffer()
    serialize(io, "deploy-projected-factors-v2")
    serialize(io, BeatEngineCore.phasor_convention())
    serialize(io, (mesh.vertices, mesh.faces))
    # All physical/discretization settings, including close-pair corrections.
    for key in ("frequency_hz", "density_kg_per_m3", "sound_speed_m_per_s",
                "quadrature_order", "singular_order", "close_pair_quadrature_order",
                "boundary")
        serialize(io, (key, deploy_canonical(get(request, key, nothing))))
    end
    # Near maps can contain millions of tiny JSON vectors. Normalize exactly as
    # the solve does, then serialize contiguous integers instead of recursively
    # allocating/serializing each vector and its diagnostic metadata.
    proximity = get(request, "proximity", Dict{String,Any}())
    default_order = Int(get(request, "close_pair_quadrature_order", 8))
    for key in ("close_face_pairs", "ground_image_close_face_pairs")
        raw = get(proximity, key, Any[])
        pairs = Matrix{Int64}(undef, 3, length(raw))
        for (i, pair) in enumerate(raw)
            pairs[1,i] = Int(pair[1])
            pairs[2,i] = Int(pair[2])
            pairs[3,i] = length(pair) >= 3 ? Int(pair[3]) : default_order
        end
        serialize(io, (key, pairs))
    end
    # Hash loaded numerical arrays, not file names, mtimes or package labels.
    for group in deploy_projection_groups(model)
        m, instance = group.model, group.instance
        serialize(io, (m.rank, m.sector_count, m.image_count, m.image_signs,
                       m.node_orbits, m.face_orbits, deploy_canonical(m.arrays),
                       instance.node_offset, instance.face_offset))
    end
    return bytes2hex(sha256(take!(io)))
end
