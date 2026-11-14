struct RocmSingularCorrectionCache{T}
    pair_offsets
    test_indices
    trial_indices
    rule_indices
    jac_scales
    normal_products
    rule_offsets
    rule_test_points
    rule_trial_points
    rule_weights
    pair_count::Int
end

"""Own the quadrature-dependent ROCm buffers for one regular rule."""
function build_rocm_quadrature_cache(mesh::BoundaryMesh{T}, p1, dp0, rule;
    singular_order, symmetry_mode, assembly_mode=nothing) where {T}
    identity = (
        assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode),
        assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode),
    )
    device_cache = field_cache = device_identity_cache = nothing
    try
        device_cache = build_rocm_regular_assembly_cache(mesh, p1, dp0, rule;
            singular_order, symmetry_mode, assembly_mode)
        field_cache = build_rocm_field_evaluation_cache(build_field_evaluation_cache(mesh, rule; symmetry_mode))
        device_identity_cache = build_rocm_burton_miller_identity_cache(identity[1], identity[2], T)
        return (; rule, identity, device_cache, field_cache, device_identity_cache)
    catch
        device_identity_cache === nothing || release_rocm_burton_miller_identity_cache!(device_identity_cache)
        field_cache === nothing || release_rocm_field_evaluation_cache!(field_cache)
        device_cache === nothing || release_rocm_regular_assembly_cache!(device_cache)
        rethrow()
    end
end

function release_rocm_quadrature_cache!(cache)
    release_rocm_burton_miller_identity_cache!(cache.device_identity_cache)
    release_rocm_field_evaluation_cache!(cache.field_cache)
    release_rocm_regular_assembly_cache!(cache.device_cache)
    return nothing
end

struct RocmRegularAssemblyCache{T,C}
    host_cache::C
    face_vertices
    normals
    areas
    faces
    curls
    rule_points
    rule_weights
    vertex_offsets
    incident_elements
    incident_local_indices
    dp0_elements
    p1_dofs
    element_dp0_dofs
    color_elements
    color_offsets::Vector{Int}
    element_indices::Vector{Int}
    face_count::Int
    p1_dof_count::Int
    dp0_dof_count::Int
    rule_count::Int
    symmetry_mode::Symbol
    image_transforms::Vector{SymmetryTransform}
    image_singular_caches::Vector{RocmSingularCorrectionCache{T}}
    image_singular_pair_count::Int
end

struct RocmFieldEvaluationCache{T}
    source_points
    source_normals
    source_weights
    source_faces
    source_elements
    basis_values
    source_count::Int
end

function _require_rocm!(; rocsolver::Bool=false)
    AMDGPU.functional() || error("ROCm solve requested, but AMDGPU.functional() is false.")
    AMDGPU.functional(:rocblas) || error("ROCm solve requested, but rocBLAS is not functional.")
    if rocsolver
        AMDGPU.functional(:rocsolver) || error("ROCm solve requested, but rocSOLVER is not functional.")
    end
    return nothing
end

function _rocm_kernel_groupsize()
    groupsize = parse(Int, get(ENV, "BLAB_ROCM_KERNEL_GROUPSIZE", "64"))
    groupsize in (32, 64, 128, 256) ||
        error("BLAB_ROCM_KERNEL_GROUPSIZE must be 32, 64, 128, or 256; got $(groupsize).")
    return groupsize
end

function _normalized_rocm_regular_kernel_mode(value=nothing)
    value === nothing && (value = get(ENV, "BLAB_ROCM_REGULAR_KERNEL_MODE", "pair_owned"))
    mode = Symbol(lowercase(strip(String(value))))
    aliases = Dict(
        :pair => :pair_owned,
        :pair_owned => :pair_owned,
        :colored => :pair_owned,
        :colored_pair_owned => :pair_owned,
        :entry => :entry_owned,
        :entry_owned => :entry_owned,
    )
    normalized = get(aliases, mode, nothing)
    normalized === nothing && error(
        "BLAB_ROCM_REGULAR_KERNEL_MODE must be pair_owned or entry_owned; got $(value).",
    )
    return normalized
end
