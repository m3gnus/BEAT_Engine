# Native ROCm exterior factors shared by fixed-Neumann and ROM feedback solves.
# ROM state and Krylov vectors stay on the host, as on the CPU path; each
# preconditioner application uses the same resident rocSOLVER factorization.
function upload_deploy_rocm_identity(matrix, ::Type{T}) where {T<:AbstractFloat}
    accelerator = BeatEngineCore.amdgpu_module()
    return accelerator.ROCArray(Complex{T}.(matrix))
end

function upload_deploy_rocm_identity(matrix::Matrix{T}, ::Type{T}) where {T<:AbstractFloat}
    accelerator = BeatEngineCore.amdgpu_module()
    # Deploy assembles real identity blocks in the requested precision. Transfer
    # those values directly; avoid a full host complex allocation and halve the
    # transfer size. The resulting resident block keeps its existing complex type.
    real_block = accelerator.ROCArray(matrix)
    try
        return Complex{T}.(real_block)
    finally
        # Conversion is asynchronous: do not free its input while it is in use.
        accelerator.synchronize()
        accelerator.unsafe_free!(real_block)
    end
end

function build_deploy_rocm_identity_cache(identity_p1_p1, identity_p1_dp0, ::Type{T}) where {T<:AbstractFloat}
    accelerator = BeatEngineCore.amdgpu_module()
    first = upload_deploy_rocm_identity(identity_p1_p1, T)
    try
        second = upload_deploy_rocm_identity(identity_p1_dp0, T)
        return BeatEngineCore.RocmBurtonMillerIdentityCache(first, second)
    catch
        accelerator.synchronize()
        accelerator.unsafe_free!(first)
        rethrow()
    end
end

function build_deploy_rocm_system(operators, identity_p1_p1, identity_p1_dp0, k::T) where {T<:AbstractFloat}
    get(operators, :on_gpu, false) && get(operators, :gpu_backend, nothing) == :rocm ||
        error("Deploy ROCm requires native device operators.")
    BeatEngineCore._require_rocm!(rocsolver=true)
    accelerator = BeatEngineCore.amdgpu_module()
    identity = build_deploy_rocm_identity_cache(identity_p1_p1, identity_p1_dp0, T)
    lhs = nothing
    factorization = nothing
    try
        lhs = BeatEngineCore.burton_miller_neumann_lhs(operators, identity.identity_p1_p1, k)
        # Preserve requested-precision assembly while using the qualified F64
        # dense factorization policy for the feedback-sensitive coupled solve.
        matrix = ComplexF64.(lhs)
        factorization = BeatEngineCore.rocm_dense_lu!(matrix)
        return (; factorization, identity, operators, wavenumber=k)
    catch
        release_rocm_burton_miller_identity_cache!(identity)
        rethrow()
    finally
        lhs === nothing || accelerator.unsafe_free!(lhs)
    end
end

function solve_deploy_rocm_system(system, q_neumann, ::Type{T}) where {T<:AbstractFloat}
    accelerator = BeatEngineCore.amdgpu_module()
    q = rhs = promoted_rhs = pressure = nothing
    try
        q = accelerator.ROCArray(Complex{T}.(q_neumann))
        rhs = BeatEngineCore._rocm_burton_miller_rhs(system.operators, system.identity, q,
            BeatEngineCore.burton_miller_coupling(system.wavenumber))
        promoted_rhs = ComplexF64.(rhs)
        pressure = BeatEngineCore.solve_rocm_dense_factorization(system.factorization, promoted_rhs)
        return Complex{T}.(Array(pressure))
    finally
        accelerator.synchronize()
        for value in (q, rhs, promoted_rhs, pressure)
            value === nothing || accelerator.unsafe_free!(value)
        end
    end
end

function release_deploy_rocm_system!(system)
    system === nothing && return nothing
    accelerator = BeatEngineCore.amdgpu_module()
    accelerator.synchronize()
    accelerator.unsafe_free!(system.factorization.factors)
    accelerator.unsafe_free!(system.factorization.ipiv)
    release_rocm_burton_miller_identity_cache!(system.identity)
    return nothing
end
