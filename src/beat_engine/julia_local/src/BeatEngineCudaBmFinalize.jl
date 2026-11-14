# Complete the system matrix whose regular entries were assembled with RHS slabs.
function complete_cuda_bm_matrix!(matrix, mesh, k, rule;
    device_cache, singular_cache, device_singular_cache,
    device_image_singular_cache=nothing, device_near_correction_cache=nothing,
    device_image_near_correction_cache=nothing, symmetry_mode=:ground, block_cache=nothing)
    T = typeof(real(zero(eltype(matrix))))
    k = outgoing_wavenumber(k)
    lanes = reinterpret(reshape, T, matrix)
    re, im = view(lanes, 1, :, :), view(lanes, 2, :, :)
    q = CUDA.zeros(Complex{T}, length(mesh.faces))
    rr, ri = CUDA.zeros(T, size(matrix,1)), CUDA.zeros(T, size(matrix,1))
    weights = nothing
    try
        add_cuda_bm_singular_corrections!(re, im, rr, ri, q, mesh, k,
            singular_cache, device_singular_cache, device_cache; block_cache=block_cache)
        for cache in (device_image_singular_cache, device_near_correction_cache,
                      device_image_near_correction_cache)
            add_cuda_bm_image_corrections!(re, im, rr, ri, q, mesh, k, rule,
                cache, device_cache; block_cache=block_cache)
        end
        CUDA.@cuda threads=256 blocks=cld(length(mesh.faces),256) _cuda_bm_identity_kernel!(
            re, im, rr, ri, device_cache.areas, device_cache.faces, q, inv(k),
            size(matrix,1), length(mesh.faces), false)
        CUDA.synchronize()
        weights = CuArray(p1_symmetry_orbit_weights(mesh, symmetry_mode))
        CUDA.@cuda threads=256 blocks=cld(length(matrix),256) _cuda_bm_scale_rows_kernel!(
            re, im, rr, ri, weights, size(matrix,1))
        CUDA.synchronize()
    finally
        weights === nothing || CUDA.unsafe_free!(weights)
        CUDA.unsafe_free!(q); CUDA.unsafe_free!(rr); CUDA.unsafe_free!(ri)
        CUDA.unsafe_free!(lanes)
    end
    return matrix
end
