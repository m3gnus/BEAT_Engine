function _cuda_field_gradient_kernel!(values, points, cache_points, normals, weights, faces, elements, basis, pressure, neumann, k)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= size(points,1)
        T = eltype(points)
        x = SVector(points[i,1], points[i,2], points[i,3])
        result = zero(SVector{4,Complex{T}})
        for j in axes(cache_points,1)
            y = SVector(cache_points[j,1],cache_points[j,2],cache_points[j,3])
            n = SVector(normals[j,1],normals[j,2],normals[j,3])
            p = basis[j,1]*pressure[faces[j,1]] + basis[j,2]*pressure[faces[j,2]] + basis[j,3]*pressure[faces[j,3]]
            result += field_value_gradient(x,y,n,p,neumann[elements[j]],k) * weights[j]
        end
        for d in 1:4
            values[i,d] = result[d]
        end
    end
    return nothing
end

function evaluate_galerkin_field_gradient_cuda(points, mesh::BoundaryMesh{T}, pressure, neumann, k::T, cache;
                                              symmetry_mode=:off, rule=triangle_rule(T,4)) where {T}
    isempty(points) && return zeros(Complex{T},0,4)
    xyz = T[points[i][d] for i in eachindex(points), d in 1:3]
    d_points, d_pressure, d_neumann = CuArray(xyz), CuArray(pressure), CuArray(neumann)
    d_values = CUDA.zeros(Complex{T},length(points),4)
    try
        CUDA.@cuda threads=128 blocks=cld(length(points),128) _cuda_field_gradient_kernel!(
            d_values,d_points,cache.source_points,cache.source_normals,cache.source_weights,
            cache.source_faces,cache.source_elements,cache.basis_values,d_pressure,d_neumann,outgoing_wavenumber(k))
        values = Array(d_values)
        return correct_field_gradient!(values,points,mesh,pressure,neumann,outgoing_wavenumber(k),rule; symmetry_mode)
    finally
        CUDA.unsafe_free!(d_values)
        CUDA.unsafe_free!(d_points)
        CUDA.unsafe_free!(d_pressure)
        CUDA.unsafe_free!(d_neumann)
    end
end
