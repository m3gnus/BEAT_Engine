function _rocm_field_gradient_kernel!(values, points, cache_points, normals, weights,
                                      faces, elements, basis, pressure, neumann, k)
    i = _rocm_global_linear_index()
    if i <= size(points, 1)
        T = eltype(points)
        x = SVector(points[i,1], points[i,2], points[i,3])
        result = zero(SVector{4,Complex{T}})
        for j in axes(cache_points, 1)
            y = SVector(cache_points[j,1], cache_points[j,2], cache_points[j,3])
            n = SVector(normals[j,1], normals[j,2], normals[j,3])
            p = basis[j,1]*pressure[faces[j,1]] + basis[j,2]*pressure[faces[j,2]] + basis[j,3]*pressure[faces[j,3]]
            result += field_value_gradient(x, y, n, p, neumann[elements[j]], k) * weights[j]
        end
        for d in 1:4
            values[i,d] = result[d]
        end
    end
    return nothing
end

function evaluate_galerkin_field_gradient_rocm(points, mesh::BoundaryMesh{T}, pressure,
        neumann, k::T, cache::RocmFieldEvaluationCache{T};
        symmetry_mode=:off, rule=triangle_rule(T,4)) where {T}
    isempty(points) && return zeros(Complex{T}, 0, 4)
    _require_rocm!()
    d_points = d_pressure = d_neumann = d_values = nothing
    try
        d_points = AMDGPU.ROCArray(_rocm_eval_point_arrays(points, T))
        d_pressure = AMDGPU.ROCArray(pressure)
        d_neumann = AMDGPU.ROCArray(neumann)
        d_values = AMDGPU.zeros(Complex{T}, length(points), 4)
        AMDGPU.@roc groupsize=128 gridsize=cld(length(points),128) _rocm_field_gradient_kernel!(
            d_values, d_points, cache.source_points, cache.source_normals, cache.source_weights,
            cache.source_faces, cache.source_elements, cache.basis_values,
            d_pressure, d_neumann, outgoing_wavenumber(k))
        values = Array(d_values)
        # Match CPU/CUDA: adaptive near-panel corrections are evaluated on the host.
        return correct_field_gradient!(values, points, mesh, pressure, neumann,
            outgoing_wavenumber(k), rule; symmetry_mode)
    finally
        AMDGPU.synchronize()
        for storage in (d_values, d_points, d_pressure, d_neumann)
            storage === nothing || AMDGPU.unsafe_free!(storage)
        end
    end
end
