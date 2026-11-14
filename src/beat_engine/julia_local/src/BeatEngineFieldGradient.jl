"""Pressure and observation-coordinate gradient of D*p - S*q, with signed k."""
@inline function field_value_gradient(x, y, normal, p, q, k::T) where {T}
    r = y - x
    radius = norm(r)
    u = r / radius
    green = exp(Complex{T}(0, k * radius)) / (T(4pi) * radius)
    a = Complex{T}(-inv(radius), k)
    b = Complex{T}(T(3) / radius^2 - k^2, -T(3) * k / radius)
    projection = dot(u, normal)
    value = green * (a * projection * p - q)
    gradient = green * (a * q * u - p * (b * projection * u + (a / radius) * normal))
    return SVector(value, gradient[1], gradient[2], gradient[3])
end

function _gradient_triangle(x, vertices, normal, p, q, k, rule)
    result = zero(SVector{4,Complex{eltype(x)}})
    jac = norm(cross(vertices[2] - vertices[1], vertices[3] - vertices[1]))
    for j in eachindex(rule.points)
        basis = p1_values(rule.points[j])
        y = local_to_global(vertices, rule.points[j])
        result += field_value_gradient(x, y, normal, sum(basis .* p), q, k) * (jac * rule.weights[j])
    end
    return result
end

function _adaptive_gradient_triangle(x, vertices, normal, p, q, k, rule, depth=0)
    center = (vertices[1] + vertices[2] + vertices[3]) / 3
    diameter = maximum((norm(vertices[1]-vertices[2]), norm(vertices[2]-vertices[3]), norm(vertices[3]-vertices[1])))
    distance = norm(x - center)
    if diameter <= 0.4 * distance && abs(k) * diameter <= 0.7
        return _gradient_triangle(x, vertices, normal, p, q, k, rule)
    end
    depth < 12 || error("BEM gradient sample is too close to a boundary for reliable evaluation.")
    a, b, c = vertices
    ab, bc, ca = (a+b)/2, (b+c)/2, (c+a)/2
    pa, pb, pc = p
    pab, pbc, pca = (pa+pb)/2, (pb+pc)/2, (pc+pa)/2
    return _adaptive_gradient_triangle(x, (a,ab,ca), normal, SVector(pa,pab,pca), q,k,rule,depth+1) +
           _adaptive_gradient_triangle(x, (ab,b,bc), normal, SVector(pab,pb,pbc), q,k,rule,depth+1) +
           _adaptive_gradient_triangle(x, (ca,bc,c), normal, SVector(pca,pbc,pc), q,k,rule,depth+1) +
           _adaptive_gradient_triangle(x, (ab,bc,ca), normal, SVector(pab,pbc,pca), q,k,rule,depth+1)
end

"""Replace near-panel regular quadrature with adaptive integration, also for CUDA output."""
function correct_field_gradient!(values, points, mesh::BoundaryMesh{T}, pressure, neumann, k, rule;
                                 symmetry_mode=:off) where {T}
    # Corrections use double precision even for single-precision field kernels.
    fine_rule = triangle_rule(Float64, 4)
    transforms = symmetry_transforms(symmetry_mode; include_identity=true)
    Threads.@threads for i in eachindex(points)
        correction = zero(SVector{4,ComplexF64})
        for face_id in eachindex(mesh.faces), transform in transforms
            vertices = reflect_vertices(transform, mesh.face_vertices[face_id])
            center = (vertices[1]+vertices[2]+vertices[3])/3
            diameter = maximum((norm(vertices[1]-vertices[2]),norm(vertices[2]-vertices[3]),norm(vertices[3]-vertices[1])))
            norm(points[i] - center) <= 3 * diameter || continue
            normal = reflect_normal(transform, mesh.normals[face_id])
            face = mesh.faces[face_id]
            p = SVector{3,Complex{T}}(pressure[face[1]], pressure[face[2]], pressure[face[3]])
            coarse = _gradient_triangle(points[i], vertices, normal, p, neumann[face_id], k, rule)
            fine = _adaptive_gradient_triangle(SVector{3,Float64}(points[i]),
                map(v -> SVector{3,Float64}(v), vertices), SVector{3,Float64}(normal),
                SVector{3,ComplexF64}(p), ComplexF64(neumann[face_id]), Float64(k), fine_rule)
            correction += fine - coarse
        end
        for d in 1:4
            values[i,d] += correction[d]
        end
    end
    return values
end

function evaluate_galerkin_field_gradient_cpu(points, mesh::BoundaryMesh{T}, pressure, neumann, k::T, cache;
                                             symmetry_mode=:off, rule=triangle_rule(T,4)) where {T}
    signed_k = outgoing_wavenumber(k)
    values = zeros(Complex{T}, length(points), 4)
    Threads.@threads for i in eachindex(points)
        result = zero(SVector{4,Complex{T}})
        for j in eachindex(cache.source_points)
            face = cache.source_faces[j]
            p = sum(cache.basis_values[j] .* SVector(pressure[face[1]], pressure[face[2]], pressure[face[3]]))
            result += field_value_gradient(points[i], cache.source_points[j], cache.source_normals[j],
                p, neumann[cache.source_elements[j]], signed_k) * cache.source_weights[j]
        end
        values[i,:] = result
    end
    return correct_field_gradient!(values, points, mesh, pressure, neumann, signed_k, rule; symmetry_mode)
end
