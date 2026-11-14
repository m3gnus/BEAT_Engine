# Pressure-only thin-boundary-layer model, Berggren et al., JCP 371 (2018),
# doi:10.1016/j.jcp.2018.06.005, equation (31). Included in BeatEngineCoupled.
# Stationary, no-slip, isothermal walls; natural zero conormal flux at patch edges.
# Boundary layers must be thin relative to gaps, radii of curvature and wavelength.
const THERMOVISCOUS_AIR = (
    dynamic_viscosity_pa_s=1.84e-5,
    thermal_conductivity_w_per_m_k=0.0257,
    specific_heat_j_per_kg_k=1005.0,
    heat_capacity_ratio=1.4,
)

"""Tangential-gradient bilinear form on affine P1/P2 boundary triangles."""
function assemble_boundary_tangential_stiffness(mesh::VolumeMesh{T}, face_indices) where {T}
    rows, cols, values = Int[], Int[], T[]
    reference_gradients = T[-1 1 0; -1 0 1]
    quadratic = is_quadratic(mesh)
    # Degree-two triangle quadrature integrates products of P2 gradients exactly.
    quadrature = (T[2/3, 1/6, 1/6], T[1/6, 2/3, 1/6], T[1/6, 1/6, 2/3])
    for index in face_indices
        corners = mesh.boundary_faces[index]
        face = quadratic ? mesh.quadratic_boundary_faces[index] : corners
        origin = mesh.vertices[corners[1]]
        jacobian = hcat(mesh.vertices[corners[2]] - origin, mesh.vertices[corners[3]] - origin)
        gram = transpose(jacobian) * jacobian
        determinant = det(gram)
        determinant > eps(T) * maximum(abs, gram)^2 || error(
            "Thermoviscous boundary contains a numerically degenerate triangle.")
        metric = inv(gram)
        area = sqrt(determinant) / T(2)
        local_matrix = zeros(T, length(face), length(face))
        if quadratic
            for lambda in quadrature
                gradients = zeros(T, 2, 6)
                for i in 1:3
                    gradients[:, i] .= (T(4) * lambda[i] - one(T)) .* reference_gradients[:, i]
                end
                for (i, (a, b)) in enumerate(((1, 2), (2, 3), (3, 1)))
                    gradients[:, i + 3] .= T(4) .* (
                        lambda[a] .* reference_gradients[:, b] + lambda[b] .* reference_gradients[:, a])
                end
                local_matrix .+= (area / T(3)) .* (transpose(gradients) * metric * gradients)
            end
        else
            local_matrix .= area .* (transpose(reference_gradients) * metric * reference_gradients)
        end
        # Use identical transpose entries, including in Float32/MUMPS symmetric solves.
        local_matrix .= (local_matrix .+ transpose(local_matrix)) ./ T(2)
        for i in eachindex(face), j in eachindex(face)
            push!(rows, face[i]); push!(cols, face[j]); push!(values, local_matrix[i, j])
        end
    end
    n = length(mesh.vertices)
    return sparse(rows, cols, values, n, n)
end

function thermoviscous_surface_operator(mesh::VolumeMesh{T}, face_indices) where {T}
    faces = sort!(unique(Int.(collect(face_indices))))
    isempty(faces) && return nothing
    return (
        stiffness=assemble_boundary_tangential_stiffness(mesh, faces),
        mass=assemble_boundary_mass_matrix(mesh, faces, collect(eachindex(mesh.vertices))),
        face_indices=faces,
        area_m2=sum(_triangle_area(mesh.vertices, mesh.boundary_faces[i]) for i in faces),
    )
end

function prepare_thermoviscous_operator(mesh::VolumeMesh{T}, walls; symmetry_mode=:off) where {T}
    isempty(walls) && return nothing
    tags = Set(Int(wall.tag) for wall in walls)
    all(tag -> tag in mesh.boundary_physical_tags, tags) || error(
        "Thermoviscous wall tag has no FEM boundary faces.")
    mode = BeatEngineCore.normalized_symmetry_mode(symmetry_mode)
    axes = mode == :x ? (1,) : mode == :xy ? (1, 2) : ()
    tolerance = symmetry_plane_tolerance(mesh.vertices)
    faces = [i for i in eachindex(mesh.boundary_faces)
             if mesh.boundary_physical_tags[i] in tags &&
                !any(axis -> all(v -> abs(mesh.vertices[v][axis]) <= tolerance,
                                 mesh.boundary_faces[i]), axes)]
    return thermoviscous_surface_operator(mesh, faces)
end

function thermoviscous_coefficients(frequency_hz::T, sound_speed, density) where {T<:AbstractFloat}
    all(x -> isfinite(x) && x > 0, (frequency_hz, sound_speed, density)) || error(
        "Thermoviscous losses require positive finite frequency, sound speed and density.")
    omega = T(2pi) * frequency_hz
    air = THERMOVISCOUS_AIR
    delta_v = sqrt(T(2 * air.dynamic_viscosity_pa_s) / (T(density) * omega))
    delta_t = sqrt(T(2 * air.thermal_conductivity_w_per_m_k / air.specific_heat_j_per_kg_k) /
                   (T(density) * omega))
    # exp(+iwt): (-1+i)/2; legacy exp(-iwt): its conjugate.
    scale = Complex{T}(-one(T), -T(propagation_sign())) / T(2)
    return (viscous=scale * delta_v,
            thermal=scale * T(air.heat_capacity_ratio - 1) * delta_t * (omega / T(sound_speed))^2,
            delta_v=delta_v, delta_t=delta_t)
end

function add_thermoviscous_terms(system, operator, frequency_hz, sound_speed, density)
    isnothing(operator) && return system
    coefficients = thermoviscous_coefficients(frequency_hz, sound_speed, density)
    return system + coefficients.viscous .* operator.stiffness + coefficients.thermal .* operator.mass
end

function thermoviscous_diagnostics(operator, walls, frequency_hz, sound_speed, density)
    isempty(walls) && return Dict{String,Any}()
    coefficients = thermoviscous_coefficients(frequency_hz, sound_speed, density)
    return Dict{String,Any}(
        "thermoviscous_wall_losses" => Dict(
            "model" => "thin_boundary_layer", "model_version" => 1,
            "boundary_ids" => [wall.boundary_id for wall in walls],
            "treated_area_m2" => isnothing(operator) ? 0.0 : operator.area_m2,
            "treated_face_count" => isnothing(operator) ? 0 : length(operator.face_indices),
            "viscous_boundary_layer_m" => coefficients.delta_v,
            "thermal_boundary_layer_m" => coefficients.delta_t,
            "air_properties" => Dict(string(k) => v for (k, v) in pairs(THERMOVISCOUS_AIR)),
        ),
    )
end
