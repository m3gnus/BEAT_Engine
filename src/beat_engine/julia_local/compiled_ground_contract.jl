"""Validate the real radiator against the rigid image plane at Y=0."""
function validate_compiled_ground_domain!(mesh, symmetry_mode; tolerance::Real=1.0e-6)
    symmetry_mode == :ground || return nothing
    isempty(mesh.faces) && error("Rigid-ground symmetry requires a mesh with faces.")

    minimum_y = minimum(
        Float64(mesh.vertices[vertex_index][2])
        for face in mesh.faces for vertex_index in face
    )
    minimum_y >= -tolerance || error(
        "Rigid-ground symmetry requires the whole radiator at Y >= 0; " *
        "the mesh reaches Y=$(minimum_y) m."
    )
    for (face_index, face) in enumerate(mesh.faces)
        all(abs(Float64(mesh.vertices[vertex_index][2])) <= tolerance for vertex_index in face) || continue
        error(
            "Rigid-ground symmetry cannot image triangle $(face_index), which lies " *
            "flat on Y=0 and would coincide with itself."
        )
    end
    return nothing
end

"""Keep the physical FEM region above the image plane in coupled ground solves."""
function validate_compiled_ground_volume!(mesh, symmetry_mode; tolerance::Real=1.0e-6)
    symmetry_mode == :ground || return nothing
    isempty(mesh.vertices) && error("Rigid-ground symmetry requires a nonempty FEM volume mesh.")
    minimum_y = minimum(Float64(vertex[2]) for vertex in mesh.vertices)
    minimum_y >= -tolerance || error(
        "Rigid-ground symmetry requires the whole FEM region at Y >= 0; " *
        "the volume mesh reaches Y=$(minimum_y) m."
    )
    return nothing
end

"""Signed face projection for an explicitly oriented rigid source."""
exterior_motion_factor(excitation, normal, ::Type{T}) where {T<:AbstractFloat} =
    get(excitation, :motion_axis, nothing) === nothing ? one(T) :
    T(dot(normal, excitation.motion_axis))

"""Radiation impedance scales by real symmetry copies, not image sources."""
function exterior_component_impedance(mesh, pressure, excitation, symmetry_mode, ::Type{T}) where {T<:AbstractFloat}
    force = zero(Complex{T})
    amplitude_by_tag = Dict(zip(excitation.tags, excitation.amplitudes))
    for face_index in eachindex(mesh.faces)
        tag = mesh.physical_tags[face_index]
        haskey(amplitude_by_tag, tag) || continue
        face = mesh.faces[face_index]
        average_pressure = (pressure[face[1]] + pressure[face[2]] + pressure[face[3]]) / T(3)
        contribution = average_pressure * T(mesh.areas[face_index]) * amplitude_by_tag[tag]
        force += get(excitation, :motion_axis, nothing) === nothing ? contribution :
            contribution * exterior_motion_factor(excitation, mesh.normals[face_index], T)
    end
    real_radiator_count = symmetry_mode == :ground ? 1 : symmetry_reduction_factor(symmetry_mode)
    return force * T(real_radiator_count)
end
