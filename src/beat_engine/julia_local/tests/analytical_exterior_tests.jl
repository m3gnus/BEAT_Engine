using Test, LinearAlgebra, StaticArrays

# Default closed-form fixture: 0.1 m radius, 1280 outward-facing triangles.
function reference_sphere(
    radius::Float64,
    subdivisions::Int;
    centre::SVector{3,Float64}=SVector{3,Float64}(0, 0, 0),
    tag::Int=1,
)
    subdivisions >= 0 || error("Icosphere subdivision level must be >= 0.")
    phi = (1 + sqrt(Float64(5))) / 2
    vertices = SVector{3,Float64}[
        SVector{3,Float64}(-1, phi, 0), SVector{3,Float64}(1, phi, 0),
        SVector{3,Float64}(-1, -phi, 0), SVector{3,Float64}(1, -phi, 0),
        SVector{3,Float64}(0, -1, phi), SVector{3,Float64}(0, 1, phi),
        SVector{3,Float64}(0, -1, -phi), SVector{3,Float64}(0, 1, -phi),
        SVector{3,Float64}(phi, 0, -1), SVector{3,Float64}(phi, 0, 1),
        SVector{3,Float64}(-phi, 0, -1), SVector{3,Float64}(-phi, 0, 1),
    ]
    faces = NTuple{3,Int}[
        (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
        (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
        (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
        (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2),
    ]

    for _ in 1:subdivisions
        midpoints = Dict{Tuple{Int,Int},Int}()
        function midpoint(a::Int, b::Int)
            key = a < b ? (a, b) : (b, a)
            index = get(midpoints, key, 0)
            index != 0 && return index
            push!(vertices, (vertices[a] + vertices[b]) / 2)
            midpoints[key] = length(vertices)
            return length(vertices)
        end
        next_faces = NTuple{3,Int}[]
        sizehint!(next_faces, 4 * length(faces))
        for (a, b, c) in faces
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            push!(next_faces, (a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))
        end
        faces = next_faces
    end

    vertices = [centre + radius * (vertex / norm(vertex)) for vertex in vertices]

    # Orient every winding outward before the mesh is built, so the normals the
    # BoundaryMesh constructor derives are outward by construction.
    oriented = similar(faces)
    for (index, face) in enumerate(faces)
        v1, v2, v3 = vertices[face[1]], vertices[face[2]], vertices[face[3]]
        outward = (v1 + v2 + v3) / 3 - centre
        oriented[index] = dot(cross(v2 - v1, v3 - v1), outward) >= 0 ?
            face : (face[1], face[3], face[2])
    end

    mesh = BoundaryMesh(vertices, oriented, fill(tag, length(oriented)))
    for index in eachindex(mesh.faces)
        dot(mesh.normals[index], mesh.centroids[index] - centre) > 0 ||
            error("Icosphere face $(index) has an inward normal.")
    end
    return mesh
end

# Keep closed forms independent of outgoing_wavenumber/neumann_scale: a
# convention defect in those helpers must not conjugate both sides of the gate.
analytic_monopole(x, source, signed_k, omega, strength) =
    -im * sign(signed_k) * 1.2041 * omega * strength *
    exp(im * signed_k * norm(x - source)) / (4pi * norm(x - source))

function analytic_monopole_gradient(x, source, signed_k, omega, strength)
    delta = x - source
    r = norm(delta)
    return analytic_monopole(x, source, signed_k, omega, strength) *
        (im * signed_k - 1/r) * delta/r
end

function analytic_exterior_system(mesh, k; symmetry_mode=:off)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(Float64, 4)
    operators = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
        skip_singular=false, singular_order=4, backend=:cpu, symmetry_mode=symmetry_mode)
    ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry_mode)
    ipq = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry_mode)
    system = build_burton_miller_neumann_cpu_system(operators, ipp, ipq, k)
    cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry_mode)
    return (; system, cache)
end

function analytic_exterior_solve(mesh, data, q, k, points)
    pressure = solve_burton_miller_neumann_cpu_system(data.system, q, Float64)
    field = evaluate_galerkin_field_cpu(points, mesh, pressure, q, k, data.cache)
    return (; pressure, field)
end

function analytic_errors(label, actual, reference)
    @test length(actual) == length(reference) > 0
    @test all(isfinite, actual)
    @test all(isfinite, reference)
    @test all(!iszero, reference)
    ratios = actual ./ reference
    level_db = maximum(abs.(20 .* log10.(abs.(ratios))))
    phase_deg = maximum(abs.(rad2deg.(angle.(ratios))))
    @info label level_db phase_deg
    return (; level_db, phase_deg)
end

function analytic_gate(label, actual, reference)
    errors = analytic_errors(label, actual, reference)
    @test errors.level_db <= 0.05
    @test errors.phase_deg <= 0.5
end

function analytic_phase_control(label, actual, reference)
    errors = analytic_errors(label, actual, reference)
    @test errors.phase_deg >= 10 * 0.5
end

@testset "closed-form exterior level and phase" begin
    frequency, radius, rho, c = 1000.0, 0.1, 1.2041, 343.0
    omega = 2pi * frequency
    k = omega/c
    centre = SVector(0., 0., 0.)
    ground_centre, image_centre = SVector(0., 0.5, 0.), SVector(0., -0.5, 0.)
    free_mesh = reference_sphere(radius, 3)
    ground_mesh = reference_sphere(radius, 3; centre=ground_centre)
    impedance_mesh = reference_sphere(radius, 3; centre=SVector(0., 20radius, 0.))
    @test length(free_mesh.faces) == 1280
    points = fibonacci_sphere(128, 3.0)
    ground_points = [x for x in fibonacci_sphere(256, 3.0) if x[2] > 0 && norm(x-ground_centre) > 4radius]
    @test length(ground_points) == 128
    @info "Analytic fixture" faces=length(free_mesh.faces) area_deficit=sum(free_mesh.areas)/(4pi*radius^2)-1
    conventions = (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
    systems = map(conventions) do convention
        with_phasor_convention(convention) do
            (free=analytic_exterior_system(free_mesh, k),
             ground=analytic_exterior_system(ground_mesh, k; symmetry_mode=:ground),
             impedance=analytic_exterior_system(impedance_mesh, k; symmetry_mode=:ground))
        end
    end
    for (index, convention) in enumerate(conventions)
        @testset "$convention" begin
            with_phasor_convention(convention) do
                @info "Analytic phasor" convention
                s = convention == NEGATIVE_TIME_PHASOR ? 1 : -1
                signed_k = s*k
                data = systems[index]
                q_uniform = fill(neumann_scale(rho, omega), length(free_mesh.faces))
                pulsating = analytic_exterior_solve(free_mesh, data.free, q_uniform, k, points)
                ka = im*signed_k*radius
                sphere_reference = [rho*c*(radius/norm(x))*ka/(ka-1)*exp(im*signed_k*(norm(x)-radius)) for x in points]
                analytic_gate("Pulsating sphere", pulsating.field, sphere_reference)

                # Upgrade the existing manufactured monopole check, rather than
                # adding a second weaker check of the same field. The faceted
                # surface is an exact boundary for this manufactured problem.
                strength = ComplexF64(4pi*radius^2)
                exact(x) = analytic_monopole(x, centre, signed_k, omega, strength)
                # dot(gradient, normal) would conjugate the complex gradient.
                q = [sum(analytic_monopole_gradient(x, centre, signed_k, omega, strength) .* n)
                     for (x,n) in zip(free_mesh.centroids, free_mesh.normals)]
                monopole = analytic_exterior_solve(free_mesh, data.free, q, k, points)
                analytic_gate("Manufactured monopole", monopole.field, exact.(points))
                @test norm(imag.(monopole.field)) > 0

                # Retain complex excitation replay and the independent second
                # source/superposition assertions from the original test.
                amplitude = s == 1 ? 0.3-0.7im : 0.3+0.7im
                replay = evaluate_galerkin_field_cpu(points, free_mesh,
                    amplitude .* monopole.pressure, amplitude .* q, k, data.free.cache)
                @test replay ≈ amplitude .* monopole.field rtol=1e-12
                source = SVector(0.02, -0.01, 0.)
                second_q = [sum(analytic_monopole_gradient(x, source, signed_k, omega, strength) .* n)
                            for (x,n) in zip(free_mesh.centroids, free_mesh.normals)]
                second = analytic_exterior_solve(free_mesh, data.free, second_q, k, points)
                second_reference = [analytic_monopole(x, source, signed_k, omega, strength) for x in points]
                analytic_gate("Second manufactured monopole", second.field, second_reference)
                mixed = analytic_exterior_solve(free_mesh, data.free, q + amplitude .* second_q, k, points)
                @test mixed.field ≈ monopole.field + amplitude .* second.field rtol=1e-12
                @test norm(second.field - monopole.field)/norm(monopole.field) > 0.01

                rigid(x) = analytic_monopole(x, ground_centre, signed_k, omega, strength) +
                           analytic_monopole(x, image_centre, signed_k, omega, strength)
                soft(x) = analytic_monopole(x, ground_centre, signed_k, omega, strength) -
                          analytic_monopole(x, image_centre, signed_k, omega, strength)
                ground_q = [sum((analytic_monopole_gradient(x, ground_centre, signed_k, omega, strength) +
                                 analytic_monopole_gradient(x, image_centre, signed_k, omega, strength)) .* n)
                            for (x,n) in zip(ground_mesh.centroids, ground_mesh.normals)]
                ground = analytic_exterior_solve(ground_mesh, data.ground, ground_q, k, ground_points)
                analytic_gate("Rigid-image monopole", ground.field, rigid.(ground_points))
                analytic_phase_control("Control: pressure-release image", ground.field, soft.(ground_points))

                # A copied k -> -k control is a no-op after official sign
                # normalisation. Instead select the opposite convention for
                # operators, coupling AND field, while q/reference stay fixed.
                @test outgoing_wavenumber(-k) == outgoing_wavenumber(k)
                opposite = conventions[3-index]
                with_phasor_convention(opposite) do
                    wrong = systems[3-index]
                    wrong_free = analytic_exterior_solve(free_mesh, wrong.free, q, k, points)
                    wrong_ground = analytic_exterior_solve(ground_mesh, wrong.ground, ground_q, k, ground_points)
                    analytic_phase_control("Control: opposite kernel, free", wrong_free.field, exact.(points))
                    analytic_phase_control("Control: opposite kernel, ground", wrong_ground.field, rigid.(ground_points))
                    wrong_sphere = analytic_exterior_solve(free_mesh, wrong.free, q_uniform, k, points)
                    analytic_phase_control("Control: opposite kernel, pulsating", wrong_sphere.field, sphere_reference)
                    # With real velocity, q is imaginary: wrong p = -conj(p).
                    # The magnitude gate therefore cannot detect this defect.
                    @test wrong_sphere.field ≈ -conj.(pulsating.field) rtol=1e-12
                    @test abs.(wrong_sphere.field) ≈ abs.(pulsating.field) rtol=1e-12
                end
                @test phasor_convention() == convention

                # Julia-side absolute phase gate for unit normal acceleration.
                # This does not exercise transport decoding in Python clients.
                acceleration_field = pulsating.field / time_derivative(omega)
                acceleration_reference = [rho*radius^2/norm(x)*exp(im*signed_k*(norm(x)-radius))/(1-im*signed_k*radius) for x in points]
                analytic_gate("Pulsating sphere per acceleration", acceleration_field, acceleration_reference)
                analytic_phase_control("Control: conjugated pressure", conj.(acceleration_field), acceleration_reference)
                @test abs.(conj.(acceleration_field)) == abs.(acceleration_field)
                analytic_phase_control("Control: wrong acceleration sign", -acceleration_field, acceleration_reference)

                # Independently score physical impedance via the compiled
                # path's production integration, not the exterior field gate.
                # At 20 radii the image loading is small; counting a fictitious
                # image as another radiator instead moves impedance by ~6 dB.
                impedance_q = fill(neumann_scale(rho, omega), length(impedance_mesh.faces))
                impedance = analytic_exterior_solve(impedance_mesh, data.impedance, impedance_q, k, SVector{3,Float64}[])
                excitation = (tags=[1], amplitudes=ComplexF64[1])
                free_z = exterior_component_impedance(free_mesh, pulsating.pressure, excitation, :off, Float64)
                ground_z = exterior_component_impedance(impedance_mesh, impedance.pressure, excitation, :ground, Float64)
                move_db = 20log10(abs(ground_z/free_z))
                defect_db = 20log10(symmetry_reduction_factor(:ground)*abs(ground_z/free_z))
                @info "Physical impedance count" move_db defect_db
                @test isfinite(move_db) && isfinite(defect_db)
                @test abs(move_db) <= 1.0
                @test abs(defect_db) > 1.0
            end
        end
    end
    @test phasor_convention() == NEGATIVE_TIME_PHASOR
end

@testset "retained complex fields match explicit symmetry images" begin
    T = Float64
    vertices = [SVector{3,T}(0.2,0.3,0.0), SVector{3,T}(0.4,0.3,0.0), SVector{3,T}(0.2,0.5,0.1)]
    mesh = BoundaryMesh(vertices, [(1,2,3)], [1])
    pressure = ComplexF64[1+0.2im, 0.5-0.3im, 0.7+0.1im]
    q = ComplexF64[0.3-0.2im]
    points = [SVector{3,T}(1,2,3), SVector{3,T}(-1,-2,3)]
    rule = triangle_rule(T, 3)
    for symmetry in (:x, :xy)
        signs = symmetry == :x ? [(1,1,1), (-1,1,1)] : [(1,1,1), (-1,1,1), (1,-1,1), (-1,-1,1)]
        full_vertices = SVector{3,T}[]
        full_faces = NTuple{3,Int}[]
        for sign in signs
            offset = length(full_vertices)
            append!(full_vertices, [SVector{3,T}(sign) .* vertex for vertex in vertices])
            face = prod(sign) == 1 ? (1,2,3) : (1,3,2)
            push!(full_faces, face .+ offset)
        end
        full = BoundaryMesh(full_vertices, full_faces, ones(Int, length(full_faces)))
        reduced_cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry)
        full_cache = build_field_evaluation_cache(full, rule)
        reduced = evaluate_galerkin_field_cpu(points, mesh, pressure, q, 0.7, reduced_cache)
        explicit = evaluate_galerkin_field_cpu(points, full, repeat(pressure, length(signs)), repeat(q, length(signs)), 0.7, full_cache)
        @test reduced ≈ explicit rtol=1e-12 atol=1e-14
        @test norm(reduced) > 0
    end
end
