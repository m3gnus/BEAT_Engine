@testset "projected DP0 interface work and admissibility" begin
    for T in (Float32, Float64)
        vertices = SVector{3,T}[
            SVector{3,T}(0, 0, 0), SVector{3,T}(1, 0, 0),
            SVector{3,T}(0, 1, 0), SVector{3,T}(0, 0, 1),
        ]
        faces = [(1, 3, 2), (1, 2, 4), (2, 3, 4), (3, 1, 4)]
        fem = VolumeMesh{T}(vertices, [(1, 2, 3, 4)], [1], faces,
                            fill(2, 4), Dict{Tuple{Int,Int},String}())
        for orientation in (1, -1)
            permuted = orientation == 1 ? [(f[2], f[3], f[1]) for f in faces] :
                                         [(f[1], f[3], f[2]) for f in faces]
            # A rigid trial face shares mouth vertices; it must receive no flux.
            bem_vertices = vcat(vertices, [SVector{3,T}(2, -1, 0)])
            bem = BoundaryMesh(bem_vertices, vcat(permuted, [(1, 2, 5)]), [2, 2, 2, 2, 1])
            map = ConformingInterfaceMap(collect(1:4), collect(1:4),
                                        collect(1:4), collect(1:4), fill(orientation, 4))
            op = assemble_interface_operators(fem, bem, map)
            p = Complex{T}[1+im, -0.3+0.2im, 0.7-0.5im, -0.8-0.1im]
            q = Complex{T}[0.2-0.1im, 1+0.3im, -0.5+0.9im, 0.4-0.8im]
            face_flux = op.bem_flux * q
            fem_work = dot(p, op.fem_load * q)
            bem_work = sum(bem.areas[a] * conj(sum(p[v] for v in faces[a])/T(3)) *
                           face_flux[a] * T(orientation) for a in 1:4)
            @test fem_work ≈ bem_work rtol=50eps(T)
            @test face_flux[5] == 0
            @test all(face_flux[a] ≈ T(orientation)*sum(q[v] for v in faces[a])/T(3) for a in 1:4)
            # Integrated constant-flux loading remains the exact P1 weak load.
            consistent = assemble_boundary_mass_matrix(fem, collect(1:4), collect(1:4))
            @test op.fem_load * ones(T, 4) ≈ consistent * ones(T, 4) rtol=50eps(T)
            @test norm(op.fem_load*q - consistent*q) > T(0.01)
            # Uniform rescaling cannot change admissibility.
            @test isnothing(BeatEngineCoupled._validate_projected_interface_load(
                op.fem_load*T(1e-6), collect(1:4), T))
        end
        if T == Float64
            perturbed_vertices = copy(vertices)
            perturbed_vertices[2] += SVector{3,T}(5e-11, 0, 0)
            perturbed_bem = BoundaryMesh(perturbed_vertices, faces, fill(2, 4))
            matched = build_conforming_interface_map(fem, perturbed_bem, 2, 2)
            op = assemble_interface_operators(fem, perturbed_bem, matched)
            p = ComplexF64[1+im, 2-im, -0.5+im, 0.3-0.2im]
            q = ComplexF64[0.2-im, -0.3+im, 2+im, 0.7-0.5im]
            Q = op.bem_flux
            @test dot(p, op.fem_load*q) ≈ dot(Q*p, Diagonal(perturbed_bem.areas)*(Q*q)) rtol=1e-14
            @test perturbed_bem.areas != BoundaryMesh(vertices, faces, fill(2, 4)).areas
        end
        # One triangle loses two coefficient modes even though geometry is valid.
        triangle_gram = sparse(fill(T(1)/T(9), 3, 3))
        @test_throws ErrorException BeatEngineCoupled._validate_projected_interface_load(
            triangle_gram, collect(1:3), T)
        # Two adjacent triangles support a nonzero three-color null mode too.
        Q = sparse(T[1 1 1 0; 1 0 1 1] ./ T(3))
        @test_throws ErrorException BeatEngineCoupled._validate_projected_interface_load(
            transpose(Q)*Q, collect(1:4), T)
        # A closed octahedron has more faces than vertices, yet a three-color
        # coefficient mode still has zero mean on all eight faces.
        octahedron_faces = [(a, b, c) for a in (1, 2), b in (3, 4), c in (5, 6)]
        octahedron_Q = sparse(repeat(collect(1:8); inner=3),
            [v for face in vec(octahedron_faces) for v in face], fill(T(1)/T(3), 24), 8, 6)
        @test_throws ErrorException BeatEngineCoupled._validate_projected_interface_load(
            transpose(octahedron_Q)*octahedron_Q, collect(1:6), T)
        threshold = max(sqrt(eps(Float64)), 1024Float64(eps(T)))
        for (margin, accepted) in ((threshold/4, false), (4threshold, true))
            # Row-stochastic Gram with eigenvalues 1 and margin.
            gram = sparse(T[(1+margin)/2 (1-margin)/2; (1-margin)/2 (1+margin)/2])
            if accepted
                @test isnothing(BeatEngineCoupled._validate_projected_interface_load(gram, [1, 2], T))
            else
                @test_throws ErrorException BeatEngineCoupled._validate_projected_interface_load(gram, [1, 2], T)
            end
        end
        @test_throws ErrorException BeatEngineCoupled._validate_projected_interface_load(spzeros(T, 2, 2), [1, 2], T)
    end
end
