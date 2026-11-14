@testset "ROCm rigid ground with no image singular pairs" begin
    for T in (Float32, Float64)
        @testset "$T" begin
            # Closed octahedron above y=0: image contributions are regular, so
            # the image singular cache legitimately contains no rules or pairs.
            vertices = SVector{3,T}[(.1, 1, 0), (-.1, 1, 0), (0, 1.1, 0),
                                     (0, .9, 0), (0, 1, .1), (0, 1, -.1)]
            faces = [(1,3,5), (3,2,5), (2,4,5), (4,1,5),
                     (3,1,6), (2,3,6), (4,2,6), (1,4,6)]
            mesh = BoundaryMesh(vertices, faces, ones(Int, length(faces)))
            p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
            rule, k = triangle_rule(T, 3), T(2pi * 500 / 343)
            singular = build_singular_correction_cache(mesh, 3)
            device_cache = build_rocm_regular_assembly_cache(mesh, p1, dp0, rule;
                singular_order=3, symmetry_mode=:ground)
            try
                @test device_cache.image_singular_pair_count == 0
                @test length(device_cache.image_singular_caches) == 1
                image = only(device_cache.image_singular_caches)
                @test image.pair_count == 0
                @test isempty(image.rule_weights)
                @test Array(image.rule_offsets) == Int32[1]
                @test Array(image.pair_offsets) == ones(Int32, length(faces) + 1)
                cpu = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                    backend=:cpu, symmetry_mode=:ground, singular_order=3,
                    singular_cache=singular, skip_singular=false)
                free = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                    backend=:cpu, symmetry_mode=:off, singular_order=3,
                    singular_cache=singular, skip_singular=false)
                @test norm(cpu.single_layer - free.single_layer) > 0
                gpu = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule;
                    backend=:rocm, symmetry_mode=:ground, singular_order=3,
                    device_cache, singular_cache=singular, skip_singular=false,
                    rocm_assembly_mode=:native)
                try
                    @test gpu.on_gpu && gpu.gpu_backend == :rocm
                    tolerance = T == Float32 ? 5e-4 : 1e-10
                    for key in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                        expected = getproperty(cpu, key)
                        @test norm(Array(getproperty(gpu, key)) - expected) / norm(expected) < tolerance
                    end
                    ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=:ground)
                    ipq = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=:ground)
                    identity = build_rocm_burton_miller_identity_cache(ipp, ipq, T)
                    try
                        q = Complex{T}[complex(sin(i), cos(i)) for i in eachindex(faces)]
                        expected = solve_burton_miller_neumann(cpu, ipp, ipq, q, k)
                        pressure = solve_burton_miller_neumann(gpu, identity, q, k)
                        @test norm(pressure - expected) / norm(expected) < tolerance
                    finally
                        release_rocm_burton_miller_identity_cache!(identity)
                    end
                finally
                    release_operator_storage!(gpu)
                end
            finally
                release_rocm_regular_assembly_cache!(device_cache)
            end
        end
    end
end
