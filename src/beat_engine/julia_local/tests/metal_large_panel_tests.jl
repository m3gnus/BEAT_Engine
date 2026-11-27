# Run explicitly on Metal hardware; unavailable hardware must not certify this gate.
using Test, StaticArrays, LinearAlgebra, Metal
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "large_panel_test_helpers.jl"))

@test Metal.functional()
@testset "Metal large-panel corrections match the CPU's" begin
    T = Float32
    for symmetry in (:off, :x, :xy), storage in ("shared", "private")
        withenv("BLAB_METAL_OPERATOR_STORAGE" => storage) do
            mesh = _large_panel_sphere(T; symmetry=symmetry)
            p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
            rule = triangle_rule(T, 4)
            k = T(_large_panel_mixed_k(mesh))
            panel_rules = LargePanelRules(mesh, 4)
            plan = large_panel_plan(panel_rules, k)
            @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
            identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
            identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
            q = ComplexF32[ComplexF32(sin(0.7f0 * row), cos(0.4f0 * row)) for row in 1:dp0.global_dof_count, _ in 1:2]
            singular_cache = build_singular_correction_cache(mesh, 4)
            reference = assemble_burton_miller_neumann_system_cpu(
                mesh, p1, dp0, q, k, rule; identity_p1_p1, identity_p1_dp0, singular_order=4, singular_cache,
                symmetry_mode=symmetry, large_panels=plan,
            )
            # The tolerance below must be far smaller than the correction
            # itself, or a Metal path that dropped it would still pass.
            unplanned = assemble_burton_miller_neumann_system_cpu(
                mesh, p1, dp0, q, k, rule; identity_p1_p1, identity_p1_dp0, singular_order=4, singular_cache,
                symmetry_mode=symmetry,
            )
            @test norm(reference.matrix - unplanned.matrix) > 20 * 5.0f-4 * norm(reference.matrix)
            @test norm(reference.rhs - unplanned.rhs) > 20 * 5.0f-4 * norm(reference.rhs)
            cache = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=4, symmetry_mode=symmetry)
            for device_drives in (false, true)
                drives = device_drives ? MtlArray(q) : q
                system = nothing
                try
                    system = assemble_burton_miller_neumann_system_metal(
                        mesh, p1, dp0, drives, k, rule; device_cache=cache, singular_cache, identity_p1_p1, identity_p1_dp0,
                        singular_order=4, symmetry_mode=symmetry, large_panels=plan,
                    )
                    host = BeatEngineCore.metal_host_burton_miller_system(system)
                    @test norm(host.matrix - reference.matrix) <= 5.0f-4 * norm(reference.matrix)
                    @test norm(host.rhs - reference.rhs) <= 5.0f-4 * norm(reference.rhs)
                finally
                    system === nothing || release_metal_burton_miller_system!(system)
                    device_drives && Metal.unsafe_free!(drives)
                end
            end
            operators = nothing
            try
                operators = assemble_regular_galerkin_operators(
                    mesh, p1, dp0, k, rule; backend=:metal, return_device=true, accelerator_quadrature=true,
                    device_cache=cache, skip_singular=false, singular_order=4, singular_cache,
                    symmetry_mode=symmetry, large_panels=plan,
                )
                operators = metal_host_operators(operators)
                expected = assemble_regular_galerkin_operators_cpu(
                    mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4, singular_cache,
                    symmetry_mode=symmetry, large_panels=plan,
                )
                unplanned_operators = assemble_regular_galerkin_operators_cpu(
                    mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4, singular_cache,
                    symmetry_mode=symmetry,
                )
                for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                    @test norm(getfield(operators, name) - getfield(expected, name)) <= 5.0f-4 * norm(getfield(expected, name))
                    @test norm(getfield(expected, name) - getfield(unplanned_operators, name)) >
                          20 * 5.0f-4 * norm(getfield(expected, name))
                end
            finally
                operators === nothing || release_operator_storage!(operators)
                release_metal_regular_assembly_cache!(cache)
            end
        end
    end
end

@testset "A launch stays below its point-pair budget" begin
    T = Float32
    mesh = _large_panel_sphere(T)
    plan = large_panel_plan(LargePanelRules(mesh), T(_large_panel_mixed_k(mesh)))
    tables = BeatEngineCore._metal_large_panel_tables(plan, triangle_rule(T, 4), eachindex(mesh.faces))
    try
        spans = BeatEngineCore._metal_large_panel_spans(tables, 5, 20)
        @test spans == [(5, 20, 1, length(mesh.faces))]
    finally
        BeatEngineCore._release_metal_large_panel_tables!(tables)
    end
    for counts in (fill(321, 20_000), [fill(6, 5_000); fill(321, 15_000)], [fill(6, 15_000); fill(321, 5_000)])
        tables = BeatEngineCore.MetalLargePanelTables(nothing, nothing, nothing, counts, sum(counts))
        chunk_start, chunk_count = 15_001, 3
        spans = BeatEngineCore._metal_large_panel_spans(tables, chunk_start, chunk_count)
        @test length(spans) > 1
        seen = zeros(Int, length(counts), chunk_count)
        for (trial_start, trial_count, test_start, test_count) in spans
            trials = trial_start:(trial_start + trial_count - 1)
            tests = test_start:(test_start + test_count - 1)
            @test sum(counts[trials]) * sum(counts[tests]) <= BeatEngineCore._METAL_LARGE_PANEL_LAUNCH_POINT_PAIRS
            @test first(trials) >= chunk_start && last(trials) < chunk_start + chunk_count
            @test first(tests) >= 1 && last(tests) <= length(counts)
            seen[tests, trials .- chunk_start .+ 1] .+= 1
        end
        @test all(==(1), seen)
        for (trial_start, trial_count) in unique([(span[1], span[2]) for span in spans])
            loads = [sum(counts[test_start:(test_start + test_count - 1)])
                     for (start, count, test_start, test_count) in spans
                     if start == trial_start && count == trial_count]
            @test maximum(loads) - minimum(loads) <= 2maximum(counts)
        end
    end
end

@testset "Splitting both dimensions preserves the pair blocks" begin
    core = BeatEngineCore
    T = Float32
    mesh = _large_panel_sphere(T; symmetry=:x)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(T, 4)
    k = T(_large_panel_mixed_k(mesh))
    plan = large_panel_plan(LargePanelRules(mesh), k)
    cache = build_metal_regular_assembly_cache(mesh, p1, dp0, rule; singular_order=4, symmetry_mode=:x)
    tables = core._metal_large_panel_tables(plan, rule, cache.element_indices)
    blocks = nothing
    try
        # Only scheduling sees these inflated costs; both launches use the same points.
        counts = fill(10_000, length(cache.element_indices))
        split = core.MetalLargePanelTables(tables.points4, tables.coords4, tables.offsets, counts, sum(counts))
        chunk_start, chunk_count = 5, 20
        spans = core._metal_large_panel_spans(split, chunk_start, chunk_count)
        @test any(span -> span[3] > 1, spans)
        @test any(span -> span[1] > chunk_start, spans)
        pair_stride = Int32(length(cache.element_indices) * chunk_count)
        blocks = Metal.zeros(Float32, core._METAL_FUSED_COMPONENTS * Int(pair_stride))
        packed = core._metal_packed_pair_tables_for(cache)
        gather = core._metal_fused_gather_tables(cache)
        transforms = Any[(cache.vertex_offsets, cache.incident_elements, Int32(0),
                          one(T), one(T), one(T), one(T), one(T), one(T))]
        for (transform, image) in zip(cache.image_transforms, cache.image_singular_caches)
            push!(transforms, (image.pair_offsets, image.trial_indices, Int32(1),
                T(transform.signs[1]), T(transform.signs[2]), T(transform.signs[3]),
                T(transform.determinant * transform.signs[1]),
                T(transform.determinant * transform.signs[2]),
                T(transform.determinant * transform.signs[3])))
        end
        results = map((tables, split)) do active
            fill!(blocks, 0.0f0)
            core._launch_metal_large_panel_kernels!(blocks, packed, cache, gather, active, k, transforms,
                chunk_start, chunk_count, pair_stride, 8, 4)
            Metal.synchronize()
            Array(blocks)
        end
        @test !iszero(norm(first(results)))
        @test reinterpret(UInt32, first(results)) == reinterpret(UInt32, last(results))
    finally
        blocks === nothing || Metal.unsafe_free!(blocks)
        core._release_metal_large_panel_tables!(tables)
        release_metal_regular_assembly_cache!(cache)
    end
end
