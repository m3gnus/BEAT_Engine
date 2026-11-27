# Gates the per-panel integration of electrically large panels; standalone or after runtests.jl.
using Test
using StaticArrays
using LinearAlgebra

if !isdefined(Main, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore

include(joinpath(@__DIR__, "large_panel_test_helpers.jl"))

const _LP = BeatEngineCore

@testset "Large-panel plan" begin
    mesh, h = _large_panel_pair_mesh(Float64)
    rules = LargePanelRules(mesh, 4)
    limit = _LP.beat_large_panel_kh_limit()
    @test large_panel_plan(rules, 0.99 * limit / h) === nothing
    plan = large_panel_plan(rules, 16.0 / h)
    @test all(>(0), plan.panel_orders)
    @test large_panel_summary(plan).panels == 2
    @test large_panel_summary(nothing) == (panels=0, max_kh=nothing, max_order=0)
    # The negative-time convention hands the kernels a negative wavenumber.
    @test large_panel_plan(rules, -16.0 / h).panel_orders == plan.panel_orders

    # Each panel decides from its own size, so a mesh can have both kinds.
    mixed = _large_panel_sphere(Float64)
    mixed_rules = LargePanelRules(mixed)
    edges = mixed_rules.max_edges
    mixed_plan = large_panel_plan(mixed_rules, 0.5 * (limit / minimum(edges) + limit / maximum(edges)))
    @test any(==(0), mixed_plan.panel_orders) && any(>(0), mixed_plan.panel_orders)
    @test issorted(_LP.large_panel_regular_order.(collect(6.5:0.5:40.0)))
    @test _LP.large_panel_regular_order(1.0e6) == _LP.BEAT_LARGE_PANEL_MAX_ORDER
    @test _LP.large_panel_singular_order(16.0, 4) == _LP.large_panel_regular_order(16.0) + 1
    @test _LP.large_panel_singular_order(1.0e6, 4) == _LP.BEAT_LARGE_PANEL_MAX_SINGULAR_ORDER == 16
    @test _LP.large_panel_singular_order(1.0e6, 20) == 20
    capped = large_panel_plan(rules, 1.0e6 / h)
    @test all(==(_LP.BEAT_LARGE_PANEL_MAX_ORDER), capped.panel_orders)
    @test all(==(_LP.BEAT_LARGE_PANEL_MAX_SINGULAR_ORDER), capped.singular_orders)
    @test all(==(20), large_panel_plan(LargePanelRules(mesh, 20), 1.0e6 / h).singular_orders)

    withenv("BLAB_BEAT_LARGE_PANEL_KH" => "0") do
        @test large_panel_plan(rules, 1000.0) === nothing
    end
    withenv("BLAB_BEAT_LARGE_PANEL_KH" => "nonsense") do
        @test_throws ErrorException large_panel_plan(rules, 1000.0)
    end
end

@testset "Correction chunks deal the expensive tests within each colour" begin
    costs = [fill(1000, 8); fill(10, 24)]
    groups = [collect(1:32), [33, 34], Int[]]
    append!(costs, [20, 10])
    original = deepcopy(groups)
    owners = zeros(Int, length(costs))
    _LP._beat_cpu_panel_regular_groups!(groups, collect(1:4), true, costs) do index, chunk
        owners[index] = chunk
    end
    @test groups == original
    @test all(>(0), owners)
    @test owners[1:32] == repeat(1:4, 8)
    loads = [sum(costs[index] for index in groups[1] if owners[index] == chunk) for chunk in 1:4]
    @test maximum(loads) == minimum(loads)
    @test owners[33:34] == [1, 2]
end

@testset "Dealt Float64 corrections are bitwise identical to serial corrections" begin
    for symmetry in (:off, :x)
        mesh = _large_panel_sphere(Float64; symmetry=symmetry, level=1)
        p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
        rule = triangle_rule(Float64, 4)
        k = _large_panel_mixed_k(mesh)
        plan = large_panel_plan(LargePanelRules(mesh), k)
        cache = _LP.build_beat_cpu_assembly_cache(mesh, p1, dp0, rule; symmetry_mode=symmetry)
        (; elements, regular_quadrature, indices, image_transforms) = cache
        passes = _LP.BeatCpuPanelPasses(mesh, plan, elements, regular_quadrature, image_transforms)
        groups = _LP._beat_cpu_element_color_groups(mesh, indices)
        costs = _LP._beat_cpu_panel_regular_costs(passes, plan, indices)
        large = [index for index in indices if plan.panel_orders[index] != 0]
        small = [index for index in indices if plan.panel_orders[index] == 0]
        @test !isempty(large) && !isempty(small)
        @test costs[first(large)] == length(passes.quadrature[first(large)].points) * sum(length(q.points) for q in passes.quadrature)
        @test costs[first(small)] == 6sum(length(passes.quadrature[index].points) for index in large) + 36length(large)
        q = ComplexF64[complex(sin(0.7 * row * drive), cos(0.4 * row + drive))
                       for row in 1:dp0.global_dof_count, drive in 1:2]
        function correction(threaded, chunk_count)
            lhs = zeros(ComplexF64, p1.global_dof_count, p1.global_dof_count)
            rhs = zeros(ComplexF64, p1.global_dof_count, size(q, 2))
            scratch = [_LP.BeatCpuRegularScratch{Float64}(_LP._BEAT_CPU_REGULAR_BLOCK_SIZE) for _ in 1:chunk_count]
            _LP._beat_cpu_panel_regular_pass!(
                elements, passes, plan, indices, image_transforms, groups, threaded, scratch,
            ) do test_data, test_quad, trial_elements, soa, work, skip_adjacent
                _LP._beat_cpu_bm_regular_test_simd!(
                    lhs, rhs, q, test_data, test_quad, trial_elements, soa, k,
                    _LP.burton_miller_coupling(k), work, skip_adjacent,
                )
            end
            return lhs, rhs
        end
        serial = correction(false, 1)
        @test all(values -> !iszero(norm(values)), serial)
        for chunk_count in 1:Threads.nthreads()
            dealt = correction(true, chunk_count)
            for (actual, expected) in zip(dealt, serial)
                @test reinterpret(UInt64, vec(actual)) == reinterpret(UInt64, vec(expected))
            end
        end
    end
end

@testset "A frequency without a plan runs the unplanned code" begin
    # One order of magnitude below the limit: nothing to correct, for any caller.
    mesh = _large_panel_sphere(Float32)
    rules = LargePanelRules(mesh)
    k = Float32(0.5 * _LP.beat_large_panel_kh_limit() / maximum(rules.max_edges))
    @test large_panel_plan(rules, k) === nothing
    rule = triangle_rule(Float32, 4)
    planned = build_field_evaluation_cache(mesh, rule; large_panels=large_panel_plan(rules, k))
    plain = build_field_evaluation_cache(mesh, rule)
    for name in fieldnames(typeof(plain))
        @test getfield(planned, name) == getfield(plain, name)
    end
end

# Each panel partitioned and summed with the base rule, projected back onto the
# original basis. This shares no code with the tensor rule, adds no solution
# freedom, and is the check the rule has to agree with.
@testset "Regular blocks against partitioned panels" begin
    for aspect in (1.0, 4.0), kh in (8.0, 16.0, 24.0)
        mesh, h = _large_panel_pair_mesh(Float64; aspect=aspect)
        k = kh / h
        reference = _large_panel_partitioned_blocks(mesh, k, 5)
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(Float64, 4)
        plan = large_panel_plan(LargePanelRules(mesh, 4), k)
        before = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4)
        after = assemble_regular_galerkin_operators_cpu(
            mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4, large_panels=plan,
        )
        for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            actual = getproperty(after, name)[_large_panel_block(name)...]
            baseline = getproperty(before, name)[_large_panel_block(name)...]
            target = reference[name]
            @test norm(actual - target) <= 1.0e-3 * norm(target)
            kh >= 16 && @test norm(baseline - target) >= 0.5 * norm(target)
        end
    end
end

@testset "Touching pairs converge with the Duffy order" begin
    for mesh_of in (_large_panel_coincident_mesh, _large_panel_edge_mesh), kh in (8.0, 16.0)
        mesh, h = mesh_of(Float64)
        k = kh / h
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(Float64, 4)
        reference = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; skip_singular=false, singular_order=16)
        before = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4)
        after = assemble_regular_galerkin_operators_cpu(
            mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4,
            large_panels=large_panel_plan(LargePanelRules(mesh, 4), k),
        )
        for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            target = getproperty(reference, name)
            @test norm(getproperty(after, name) - target) <= 1.0e-3 * norm(target)
            @test norm(getproperty(after, name) - target) <= norm(getproperty(before, name) - target)
        end
    end
end

# The correction goes through the fused and the four-operator kernels, the
# image passes and the singular deltas, and a slip in any sign shows up here.
@testset "Fused and four-operator corrections agree" begin
    for symmetry in (:off, :x, :xy, :ground), T in (Float64, Float32), kernel in (:simd, :scalar)
        mesh = _large_panel_sphere(T; symmetry=symmetry, level=symmetry == :ground ? 1 : 2)
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(T, 4)
        k = T(_large_panel_mixed_k(mesh))
        plan = large_panel_plan(LargePanelRules(mesh, 4), k)
        @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
        q = Complex{T}[Complex{T}(sin(T(0.7 * row)), cos(T(0.4 * row))) for row in 1:dp0.global_dof_count, _ in 1:2]
        fused = assemble_burton_miller_neumann_system_cpu(
            mesh, p1, dp0, q, k, rule; identity_p1_p1, identity_p1_dp0, singular_order=4,
            symmetry_mode=symmetry, regular_kernel=kernel, singular_kernel=kernel, large_panels=plan,
        )
        operators = assemble_regular_galerkin_operators_cpu(
            mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4, symmetry_mode=symmetry,
            regular_kernel=kernel, large_panels=plan,
        )
        eta = BeatEngineCore.burton_miller_coupling(outgoing_wavenumber(k))
        matrix = T(0.5) * identity_p1_p1 - operators.double_layer + eta * operators.hypersingular
        rhs = (-operators.single_layer - eta * (operators.adjoint_double_layer + T(0.5) * identity_p1_dp0)) * q
        tol = T === Float32 ? 5.0e-4 : 1.0e-11
        @test maximum(abs, fused.matrix - matrix) <= tol * maximum(abs, matrix)
        @test maximum(abs, fused.rhs - rhs) <= tol * maximum(abs, rhs)
    end
end

# A reduced mesh and its explicit full mesh pair the same panels, except that the
# reduced one meets half of them through the image passes. They are different
# discretizations of touching image pairs (Duffy orientation), so they agree to
# the integration error, not to rounding: 2e-4 under the base rules on this mesh.
@testset "Symmetry reduction under a plan" begin
    T = Float64
    full = _large_panel_sphere(T; symmetry=:off)
    k = T(_large_panel_mixed_k(full))
    points = [SVector{3,T}(3cos(a), 3sin(a), 1.0) for a in (0.3, 1.1, 2.0, 3.7, 5.1)]
    reference = _large_panel_field(full, :off, k, points)
    for symmetry in (:x, :xy)
        @test _large_panel_field(_large_panel_sphere(T; symmetry=symmetry), symmetry, k, points) ≈ reference rtol = 1.0e-5
    end
    # A request's Duffy order can already exceed what a panel asks for. The
    # image-touching pairs' regular correction still has to come back out.
    high_order = _large_panel_field(full, :off, k, points; singular_order=12)
    @test _large_panel_field(_large_panel_sphere(T; symmetry=:x), :x, k, points; singular_order=12) ≈ high_order rtol = 1.0e-5
end

@testset "Large-panel blocks under both phasor conventions" begin
    for symmetry in (:off, :x)
        mesh = _large_panel_sphere(Float64; symmetry=symmetry, level=1)
        p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
        rule = triangle_rule(Float64, 4)
        k = _large_panel_mixed_k(mesh)
        rules = LargePanelRules(mesh, 4)
        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
        # Real, nonuniform data isolate conjugation of the assembled blocks.
        q = ComplexF64[sin(0.7 * row) for row in 1:dp0.global_dof_count, _ in 1:1]
        results = map((NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)) do convention
            with_phasor_convention(convention) do
                plan = large_panel_plan(rules, outgoing_wavenumber(k))
                @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
                fused = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; identity_p1_p1, identity_p1_dp0,
                    singular_order=4, symmetry_mode=symmetry, large_panels=plan,
                )
                operators = assemble_regular_galerkin_operators_cpu(
                    mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4,
                    symmetry_mode=symmetry, large_panels=plan,
                )
                return (; plan, fused, operators)
            end
        end
        negative, positive = results
        @test positive.plan.k == negative.plan.k
        @test positive.plan.panel_orders == negative.plan.panel_orders
        @test positive.plan.singular_orders == negative.plan.singular_orders
        @test positive.plan.panel_rules == negative.plan.panel_rules
        for name in (:matrix, :rhs)
            @test getproperty(positive.fused, name) ≈ conj.(getproperty(negative.fused, name)) rtol = 1.0e-11
        end
        for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            @test getproperty(positive.operators, name) ≈ conj.(getproperty(negative.operators, name)) rtol = 1.0e-11
        end
    end
end

@testset "Independent drives in the fused correction" begin
    for symmetry in (:off, :x)
        mesh = _large_panel_sphere(Float64; symmetry=symmetry, level=1)
        p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
        rule = triangle_rule(Float64, 4)
        k = _large_panel_mixed_k(mesh)
        plan = large_panel_plan(LargePanelRules(mesh, 4), k)
        @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
        identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
        identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
        q = ComplexF64[complex(sin(0.7 * row * column), cos(0.4 * row + column))
                       for row in 1:dp0.global_dof_count, column in 1:2]
        assemble(drives) = assemble_burton_miller_neumann_system_cpu(
            mesh, p1, dp0, drives, k, rule; identity_p1_p1, identity_p1_dp0,
            singular_order=4, symmetry_mode=symmetry, large_panels=plan,
        )
        together = assemble(q)
        for column in axes(q, 2)
            separate = assemble(q[:, column:column])
            @test together.matrix ≈ separate.matrix rtol = 1.0e-12
            @test together.rhs[:, column:column] ≈ separate.rhs rtol = 1.0e-12
        end
    end
end

@testset "Large panels refuse a second correction of the same pairs" begin
    mesh, h = _large_panel_pair_mesh(Float64)
    k = 16.0 / h
    plan = large_panel_plan(LargePanelRules(mesh, 4), k)
    near = build_near_correction_cache(mesh, [(1, 2)], 6)
    @test_throws ErrorException assemble_regular_galerkin_operators_cpu(
        mesh, build_p1_space(mesh), build_dp0_space(mesh), k, triangle_rule(Float64, 4);
        skip_singular=false, singular_order=4, near_correction_cache=near, large_panels=plan,
    )
end

@testset "One rules object serves different element subsets" begin
    T = Float64
    mesh = _large_panel_sphere(T)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(T, 4)
    k = T(_large_panel_mixed_k(mesh))
    shared = LargePanelRules(mesh, 4)
    subset(indices, rules) = assemble_regular_galerkin_operators_cpu(
        mesh, p1, dp0, k, rule; skip_singular=false, singular_order=4, element_indices=indices,
        large_panels=large_panel_plan(rules, k),
    ).hypersingular
    first_subset, second_subset = collect(1:40), collect(41:length(mesh.faces))
    subset(first_subset, shared)
    @test subset(second_subset, shared) == subset(second_subset, LargePanelRules(mesh, 4))
end

@testset "Field sources" begin
    for T in (Float64, Float32)
        mesh = _large_panel_sphere(T)
        rule = triangle_rule(T, 4)
        plan = large_panel_plan(LargePanelRules(mesh), T(_large_panel_mixed_k(mesh)))
        for symmetry in (:off,)
            cache = build_field_evaluation_cache(mesh, rule; symmetry_mode=symmetry, large_panels=plan)
            expected = sum(length(_LP._large_panel_rule(plan, rule, element).points) for element in eachindex(mesh.faces))
            @test length(cache.source_points) == expected
            for element in eachindex(mesh.faces)
                sources = findall(==(element), cache.source_elements)
                @test sum(cache.source_weights[sources]) ≈ mesh.areas[element] rtol = T === Float32 ? 1.0e-5 : 1.0e-12
            end
        end
    end
    # Same partitioned reference as the blocks: P1 and DP0 data are carried over exactly.
    mesh, h = _large_panel_pair_mesh(Float64)
    k = 16.0 / h
    pressure = ComplexF64[cis(0.4 * i) for i in eachindex(mesh.vertices)]
    neumann = ComplexF64[cis(0.9 * i) for i in eachindex(mesh.faces)]
    points = [SVector(1.0, 0.4, 0.2), SVector(-0.3, 0.9, 1.1), SVector(0.2, -1.2, -0.3), SVector(0.8, 0.8, 0.8)]
    fine = _large_panel_partitioned_field(mesh, pressure, neumann, k, points, 5)
    rule = triangle_rule(Float64, 4)
    before = evaluate_galerkin_field_cpu(points, mesh, pressure, neumann, k, build_field_evaluation_cache(mesh, rule))
    after = evaluate_galerkin_field_cpu(points, mesh, pressure, neumann, k, build_field_evaluation_cache(
        mesh, rule; large_panels=large_panel_plan(LargePanelRules(mesh), k),
    ))
    @test norm(after - fine) <= 1.0e-3 * norm(fine)
    @test norm(before - fine) >= 0.5 * norm(fine)
end

# Load the production entry points without starting a worker, as in the
# regular-quadrature driver gates. Direct includes avoid bundle precompilation.
module LargePanelCompiledDriver
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))
end
const _LargePanelCompiledDriver = LargePanelCompiledDriver

module LargePanelSourceDriver
using ..BeatEngineCore
include(joinpath(@__DIR__, "..", "BeatEngineDriver.jl"))
end

@testset "Retained pressure requests use this frequency's plan" begin
    driver = _LargePanelCompiledDriver
    core = driver.BeatEngineCore
    mktempdir() do directory
        # Pressure requests store geometry and observation points as Float32,
        # even when their coefficients and returned pressure are Float64.
        original = _large_panel_sphere(Float32; level=1)
        mesh = core.BoundaryMesh(SVector{3,Float64}.(original.vertices), original.faces, original.physical_tags)
        points = SVector{3,Float32}[(4, 1, 5), (-3, 2, 4)]
        pressure = ComplexF64[cis(0.4 * row) for row in eachindex(mesh.vertices)]
        neumann = ComplexF64[cis(0.9 * row) for row in eachindex(mesh.faces)]
        arrays = Dict(
            "boundary_points_m" => reduce(vcat, transpose.(collect.(original.vertices))),
            "boundary_triangles" => reduce(vcat, [transpose(collect(Int64, face) .- 1) for face in mesh.faces]),
            "points_m" => reduce(vcat, transpose.(collect.(points))),
            "boundary_pressure" => pressure, "boundary_normal_derivative" => neumann,
        )
        request = Dict{String,Any}(
            "domain_key" => basename(directory), "precision" => "float64", "bem_backend" => "cpu",
            "symmetry" => "off", "field_quantity" => "pressure", "sound_speed_m_per_s" => 343.0,
            "binary_array_schema_version" => 1,
            "binary_arrays" => Dict(name => _large_panel_binary_array(directory, name, values, driver)
                                   for (name, values) in arrays),
        )
        request_path = joinpath(directory, "request.json")
        high_k = _large_panel_mixed_k(original)
        low_k = 0.5 * _LP.beat_large_panel_kh_limit() / maximum(LargePanelRules(original).max_edges)
        cache_keys = String[]
        try
            for order in (nothing, 4)
                order === nothing ? delete!(request, "quadrature_order") : (request["quadrature_order"] = order)
                base_order = order === nothing ? 2 : order
                push!(cache_keys, "$(request["domain_key"])|Float64|cpu|off|q$base_order|pressure")
                retained = driver.retained_bem_field_evaluation_cache(request, request_path, Float64, :cpu, :off)
                @test retained.rule.points == core.triangle_rule(Float64, base_order).points
                @test retained.rule.weights == core.triangle_rule(Float64, base_order).weights
                if order === nothing
                    explicit = merge(request, Dict("quadrature_order" => 2))
                    @test driver.retained_bem_field_evaluation_cache(explicit, request_path, Float64, :cpu, :off) === retained
                end
                # Revisit the high frequency after the unplanned one on the same
                # retained entry: a plan must neither leak nor be lost to reuse.
                for target_k in (high_k, low_k, high_k)
                    request["frequency_hz"] = target_k * 343.0 / (2pi)
                    k = 2pi * request["frequency_hz"] / 343.0
                    plan = core.large_panel_plan(core.LargePanelRules(mesh), k)
                    if target_k == high_k
                        @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
                    else
                        @test plan === nothing
                    end
                    rule = core.triangle_rule(Float64, base_order)
                    cache = core.build_field_evaluation_cache(mesh, rule; large_panels=plan)
                    expected = core.evaluate_galerkin_field_cpu(SVector{3,Float64}.(points), mesh, pressure, neumann, k, cache)
                    actual = driver.evaluate_bem_field_request(request, request_path)
                    @test actual ≈ expected rtol = 1.0e-12
                    @test driver.retained_bem_field_evaluation_cache(request, request_path, Float64, :cpu, :off) === retained
                    if plan !== nothing
                        plain = core.evaluate_galerkin_field_cpu(
                            SVector{3,Float64}.(points), mesh, pressure, neumann, k, retained.field_cache,
                        )
                        @test norm(actual - plain) > 1.0e-5 * norm(actual)
                    end
                end
            end
        finally
            foreach(driver.release_bem_field_evaluation_cache!, cache_keys)
        end
    end
end

@testset "CPU compiled and source requests complete with large panels" begin
    driver = _LargePanelCompiledDriver
    json = driver.JSON
    mktempdir() do directory
        mesh = _large_panel_sphere(Float32; level=0)
        k = _large_panel_mixed_k(mesh)
        plan = large_panel_plan(LargePanelRules(mesh, 4), k)
        @test any(==(0), plan.panel_orders) && any(>(0), plan.panel_orders)
        path = joinpath(directory, "sphere.msh")
        _large_panel_write_mesh(path, mesh)
        frequency = k * 343.0 / (2pi)
        compiled = json.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "example-exterior-request.json"))
        compiled["compiled_system"]["meshes"][1]["file"] = path
        compiled["compiled_system"]["meshes"][1]["scale_to_m"] = 1.0
        compiled["compiled_system"]["boundaries"][1]["group"]["tag"] = 1
        compiled["frequencies_hz"] = [frequency]
        compiled["outputs"][1]["options"]["points_m"] = [[4.0, 1.0, 5.0]]
        compiled["solver_options"] = Dict(
            "precision" => "float32", "bem_backend" => "cpu", "symmetry" => "off",
            "quadrature_order" => 4, "singular_order" => 4,
            "regular_quadrature_mode" => "fixed", "burton_miller_assembly" => "direct_system",
        )
        source = json.parse(json.json(Dict(
            "beat_engine_backend" => "cpu", "frequencies_hz" => [frequency],
            "config" => Dict("mesh_file" => path, "scale_factor" => 1.0, "tag_throat" => 1,
                "symmetry" => "off", "min_angle" => 0.0, "max_angle" => 30.0, "step_size" => 30.0,
                "distance" => 6.0, "quadrature_order" => 4, "singular_order" => 4,
                "regular_quadrature_mode" => "fixed", "sound_speed" => 343.0),
        )))
        # The last solve's warning events, read after each solve below.
        warnings = Any[]
        record_warnings!(events) = append!(empty!(warnings),
            [event for event in events if event["type"] == "status" && get(event, "level", "") == "warning"])
        solve_compiled() = begin
            outcome, events = _large_panel_capture(json) do
                driver.solve_request(compiled; event_mode=true)
            end
            @test !outcome.cancelled && outcome.solved_count == 1
            @test last(events)["type"] == "result"
            record_warnings!(events)
            only(event["result"] for event in events if event["type"] == "result")
        end
        solve_source() = begin
            _, events = _large_panel_capture(json) do
                LargePanelSourceDriver.solve_request(source)
            end
            @test last(events)["type"] == "completed"
            record_warnings!(events)
            only(event["result"] for event in events if event["type"] == "result")
        end
        max_kh = k * maximum(LargePanelRules(mesh).max_edges)
        withenv("BLAB_BEAT_FUSED_BM" => "1", "BLAB_BEAT_DENSE_SOLVE" => "lu") do
            for (solve, values) in (
                (solve_compiled, result -> result["quantities"]),
                (solve_source, result -> [result[key] for key in ("horizontal_pressure", "vertical_pressure", "impedance")]),
            )
                planned = withenv(solve, "BLAB_BEAT_LARGE_PANEL_KH" => "6")
                @test planned["diagnostics"]["large_panel_count"] == count(>(0), plan.panel_orders)
                @test planned["diagnostics"]["large_panel_max_order"] == maximum(plan.panel_orders)
                @test planned["diagnostics"]["large_panel_correction"] == "applied"
                @test planned["diagnostics"]["large_panel_max_kh"] ≈ max_kh rtol = 1.0e-5
                @test isempty(warnings)
                disabled = withenv(solve, "BLAB_BEAT_LARGE_PANEL_KH" => "0")
                @test disabled["diagnostics"]["large_panel_count"] == 0
                @test disabled["diagnostics"]["large_panel_correction"] == "disabled"
                @test disabled["diagnostics"]["large_panel_max_kh"] ≈ max_kh rtol = 1.0e-5
                @test isempty(warnings)
                # Same frequency and physics, with every panel below the limit.
                below = withenv(solve, "BLAB_BEAT_LARGE_PANEL_KH" => string(2max_kh))
                @test below["diagnostics"]["large_panel_count"] == 0
                @test below["diagnostics"]["large_panel_correction"] == "not_needed"
                @test isempty(warnings)
                @test values(disabled) == values(below)
                @test values(planned) != values(disabled)
            end
            source["config"]["near_correction_enabled"] = true
            source["config"]["near_correction_order"] = 4
            near = withenv(solve_source, "BLAB_BEAT_LARGE_PANEL_KH" => "6")
            @test near["diagnostics"]["near_correction_enabled"]
            @test near["diagnostics"]["large_panel_count"] == count(>(0), plan.panel_orders)
            @test near["diagnostics"]["large_panel_max_order"] == 0
            @test near["diagnostics"]["large_panel_correction"] == "near_correction"
            @test length(warnings) == 1
            @test occursin("near corrections", only(warnings)["message"])
        end
    end
end

@testset "Large-panel report and warning on every path" begin
    mesh, h = _large_panel_pair_mesh(Float64)
    rules = LargePanelRules(mesh, 4)
    limit = _LP.beat_large_panel_kh_limit()
    k = 16.0 / h
    plan = large_panel_plan(rules, k)
    applied = large_panel_report(rules, k, plan, LARGE_PANEL_UNSUPPORTED_BACKEND)
    @test applied.correction == "applied"
    @test applied.panels == count(>(0), plan.panel_orders) > 0
    @test applied.max_order == maximum(plan.panel_orders)
    @test applied.max_kh ≈ k * maximum(rules.max_edges)
    @test large_panel_warning(applied, 1000.0) === nothing
    # The selection the drivers make for CUDA, ROCm, coupled and Deploy solves:
    # no plan, the same panels counted, and a warning naming why.
    for (reason, phrase) in ((LARGE_PANEL_UNSUPPORTED_BACKEND, "backend or solve path"),
                             (LARGE_PANEL_NEAR_CORRECTION, "near corrections"))
        skipped = large_panel_report(rules, k, nothing, reason)
        @test skipped.correction == reason
        @test skipped.panels == applied.panels
        @test skipped.max_order == 0
        @test skipped.max_kh == applied.max_kh
        message = large_panel_warning(skipped, 1000.0)
        @test occursin(phrase, message) && occursin("1000.0 Hz", message)
        @test large_panel_diagnostics(skipped)["large_panel_correction"] == reason
    end
    below = large_panel_report(rules, 0.5 * limit / h, nothing, LARGE_PANEL_UNSUPPORTED_BACKEND)
    @test below.correction == "not_needed" && below.panels == 0
    @test large_panel_warning(below, 1000.0) === nothing
    withenv("BLAB_BEAT_LARGE_PANEL_KH" => "0") do
        disabled = large_panel_report(rules, k, nothing, LARGE_PANEL_UNSUPPORTED_BACKEND)
        @test disabled.correction == "disabled" && disabled.panels == 0
        @test large_panel_warning(disabled, 1000.0) === nothing
    end
    @test_throws ArgumentError _LP._large_panel_require_singular_order(plan, 6)
    @test _LP._large_panel_require_singular_order(plan, 4) === nothing
    @test _LP._large_panel_require_singular_order(nothing, 6) === nothing

    # The compiled driver warns once per request, as an event or on stderr.
    driver = _LargePanelCompiledDriver
    skipped = large_panel_report(rules, k, nothing, LARGE_PANEL_UNSUPPORTED_BACKEND)
    warned, events = _large_panel_capture(driver.JSON) do
        first = driver.emit_large_panel_warning(skipped, 1000.0, false, true)
        second = driver.emit_large_panel_warning(skipped, 2000.0, first, true)
        quiet = driver.emit_large_panel_warning(applied, 1000.0, false, true)
        (first, second, quiet)
    end
    @test warned == (true, true, false)
    @test length(events) == 1
    @test only(events)["type"] == "status" && only(events)["level"] == "warning"
end

@testset "Assembly refuses a plan built for another singular order" begin
    mesh = _large_panel_sphere(Float64; level=0)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(Float64, 4)
    k = _large_panel_mixed_k(mesh)
    plan = large_panel_plan(LargePanelRules(mesh, 4), k)
    @test_throws ArgumentError assemble_regular_galerkin_operators_cpu(
        mesh, p1, dp0, k, rule; skip_singular=false, singular_order=6, large_panels=plan,
    )
    q = ComplexF64[1.0 + 0.0im for _ in 1:dp0.global_dof_count, _ in 1:1]
    @test_throws ArgumentError assemble_burton_miller_neumann_system_cpu(
        mesh, p1, dp0, q, k, rule; singular_order=6, large_panels=plan,
        identity_p1_p1=assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1),
        identity_p1_dp0=assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0),
    )
end
