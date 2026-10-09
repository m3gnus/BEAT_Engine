include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
include(joinpath(@__DIR__, "condensed_backend_tests.jl"))

@testset "Schur condensation algebra" begin
    system, operators, retained = condensed_synthetic_case(Float64)
    vertex_count = size(system, 1)
    interior = setdiff(1:vertex_count, retained)

    condensation = BeatEngineCoupledCondensed._build_condensation(system, operators, retained)

    @test condensation.interior_count == length(interior)
    @test condensation.retained_count == length(retained)
    @test condensation.interior_vertices == interior
    @test condensation.retained_vertices == retained
    @test eltype(condensation.schur) == ComplexF64
    @test size(condensation.schur) == (length(retained), length(retained))

    # The Schur complement is unnegated, in the same sign convention as `system`. Computed here
    # independently and densely.
    expected_schur = (
        Matrix(system[retained, retained]) -
        Matrix(system[retained, interior]) *
        (Matrix(system[interior, interior]) \ Matrix(system[interior, retained]))
    )
    @test condensation.schur ≈ expected_schur rtol = 1e-10

    # Manufactured solution: plant u = 1, form f = A u, and require the full round trip
    # (forward -> reduced dense solve -> backward) to recover it.
    planted = ones(ComplexF64, vertex_count, 1)
    forcing = Matrix(system * planted)
    reduced_rhs, interior_rhs = BeatEngineCoupledCondensed._forward_schur(condensation, forcing)
    @test size(reduced_rhs) == (length(retained), 1)
    recovered, interior_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        interior_rhs,
        condensation.schur \ reduced_rhs,
    )
    @test size(recovered) == (vertex_count, 1)
    @test norm(recovered - planted) / norm(planted) < 1e-10
    @test length(interior_residual) == 1
    @test maximum(interior_residual) < 1e-10

    # Two excitations at once, the second scaled, must scale linearly.
    multi_forcing = hcat(forcing, 0.5 .* forcing)
    multi_reduced, multi_interior = BeatEngineCoupledCondensed._forward_schur(condensation, multi_forcing)
    multi_recovered, multi_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        multi_interior,
        condensation.schur \ multi_reduced,
    )
    @test norm(multi_recovered[:, 1] - planted[:, 1]) / norm(planted) < 1e-10
    @test norm(multi_recovered[:, 2] - 0.5 .* planted[:, 1]) / norm(planted) < 1e-10
    @test length(multi_residual) == 2
    @test maximum(multi_residual) < 1e-10

    # A voltage-driven transducer forces the coupled system entirely through the electrical
    # rows, so the FEM interior sees f_I = 0. The residual must stay at round-off there rather
    # than normalizing against nothing.
    gamma_only_forcing = zeros(ComplexF64, vertex_count, 1)
    gamma_only_forcing[retained, 1] .= 1 .+ 0.5im
    @test iszero(gamma_only_forcing[interior, 1])
    gamma_reduced, gamma_interior = BeatEngineCoupledCondensed._forward_schur(
        condensation,
        gamma_only_forcing,
    )
    gamma_recovered, gamma_residual = BeatEngineCoupledCondensed._backward_schur(
        condensation,
        gamma_interior,
        condensation.schur \ gamma_reduced,
    )
    @test maximum(gamma_residual) < 1e-10
    # With f_I = 0 the interior rows must annihilate the recovered solution, so measure that
    # absolutely against the solution magnitude rather than against the zero forcing.
    @test norm(system[interior, :] * gamma_recovered[:, 1]) / norm(gamma_recovered[:, 1]) < 1e-10
    @test norm(gamma_recovered[:, 1]) > 0

    # The chunked column sweep runs one task per block against its own copy of the UMFPACK
    # factorization, so the partition must not move the answer. Every column is solved and
    # accumulated independently of the block it lands in, which makes the agreement exact and not
    # merely close -- two tasks sharing a solve workspace could not survive that. Run this file
    # under `julia -t <n>` for the comparison to actually cover concurrent tasks; at the default
    # single thread it still checks that the block partition itself is neutral.
    single_shot = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=length(retained),
    )
    narrow = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=3,
    )
    # One block per column: more blocks than threads, so the tasks stride over several each.
    per_column = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=1,
    )
    @test single_shot.schur ≈ condensation.schur rtol = 1e-12
    @test narrow.schur ≈ condensation.schur rtol = 1e-12
    @test single_shot.schur == condensation.schur
    @test narrow.schur == condensation.schur
    @test per_column.schur == condensation.schur
    # A block wider than Γ is clamped to it rather than overrunning the accumulator.
    wide = BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=4 * length(retained),
    )
    @test wide.schur == condensation.schur
    # The requested width is an upper bound. It is narrowed until Γ splits into at least one
    # block per thread, so no thread sits idle through the sweep whatever the interface size.
    # Stated as a property because the answer depends on the thread count this file is run under.
    for requested in (1, 3, length(retained), 4 * length(retained))
        built = BeatEngineCoupledCondensed._build_condensation(
            system,
            operators,
            retained;
            schur_block_columns=requested,
        )
        @test built.schur == condensation.schur
        @test 1 <= built.schur_block_columns <= requested
        @test cld(length(retained), built.schur_block_columns) >=
              min(Threads.nthreads(), length(retained))
    end
    @test_throws ErrorException BeatEngineCoupledCondensed._build_condensation(
        system,
        operators,
        retained;
        schur_block_columns=0,
    )

    # Mixed precision: the interior is factored in ComplexF64 whatever T is, and only the
    # assembled Schur complement is demoted.
    system32, operators32, retained32 = condensed_synthetic_case(Float32)
    condensation32 = BeatEngineCoupledCondensed._build_condensation(system32, operators32, retained32)
    @test eltype(condensation32.schur) == ComplexF32
    @test eltype(condensation32.interior_system) == ComplexF64
    @test eltype(condensation32.factorization) == ComplexF64
    planted32 = ones(ComplexF32, size(system32, 1), 1)
    forcing32 = Matrix(system32 * planted32)
    reduced32, interior32 = BeatEngineCoupledCondensed._forward_schur(condensation32, forcing32)
    @test eltype(reduced32) == ComplexF32
    # Under double dense assembly the reduced right-hand side is returned before demotion.
    reduced32_double, _ = BeatEngineCoupledCondensed._forward_schur(condensation32, forcing32; result_type=Float64)
    @test eltype(reduced32_double) == ComplexF64
    @test ComplexF32.(reduced32_double) == reduced32
    @test any(value -> ComplexF64(ComplexF32(value)) != value, reduced32_double)
    recovered32, residual32 = BeatEngineCoupledCondensed._backward_schur(
        condensation32,
        interior32,
        condensation32.schur \ reduced32,
    )
    @test eltype(recovered32) == ComplexF32
    @test norm(recovered32 - planted32) / norm(planted32) < 1e-4
    @test eltype(residual32) == Float32
    @test maximum(residual32) < 1e-5

    # A mesh whose every FEM vertex lies on a retained interface or moving surface has no
    # interior block. Condensation is then an exact no-op rather than an invalid 0x0 UMFPACK
    # factorization.
    all_retained_system, all_retained_operators, _ = condensed_synthetic_case(
        Float64;
        vertex_count=10,
        retained_count=10,
    )
    all_retained = collect(1:size(all_retained_system, 1))
    all_retained_condensation = BeatEngineCoupledCondensed._build_condensation(
        all_retained_system,
        all_retained_operators,
        all_retained,
    )
    @test all_retained_condensation.backend == :cpu_noop
    @test all_retained_condensation.interior_count == 0
    @test all_retained_condensation.schur == Matrix(all_retained_system)
    all_retained_planted = ones(ComplexF64, size(all_retained_system, 1), 1)
    all_retained_forcing = Matrix(all_retained_system * all_retained_planted)
    all_retained_rhs, all_retained_interior = BeatEngineCoupledCondensed._forward_schur(
        all_retained_condensation,
        all_retained_forcing,
    )
    all_retained_recovered, all_retained_residual = BeatEngineCoupledCondensed._backward_schur(
        all_retained_condensation,
        all_retained_interior,
        all_retained_condensation.schur \ all_retained_rhs,
    )
    @test all_retained_recovered ≈ all_retained_planted rtol = 1e-10
    @test maximum(all_retained_residual) == 0
    BeatEngineCoupledCondensed._release_condensation!(all_retained_condensation)

    # The structural guard that makes the condensation valid: interface loads must not touch
    # eliminated interior vertices.
    violating_system, violating_operators, violating_retained =
        condensed_synthetic_case(Float64; interior_load=true)
    @test_throws ErrorException BeatEngineCoupledCondensed._build_condensation(
        violating_system,
        violating_operators,
        violating_retained,
    )
end

if get(ENV, "BLAB_RUN_COUPLED_REFERENCE", "0") == "1"
    with_baseline_cpu_coupled() do
        @testset "Condensed coupled solver matches monolithic" begin
            fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
            bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
            interface_map = build_conforming_interface_map(
                fem_mesh,
                bem_mesh,
                physical_tag(fem_mesh, 2, "Interface"),
                2,
            )
            radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
            fem_mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
            bem_mesh32 = load_gmsh22_with_tags(
                joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"),
                Float32(0.001),
            )
            interface_map32 = build_conforming_interface_map(
                fem_mesh32,
                bem_mesh32,
                physical_tag(fem_mesh32, 2, "Interface"),
                2,
            )

            function monolithic_system(::Type{T}, frequency; transducers=ElectrodynamicTransducer{T}[]) where {T}
                mesh, boundary, mapping = T === Float32 ?
                                          (fem_mesh32, bem_mesh32, interface_map32) :
                                          (fem_mesh, bem_mesh, interface_map)
                return build_coupled_system(
                    mesh,
                    boundary,
                    mapping,
                    T(frequency),
                    T(343.0),
                    T(1.21);
                    quadrature_order=CONDENSED_QUADRATURE_ORDER,
                    singular_order=CONDENSED_SINGULAR_ORDER,
                    validation_diagnostics=false,
                    bem_backend=:cpu,
                    transducers=transducers,
                )
            end

            function condensed_system_at(::Type{T}, frequency; transducers=ElectrodynamicTransducer{T}[], extra...) where {T}
                mesh, boundary, mapping = T === Float32 ?
                                          (fem_mesh32, bem_mesh32, interface_map32) :
                                          (fem_mesh, bem_mesh, interface_map)
                return build_condensed_coupled_system(
                    mesh,
                    boundary,
                    mapping,
                    T(frequency),
                    T(343.0),
                    T(1.21);
                    quadrature_order=CONDENSED_QUADRATURE_ORDER,
                    singular_order=CONDENSED_SINGULAR_ORDER,
                    transducers=transducers,
                    extra...,
                )
            end

            relative_error(reference, candidate) = norm(
                ComplexF64.(candidate) .- ComplexF64.(reference),
            ) / norm(ComplexF64.(reference))

            monolithic = monolithic_system(Float64, 500.0)
            condensed = condensed_system_at(Float64, 500.0)
            try
                @test monolithic.formulation == :monolithic
                @test condensed.formulation == :fem_interface_condensed
                @test condensed.solved_system_order < condensed.full_system_order
                @test condensed.full_system_order == monolithic.full_system_order
                @test condensed.linear_backend == :cpu
                @test eltype(condensed.condensation.schur) == ComplexF64
                @test condensed.condensation.retained_count == length(interface_map.fem_vertex_indices)

                reference = only(solve_coupled_systems(monolithic, [radiator_tag]))
                candidates = solve_condensed_coupled_systems(
                    condensed,
                    [radiator_tag, radiator_tag];
                    radiator_velocities=ComplexF64[1, 0.5],
                )
                candidate = candidates[1]

                # Both sides are CPU double and differ only in elimination order.
                @test relative_error(reference.fem_pressure, candidate.fem_pressure) < 1e-9
                @test relative_error(reference.bem_pressure, candidate.bem_pressure) < 1e-9
                @test relative_error(reference.interface_flux, candidate.interface_flux) < 1e-9
                @test relative_error(reference.bem_neumann, candidate.bem_neumann) < 1e-9
                @test candidate.fem_interior_residual < 1e-10
                @test isnothing(candidate.relative_residual)
                @test candidate.pressure_continuity_error < 1e-8
                @test candidate.flux_conservation_error < 1e-10
                @test relative_error(0.5 .* candidate.bem_pressure, candidates[2].bem_pressure) < 1e-9
                @test candidates[2].fem_interior_residual < 1e-10

                condensed32 = condensed_system_at(Float32, 500.0)
                try
                    @test eltype(condensed32.condensation.schur) == ComplexF32
                    # The interior factorization stays double whatever T is.
                    @test eltype(condensed32.condensation.interior_system) == ComplexF64
                    solution32 = solve_condensed_coupled_system(
                        condensed32,
                        physical_tag(fem_mesh32, 2, "Radiator"),
                    )
                    @test relative_error(candidate.fem_pressure, solution32.fem_pressure) < 1e-4
                    @test relative_error(candidate.bem_pressure, solution32.bem_pressure) < 1e-4
                    @test relative_error(candidate.interface_flux, solution32.interface_flux) < 1e-4
                    @test solution32.fem_interior_residual < 1e-5
                finally
                    release_condensed_coupled_system!(condensed32)
                end
            finally
                release_coupled_system!(monolithic)
                release_condensed_coupled_system!(condensed)
            end

            # Γ is `interface ∪ transducer_fem_vertices`, so a transducer widens the retained set
            # and exercises the retained-vertex slicing of the transducer blocks.
            transducer = ElectrodynamicTransducer{Float64}(
                "component:test",
                [radiator_tag],
                [1.0],
                [1],
                [-1.0],
                SVector(0.0, 0.0, 1.0),
                2.0,
                1,
                6.0,
                0.0005,
                7.0,
                0.015,
                0.0005,
                1.0,
            )
            transducer_monolithic = monolithic_system(Float64, 500.0; transducers=[transducer])
            transducer_condensed = condensed_system_at(Float64, 500.0; transducers=[transducer])
            try
                @test transducer_condensed.condensation.retained_count >
                      length(interface_map.fem_vertex_indices)
                @test transducer_condensed.solved_system_order < transducer_condensed.full_system_order
                voltage = (
                    kind=:voltage,
                    radiator_tag=0,
                    transducer_index=1,
                    amplitude=ComplexF64(1, 0),
                )
                transducer_reference = only(solve_coupled_excitations(transducer_monolithic, [voltage]))
                transducer_candidate = only(
                    solve_condensed_coupled_excitations(transducer_condensed, [voltage]),
                )
                @test relative_error(
                    transducer_reference.fem_pressure,
                    transducer_candidate.fem_pressure,
                ) < 1e-9
                @test relative_error(
                    transducer_reference.bem_pressure,
                    transducer_candidate.bem_pressure,
                ) < 1e-9
                @test relative_error(
                    transducer_reference.diaphragm_velocity,
                    transducer_candidate.diaphragm_velocity,
                ) < 1e-9
                @test relative_error(
                    transducer_reference.voice_coil_current,
                    transducer_candidate.voice_coil_current,
                ) < 1e-9
                @test transducer_candidate.fem_interior_residual < 1e-10

                # Γ is built from the interface map and transducer surfaces only, neither of which
                # depends on symmetry — symmetry enters only the BEM operators — so the retained set,
                # and with it the condensation, is invariant under :x and :xy.
                transducer_fem_vertices = unique(
                    findnz(
                        assemble_transducer_operators(fem_mesh, bem_mesh, [transducer]).fem_surface,
                    )[1],
                )
                @test transducer_condensed.retained_fem_vertices ==
                      sort(unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)))
                @test transducer_condensed.retained_fem_vertices ==
                      transducer_monolithic.retained_fem_vertices
            finally
                release_coupled_system!(transducer_monolithic)
                release_condensed_coupled_system!(transducer_condensed)
            end

            # Transducer condensation eliminates the transducer surfaces with the interior and carries
            # their rank-one coupling as low-rank columns. It must reproduce the monolithic solve to
            # round-off, including a prescribed-velocity load that lands on the eliminated vertices.
            voltage_excitation = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))
            velocity_excitation = (
                kind=:normal_velocity,
                radiator_tag=radiator_tag,
                transducer_index=0,
                amplitude=ComplexF64(0.3, -0.2),
            )
            reference_monolithic = monolithic_system(Float64, 500.0; transducers=[transducer])
            retained_condensed = condensed_system_at(Float64, 500.0; transducers=[transducer])
            eliminated_condensed = withenv("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1") do
                condensed_system_at(Float64, 500.0; transducers=[transducer])
            end
            try
                @test !retained_condensed.transducer_condensation
                @test eliminated_condensed.transducer_condensation
                @test eliminated_condensed.condensation.transducer_condensed
                @test eliminated_condensed.condensation.retained_count ==
                      length(unique(interface_map.fem_vertex_indices))
                @test eliminated_condensed.gamma_fem_vertices == sort(unique(interface_map.fem_vertex_indices))
                @test eliminated_condensed.retained_fem_vertices == retained_condensed.retained_fem_vertices
                @test eliminated_condensed.solved_system_order ==
                      retained_condensed.solved_system_order -
                      (retained_condensed.condensation.retained_count -
                       eliminated_condensed.condensation.retained_count)
                @test eliminated_condensed.solved_system_order < retained_condensed.solved_system_order
                excitations = [voltage_excitation, velocity_excitation]
                references = solve_coupled_excitations(reference_monolithic, excitations)
                retained_solutions = solve_condensed_coupled_excitations(retained_condensed, excitations)
                eliminated_solutions = solve_condensed_coupled_excitations(eliminated_condensed, excitations)
                for (reference, retained_solution, candidate) in
                    zip(references, retained_solutions, eliminated_solutions)
                    for field in (:fem_pressure, :bem_pressure, :interface_flux, :diaphragm_velocity, :voice_coil_current)
                        @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                        @test relative_error(getproperty(retained_solution, field), getproperty(candidate, field)) < 1e-9
                    end
                    @test candidate.fem_interior_residual < 1e-10
                end
            finally
                release_coupled_system!(reference_monolithic)
                release_condensed_coupled_system!(retained_condensed)
                release_condensed_coupled_system!(eliminated_condensed)
            end

            # Interface elimination substitutes p_Γ = P p_B (pressure) and then
            # q = M_Γ⁻¹ (S P p_B + E y - g) (flux). Both are exact, so every output must match the
            # monolithic solve and the unmodified condensed layout to round-off.
            interface_count = length(interface_map.fem_vertex_indices)
            elimination_fields = (
                :fem_pressure, :bem_pressure, :interface_flux, :bem_neumann,
                :diaphragm_velocity, :voice_coil_current,
            )
            lever_one = "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1"
            pressure_switch = "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION" => "1"
            flux_switch = "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1"
            elimination_reference = monolithic_system(Float64, 700.0; transducers=[transducer])
            elimination_baseline = withenv(lever_one) do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            pressure_eliminated = withenv(lever_one, pressure_switch) do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            flux_eliminated = withenv(lever_one, flux_switch) do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            try
                @test elimination_baseline.interface_elimination == :none
                @test pressure_eliminated.interface_elimination == :pressure
                @test flux_eliminated.interface_elimination == :flux
                # Only the flux elimination solves with M_Γ; the unspecialized path is one LU.
                for (system, requested, solver, block_count) in (
                    (elimination_baseline, nothing, nothing, 0),
                    (pressure_eliminated, nothing, nothing, 0),
                    (flux_eliminated, "lu", "lu", 1),
                )
                    diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics(system)
                    @test diagnostics["interface_mass_solver_requested"] == requested
                    @test diagnostics["interface_mass_solver"] == solver
                    @test diagnostics["interface_mass_block_count"] == block_count
                    @test isempty(diagnostics["interface_mass_fallback_reasons"])
                end
                gamma_count = elimination_baseline.condensation.retained_count
                @test gamma_count == interface_count
                @test pressure_eliminated.solved_system_order ==
                      elimination_baseline.solved_system_order - gamma_count
                @test flux_eliminated.solved_system_order ==
                      elimination_baseline.solved_system_order - gamma_count - interface_count
                @test flux_eliminated.solved_system_order == length(bem_mesh.vertices) + 2
                excitations = [
                    voltage_excitation,
                    velocity_excitation,
                    (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(-0.4, 1.3)),
                ]
                references = solve_coupled_excitations(elimination_reference, excitations)
                baselines = solve_condensed_coupled_excitations(elimination_baseline, excitations)
                for candidate_system in (pressure_eliminated, flux_eliminated)
                    candidates = solve_condensed_coupled_excitations(candidate_system, excitations)
                    for (reference, baseline, candidate) in zip(references, baselines, candidates)
                        for field in elimination_fields
                            @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                            @test relative_error(getproperty(baseline, field), getproperty(candidate, field)) < 1e-9
                        end
                        @test candidate.fem_interior_residual < 1e-10
                        @test candidate.pressure_continuity_error == 0
                        @test candidate.flux_conservation_error < 1e-10
                    end
                end
                # The mass factorization is geometry-only and reused by the next frequency.
                @test !flux_eliminated.timings.interface_mass_cached
                flux_eliminated_next = withenv(lever_one, flux_switch) do
                    build_condensed_coupled_system(
                        fem_mesh, bem_mesh, interface_map, 900.0, 343.0, 1.21;
                        quadrature_order=CONDENSED_QUADRATURE_ORDER,
                        singular_order=CONDENSED_SINGULAR_ORDER,
                        transducers=[transducer],
                        cache=flux_eliminated.cache,
                    )
                end
                baseline_next = withenv(lever_one) do
                    condensed_system_at(Float64, 900.0; transducers=[transducer])
                end
                try
                    @test flux_eliminated_next.timings.interface_mass_cached
                    for (baseline, candidate) in zip(
                        solve_condensed_coupled_excitations(baseline_next, excitations),
                        solve_condensed_coupled_excitations(flux_eliminated_next, excitations),
                    )
                        for field in elimination_fields
                            @test relative_error(getproperty(baseline, field), getproperty(candidate, field)) < 1e-9
                        end
                    end
                finally
                    release_condensed_coupled_system!(flux_eliminated_next)
                    release_condensed_coupled_system!(baseline_next)
                end
            finally
                release_coupled_system!(elimination_reference)
                release_condensed_coupled_system!(elimination_baseline)
                release_condensed_coupled_system!(pressure_eliminated)
                release_condensed_coupled_system!(flux_eliminated)
            end

            # Specialized flux elimination: Cholesky mass solves, per-component blocks, mass solves in
            # the FEM stage, and the demand-driven interior reconstruction. All exact: every output
            # must match the unmodified flux elimination and the monolithic solve to round-off.
            specialized_switch_sets = (
                ("BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod",),
                ("BLAB_COUPLED_INTERFACE_BLOCKS" => "1",),
                ("BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1", "BLAB_COUPLED_STAGE_OVERLAP" => "on"),
                ("BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1", "BLAB_COUPLED_STAGE_OVERLAP" => "off"),
                (
                    "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod",
                    "BLAB_COUPLED_INTERFACE_BLOCKS" => "1",
                    "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1",
                    "BLAB_COUPLED_STAGE_OVERLAP" => "on",
                ),
            )
            specialized_excitations = [
                voltage_excitation,
                velocity_excitation,
                (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(-0.4, 1.3)),
            ]
            specialized_reference = monolithic_system(Float64, 700.0; transducers=[transducer])
            specialized_baseline = withenv(lever_one, flux_switch) do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            try
                references = solve_coupled_excitations(specialized_reference, specialized_excitations)
                baselines = solve_condensed_coupled_excitations(specialized_baseline, specialized_excitations)
                @test !hasproperty(specialized_baseline.interface_elimination_data, :mass_operator)
                for switches in specialized_switch_sets
                    candidate_system = withenv(lever_one, flux_switch, switches...) do
                        condensed_system_at(Float64, 700.0; transducers=[transducer])
                    end
                    try
                        @test hasproperty(candidate_system.interface_elimination_data, :mass_operator)
                        operator = candidate_system.interface_elimination_data.mass_operator
                        @test operator.count == interface_count
                        @test all(isnothing(block.fallback_reason) for block in operator.blocks)
                        @test operator.solver == (("BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod") in switches ? :cholmod : :lu)
                        diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics(candidate_system)
                        @test diagnostics["interface_mass_solver_requested"] == String(operator.solver)
                        @test diagnostics["interface_mass_solver"] == String(operator.solver)
                        @test diagnostics["interface_mass_block_count"] == length(operator.blocks)
                        @test isempty(diagnostics["interface_mass_fallback_reasons"])
                        in_stage = ("BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1") in switches
                        split = candidate_system.timings.interface_elimination_split
                        @test haskey(split, :fem_stage_mass_solve) == in_stage
                        @test haskey(split, :mass_solve) == !in_stage
                        candidates = solve_condensed_coupled_excitations(candidate_system, specialized_excitations)
                        skipped = solve_condensed_coupled_excitations(
                            candidate_system, specialized_excitations; reconstruct_interior=false,
                        )
                        interior = candidate_system.condensation.interior_vertices
                        retained = candidate_system.condensation.retained_vertices
                        for (reference, baseline, candidate, lean) in zip(references, baselines, candidates, skipped)
                            for field in elimination_fields
                                @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                                @test relative_error(getproperty(baseline, field), getproperty(candidate, field)) < 1e-9
                            end
                            @test candidate.fem_interior_residual < 1e-10
                            @test candidate.flux_conservation_error < 1e-10
                            # Skipped reconstruction: interior not evaluated, everything else identical.
                            @test all(isnan, lean.fem_pressure[interior])
                            @test lean.fem_pressure[retained] == candidate.fem_pressure[retained]
                            @test isnan(lean.fem_interior_residual)
                            for field in (:bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
                                @test getproperty(lean, field) == getproperty(candidate, field)
                            end
                            @test lean.pressure_continuity_error == candidate.pressure_continuity_error
                        end
                        # The operator is geometry-only and reused by the next frequency.
                        @test !candidate_system.timings.interface_mass_cached
                        next_system = withenv(lever_one, flux_switch, switches...) do
                            build_condensed_coupled_system(
                                fem_mesh, bem_mesh, interface_map, 900.0, 343.0, 1.21;
                                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                                singular_order=CONDENSED_SINGULAR_ORDER,
                                transducers=[transducer],
                                cache=candidate_system.cache,
                            )
                        end
                        next_baseline = withenv(lever_one, flux_switch) do
                            condensed_system_at(Float64, 900.0; transducers=[transducer])
                        end
                        try
                            @test next_system.timings.interface_mass_cached
                            @test next_system.interface_elimination_data.mass_operator === operator
                            for (baseline, candidate) in zip(
                                solve_condensed_coupled_excitations(next_baseline, specialized_excitations),
                                solve_condensed_coupled_excitations(next_system, specialized_excitations),
                            )
                                for field in elimination_fields
                                    @test relative_error(getproperty(baseline, field), getproperty(candidate, field)) < 1e-9
                                end
                            end
                        finally
                            release_condensed_coupled_system!(next_system)
                            release_condensed_coupled_system!(next_baseline)
                        end
                    finally
                        release_condensed_coupled_system!(candidate_system)
                    end
                end
            finally
                release_coupled_system!(specialized_reference)
                release_condensed_coupled_system!(specialized_baseline)
            end
            @test_throws "BLAB_COUPLED_INTERFACE_MASS_SOLVER" withenv(
                BeatEngineCoupledCondensed._interface_mass_solver_selection,
                "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "dense",
            )

            # Retained transducer surfaces make Γ wider than the interface: refuse, do not guess.
            for switch in (pressure_switch, flux_switch)
                @test_throws ErrorException withenv(switch) do
                    condensed_system_at(Float64, 700.0; transducers=[transducer])
                end
            end

            # Without transducers Γ is the interface already; both eliminations apply directly.
            # A prescribed-velocity load on the radiator exercises the g term with a nonzero f.
            plain_reference = monolithic_system(Float64, 650.0)
            plain_eliminated = [
                withenv(switch) do
                    condensed_system_at(Float64, 650.0)
                end
                for switch in (pressure_switch, flux_switch)
            ]
            try
                reference = only(solve_coupled_systems(plain_reference, [radiator_tag]))
                for candidate_system in plain_eliminated
                    @test candidate_system.solved_system_order < plain_reference.full_system_order
                    candidate = only(solve_condensed_coupled_systems(candidate_system, [radiator_tag]))
                    for field in (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann)
                        @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                    end
                end
            finally
                release_coupled_system!(plain_reference)
                foreach(release_condensed_coupled_system!, plain_eliminated)
            end

            # A multi-interface request concatenates per-interface maps, so interface-dof order need
            # not be sorted FEM vertex order. Shuffle the map to exercise the Γ-to-dof permutation.
            shuffle_order = randperm(Random.MersenneTwister(20260916), interface_count)
            shuffled_map = ConformingInterfaceMap(
                interface_map.fem_vertex_indices[shuffle_order],
                interface_map.fem_to_bem_vertex_indices[shuffle_order],
                interface_map.fem_face_indices,
                interface_map.bem_face_indices,
                interface_map.normal_sign,
            )
            @test shuffled_map.fem_vertex_indices != sort(shuffled_map.fem_vertex_indices)
            shuffled_options = (
                quadrature_order=CONDENSED_QUADRATURE_ORDER,
                singular_order=CONDENSED_SINGULAR_ORDER,
            )
            shuffled_reference = build_coupled_system(
                fem_mesh, bem_mesh, shuffled_map, 650.0, 343.0, 1.21;
                shuffled_options..., validation_diagnostics=false, bem_backend=:cpu,
            )
            shuffled_eliminated = [
                withenv(switch) do
                    build_condensed_coupled_system(
                        fem_mesh, bem_mesh, shuffled_map, 650.0, 343.0, 1.21; shuffled_options...,
                    )
                end
                for switch in (pressure_switch, flux_switch)
            ]
            push!(shuffled_eliminated, withenv(
                flux_switch,
                "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod",
                "BLAB_COUPLED_INTERFACE_BLOCKS" => "1",
                "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "1",
            ) do
                build_condensed_coupled_system(
                    fem_mesh, bem_mesh, shuffled_map, 650.0, 343.0, 1.21; shuffled_options...,
                )
            end)
            try
                @test hasproperty(last(shuffled_eliminated).interface_elimination_data, :mass_operator)
                # One block always spans every dof; the shuffled map makes its Γ row order non-trivial.
                @test only(last(shuffled_eliminated).interface_elimination_data.mass_operator.blocks).rows != 1:interface_count
                reference = only(solve_coupled_systems(shuffled_reference, [radiator_tag]))
                for candidate_system in shuffled_eliminated
                    candidate = only(solve_condensed_coupled_systems(candidate_system, [radiator_tag]))
                    for field in (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann)
                        @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < 1e-9
                    end
                end
            finally
                release_coupled_system!(shuffled_reference)
                foreach(release_condensed_coupled_system!, shuffled_eliminated)
            end

            # Composes with the single-precision operators and the double-precision dense LU.
            transducer32 = ElectrodynamicTransducer{Float32}(
                "component:test",
                [physical_tag(fem_mesh32, 2, "Radiator")],
                Float32[1],
                [1],
                Float32[-1],
                SVector(0f0, 0f0, 1f0),
                2f0,
                1,
                6f0,
                0.0005f0,
                7f0,
                0.015f0,
                0.0005f0,
                1f0,
            )
            composed_baseline = withenv(lever_one, "BLAB_COUPLED_DENSE_FLOAT64" => "1") do
                condensed_system_at(Float32, 700.0; transducers=[transducer32])
            end
            composed_flux = withenv(lever_one, flux_switch, "BLAB_COUPLED_DENSE_FLOAT64" => "1") do
                condensed_system_at(Float32, 700.0; transducers=[transducer32])
            end
            composed_fem64 = withenv(lever_one, flux_switch, "BLAB_COUPLED_DENSE_FLOAT64" => "1",
                                     "BLAB_COUPLED_FEM_FLOAT64" => "1") do
                condensed_system_at(Float32, 700.0; transducers=[transducer32])
            end
            try
                @test composed_flux.fem_scalar_type == Float32
                @test composed_fem64.fem_scalar_type == Float64
                fem64_solution = only(solve_condensed_coupled_excitations(
                    composed_fem64, [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))],
                ))
                flux_solution = only(solve_condensed_coupled_excitations(
                    composed_flux, [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))],
                ))
                @test eltype(fem64_solution.bem_pressure) == ComplexF32
                for field in elimination_fields
                    @test relative_error(getproperty(flux_solution, field), getproperty(fem64_solution, field)) < 1e-5
                end
            finally
                release_condensed_coupled_system!(composed_fem64)
            end
            composed_refined = withenv(lever_one, flux_switch, "BLAB_COUPLED_DENSE_REFINEMENT" => "1") do
                condensed_system_at(Float32, 700.0; transducers=[transducer32])
            end
            try
                @test composed_refined.dense_scalar_type == Float64
                @test composed_refined.factorization isa BeatEngineCoupledCondensed.RefinedDenseLU
                @test eltype(composed_refined.condensation.schur) == ComplexF64
                # The mechanical block (mechanical impedance minus the condensed air spring) is formed
                # in Float64 and must reach the Float64 dense matrix unrounded.
                mechanical = composed_refined.factorization.matrix[composed_refined.mechanical_range, composed_refined.mechanical_range]
                @test any(value -> ComplexF64(ComplexF32(value)) != value, mechanical)
                @test composed_flux.factorization isa LinearAlgebra.LU{ComplexF64}
                excitations_refined = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))]
                refined_solution = only(solve_condensed_coupled_excitations(composed_refined, excitations_refined))
                dense64_solution = only(solve_condensed_coupled_excitations(composed_flux, excitations_refined))
                for field in elimination_fields
                    @test relative_error(getproperty(dense64_solution, field), getproperty(refined_solution, field)) < 1e-6
                end
                @test BeatEngineCoupledCondensed.dense_solver_diagnostics(composed_refined)["dense_solver"] == "lu_float32_refined"
                @test composed_refined.factorization.iterations >= 1
            finally
                release_condensed_coupled_system!(composed_refined)
            end
            refined_double = withenv(lever_one, flux_switch, "BLAB_COUPLED_DENSE_REFINEMENT" => "1") do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            try
                @test refined_double.factorization isa LinearAlgebra.LU{ComplexF64}
            finally
                release_condensed_coupled_system!(refined_double)
            end
            # `auto` optimizations fall back to the established path, with a reason, where the model's
            # structure does not allow them; `on` refuses (tested above). All results must equal the
            # established path exactly, since the fallback *is* that path.
            voltage_auto = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))]
            auto_names = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION")
            plain_retained = withenv(lever_one.first => "0") do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            flux_without_condensation = withenv(lever_one.first => "0", flux_switch.first => "auto") do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            rom_request = withenv(lever_one.first => "auto", flux_switch.first => "auto") do
                condensed_system_at(Float64, 700.0; transducers=[transducer], allow_transducer_condensation=false)
            end
            try
                @test flux_without_condensation.interface_elimination == :none
                @test only(flux_without_condensation.optimization_fallback_reasons) |> r -> startswith(r, "interface elimination not used")
                @test !rom_request.transducer_condensation && rom_request.interface_elimination == :none
                @test length(rom_request.optimization_fallback_reasons) == 2
                @test any(r -> occursin("speaker ROM", r), rom_request.optimization_fallback_reasons)
                @test isempty(plain_retained.optimization_fallback_reasons)
                plain = only(solve_condensed_coupled_excitations(plain_retained, voltage_auto))
                for candidate_system in (flux_without_condensation, rom_request)
                    candidate = only(solve_condensed_coupled_excitations(candidate_system, voltage_auto))
                    @test candidate.bem_pressure == plain.bem_pressure
                    @test candidate.diaphragm_velocity == plain.diaphragm_velocity
                end
            finally
                foreach(release_condensed_coupled_system!, (plain_retained, flux_without_condensation, rom_request))
            end
            # Every optimization `auto` on the CPU backend reproduces what an unset environment selects on
            # Metal (MUMPS aside, which julia_local does not ship): exact in Float64, and in Float32 the
            # precision fixes engage.
            metal_like = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "auto", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "auto",
                          "BLAB_COUPLED_INTERFACE_MASS_SOLVER" => "cholmod", "BLAB_COUPLED_INTERFACE_MASS_OVERLAP" => "auto",
                          "BLAB_COUPLED_INTERFACE_BLOCKS" => "auto", "BLAB_COUPLED_DENSE_REFINEMENT" => "auto",
                          "BLAB_COUPLED_FEM_FLOAT64" => "auto")
            metal_like_reference = monolithic_system(Float64, 700.0; transducers=[transducer])
            metal_like64 = withenv(() -> condensed_system_at(Float64, 700.0; transducers=[transducer]), metal_like...)
            metal_like32 = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]), metal_like...)
            try
                @test metal_like64.transducer_condensation && metal_like64.interface_elimination == :flux
                @test isempty(metal_like64.optimization_fallback_reasons)
                reference_solution = only(solve_coupled_excitations(metal_like_reference, voltage_auto))
                metal_like_solution = only(solve_condensed_coupled_excitations(metal_like64, voltage_auto))
                for field in elimination_fields
                    @test relative_error(getproperty(reference_solution, field), getproperty(metal_like_solution, field)) < 1e-9
                end
                @test metal_like32.factorization isa BeatEngineCoupledCondensed.RefinedDenseLU
                @test metal_like32.fem_scalar_type == Float64
                @test BeatEngineCoupledCondensed.interface_mass_diagnostics(metal_like32)["interface_mass_solver"] == "cholmod"
            finally
                release_coupled_system!(metal_like_reference)
                release_condensed_coupled_system!(metal_like64)
                release_condensed_coupled_system!(metal_like32)
            end
            # Under precision=float64 the switch changes nothing.
            fem64_double = withenv(lever_one, flux_switch, "BLAB_COUPLED_FEM_FLOAT64" => "1") do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            fem64_plain = withenv(lever_one, flux_switch) do
                condensed_system_at(Float64, 700.0; transducers=[transducer])
            end
            try
                voltage64 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))]
                @test only(solve_condensed_coupled_excitations(fem64_double, voltage64)).bem_pressure ==
                      only(solve_condensed_coupled_excitations(fem64_plain, voltage64)).bem_pressure
            finally
                release_condensed_coupled_system!(fem64_double)
                release_condensed_coupled_system!(fem64_plain)
            end
            try
                @test composed_flux.dense_scalar_type == Float64
                excitations32 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))]
                baseline = only(solve_condensed_coupled_excitations(composed_baseline, excitations32))
                candidate = only(solve_condensed_coupled_excitations(composed_flux, excitations32))
                @test eltype(candidate.bem_pressure) == ComplexF32
                for field in elimination_fields
                    @test relative_error(getproperty(baseline, field), getproperty(candidate, field)) < 1e-5
                end
            finally
                release_condensed_coupled_system!(composed_baseline)
                release_condensed_coupled_system!(composed_flux)
            end

            # Precision switches on the fixture with a transducer. Unset CPU precision switches
            # refine the dense LU while retaining Float32 FEM assembly. Explicit `auto` FEM assembly
            # gives the Metal precision configuration, agreeing with plain Float64 LU of that system.
            transducer32 = ElectrodynamicTransducer{Float32}(
                "component:test", [physical_tag(fem_mesh32, 2, "Radiator")], Float32[1], [1], Float32[-1],
                SVector(0f0, 0f0, 1f0), 2f0, 1, 6f0, 0.0005f0, 7f0, 0.015f0, 0.0005f0, 1f0,
            )
            precision_names = ("BLAB_COUPLED_DENSE_FLOAT64", "BLAB_COUPLED_DENSE_REFINEMENT", "BLAB_COUPLED_FEM_FLOAT64")
            voltage32 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF32(1, 0))]
            baseline32 = condensed_system_at(Float32, 700.0; transducers=[transducer32])
            default32 = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                                (name => nothing for name in precision_names)...)
            refined32 = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                                "BLAB_COUPLED_DENSE_REFINEMENT" => "auto", "BLAB_COUPLED_FEM_FLOAT64" => "auto")
            plain64lu = withenv(() -> condensed_system_at(Float32, 700.0; transducers=[transducer32]),
                                "BLAB_COUPLED_DENSE_FLOAT64" => "1", "BLAB_COUPLED_FEM_FLOAT64" => "1",
                                "BLAB_COUPLED_DENSE_REFINEMENT" => "0")
            try
                @test baseline32.factorization isa LinearAlgebra.LU{ComplexF32}
                @test baseline32.dense_scalar_type == Float32 && baseline32.fem_scalar_type == Float32
                @test default32.factorization isa BeatEngineCoupledCondensed.RefinedDenseLU
                @test default32.dense_scalar_type == Float64 && default32.fem_scalar_type == Float32
                @test refined32.factorization isa BeatEngineCoupledCondensed.RefinedDenseLU
                @test refined32.dense_scalar_type == Float64 && refined32.fem_scalar_type == Float64
                # The Schur block is formed in Float64 and must not be demoted on the way to the dense matrix.
                @test eltype(refined32.condensation.schur) == ComplexF64
                @test any(value -> ComplexF64(ComplexF32(value)) != value,
                          refined32.factorization.matrix[refined32.gamma_range, refined32.gamma_range])
                @test plain64lu.factorization isa LinearAlgebra.LU{ComplexF64}
                @test BeatEngineCoupledCondensed.dense_solver_diagnostics(refined32)["dense_solver"] == "lu_float32_refined"
                refined_solution = only(solve_condensed_coupled_excitations(refined32, voltage32))
                plain_solution = only(solve_condensed_coupled_excitations(plain64lu, voltage32))
                @test eltype(refined_solution.bem_pressure) == ComplexF32
                for field in (:fem_pressure, :bem_pressure, :interface_flux, :diaphragm_velocity, :voice_coil_current)
                    @test relative_error(getproperty(plain_solution, field), getproperty(refined_solution, field)) < 1e-6
                end
            finally
                foreach(release_condensed_coupled_system!, (baseline32, default32, refined32, plain64lu))
            end

            @testset "Unset CPU defaults reduce the system and match the pinned baseline" begin
                # Clear the enclosing baseline pins and any ambient coupled overrides. Keep the
                # environment unset through both construction and solving, including reconstruction.
                names = filter(name -> startswith(name, "BLAB_COUPLED_"), collect(keys(ENV)))
                withenv((name => nothing for name in names)...) do
                    CC = BeatEngineCoupledCondensed
                    @test CC._demand_reconstruction_enabled(:cpu)
                    for (T, driver, tolerance) in ((Float64, transducer, 1e-9), (Float32, transducer32, 1e-5))
                        # The unreduced baseline with a ComplexF64 dense LU: a plain Float32 dense LU
                        # would be the less accurate side of the comparison.
                        baseline = with_baseline_cpu_coupled() do
                            withenv(() -> condensed_system_at(T, 700.0; transducers=[driver]),
                                    "BLAB_COUPLED_DENSE_FLOAT64" => "1")
                        end
                        defaults = condensed_system_at(T, 700.0; transducers=[driver])
                        try
                            @test defaults.bem_backend == :cpu
                            @test !baseline.transducer_condensation && baseline.interface_elimination == :none
                            @test defaults.transducer_condensation && defaults.condensation.transducer_condensed
                            @test defaults.interface_elimination == :flux
                            @test defaults.condensation.retained_count == length(unique(interface_map.fem_vertex_indices))
                            @test defaults.solved_system_order == length(bem_mesh.vertices) + 2
                            @test defaults.solved_system_order < baseline.solved_system_order
                            @test defaults.dense_scalar_type == Float64 && defaults.fem_scalar_type == T
                            @test eltype(defaults.condensation.schur) == ComplexF64
                            @test defaults.condensation.backend == :cpu_umfpack
                            @test !defaults.timings.stage_overlap
                            @test isempty(defaults.optimization_fallback_reasons)
                            @test hasproperty(defaults.interface_elimination_data, :mass_operator)
                            @test haskey(defaults.timings.interface_elimination_split, :components)
                            @test !haskey(defaults.timings.interface_elimination_split, :fem_stage_mass_solve)
                            mass = CC.interface_mass_diagnostics(defaults)
                            @test mass["interface_mass_solver_requested"] == "cholmod"
                            @test mass["interface_mass_solver"] == "cholmod"
                            @test mass["interface_mass_block_count"] == 1
                            @test isempty(mass["interface_mass_fallback_reasons"])
                            @test baseline.factorization isa LinearAlgebra.LU{ComplexF64}
                            if T === Float32
                                @test defaults.factorization isa CC.RefinedDenseLU
                            else
                                @test defaults.factorization isa LinearAlgebra.LU{ComplexF64}
                            end
                            excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0))]
                            reference = only(solve_condensed_coupled_excitations(baseline, excitations))
                            candidate = only(solve_condensed_coupled_excitations(defaults, excitations))
                            for field in elimination_fields
                                @test relative_error(getproperty(reference, field), getproperty(candidate, field)) < tolerance
                            end
                            if T === Float32
                                @test CC.dense_solver_diagnostics(defaults)["dense_solver"] == "lu_float32_refined"
                                @test defaults.factorization.iterations >= 1
                            end
                            lean = only(solve_condensed_coupled_excitations(defaults, excitations; reconstruct_interior=false))
                            @test all(isnan, lean.fem_pressure[defaults.condensation.interior_vertices])
                            @test lean.fem_pressure[defaults.condensation.retained_vertices] ==
                                  candidate.fem_pressure[defaults.condensation.retained_vertices]
                            @test isnan(lean.fem_interior_residual)
                            for field in (:bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
                                @test getproperty(lean, field) == getproperty(candidate, field)
                            end
                        finally
                            release_condensed_coupled_system!(baseline)
                            release_condensed_coupled_system!(defaults)
                        end
                    end
                end
            end
            # Under precision=float64 the switches change nothing.
            switched64 = withenv(() -> condensed_system_at(Float64, 700.0; transducers=[transducer]),
                                 "BLAB_COUPLED_DENSE_REFINEMENT" => "1", "BLAB_COUPLED_FEM_FLOAT64" => "1")
            unswitched64 = withenv(() -> condensed_system_at(Float64, 700.0; transducers=[transducer]),
                                   (name => nothing for name in precision_names)...)
            try
                voltage64 = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))]
                @test switched64.factorization isa LinearAlgebra.LU{ComplexF64}
                @test only(solve_condensed_coupled_excitations(switched64, voltage64)).bem_pressure ==
                      only(solve_condensed_coupled_excitations(unswitched64, voltage64)).bem_pressure
            finally
                release_condensed_coupled_system!(switched64)
                release_condensed_coupled_system!(unswitched64)
            end
        end

        @testset "Condensed coupled solver interior resonance" begin
            sound_speed = 343.0
            fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
            bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
            interface_map = build_conforming_interface_map(
                fem_mesh,
                bem_mesh,
                physical_tag(fem_mesh, 2, "Interface"),
                2,
            )
            radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
            fem_mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
            bem_mesh32 = load_gmsh22_with_tags(
                joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"),
                Float32(0.001),
            )
            interface_map32 = build_conforming_interface_map(
                fem_mesh32,
                bem_mesh32,
                physical_tag(fem_mesh32, 2, "Interface"),
                2,
            )

            # The Schur complement has poles at the eigenvalues of A_II. Restricting the FEM operator
            # to interior rows *and* columns imposes u_Γ = 0, so these are the cavity modes with a
            # pressure-release interface — a different, lower set than the rigid-wall modes
            # `sealed_cavity_modes` reports.
            retained = sort(unique(interface_map.fem_vertex_indices))
            interior = setdiff(1:length(fem_mesh.vertices), retained)
            stiffness, mass = assemble_p1_fem_matrices(fem_mesh)
            interior_eigenvalues = eigen(
                Symmetric(Matrix(stiffness[interior, interior])),
                Symmetric(Matrix(mass[interior, interior])),
            ).values
            scale = maximum(abs, interior_eigenvalues)
            positive = filter(value -> value > 1e-8 * scale, sort(real.(interior_eigenvalues)))
            pole_hz = sound_speed * sqrt(first(positive)) / (2pi)
            @test pole_hz > 0
            @test pole_hz < first(sealed_cavity_modes(fem_mesh, sound_speed; count=1))

            function resonance_solution(::Type{T}, frequency, bulk_loss) where {T}
                mesh, boundary, mapping = T === Float32 ?
                                          (fem_mesh32, bem_mesh32, interface_map32) :
                                          (fem_mesh, bem_mesh, interface_map)
                system = build_condensed_coupled_system(
                    mesh,
                    boundary,
                    mapping,
                    T(frequency),
                    T(sound_speed),
                    T(1.21);
                    quadrature_order=CONDENSED_QUADRATURE_ORDER,
                    singular_order=CONDENSED_SINGULAR_ORDER,
                    bulk_loss_factor=T(bulk_loss),
                )
                try
                    return solve_condensed_coupled_system(
                        system,
                        T === Float32 ? physical_tag(fem_mesh32, 2, "Radiator") : radiator_tag,
                    )
                finally
                    release_condensed_coupled_system!(system)
                end
            end

            function monolithic_solution(frequency, bulk_loss)
                system = build_coupled_system(
                    fem_mesh,
                    bem_mesh,
                    interface_map,
                    frequency,
                    sound_speed,
                    1.21;
                    quadrature_order=CONDENSED_QUADRATURE_ORDER,
                    singular_order=CONDENSED_SINGULAR_ORDER,
                    validation_diagnostics=false,
                    bem_backend=:cpu,
                    bulk_loss_factor=bulk_loss,
                )
                try
                    return solve_coupled_system(system, radiator_tag)
                finally
                    release_coupled_system!(system)
                end
            end

            relative_error(reference, candidate) = norm(
                ComplexF64.(candidate) .- ComplexF64.(reference),
            ) / norm(ComplexF64.(reference))

            for bulk_loss in (0.0, 0.02)
                observed = NamedTuple[]
                for offset in (-0.005, 0.0, 0.005)
                    frequency = pole_hz * (1 + offset)
                    reference = monolithic_solution(frequency, bulk_loss)
                    candidate = resonance_solution(Float64, frequency, bulk_loss)
                    candidate32 = resonance_solution(Float32, frequency, bulk_loss)
                    error64 = relative_error(reference.fem_pressure, candidate.fem_pressure)
                    error32 = relative_error(reference.fem_pressure, candidate32.fem_pressure)
                    push!(
                        observed,
                        (
                            offset=offset,
                            frequency_hz=frequency,
                            error64=error64,
                            error32=error32,
                            residual64=candidate.fem_interior_residual,
                            residual32=candidate32.fem_interior_residual,
                        ),
                    )

                    # The interior residual only validates the back-substitution, so it stays small
                    # even where the Schur complement is badly conditioned. It is not a resonance
                    # detector — hence the separate accuracy assertions.
                    @test candidate.fem_interior_residual < 1e-10
                    @test candidate32.fem_interior_residual < 1e-4
                    @test all(isfinite, real.(candidate.fem_pressure))
                    @test all(isfinite, real.(candidate32.fem_pressure))

                    if offset == 0.0 && bulk_loss == 0.0
                        # Straddling the pole with no loss is where condensation is weakest and the
                        # tight tolerance genuinely does not hold. Assert only that it degrades
                        # gracefully rather than diverging; the measured value is reported below as
                        # the documented limit rather than pinned here.
                        @test error64 < 1e-1
                        @test error32 < 1e-1
                    else
                        @test error64 < 1e-9
                        @test error32 < 1e-3
                    end
                end
                @info "Schur condensation interior-resonance sweep" bulk_loss pole_hz observed
            end
        end
    end
else
    @info "Set BLAB_RUN_COUPLED_REFERENCE=1 to run the condensed coupled solver validation."
end

@testset "Condensed regular assembly matches the shared CPU assembly" begin
    # The condensed solver runs its own fork of the CPU regular assembly so it can be optimised
    # without touching the path every other backend shares. This is the contract that makes that
    # safe: operators agree to round-off, across every symmetry mode, cached and uncached. Any
    # optimisation added to the fork has to keep this passing.
    entrywise_assembly_matches(candidate, reference) =
        maximum(abs, candidate .- reference) <= 1.0f-5 * maximum(abs, reference)
    rule = triangle_rule(Float32, 2)
    k = Float32(2pi * 1000.0 / 343.0)
    # Each symmetry mode needs a mesh that actually lies in its fundamental domain: the half mesh
    # straddles y<0 so :xy rejects it. :xy matters most here -- it is four image sweeps, which is
    # where a fused assembly has the most to gain and the most to get wrong.
    for (symmetry, mesh_name) in ((:off, "sample.msh"), (:x, "sample_half.msh"), (:xy, "sample_quarter.msh"))
        mesh = load_gmsh22_with_tags(
            joinpath(@__DIR__, "..", "test_meshes", mesh_name), Float32(0.001),
        )
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        element_indices = 1:min(24, length(mesh.faces))
        singular_cache = build_singular_correction_cache(mesh, 2, element_indices)
        symmetry == :off || validate_symmetry_fundamental_domain!(mesh, symmetry)
        shared = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, element_indices=element_indices,
            backend=:cpu, singular_cache=singular_cache, symmetry_mode=symmetry,
        )
        forked = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, element_indices=element_indices,
            singular_cache=singular_cache, symmetry_mode=symmetry,
        )
        # With no images the two loops do the same arithmetic but are compiled separately.
        # Paired AVX-512/masked runs measured 201 differing double-layer entries (up to 256
        # ULP) and 205 adjoint entries (up to 148 ULP) from near-cancelling dot(r_vec, normal).
        # A sapphirerapids probe measured a worst double-layer gap of 3.55e-15 against a
        # 2.0e-12 floor; see scripts/avx512_condensed_probe.jl and avx512_codegen_assay.jl.
        # Bound EVERY entry absolutely by 1e-5 of the reference operator's largest entry:
        # a norm can hide isolated defects, while a per-entry relative bound over-penalises
        # near-cancellation (the measured worst relative gap was 1.57e-5).
        # With images, summing contributions before scattering reorders the arithmetic by
        # design, so keep the existing norm bound there.
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            if symmetry == :off
                @test entrywise_assembly_matches(getproperty(forked, op), getproperty(shared, op))
            else
                @test getproperty(forked, op) ≈ getproperty(shared, op) rtol = 1.0f-5
            end
        end
        for count in (:regular_pairs, :singular_pairs, :skipped_pairs, :image_singular_pairs)
            @test getproperty(forked, count) == getproperty(shared, count)
        end

        # Again through a prebuilt cache, which is how the solver actually calls it.
        cache = build_beat_cpu_assembly_cache(
            mesh, p1, dp0, rule;
            singular_order=2, element_indices=element_indices, symmetry_mode=symmetry,
        )
        shared_cached = assemble_regular_galerkin_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2, backend=:cpu,
            singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=symmetry,
        )
        forked_cached = BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
            mesh, p1, dp0, k, rule;
            skip_singular=false, singular_order=2,
            singular_cache=singular_cache, cpu_cache=cache, symmetry_mode=symmetry,
        )
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            if symmetry == :off
                @test entrywise_assembly_matches(getproperty(forked_cached, op), getproperty(shared_cached, op))
            else
                @test getproperty(forked_cached, op) ≈ getproperty(shared_cached, op) rtol = 1.0f-5
            end
        end
        # The cached and uncached fused paths must agree with each other exactly: same order, same
        # data, only the provenance of the reflected element sets differs.
        for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
            @test getproperty(forked_cached, op) == getproperty(forked, op)
        end

        if symmetry == :off
            # A missing quadrature contribution could corrupt just one entry. A 0.1% error
            # at the operator's largest entry is 100 times the floor and must be rejected,
            # for every operator in both paths, using the same predicate as the positive gate.
            for (candidate, reference) in ((forked, shared), (forked_cached, shared_cached))
                for op in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                    expected = getproperty(reference, op)
                    corrupted = copy(getproperty(candidate, op))
                    corrupted[argmax(abs.(expected))] += 1.0f-3 * maximum(abs, expected)
                    @test !entrywise_assembly_matches(corrupted, expected)
                end
            end
        end
    end

    # The fork keeps the strict stale-cache guard: a value-equal but distinct rule is rejected.
    guard_mesh = load_gmsh22_with_tags(
        joinpath(@__DIR__, "..", "test_meshes", "sample.msh"), Float32(0.001),
    )
    guard_p1 = build_p1_space(guard_mesh)
    guard_dp0 = build_dp0_space(guard_mesh)
    guard_indices = 1:min(24, length(guard_mesh.faces))
    guard_cache = build_beat_cpu_assembly_cache(
        guard_mesh, guard_p1, guard_dp0, rule;
        singular_order=2, element_indices=guard_indices, symmetry_mode=:off,
    )
    @test_throws "cache quadrature rule does not match" BeatEngineCoupledCondensed.assemble_condensed_regular_operators(
        guard_mesh, guard_p1, guard_dp0, k, triangle_rule(Float32, 2);
        skip_singular=false, singular_order=2,
        singular_cache=build_singular_correction_cache(guard_mesh, 2, guard_indices),
        cpu_cache=guard_cache, symmetry_mode=:off,
    )
end

@testset "Wavelength quadrature selection" begin
    # h = sqrt(area) = 0.01 m exactly, so kh = 2*pi*f/c * 0.01 and a frequency can be placed
    # either side of a cutoff by hand.
    areas = fill(1.0e-4, 16)
    c = 343.0
    kh(f) = 2pi * f / c * 0.01
    low, high = 2000.0, 20000.0
    @test kh(low) < 2.0 && kh(high) > 2.0

    below = wavelength_quadrature_order(areas, low, c, 4)
    above = wavelength_quadrature_order(areas, high, c, 4)
    @test below.order == 2
    @test above.order == 4
    @test above.base_order == 4
    @test below.length ≈ 0.01
    @test below.kh ≈ kh(low)
    @test below.q1_max == 0.0 && below.q2_max == 2.0

    # The one-point tier is unreachable at the shipped cutoff and reachable once it is raised.
    @test wavelength_quadrature_order(areas, low, c, 4; q1_max=0.0).order == 2
    @test wavelength_quadrature_order(areas, low, c, 4; q1_max=kh(low) + 0.1).order == 1

    spread = [1.0e-4, 4.0e-4, 9.0e-4, 1.6e-3]
    stats = [wavelength_quadrature_order(spread, low, c, 4; mesh_stat=s).area
             for s in ("median", "p75", "p90", "max")]
    @test issorted(stats)
    @test stats[end] == maximum(spread)

    @test_throws "Unsupported wavelength mesh stat" wavelength_quadrature_order(
        areas, low, c, 4; mesh_stat="p50",
    )
    @test_throws "must be non-negative" wavelength_quadrature_order(
        areas, low, c, 4; q1_max=-1.0,
    )
    @test_throws "must exceed" wavelength_quadrature_order(
        areas, low, c, 4; q1_max=3.0, q2_max=2.0,
    )
    @test_throws "empty mesh" wavelength_quadrature_order(Float64[], low, c, 4)
end

@testset "Condensed per-order quadrature bundles" begin
    fem_mesh = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem_mesh = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_map = build_conforming_interface_map(
        fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2,
    )

    cache = prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=2,
        regular_quadrature_orders=[2, 4],
        singular_order=CONDENSED_SINGULAR_ORDER,
    )
    @test sort(collect(keys(cache.quadrature_bundles))) == [2, 4]
    @test cache.base_quadrature_order == 2
    @test cache.singular_order == CONDENSED_SINGULAR_ORDER
    # The base bundle reuses what the underlying coupled cache already built, so a single-order
    # sweep is identical to using that cache directly.
    @test cache.quadrature_bundles[2].rule === cache.base.rule
    @test cache.quadrature_bundles[2].cpu_assembly_cache === cache.base.cpu_assembly_cache
    for order in (2, 4)
        bundle = cache.quadrature_bundles[order]
        @test bundle.order == order
        @test length(bundle.rule.points) == length(triangle_rule(Float64, order).points)
        # The bundle owns its rule: the CPU assembly guard compares rules by pointer identity.
        @test bundle.rule === bundle.cpu_assembly_cache.rule
    end
    # The base order is carried even when no frequency selects it: base 4 with every frequency
    # picking 1 or 2 must not fail at cache setup.
    absent_base = prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=4, regular_quadrature_orders=[1, 2],
        singular_order=CONDENSED_SINGULAR_ORDER,
    )
    @test sort(collect(keys(absent_base.quadrature_bundles))) == [1, 2, 4]
    @test absent_base.base_quadrature_order == 4
    # A q1 bundle must NOT take its identity matrices from the 1-point rule: that rule integrates
    # the P1xP1 mass matrix inexactly (area/9 uniform instead of area/6, area/12) and would
    # silently degrade the Burton-Miller 0.5*I term. Clamped to order 2, so it matches exactly.
    exact_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, absent_base.base.p1, absent_base.base.dp0,
        triangle_rule(Float64, 2), :p1, :p1,
    )
    wrong_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, absent_base.base.p1, absent_base.base.dp0,
        triangle_rule(Float64, 1), :p1, :p1,
    )
    @test absent_base.quadrature_bundles[1].identity_p1_p1 == exact_p1_p1
    @test wrong_p1_p1 != exact_p1_p1          # the 1-point rule really is wrong here
    # Clamping the identity rule must not disturb the regular rule, which stays at order 1.
    @test length(absent_base.quadrature_bundles[1].rule.points) == 1
    # Orders >= 2 are exact in exact arithmetic but not bitwise: different points and weights
    # round differently, so they agree to round-off rather than identically.
    @test absent_base.quadrature_bundles[4].identity_p1_p1 ≈ exact_p1_p1 rtol = 1e-13
    @test absent_base.quadrature_bundles[4].identity_p1_p1 != exact_p1_p1
    release_condensed_coupled_cache!(absent_base)

    @test_throws "must be positive" prepare_condensed_coupled_cache(
        fem_mesh, bem_mesh, interface_map;
        quadrature_order=2, regular_quadrature_orders=[0], singular_order=CONDENSED_SINGULAR_ORDER,
    )

    # Stale-cache guards, compared as integers rather than as rules. Both fire before assembly.
    for (order, singular, message) in (
        (4, CONDENSED_SINGULAR_ORDER, "base quadrature order does not match"),
        (2, CONDENSED_SINGULAR_ORDER + 1, "singular order does not match"),
    )
        @test_throws message build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=order, singular_order=singular, cache=cache,
        )
    end
    @test_throws "holds no quadrature bundle for order 3" build_condensed_coupled_system(
        fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
        quadrature_order=2, regular_quadrature_order=3,
        singular_order=CONDENSED_SINGULAR_ORDER, cache=cache,
    )

    # A multi-order cache must give each order exactly what a dedicated single-order cache gives.
    # This is what catches a bundle being shared or looked up by the wrong key. The Schur solver
    # never retains the assembled matrix, so the LU factors are the observable -- deterministic
    # for identical input in one process, and order-sensitive through `bem_lhs`.
    factors = Dict{Int,Any}()
    for order in (2, 4)
        dedicated = prepare_condensed_coupled_cache(
            fem_mesh, bem_mesh, interface_map;
            quadrature_order=order, singular_order=CONDENSED_SINGULAR_ORDER,
        )
        shared_system = build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=2, regular_quadrature_order=order,
            singular_order=CONDENSED_SINGULAR_ORDER, cache=cache,
        )
        dedicated_system = build_condensed_coupled_system(
            fem_mesh, bem_mesh, interface_map, 1000.0, 343.0, 1.2;
            quadrature_order=order, singular_order=CONDENSED_SINGULAR_ORDER, cache=dedicated,
        )
        @test shared_system.regular_quadrature_order == order
        @test dedicated_system.regular_quadrature_order == order
        @test shared_system.factorization.factors == dedicated_system.factorization.factors
        factors[order] = copy(shared_system.factorization.factors)
        release_condensed_coupled_system!(shared_system)
        release_condensed_coupled_system!(dedicated_system)
        release_condensed_coupled_cache!(dedicated)
    end
    # Changing order must actually change the operator, or the equality above holds trivially.
    @test factors[2] != factors[4]

    release_condensed_coupled_cache!(cache)
end

@testset "Condensed coupled precision switches: CPU and Metal defaults" begin
    CC = BeatEngineCoupledCondensed
    names = ("BLAB_COUPLED_DENSE_FLOAT64", "BLAB_COUPLED_DENSE_REFINEMENT", "BLAB_COUPLED_FEM_FLOAT64")
    withenv((name => nothing for name in names)...) do
        for backend in (:cuda, :rocm)
            @test !CC._dense_double_assembly(backend) && !CC._dense_refinement_enabled(backend)
            @test !CC._fem_float64_enabled(backend)
        end
        @test CC._dense_double_assembly(:cpu) && CC._dense_refinement_enabled(:cpu)
        @test !CC._dense_float64_enabled(:cpu) && !CC._fem_float64_enabled(:cpu)
        @test !CC._fem_float64_enabled() && CC._dense_double_assembly() && CC._dense_refinement_enabled()
        @test CC._dense_refinement_enabled(:metal) && !CC._dense_float64_enabled(:metal) && CC._dense_double_assembly(:metal)
        @test CC._fem_float64_enabled(:metal)
    end
    withenv("BLAB_COUPLED_DENSE_REFINEMENT" => "off", "BLAB_COUPLED_FEM_FLOAT64" => "AUTO", "BLAB_COUPLED_DENSE_FLOAT64" => "1") do
        @test !CC._dense_refinement_enabled(:metal) && CC._dense_float64_enabled(:cpu)
        @test CC._coupled_mode("BLAB_COUPLED_FEM_FLOAT64", :cpu) == :auto
    end
    @test_throws "BLAB_COUPLED_FEM_FLOAT64" withenv(() -> CC._fem_float64_enabled(:metal), "BLAB_COUPLED_FEM_FLOAT64" => "maybe")
end

@testset "Condensed coupled optimization switches: CPU and Metal defaults, auto and on" begin
    CC = BeatEngineCoupledCondensed
    names = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION", "BLAB_COUPLED_DENSE_FLOAT64", "BLAB_COUPLED_DENSE_REFINEMENT",
             "BLAB_COUPLED_FEM_FLOAT64", "BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION",
             "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION", "BLAB_COUPLED_FEM_SOLVER", "BLAB_COUPLED_INTERFACE_MASS_SOLVER",
             "BLAB_COUPLED_INTERFACE_MASS_OVERLAP", "BLAB_COUPLED_INTERFACE_BLOCKS", "BLAB_COUPLED_DEMAND_RECONSTRUCTION")
    withenv((name => nothing for name in names)...) do
        # Unset: CUDA and ROCm keep their established paths.
        for backend in (:cuda, :rocm)
            @test CC._transducer_condensation_mode(backend) == :off
            @test CC._interface_elimination_request(backend) == (:none, false)
            @test !CC._dense_double_assembly(backend) && !CC._dense_refinement_enabled(backend)
            @test !CC._fem_float64_enabled(backend) && !CC._demand_reconstruction_enabled(backend)
            @test CC._fem_solver_selection(backend) == :umfpack
            @test CC._interface_mass_solver_selection(backend) == :lu
            @test !CC._interface_mass_specialized(backend)
            @test CC._interface_pressure_elimination_mode(backend) == :off
            @test !CC._interface_mass_overlap_enabled(backend) && CC._interface_blocks_mode(backend) == :off
        end
        # Unset CPU: the same reductions and refined LU, with UMFPACK, Float32 FEM assembly,
        # and no interface mass overlap.
        @test CC._transducer_condensation_mode(:cpu) == :auto
        @test CC._interface_elimination_request(:cpu) == (:flux, false)
        @test CC._interface_pressure_elimination_mode(:cpu) == :off
        @test CC._dense_refinement_enabled(:cpu) && !CC._dense_float64_enabled(:cpu) && CC._dense_double_assembly(:cpu)
        @test !CC._fem_float64_enabled(:cpu) && CC._demand_reconstruction_enabled(:cpu)
        @test CC._fem_solver_selection(:cpu) == :umfpack
        @test CC._interface_mass_solver_selection(:cpu) == :cholmod
        @test CC._interface_mass_specialized(:cpu)
        @test !CC._interface_mass_overlap_enabled(:cpu) && CC._interface_blocks_mode(:cpu) == :auto
        @test CC._fem_solver_selection() == :umfpack && CC._transducer_condensation_mode() == :auto
        # Unset on Metal: the optimized configuration, each switch in `auto`.
        @test CC._transducer_condensation_mode(:metal) == :auto
        @test CC._interface_elimination_request(:metal) == (:flux, false)
        @test CC._interface_pressure_elimination_mode(:metal) == :off
        @test CC._dense_refinement_enabled(:metal) && !CC._dense_float64_enabled(:metal) && CC._dense_double_assembly(:metal)
        @test CC._fem_float64_enabled(:metal) && CC._demand_reconstruction_enabled(:metal)
        @test CC._fem_solver_selection(:metal) == :mumps
        @test CC._interface_mass_solver_selection(:metal) == :cholmod
        @test CC._interface_mass_overlap_enabled(:metal) && CC._interface_blocks_mode(:metal) == :auto
    end
    # Explicit values win on every backend.
    withenv("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1", "BLAB_COUPLED_FEM_SOLVER" => "umfpack",
            "BLAB_COUPLED_DENSE_REFINEMENT" => "off", "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "AUTO") do
        @test CC._interface_elimination_request(:cpu) == (:flux, true)
        @test CC._interface_elimination_request(:metal) == (:flux, true)
        @test CC._fem_solver_selection(:metal) == :umfpack
        @test !CC._dense_refinement_enabled(:metal)
        @test CC._transducer_condensation_mode(:cpu) == :auto
    end
    @test_throws "BLAB_COUPLED_INTERFACE_BLOCKS" withenv(() -> CC._interface_blocks_mode(:metal), "BLAB_COUPLED_INTERFACE_BLOCKS" => "maybe")
    @test_throws "BLAB_COUPLED_FEM_SOLVER" withenv(() -> CC._fem_solver_selection(:metal), "BLAB_COUPLED_FEM_SOLVER" => "pardiso")

    # Blocks: a mass matrix coupling two components is one block under `auto`, refused under `on`.
    load = sparse([2.0 0.1 0.01 0.0; 0.1 2.0 0.0 0.0; 0.01 0.0 2.0 0.1; 0.0 0.0 0.1 2.0])
    operators = InterfaceOperators(load, spzeros(1, 4), spzeros(4, 4), spzeros(4, 1))
    labels = [1, 1, 2, 2]
    fallbacks = String[]
    operator, _ = CC._interface_mass_operator_or_single_block(nothing, operators, 1:4, collect(1:4), labels, :lu, :auto, fallbacks)
    @test length(operator.blocks) == 1
    @test startswith(only(fallbacks), "interface blocks not used")
    rhs = randn(ComplexF64, 4, 2)
    @test norm(CC._interface_mass_apply(operator, rhs) - Matrix(load) \ rhs) < 1e-12
    @test_throws "different FEM components" CC._interface_mass_operator_or_single_block(
        nothing, operators, 1:4, collect(1:4), labels, :lu, :on, String[],
    )
    decoupled = copy(load); decoupled[1, 3] = decoupled[3, 1] = 0
    kept = String[]
    two, _ = CC._interface_mass_operator_or_single_block(nothing, InterfaceOperators(dropzeros(decoupled), spzeros(1, 4), spzeros(4, 4), spzeros(4, 1)),
                                                         1:4, collect(1:4), labels, :lu, :auto, kept)
    @test length(two.blocks) == 2 && isempty(kept)

    # A singular interface mass cannot be eliminated: `auto` solves unreduced and says why, `on` raises.
    singular = copy(load); singular[4, :] .= 0; singular[:, 4] .= 0
    singular_operators = InterfaceOperators(dropzeros(singular), spzeros(1, 4), spzeros(4, 4), spzeros(4, 1))
    for solver in (:lu, :cholmod)
        reasons = String[]
        unreduced, cached = CC._interface_mass_operator_or_unreduced(
            nothing, singular_operators, 1:4, collect(1:4), nothing, solver, :off, false, reasons,
        )
        @test isnothing(unreduced) && !cached
        @test startswith(only(reasons), "interface elimination not used")
        @test_throws LinearAlgebra.SingularException CC._interface_mass_operator_or_unreduced(
            nothing, singular_operators, 1:4, collect(1:4), nothing, solver, :off, true, String[],
        )
    end
    # A structural refusal is not a factorization failure: blocks `on` still raises under flux `auto`.
    @test_throws "different FEM components" CC._interface_mass_operator_or_unreduced(
        nothing, operators, 1:4, collect(1:4), labels, :lu, :on, false, String[],
    )
    regular_reasons = String[]
    regular, _ = CC._interface_mass_operator_or_unreduced(nothing, operators, 1:4, collect(1:4), nothing, :cholmod, :off, false, regular_reasons)
    @test only(regular.blocks).kind == :cholmod && isempty(regular_reasons)
end

@testset "Float32 dense LU with Float64 refinement (BLAB_COUPLED_DENSE_REFINEMENT)" begin
    Random.seed!(20260917)
    n = 240
    relative(reference, candidate) = norm(candidate - reference) / norm(reference)
    # Moderately conditioned, like the coupled systems (κ ~ 1e5): refinement converges.
    basis = qr(randn(ComplexF64, n, n)).Q
    singular_values = exp10.(range(0, -5; length=n))
    matrix = Matrix(basis * Diagonal(ComplexF64.(singular_values)) * qr(randn(ComplexF64, n, n)).Q')
    rhs = randn(ComplexF64, n, 3)
    reference = lu(matrix) \ rhs
    refined = BeatEngineCoupledCondensed.RefinedDenseLU(matrix)
    @test size(refined, 1) == n
    single = ComplexF64.(refined.factor \ ComplexF32.(rhs))
    @test relative(reference, single) > 1e-6
    solution = refined \ rhs
    @test eltype(solution) == ComplexF64
    @test relative(reference, solution) < 1e-10
    @test 1 <= refined.iterations <= 4
    @test isnothing(refined.fallback) && isnothing(refined.fallback_reason)
    @test relative(reference[:, 2], refined \ rhs[:, 2]) < 1e-10
    diagnostics = BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=refined,))
    @test diagnostics["dense_solver"] == "lu_float32_refined"
    @test diagnostics["dense_refinement_iterations"] == refined.iterations
    @test isnothing(diagnostics["dense_refinement_fallback_reason"])
    @test BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=lu(matrix),))["dense_solver"] == "lu_float64"

    # κ ~ 1e10 is beyond what a Float32 factor can refine: fall back to Float64 LU and say why.
    hard_values = exp10.(range(0, -10; length=n))
    hard = Matrix(basis * Diagonal(ComplexF64.(hard_values)) * qr(randn(ComplexF64, n, n)).Q')
    hard_refined = BeatEngineCoupledCondensed.RefinedDenseLU(hard)
    hard_solution = @test_logs (:warn, r"fell back to a Float64 LU") match_mode=:any hard_refined \ rhs
    @test hard_solution == lu(hard) \ rhs
    @test !isnothing(hard_refined.fallback)
    @test occursin("fell back to a Float64 LU", hard_refined.fallback_reason)
    @test hard_refined \ rhs == hard_solution
    hard_diagnostics = BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=hard_refined,))
    @test hard_diagnostics["dense_solver"] == "lu_float64_fallback"
    @test hard_diagnostics["dense_refinement_fallback_reason"] == hard_refined.fallback_reason

    # Accepted solves record their backward error (in units of the Float64 attainable one).
    @test 0 <= refined.backward_error <= 1
    @test diagnostics["dense_refinement_backward_error"] === nothing ||
          BeatEngineCoupledCondensed.dense_solver_diagnostics((factorization=refined,))["dense_refinement_backward_error"] <= 1

    # Right-hand sides of very different scale, including an all-zero column: every column must
    # meet the test on its own, and a zero column solves to exactly zero.
    mixed_rhs = hcat(rhs[:, 1] .* 1e-20, zeros(ComplexF64, n), rhs[:, 2] .* 1e20)
    mixed_reference = lu(matrix) \ mixed_rhs
    mixed = BeatEngineCoupledCondensed.RefinedDenseLU(matrix) \ mixed_rhs
    @test all(iszero, mixed[:, 2])
    @test relative(mixed_reference[:, 1], mixed[:, 1]) < 1e-10
    @test relative(mixed_reference[:, 3], mixed[:, 3]) < 1e-10

    # Entries outside the Float32 range cannot be narrowed: straight to Float64, with the reason.
    huge = matrix .* 1e40
    huge_refined = @test_logs (:warn, r"outside the Float32 range") BeatEngineCoupledCondensed.RefinedDenseLU(huge)
    @test isnothing(huge_refined.factor) && !isnothing(huge_refined.fallback)
    @test huge_refined \ rhs == lu(huge) \ rhs
    @test huge_refined.backward_error <= 1

    # Singular in Float32 but not in Float64: a pivot below Float32's range.
    tiny_pivot = Matrix{ComplexF64}(I, 4, 4)
    tiny_pivot[4, 4] = 1e-50
    tiny_refined = @test_logs (:warn, r"singular or not finite") BeatEngineCoupledCondensed.RefinedDenseLU(tiny_pivot)
    @test tiny_refined \ ones(ComplexF64, 4) == lu(tiny_pivot) \ ones(ComplexF64, 4)

    # Singular in Float64 too: a structured failure, not a silent result.
    singular = copy(matrix); singular[:, 1] .= 0
    @test_throws SingularException BeatEngineCoupledCondensed.RefinedDenseLU(singular)
    # Non-finite inputs are rejected.
    bad_matrix = copy(matrix); bad_matrix[1, 1] = NaN
    @test_throws ArgumentError BeatEngineCoupledCondensed.RefinedDenseLU(bad_matrix)
    bad_rhs = copy(rhs); bad_rhs[2, 2] = Inf
    @test_throws ArgumentError refined \ bad_rhs
end

@testset "Double-precision FEM matrices (BLAB_COUPLED_FEM_FLOAT64)" begin
    mesh32 = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), Float32(0.001))
    vertex_count = length(mesh32.vertices)
    mesh64 = VolumeMesh{Float64}(
        SVector{3,Float64}.(mesh32.vertices), mesh32.tetrahedra, mesh32.tetra_physical_tags, mesh32.boundary_faces,
        mesh32.boundary_physical_tags, mesh32.physical_names, mesh32.quadratic_tetrahedra, mesh32.quadratic_boundary_faces,
    )
    stiffness32, mass32 = assemble_p1_fem_matrices(mesh32)
    stiffness64, mass64 = assemble_p1_fem_matrices(mesh64)
    radiator_faces = findall(==(physical_tag(mesh32, 2, "Radiator")), mesh32.boundary_physical_tags)
    Random.seed!(20260917)
    loss = rand(Float32, vertex_count) .* 0.05f0
    wall = assemble_boundary_mass_matrix(mesh32, radiator_faces, collect(1:vertex_count))
    prepared = (
        stiffness=stiffness32,
        bulk_loss_factor_by_vertex=loss,
        wall_impedance_operators=[(matrix=wall, thickness_m=0.02f0, flow_resistivity_pa_s_per_m2=12000f0)],
    )
    store = Dict{Symbol,Any}()
    frequency, sound_speed, density = 20f0, 343f0, 1.21f0
    system = BeatEngineCoupledCondensed._fem_system_float64(store, mesh32, prepared, frequency, sound_speed, density)
    omega = 2pi * Float64(frequency)
    expected = assemble_fem_dynamic_stiffness(
        stiffness64, mass64, omega / Float64(sound_speed);
        bulk_loss_mass=spdiagm(0 => Float64.(loss)) * mass64,
    ) - BeatEngineCoupledCondensed.neumann_scale(Float64(density), omega) *
        miki_rigid_backed_surface_admittance(Float64(frequency), Float64(sound_speed), Float64(density),
                                             Float64(0.02f0), Float64(12000f0)) .*
        SparseMatrixCSC{Float64,Int}(wall)
    @test eltype(system) == ComplexF64
    @test system == expected
    # The matrices are cached by what they depend on, across frequencies and mesh objects, and are
    # rebuilt when a dependency changes in place (a persistent worker reusing its cache).
    cached = store[:matrices]
    BeatEngineCoupledCondensed._fem_system_float64(store, mesh32, prepared, 200f0, sound_speed, density)
    @test store[:matrices] === cached
    equal_mesh = deepcopy(mesh32)
    BeatEngineCoupledCondensed._fem_system_float64(store, equal_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] === cached
    moved_mesh = deepcopy(mesh32)
    moved_mesh.vertices[1] = moved_mesh.vertices[1] .+ 1f-4
    moved = BeatEngineCoupledCondensed._fem_system_float64(store, moved_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] !== cached && moved != system
    rebuilt = store[:matrices]
    prepared.bulk_loss_factor_by_vertex[2] += 0.01f0
    BeatEngineCoupledCondensed._fem_system_float64(store, moved_mesh, prepared, frequency, sound_speed, density)
    @test store[:matrices] !== rebuilt
    prepared.bulk_loss_factor_by_vertex[2] -= 0.01f0
    # Matrices with another element structure than the cached system are refused.
    wrong_structure = (stiffness=sparse(1.0f0 * I, vertex_count, vertex_count), bulk_loss_factor_by_vertex=loss,
                       wall_impedance_operators=NamedTuple[])
    @test_throws "structure" BeatEngineCoupledCondensed._fem_system_float64(nothing, mesh32, wrong_structure, frequency, sound_speed, density)

    # Why: the air spring of a cavity with nothing retained, `Cᵀ A⁻¹ C` for a transducer surface load
    # on the whole volume (the sealed-chamber case). Through the Float32-assembled K it loses digits
    # like 1/k^2; through the double-precision matrices it matches the Float64 assembly exactly.
    surface_load = assemble_boundary_mass_matrix(mesh64, radiator_faces, collect(1:vertex_count)) * ones(vertex_count)
    surface = SparseMatrixCSC{ComplexF64,Int}(reshape(ComplexF64.(surface_load), vertex_count, 1))
    air_spring(fem_system) = transpose(surface) * (lu(SparseMatrixCSC{ComplexF64,Int}(fem_system)) \ Matrix(surface))
    lossless = (stiffness=stiffness32, bulk_loss_factor_by_vertex=zeros(Float32, vertex_count), wall_impedance_operators=NamedTuple[])
    reference = air_spring(assemble_fem_dynamic_stiffness(stiffness64, mass64, omega / Float64(sound_speed)))
    single = air_spring(SparseMatrixCSC{ComplexF64,Int}(
        assemble_fem_dynamic_stiffness(stiffness32, mass32, Float32(2pi) * frequency / sound_speed),
    ))
    double = air_spring(BeatEngineCoupledCondensed._fem_system_float64(nothing, mesh32, lossless, frequency, sound_speed, density))
    @test norm(single - reference) / norm(reference) > 1e-4
    @test norm(double - reference) / norm(reference) < 1e-12
end

@testset "Specialized interface flux elimination algebra" begin
    Random.seed!(20260916)
    # Three FEM components with interleaved vertex numbering.
    vertex_count = 90
    labels = rand(1:3, vertex_count)
    labels[1:3] = [1, 2, 3]
    fem_rows, fem_columns = Int[], Int[]
    for component in 1:3
        members = findall(==(component), labels)
        for (a, b) in zip(members[1:(end - 1)], members[2:end])
            push!(fem_rows, a, b); push!(fem_columns, b, a)
        end
        for _ in 1:length(members)
            a, b = rand(members, 2)
            push!(fem_rows, a); push!(fem_columns, b)
        end
    end
    fem_graph = sparse(vcat(fem_rows, 1:vertex_count), vcat(fem_columns, 1:vertex_count), ones(ComplexF64, length(fem_rows) + vertex_count))
    computed = BeatEngineCoupledCondensed._fem_component_labels(fem_graph)
    @test length(unique(computed)) == 3
    @test all((computed[a] == computed[b]) == (labels[a] == labels[b]) for a in 1:vertex_count, b in 1:vertex_count)

    # Γ: a sorted subset; interface dofs a shuffled numbering of it. Components 1 and 3 carry Γ.
    gamma = sort(vcat(findall(==(1), labels)[1:14], findall(==(3), labels)[1:9]))
    n = length(gamma)
    gamma_labels = computed[gamma]
    gamma_dof = randperm(n)
    # Symmetric positive definite mass, block diagonal between components, in dof space.
    mass_dof = zeros(n, n)
    for i in 1:n, j in 1:n
        gamma_labels[i] == gamma_labels[j] || continue
        a, b = gamma_dof[i], gamma_dof[j]
        mass_dof[a, b] = i == j ? 2.0 + rand() : (rand() < 0.3 ? 0.2 * rand() : 0.0)
    end
    mass_dof = (mass_dof + mass_dof') / 2 + n * 0.05 * I
    fem_load = spzeros(Float64, vertex_count, n)
    for (index, vertex) in enumerate(gamma)
        fem_load[vertex, :] = mass_dof[gamma_dof[index], :]
    end
    operators = InterfaceOperators(fem_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    mass_gamma = Matrix(fem_load[gamma, :])
    schur = zeros(ComplexF64, n, n)
    for i in 1:n, j in 1:i
        gamma_labels[i] == gamma_labels[j] || continue
        schur[i, j] = schur[j, i] = complex(randn(), randn())
    end
    motion = randn(ComplexF64, n, 2)
    coupling = randn(ComplexF64, 7, n)
    reference_w = mass_gamma \ schur
    reference_v = mass_gamma \ motion
    # Also a dof numbering with each component contiguous, the layout the strided view serves.
    contiguous_dof = invperm(sortperm(collect(zip(gamma_labels, randperm(n)))))
    layouts = ((gamma_dof, fem_load), (contiguous_dof, begin
        load = spzeros(Float64, vertex_count, n)
        permuted = zeros(n, n)
        for i in 1:n, j in 1:n
            permuted[contiguous_dof[i], contiguous_dof[j]] = mass_dof[gamma_dof[i], gamma_dof[j]]
        end
        for (index, vertex) in enumerate(gamma)
            load[vertex, :] = permuted[contiguous_dof[index], :]
        end
        load
    end))
    for (layout_dof, layout_load) in layouts, solver in (:lu, :cholmod), block_labels in (nothing, gamma_labels)
    gamma_dof = layout_dof
    operators = InterfaceOperators(layout_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    mass_gamma = Matrix(layout_load[gamma, :])
    reference_w = mass_gamma \ schur
    reference_v = mass_gamma \ motion
        store = Dict{Symbol,Any}()
        operator, cached = BeatEngineCoupledCondensed._interface_mass_operator(store, operators, gamma, gamma_dof, block_labels, solver)
        @test !cached
        @test length(operator.blocks) == (isnothing(block_labels) ? 1 : 2)
        @test all(block.kind == solver && isnothing(block.fallback_reason) for block in operator.blocks)
        diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics((interface_elimination=:flux, interface_elimination_data=(mass_operator=operator,)))
        @test diagnostics["interface_mass_solver_requested"] == String(solver)
        @test diagnostics["interface_mass_solver"] == String(solver)
        @test diagnostics["interface_mass_block_count"] == length(operator.blocks)
        @test isempty(diagnostics["interface_mass_fallback_reasons"])
        @test sort(vcat([block.rows for block in operator.blocks]...)) == 1:n
        @test all(issorted(block.dofs) && block.dofs == gamma_dof[block.rows] for block in operator.blocks)
        again, cached_again = BeatEngineCoupledCondensed._interface_mass_operator(store, operators, gamma, gamma_dof, block_labels, solver)
        @test cached_again && again === operator
        other = solver == :lu ? :cholmod : :lu
        @test !last(BeatEngineCoupledCondensed._interface_mass_operator(store, operators, gamma, gamma_dof, block_labels, other))

        presolve = BeatEngineCoupledCondensed._flux_mass_presolve(operator, schur, motion)
        assembled = zeros(ComplexF64, n, n)
        for (block, w) in zip(operator.blocks, presolve.schur_blocks)
            assembled[block.dofs, block.rows] = w
        end
        @test norm(assembled - reference_w) / norm(reference_w) < 1e-12
        @test norm(presolve.motion_solution - reference_v) / norm(reference_v) < 1e-12
        rhs = randn(ComplexF64, n, 3)
        @test norm(BeatEngineCoupledCondensed._interface_mass_apply(operator, rhs) - mass_gamma \ rhs) / norm(mass_gamma \ rhs) < 1e-12
        # Per-block products scattered into Γ columns reproduce the full product.
        product = zeros(ComplexF64, 7, n)
        for (block, w) in zip(operator.blocks, presolve.schur_blocks)
            product[:, block.rows] .+= coupling[:, block.dofs] * w
        end
        @test norm(product - coupling * reference_w) / norm(coupling * reference_w) < 1e-12
        @test haskey(presolve.split, :block_check) == !isnothing(block_labels)
        # The engine's block products and scatter: Γ column j lands on column target[j] (with a
        # shared target summing), next to an untouched offset row range.
        targets = vcat(randperm(n + 2)[1:(n - 1)], 0)
        targets[end] = targets[1]
        dense = zeros(ComplexF64, 9, n + 2)
        BeatEngineCoupledCondensed._flux_block_products!(
            dense, 2:8, targets, operator, presolve.schur_blocks, coupling, Dict{Symbol,Float64}(),
        )
        expected = zeros(ComplexF64, 9, n + 2)
        full_product = coupling * reference_w
        for j in 1:n
            expected[2:8, targets[j]] .+= full_product[:, j]
        end
        @test norm(dense - expected) / norm(expected) < 1e-12
        @test all(block.contiguous for block in operator.blocks) == (layout_dof === contiguous_dof || isnothing(block_labels))
    end
    gamma_dof = first(first(layouts))
    operators = InterfaceOperators(fem_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    mass_gamma = Matrix(fem_load[gamma, :])
    reference_w = mass_gamma \ schur
    reference_v = mass_gamma \ motion
    # Structure violations are refused, not averaged away.
    blocks_operator, _ = BeatEngineCoupledCondensed._interface_mass_operator(nothing, operators, gamma, gamma_dof, gamma_labels, :cholmod)
    cross = findfirst(j -> gamma_labels[j] != gamma_labels[1], 1:n)
    coupled_schur = copy(schur); coupled_schur[1, cross] = coupled_schur[cross, 1] = 1e-30
    @test_throws "different FEM components" BeatEngineCoupledCondensed._flux_mass_presolve(blocks_operator, coupled_schur, motion)
    cross_load = copy(fem_load); cross_load[gamma[1], gamma_dof[cross]] = 0.01
    cross_operators = InterfaceOperators(cross_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    @test_throws "different FEM components" BeatEngineCoupledCondensed._interface_mass_operator(nothing, cross_operators, gamma, gamma_dof, gamma_labels, :lu)
    # An explicitly stored zero between components couples nothing: the check reads values, as
    # the Schur check does, not the stored pattern.
    stored_zero_load = copy(fem_load)
    stored_zero_load[gamma[1], gamma_dof[cross]] = 1.0
    stored_zero_column = nzrange(stored_zero_load, gamma_dof[cross])
    stored_zero_load.nzval[stored_zero_column[findfirst(==(gamma[1]), rowvals(stored_zero_load)[stored_zero_column])]] = 0.0
    @test nnz(stored_zero_load) == nnz(fem_load) + 1
    @test stored_zero_load == fem_load
    stored_zero_operators = InterfaceOperators(stored_zero_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    for solver in (:lu, :cholmod)
        stored_zero, _ = BeatEngineCoupledCondensed._interface_mass_operator(nothing, stored_zero_operators, gamma, gamma_dof, gamma_labels, solver)
        @test length(stored_zero.blocks) == 2
        @test all(block.kind == solver && isnothing(block.fallback_reason) for block in stored_zero.blocks)
        @test norm(BeatEngineCoupledCondensed._interface_mass_apply(stored_zero, motion) - reference_v) / norm(reference_v) < 1e-12
    end
    # An indefinite block cannot be Cholesky-factored: LU takes over and says why.
    indefinite_load = copy(fem_load); indefinite_load[gamma, :] .*= -1
    indefinite_operators = InterfaceOperators(indefinite_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    fallback, _ = BeatEngineCoupledCondensed._interface_mass_operator(nothing, indefinite_operators, gamma, gamma_dof, nothing, :cholmod)
    @test only(fallback.blocks).kind == :lu
    @test occursin("CHOLMOD", only(fallback.blocks).fallback_reason)
    @test norm(BeatEngineCoupledCondensed._interface_mass_apply(fallback, motion) + reference_v) / norm(reference_v) < 1e-12
    diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics((interface_elimination=:flux, interface_elimination_data=(mass_operator=fallback,)))
    @test diagnostics["interface_mass_solver_requested"] == "cholmod"
    @test diagnostics["interface_mass_solver"] == "lu"
    @test diagnostics["interface_mass_block_count"] == 1
    @test only(diagnostics["interface_mass_fallback_reasons"]) == only(fallback.blocks).fallback_reason
    # One indefinite block of two: that block falls back, the other keeps Cholesky, and the
    # diagnostics say both.
    first_block = findall(==(gamma_labels[1]), gamma_labels)
    partial_load = copy(fem_load); partial_load[gamma[first_block], :] .*= -1; dropzeros!(partial_load)
    partial_operators = InterfaceOperators(partial_load, spzeros(4, n), spzeros(n, vertex_count), spzeros(n, 4))
    partial, _ = BeatEngineCoupledCondensed._interface_mass_operator(nothing, partial_operators, gamma, gamma_dof, gamma_labels, :cholmod)
    @test sort([String(block.kind) for block in partial.blocks]) == ["cholmod", "lu"]
    partial_diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics((interface_elimination=:flux, interface_elimination_data=(mass_operator=partial,)))
    @test partial_diagnostics["interface_mass_solver"] == "mixed"
    @test partial_diagnostics["interface_mass_block_count"] == 2
    @test length(partial_diagnostics["interface_mass_fallback_reasons"]) == 1
    @test occursin("CHOLMOD", only(partial_diagnostics["interface_mass_fallback_reasons"]))
    # An operator over no interface vertices has no blocks and solved nothing.
    empty_operator = (count=0, blocks=NamedTuple[], solver=:cholmod)
    empty_diagnostics = BeatEngineCoupledCondensed.interface_mass_diagnostics((interface_elimination=:flux, interface_elimination_data=(mass_operator=empty_operator,)))
    @test isnothing(empty_diagnostics["interface_mass_solver"])
    @test empty_diagnostics["interface_mass_block_count"] == 0
    # Pressure elimination never solves with M_Γ, whatever data it carries.
    @test isnothing(BeatEngineCoupledCondensed.interface_mass_diagnostics((interface_elimination=:pressure, interface_elimination_data=(mass_operator=partial,)))["interface_mass_solver"])
end

# MUMPS ships only with the macOS Metal environment (see BeatEngineMumps), and its tests run there
# through mumps_tests.jl. Anywhere else the switch has to fall back to UMFPACK and say why.
if isnothing(Base.locate_package(BeatEngineCoupledCondensed.BeatEngineMumps.MUMPS_SEQ_PKGID))
    @testset "MUMPS absent from the environment falls back to UMFPACK" begin
        base, operators, retained = condensed_synthetic_case(Float64)
        symmetric = SparseMatrixCSC{ComplexF64,Int}(base + transpose(base))
        library = BeatEngineCoupledCondensed.BeatEngineMumps.mumps_library()
        @test !library.available
        @test occursin("MUMPS_seq_jll is not in this Julia environment", library.reason)
        umfpack = BeatEngineCoupledCondensed._build_condensation(symmetric, operators, retained)
        fallback = BeatEngineCoupledCondensed._build_condensation(symmetric, operators, retained; fem_solver=:mumps)
        @test fallback.backend == :cpu_umfpack
        @test fallback.fem_solver_requested == :mumps
        @test startswith(fallback.fem_solver_fallback_reason, "MUMPS unavailable: MUMPS_seq_jll is not in this Julia environment")
        @test fallback.schur == umfpack.schur
    end
else
    include(joinpath(@__DIR__, "coupled_mumps_tests.jl"))
end

@testset "host ZGEMM for the flux-elimination products" begin
    zgemm = BeatEngineCoupledCondensed._host_zgemm
    A = ComplexF64[sin(i + 2j) + im * cos(i * j) for i in 1:37, j in 1:23]
    B = ComplexF64[cos(i - j) - im * sin(2i + j) for i in 1:23, j in 1:11]
    reference = A * B
    for setting in ("auto", "blas")
        withenv("BLAB_COUPLED_HOST_ZGEMM" => setting) do
            @test isapprox(zgemm(A, B), reference; rtol=1e-13)
            # A contiguous column range is read in place, as `_flux_block_products!` does.
            wide = hcat(zeros(ComplexF64, 37, 4), A, zeros(ComplexF64, 37, 3))
            @test isapprox(zgemm(view(wide, :, 5:27), B), reference; rtol=1e-13)
            # Non-unit row stride and other element types take Julia's `*`.
            strided = view(vcat(A, A), 1:2:74, :)
            @test isapprox(zgemm(strided, B), Matrix(strided) * B; rtol=1e-13)
            @test zgemm(ComplexF32.(A), ComplexF32.(B)) == ComplexF32.(A) * ComplexF32.(B)
            @test size(zgemm(zeros(ComplexF64, 5, 0), zeros(ComplexF64, 0, 3))) == (5, 3)
            @test iszero(zgemm(zeros(ComplexF64, 5, 0), zeros(ComplexF64, 0, 3)))
            @test size(zgemm(zeros(ComplexF64, 0, 4), zeros(ComplexF64, 4, 3))) == (0, 3)
            @test size(zgemm(zeros(ComplexF64, 5, 4), zeros(ComplexF64, 4, 0))) == (5, 0)
            # A row sub-range of a taller parent: the leading dimension exceeds the row count.
            tall = vcat(A, ComplexF64.(randn(11, 23)))
            @test isapprox(zgemm(view(tall, 1:37, :), B), reference; rtol=1e-13)
            taller = hcat(zeros(ComplexF64, 48, 2), tall)
            @test isapprox(zgemm(view(taller, 1:37, 3:25), B), reference; rtol=1e-13)
            # A reversed column range is a StridedMatrix with a negative column stride: `*` handles it.
            @test isapprox(zgemm(view(A, :, 23:-1:1), B[end:-1:1, :]), reference; rtol=1e-13)
            @test isapprox(zgemm(A[:, end:-1:1], view(B, 23:-1:1, :)), reference; rtol=1e-13)
            @test isapprox(zgemm(A, view(B, :, 11:-1:1)), reference[:, end:-1:1]; rtol=1e-13)
            # A product at the solver's shape (about 3,100 rows), where threading engages.
            big_A = ComplexF64[sin(0.01i + 0.3j) + im * cos(0.02i * j) for i in 1:3100, j in 1:300]
            big_B = ComplexF64[cos(0.1i - 0.2j) - im * sin(0.05i + j) for i in 1:300, j in 1:300]
            @test isapprox(zgemm(big_A, big_B), big_A * big_B; rtol=1e-12)
        end
    end
    # Availability-aware: macOS before 13.3 lacks the new-LAPACK entry point, and `auto` then uses BLAS.
    if BeatEngineCoupledCondensed._accelerate_zgemm() != C_NULL
        withenv("BLAB_COUPLED_HOST_ZGEMM" => "accelerate") do
            @test isapprox(zgemm(A, B), reference; rtol=1e-13)
        end
    else
        @test_throws "not available" withenv(() -> BeatEngineCoupledCondensed._host_zgemm_symbol(),
            "BLAB_COUPLED_HOST_ZGEMM" => "accelerate")
    end
    @test withenv(() -> BeatEngineCoupledCondensed._host_zgemm_symbol(), "BLAB_COUPLED_HOST_ZGEMM" => "blas") == C_NULL
    @test_throws "BLAB_COUPLED_HOST_ZGEMM" withenv(() -> BeatEngineCoupledCondensed._host_zgemm_symbol(),
        "BLAB_COUPLED_HOST_ZGEMM" => "mkl")
    # The diagnostic reports instead of throwing, so a solve without products still returns.
    path(setting) = withenv(BeatEngineCoupledCondensed.host_zgemm_path, "BLAB_COUPLED_HOST_ZGEMM" => setting)
    @test path("blas") == "blas"
    @test startswith(path("mkl"), "unavailable: ") && occursin("BLAB_COUPLED_HOST_ZGEMM", path("mkl"))
    @test path("auto") in ("blas", "accelerate")
end
