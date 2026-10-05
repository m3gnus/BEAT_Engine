# Gates the vectorised CPU regular kernel against the scalar one; standalone or after runtests.jl.
using Test
using StaticArrays
using LinearAlgebra

if !isdefined(Main, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore

function _beat_cpu_simd_test_mesh(::Type{T}, symmetry::Symbol) where {T<:AbstractFloat}
    if symmetry == :ground
        # The rigid y0 half-space fixture from runtests.jl.
        vertices = [
            SVector{3,T}(0, 0.2, 0),
            SVector{3,T}(0.04, 0.2, 0),
            SVector{3,T}(0, 0.2, 0.04),
        ]
        return BoundaryMesh(vertices, [(1, 2, 3)], [1])
    end
    name = symmetry == :x ? "sample_half.msh" :
        symmetry == :xy ? "sample_quarter.msh" : "sample.msh"
    mesh = load_gmsh22_with_tags(joinpath(@__DIR__, "..", "test_meshes", name), T(0.001))
    mesh = snap_symmetry_planes(mesh, symmetry)
    validate_symmetry_fundamental_domain!(mesh, symmetry)
    return mesh
end

function _beat_cpu_simd_test_entrywise(actual, scalar, tol)
    @test size(actual) == size(scalar)
    @test all(isfinite, actual)
    @test maximum(abs, actual .- scalar) <= tol * maximum(abs, scalar)
end

@testset "CPU fast sincos" begin
    for T in (Float32, Float64)
        @test BeatEngineCore._beat_cpu_fast_sincos(zero(T)) == (zero(T), one(T))
        sine_error = zero(T)
        cosine_error = zero(T)
        for value in range(0, 2500; length=1_000_001)
            x = T(value)
            s, c = BeatEngineCore._beat_cpu_fast_sincos(x)
            bs, bc = sincos(x)
            sine_error = max(sine_error, abs(s - bs))
            cosine_error = max(cosine_error, abs(c - bc))
        end
        @test sine_error <= 2 * eps(T)
        @test cosine_error <= 2 * eps(T)
    end
end

@testset "CPU regular kernel selection" begin
    withenv("BLAB_BEAT_CPU_REGULAR_KERNEL" => nothing) do
        @test BeatEngineCore.beat_cpu_regular_kernel() === :simd
    end
    for (value, expected) in (("scalar", :scalar), (" SIMD ", :simd), (" Scalar ", :scalar))
        withenv("BLAB_BEAT_CPU_REGULAR_KERNEL" => value) do
            @test BeatEngineCore.beat_cpu_regular_kernel() === expected
        end
    end
    withenv("BLAB_BEAT_CPU_REGULAR_KERNEL" => "invalid") do
        @test_throws ErrorException BeatEngineCore.beat_cpu_regular_kernel()
    end
    @test_throws ErrorException BeatEngineCore._beat_cpu_validated_regular_kernel(:invalid)
end

@testset "CPU SIMD fused systems" begin
    for T in (Float32, Float64), order in (2, 4), symmetry in (:off, :x, :xy, :ground)
        @testset "$T order=$order symmetry=$symmetry" begin
            mesh = _beat_cpu_simd_test_mesh(T, symmetry)
            p1 = build_p1_space(mesh)
            dp0 = build_dp0_space(mesh)
            rule = triangle_rule(T, order)
            k = T(2pi * 1500 / 343)
            tol = T === Float32 ? 2f-6 : 1e-13
            identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
            identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
            q = Complex{T}[
                Complex{T}(sin(T(0.7 * row + 1.3 * drive)), cos(T(0.4 * row - 0.9 * drive)))
                for row in 1:dp0.global_dof_count, drive in 1:3
            ]
            selections = symmetry == :ground ? [collect(eachindex(mesh.faces))] :
                [collect(1:min(48, length(mesh.faces))), collect(5:2:min(75, length(mesh.faces)))]
            for indices in selections
                singular_cache = build_singular_correction_cache(mesh, 2, indices)
                cache = build_beat_cpu_assembly_cache(
                    mesh, p1, dp0, rule; element_indices=indices, symmetry_mode=symmetry,
                )
                kwargs = (;
                    identity_p1_p1, identity_p1_dp0, element_indices=indices,
                    skip_singular=false, singular_cache, symmetry_mode=symmetry,
                )
                reference_simd = nothing
                for cpu_cache in (nothing, cache)
                    scalar = assemble_burton_miller_neumann_system_cpu(
                        mesh, p1, dp0, q, k, rule; kwargs..., cpu_cache, regular_kernel=:scalar,
                    )
                    simd = assemble_burton_miller_neumann_system_cpu(
                        mesh, p1, dp0, q, k, rule; kwargs..., cpu_cache, regular_kernel=:simd,
                    )
                    @test simd.regular_kernel === :simd
                    unselected = withenv("BLAB_BEAT_CPU_REGULAR_KERNEL" => "simd") do
                        assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, k, rule; kwargs..., cpu_cache)
                    end
                    @test unselected.regular_kernel === :scalar
                    @test unselected.matrix == scalar.matrix
                    @test unselected.rhs == scalar.rhs
                    @test scalar.regular_kernel === :scalar
                    @test simd.drive_count == 3
                    _beat_cpu_simd_test_entrywise(simd.matrix, scalar.matrix, tol)
                    _beat_cpu_simd_test_entrywise(simd.rhs, scalar.rhs, tol)
                    if reference_simd === nothing
                        reference_simd = simd
                    else
                        @test simd.matrix == reference_simd.matrix
                        @test simd.rhs == reference_simd.rhs
                    end
                    if T === Float32 && order == 2 && symmetry == :off && indices[1] == 1 && cpu_cache === nothing
                        scalar_pressure = scalar.matrix \ scalar.rhs
                        simd_pressure = simd.matrix \ simd.rhs
                        @test norm(simd_pressure - scalar_pressure) <= 1f-4 * norm(scalar_pressure)
                    end
                end

                # Isolate the regular pass: unchanged singular scatter visits
                # shared rows in different orders in serial and coloured mode.
                regular_kwargs = merge(kwargs, (; skip_singular=true))
                serial = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; regular_kwargs..., threaded=false, regular_kernel=:simd,
                )
                parallel = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; regular_kwargs..., threaded=true, regular_kernel=:simd,
                )
                @test serial.matrix == parallel.matrix
                @test serial.rhs == parallel.rhs
                serial_cache = build_beat_cpu_assembly_cache(
                    mesh, p1, dp0, rule; element_indices=indices, symmetry_mode=symmetry, threaded=false,
                )
                for cpu_cache in (cache, serial_cache)
                    cached_regular = assemble_burton_miller_neumann_system_cpu(
                        mesh, p1, dp0, q, k, rule; regular_kwargs..., cpu_cache, regular_kernel=:simd,
                    )
                    @test cached_regular.matrix == serial.matrix
                    @test cached_regular.rhs == serial.rhs
                end
                full_serial = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; kwargs..., threaded=false, regular_kernel=:simd,
                )
                _beat_cpu_simd_test_entrywise(full_serial.matrix, reference_simd.matrix, tol)
                _beat_cpu_simd_test_entrywise(full_serial.rhs, reference_simd.rhs, tol)
                scalar_serial = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; kwargs..., threaded=false, regular_kernel=:scalar,
                )
                _beat_cpu_simd_test_entrywise(full_serial.matrix, scalar_serial.matrix, tol)
                _beat_cpu_simd_test_entrywise(full_serial.rhs, scalar_serial.rhs, tol)
                cached_serial = assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; kwargs..., cpu_cache=serial_cache, regular_kernel=:simd,
                )
                @test cached_serial.matrix == full_serial.matrix
                @test cached_serial.rhs == full_serial.rhs
            end
        end
    end
end

@testset "CPU SIMD trial block boundaries" begin
    for T in (Float32, Float64), order in (2, 4)
        mesh = _beat_cpu_simd_test_mesh(T, :off)
        rule = triangle_rule(T, order)
        source = BeatEngineCore._beat_cpu_element_data(mesh, build_p1_space(mesh), build_dp0_space(mesh))
        quadrature = BeatEngineCore._beat_cpu_regular_quadrature_data(mesh, rule)
        count = BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE + 4
        # Preserve the bundled geometry but give every trial its own columns.
        elements = [
            BeatEngineCore.BeatCpuElementData(
                (3 * j - 2, 3 * j - 1, 3 * j), source[j].vertices, source[j].normal,
                source[j].curls, (3 * j - 2, 3 * j - 1, 3 * j), j, source[j].area,
            )
            for j in 1:count
        ]
        indices = reverse(collect(2:count))
        soa = BeatEngineCore.BeatCpuRegularSoA(elements, quadrature, indices)
        k = T(20)
        coupling = Complex{T}(0, 1) / k
        q = Complex{T}[Complex{T}(T(j / count), T(drive)) for j in 1:count, drive in 1:3]
        lhs = zeros(Complex{T}, 3, 3 * count)
        rhs = zeros(Complex{T}, 3, 3)
        scalar_lhs = copy(lhs)
        scalar_rhs = copy(rhs)
        scratch = BeatEngineCore.BeatCpuRegularScratch{T}(BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE)
        BeatEngineCore._beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q, elements[1], quadrature[1], elements, soa, k, coupling, scratch, true,
        )
        BeatEngineCore._beat_cpu_bm_regular_test!(
            scalar_lhs, scalar_rhs, q, elements, 1, indices, k, quadrature, coupling,
        )
        tol = T === Float32 ? 2f-6 : 1e-13
        _beat_cpu_simd_test_entrywise(lhs, scalar_lhs, tol)
        _beat_cpu_simd_test_entrywise(rhs, scalar_rhs, tol)
        operators = (zeros(Complex{T}, 3, count), zeros(Complex{T}, 3, 3 * count),
                     zeros(Complex{T}, 3, count), zeros(Complex{T}, 3, 3 * count))
        reference = map(copy, operators)
        work = BeatEngineCore.BeatCpuOperatorRegularScratch{T}(BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE)
        BeatEngineCore._beat_cpu_accumulate_regular_test_simd!(
            operators..., elements[1], quadrature[1], elements, soa, k, work, true,
        )
        BeatEngineCore._beat_cpu_accumulate_regular_test!(
            reference..., elements, 1, indices, k, quadrature,
        )
        for (actual, scalar) in zip(operators, reference)
            _beat_cpu_simd_test_entrywise(actual, scalar, tol)
        end
    end
end

@testset "CPU SIMD four operators" begin
    for T in (Float32, Float64), order in (2, 4), symmetry in (:off, :x)
        mesh = _beat_cpu_simd_test_mesh(T, symmetry)
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(T, order)
        k = T(2pi * 1500 / 343)
        tol = T === Float32 ? 2f-6 : 1e-13
        indices = collect(5:2:min(75, length(mesh.faces)))
        cache = build_beat_cpu_assembly_cache(
            mesh, p1, dp0, rule; element_indices=indices, symmetry_mode=symmetry,
        )
        for cpu_cache in (nothing, cache)
            kwargs = (; skip_singular=false, element_indices=indices, symmetry_mode=symmetry, cpu_cache)
            scalar = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; kwargs..., regular_kernel=:scalar)
            simd = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; kwargs..., regular_kernel=:simd)
            withenv("BLAB_BEAT_CPU_REGULAR_KERNEL" => "simd") do
                default = assemble_regular_galerkin_operators_cpu(mesh, p1, dp0, k, rule; kwargs...)
                for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                    @test getproperty(default, name) == getproperty(scalar, name)
                    _beat_cpu_simd_test_entrywise(getproperty(simd, name), getproperty(scalar, name), tol)
                end
                # The environment selects nothing by itself: the dispatcher is
                # scalar too unless a CPU entry point passes the kernel.
                dispatched = assemble_regular_galerkin_operators(mesh, p1, dp0, k, rule; kwargs..., backend=:cpu)
                opted_in = assemble_regular_galerkin_operators(
                    mesh, p1, dp0, k, rule; kwargs..., backend=:cpu, cpu_regular_kernel=:simd,
                )
                for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                    @test getproperty(dispatched, name) == getproperty(scalar, name)
                    @test getproperty(opted_in, name) == getproperty(simd, name)
                end
            end
        end
        @test_throws ErrorException assemble_regular_galerkin_operators_cpu(
            mesh, p1, dp0, k, rule; regular_kernel=:invalid,
        )
    end
end

@testset "CPU SIMD coincident regular points and empty subsets" begin
    for T in (Float32, Float64), order in (2, 4)
        vertices = [SVector{3,T}(0, 0, 0), SVector{3,T}(0.04, 0, 0), SVector{3,T}(0, 0.04, 0)]
        mesh = BoundaryMesh(vcat(vertices, vertices), [(1, 2, 3), (4, 5, 6)], [1, 1])
        p1 = build_p1_space(mesh)
        dp0 = build_dp0_space(mesh)
        rule = triangle_rule(T, order)
        elements = BeatEngineCore._beat_cpu_element_data(mesh, p1, dp0)
        quad = BeatEngineCore._beat_cpu_regular_quadrature_data(mesh, rule)
        @test !BeatEngineCore.elements_are_adjacent(elements[1].face, elements[2].face)
        @test quad[1].points == quad[2].points
        soa = BeatEngineCore.BeatCpuRegularSoA(elements, quad, [2])
        @test soa.indices == [2]
        @test vec(soa.px) == getindex.(quad[2].points, 1)
        @test BeatEngineCore.BeatCpuRegularSoA(elements, quad, Int[]).n == 0
        @test BeatEngineCore.BeatCpuRegularSoA(
            BeatEngineCore.BeatCpuElementData{T}[], BeatEngineCore.BeatCpuRegularQuadratureData{T}[], Int[],
        ).n == 0
        k = T(20)
        coupling = Complex{T}(0, 1) / k
        q = fill(Complex{T}(1, 0.5), 2, 3)
        lhs = zeros(Complex{T}, 6, 6)
        rhs = zeros(Complex{T}, 6, 3)
        scalar_lhs = similar(lhs); fill!(scalar_lhs, 0)
        scalar_rhs = similar(rhs); fill!(scalar_rhs, 0)
        scratch = BeatEngineCore.BeatCpuRegularScratch{T}(BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE)
        BeatEngineCore._beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q, elements[1], quad[1], elements, soa, k, coupling, scratch, true,
        )
        lb, rb = BeatEngineCore._beat_cpu_bm_regular_pair_blocks(
            elements[1], elements[2], quad[1], quad[2], dot(elements[1].normal, elements[2].normal),
            T(4) * elements[1].area * elements[2].area, k, coupling,
        )
        BeatEngineCore._beat_cpu_bm_scatter!(scalar_lhs, scalar_rhs, q, (1, 2, 3), (4, 5, 6), 2, lb, rb, Val(false))
        tol = T === Float32 ? 2f-6 : 1e-13
        _beat_cpu_simd_test_entrywise(lhs, scalar_lhs, tol)
        _beat_cpu_simd_test_entrywise(rhs, scalar_rhs, tol)
        operators = (zeros(Complex{T}, 6, 2), zeros(Complex{T}, 6, 6),
                     zeros(Complex{T}, 6, 2), zeros(Complex{T}, 6, 6))
        reference = map(copy, operators)
        work = BeatEngineCore.BeatCpuOperatorRegularScratch{T}(BeatEngineCore._BEAT_CPU_REGULAR_BLOCK_SIZE)
        BeatEngineCore._beat_cpu_accumulate_regular_test_simd!(
            operators..., elements[1], quad[1], elements, soa, k, work, true,
        )
        BeatEngineCore._beat_cpu_accumulate_regular_pair!(
            reference..., elements[1], elements[2], quad[1], quad[2],
            dot(elements[1].normal, elements[2].normal), T(4) * elements[1].area * elements[2].area, k,
        )
        for (actual, scalar) in zip(operators, reference)
            _beat_cpu_simd_test_entrywise(actual, scalar, tol)
        end
        BeatEngineCore._beat_cpu_bm_regular_test_simd!(
            lhs, rhs, q, elements[1], quad[1], elements,
            BeatEngineCore.BeatCpuRegularSoA(elements, quad, Int[]), k, coupling, scratch, true,
        )
        _beat_cpu_simd_test_entrywise(lhs, scalar_lhs, tol)
        @test_throws ErrorException assemble_burton_miller_neumann_system_cpu(
            mesh, p1, dp0, q, k, rule; identity_p1_p1=zeros(T, 6, 6),
            identity_p1_dp0=zeros(T, 6, 2), regular_kernel=:invalid,
        )
    end
end

# The positive-time convention hands the kernels a negative wavenumber, so the
# phase, the gradient term and the Burton-Miller coupling all change sign.
@testset "CPU SIMD kernels under both phasor conventions" begin
    for T in (Float32, Float64), convention in (BeatEngineCore.NEGATIVE_TIME_PHASOR, BeatEngineCore.POSITIVE_TIME_PHASOR)
        BeatEngineCore.with_phasor_convention(convention) do
            mesh = _beat_cpu_simd_test_mesh(T, :off)
            p1 = build_p1_space(mesh)
            dp0 = build_dp0_space(mesh)
            rule = triangle_rule(T, 2)
            k = T(2pi * 1500 / 343)
            tol = T === Float32 ? 2f-6 : 1e-13
            indices = collect(1:min(48, length(mesh.faces)))
            identity_p1_p1 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
            identity_p1_dp0 = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
            q = Complex{T}[Complex{T}(sin(T(0.7 * row)), cos(T(0.4 * row))) for row in 1:dp0.global_dof_count, _ in 1:1]
            fused = map((:scalar, :simd)) do kernel
                assemble_burton_miller_neumann_system_cpu(
                    mesh, p1, dp0, q, k, rule; identity_p1_p1, identity_p1_dp0,
                    element_indices=indices, regular_kernel=kernel,
                )
            end
            _beat_cpu_simd_test_entrywise(fused[2].matrix, fused[1].matrix, tol)
            _beat_cpu_simd_test_entrywise(fused[2].rhs, fused[1].rhs, tol)
            operators = map((:scalar, :simd)) do kernel
                assemble_regular_galerkin_operators_cpu(
                    mesh, p1, dp0, k, rule; skip_singular=false, element_indices=indices, regular_kernel=kernel,
                )
            end
            for name in (:single_layer, :double_layer, :adjoint_double_layer, :hypersingular)
                _beat_cpu_simd_test_entrywise(getproperty(operators[2], name), getproperty(operators[1], name), tol)
            end
        end
    end
end
