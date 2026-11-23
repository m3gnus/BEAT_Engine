# Complex-wavenumber ("complex_k") damping of the exterior Burton-Miller operator.
#
# With solver_options.wavenumber_shift = delta the surface operators use
# k~ = k + i*alpha, alpha = delta*|k| (k signed by the phasor convention), which
# is hornlab Metal's `complex_k` formulation. The closed forms below are written
# out here and do not call BEAT's Green function, phasor helpers or kernels.
#
# Standalone: julia --project=<engine>/julia_local tests/complex_wavenumber_tests.jl
# (set BEAT_TEST_CUDA=1 with CUDA.jl in the environment for the CPU/CUDA check).

using Test, LinearAlgebra, StaticArrays
if !isdefined(Main, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore
const CWK = BeatEngineCore

# Octahedron-refined sphere: every face lies in one octant, so the x >= 0,
# y >= 0 faces are exactly the :xy-reduced model of the full sphere.
function cwk_sphere(radius, refinements; quadrant::Bool=false)
    vertices = SVector{3,Float64}[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    faces = [(1,3,5), (3,2,5), (2,4,5), (4,1,5), (3,1,6), (2,3,6), (4,2,6), (1,4,6)]
    for _ in 1:refinements
        midpoints = Dict{Tuple{Int,Int},Int}()
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3,Int}[]
        for (a, b, c) in faces
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            append!(refined, [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        end
        faces = refined
    end
    if quadrant
        keep = [f for f in faces if all(vertices[i][1] >= -1e-12 && vertices[i][2] >= -1e-12 for i in f)]
        used = sort(unique(reduce(vcat, [collect(f) for f in keep])))
        index = Dict(v => i for (i, v) in enumerate(used))
        vertices = vertices[used]
        faces = [(index[a], index[b], index[c]) for (a, b, c) in keep]
    end
    return BoundaryMesh([radius * v for v in vertices], faces, fill(1, length(faces)))
end

function cwk_direct_solve(mesh, q, k, shift; symmetry=:off, T=Float64)
    mesh = BoundaryMesh([SVector{3,T}(v) for v in mesh.vertices], mesh.faces, mesh.physical_tags)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    rule = triangle_rule(T, 4)
    ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    ipq = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    kwargs = (identity_p1_p1=ipp, identity_p1_dp0=ipq, singular_order=4, symmetry_mode=symmetry)
    system = shift === nothing ?
        assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, T(k), rule; kwargs...) :
        assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, T(k), rule; kwargs..., wavenumber_shift=shift)
    return (system=system, pressure=vec(system.matrix \ system.rhs), p1=p1)
end

# Surface pressure at each P1 dof's vertex.
function cwk_dof_points(mesh, p1)
    points = Vector{SVector{3,Float64}}(undef, p1.global_dof_count)
    for (face_index, face) in enumerate(mesh.faces), local_index in 1:3
        points[p1.local_to_global[face_index][local_index]] = SVector{3,Float64}(mesh.vertices[face[local_index]])
    end
    return points
end

cwk_level_db(actual, reference) = maximum(abs.(20 .* log10.(abs.(actual ./ reference))))
cwk_phase_deg(actual, reference) = maximum(abs.(rad2deg.(angle.(actual ./ reference))))

@testset "complex wavenumber damping (wavenumber_shift)" begin
    rho, c, radius = 1.2041, 343.0, 0.1
    frequency = 1500.0
    omega = 2pi * frequency
    k = omega / c
    mesh = cwk_sphere(radius, 4)          # 2048 faces
    @test length(mesh.faces) == 2048

    @testset "zero shift is the undamped operator, bit for bit" begin
        q = ComplexF64[cis(0.37 * i) for i in eachindex(mesh.faces)]
        for T in (Float64, Float32)
            base = cwk_direct_solve(mesh, Complex{T}.(q), k, nothing; T=T).system
            for zero_shift in (0, 0.0, 0.0f0)
                shifted = cwk_direct_solve(mesh, Complex{T}.(q), k, zero_shift; T=T).system
                @test shifted.matrix == base.matrix
                @test shifted.rhs == base.rhs
            end
        end
        @test CWK.wavenumber_damping(k, 0) === nothing
        @test CWK.wavenumber_damping(Float32(k), 0.005) === Float32(0.005 * k)
        @test_throws ErrorException CWK.wavenumber_damping(k, -0.001)
        @test_throws ErrorException CWK.wavenumber_damping(k, Inf)
    end

    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        @testset "$convention: damped closed forms" begin
            with_phasor_convention(convention) do
                s = convention == NEGATIVE_TIME_PHASOR ? 1 : -1
                for shift in (0.005, 0.1)
                    alpha = shift * k
                    kt = s * k + im * alpha           # damped, signed wavenumber
                    ks = s * k + 0im                  # undamped control
                    # Pulsating sphere, unit outward velocity: q = i*s*rho*omega.
                    q_uniform = fill(complex(0.0, s * rho * omega), length(mesh.faces))
                    sol = cwk_direct_solve(mesh, q_uniform, k, shift)
                    points = cwk_dof_points(mesh, sol.p1)
                    surface(kk) = [q_uniform[1] * norm(x) / (im * kk * norm(x) - 1) for x in points]
                    damped_db = cwk_level_db(sol.pressure, surface(kt))
                    damped_deg = cwk_phase_deg(sol.pressure, surface(kt))
                    control_db = cwk_level_db(sol.pressure, surface(ks))
                    control_deg = cwk_phase_deg(sol.pressure, surface(ks))
                    @info "Pulsating sphere, complex k" convention shift damped_db damped_deg control_db control_deg
                    @test damped_db <= 0.03
                    @test damped_deg <= 0.2
                    if shift >= 0.1
                        # The gate discriminates: the undamped answer misses it.
                        @test control_db + control_deg / 10 >= 5 * (damped_db + damped_deg / 10)
                    end

                    # Off-centre manufactured monopole exp(i k~ r)/(4 pi r):
                    # non-uniform data exercising every damped kernel term.
                    source = SVector(0.02, -0.01, 0.015)
                    green(x, kk) = exp(im * kk * norm(x - source)) / (4pi * norm(x - source))
                    gradient(x, kk) = green(x, kk) * (im * kk - 1 / norm(x - source)) * (x - source) / norm(x - source)
                    q = [sum(gradient(x, kt) .* n) for (x, n) in zip(mesh.centroids, mesh.normals)]
                    mono = cwk_direct_solve(mesh, q, k, shift)
                    reference = [green(x, kt) for x in points]
                    mono_db = cwk_level_db(mono.pressure, reference)
                    mono_deg = cwk_phase_deg(mono.pressure, reference)
                    undamped = cwk_direct_solve(mesh, q, k, nothing)
                    wrong_db = cwk_level_db(undamped.pressure, reference)
                    wrong_deg = cwk_phase_deg(undamped.pressure, reference)
                    @info "Monopole, complex k" convention shift mono_db mono_deg wrong_db wrong_deg
                    @test mono_db <= 0.1
                    @test mono_deg <= 1.0
                    if shift >= 0.1
                        @test wrong_db + wrong_deg / 10 >= 5 * (mono_db + mono_deg / 10)
                    end
                end
            end
        end
    end

    @testset "damped image kernels: :xy reduced model equals the full sphere" begin
        shift = 0.05
        quarter = cwk_sphere(radius, 4; quadrant=true)
        @test 4 * length(quarter.faces) == length(mesh.faces)
        drive(x) = complex(1.0 + 0.3 * x[3] / radius, 0.2 * (x[1] / radius)^2)   # even in x and y
        relative = Dict{Any,Float64}()
        for s in (nothing, shift)
            full = cwk_direct_solve(mesh, [drive(x) for x in mesh.centroids], k, s)
            reduced = cwk_direct_solve(quarter, [drive(x) for x in quarter.centroids], k, s; symmetry=:xy)
            full_points = cwk_dof_points(mesh, full.p1)
            lookup = Dict(round.(Tuple(x); digits=9) => i for (i, x) in enumerate(full_points))
            reduced_points = cwk_dof_points(quarter, reduced.p1)
            matched = [full.pressure[lookup[round.(Tuple(x); digits=9)]] for x in reduced_points]
            relative[s] = norm(reduced.pressure - matched) / norm(matched)
        end
        @info "XY symmetry, reduced vs full" undamped=relative[nothing] damped=relative[shift]
        @test relative[shift] <= 1e-6
    end

    @testset "compiled option validation" begin
        shift_of = CWK.exterior_wavenumber_shift
        begin
            @test shift_of(Dict{String,Any}(), false) == 0.0
            @test shift_of(Dict{String,Any}("wavenumber_shift" => 0), false) == 0.0
            @test shift_of(Dict{String,Any}("wavenumber_shift" => 0.005), true) == 0.005
            @test_throws ErrorException shift_of(Dict{String,Any}("wavenumber_shift" => 0.005), false)
            @test_throws ErrorException shift_of(Dict{String,Any}("wavenumber_shift" => -1.0), true)
            @test_throws ErrorException shift_of(Dict{String,Any}("wavenumber_shift" => true), true)
            @test_throws ErrorException shift_of(Dict{String,Any}("wavenumber_shift" => "0.005"), true)
        end
    end

    if get(ENV, "BEAT_TEST_CUDA", "0") == "1" || (isdefined(Main, :cuda_available) && Main.cuda_available())
        @testset "CPU and CUDA damped operators agree" begin
            cuda = CWK.CUDA_MODULE
            @test cuda !== nothing && cuda.functional()
            T = Float32
            shift = 0.005
            for symmetry in (:off, :xy)
                m64 = symmetry == :off ? mesh : cwk_sphere(radius, 4; quadrant=true)
                m = BoundaryMesh([SVector{3,T}(v) for v in m64.vertices], m64.faces, m64.physical_tags)
                q = Complex{T}[cis(0.37 * i) for i in eachindex(m.faces)]
                p1, dp0 = build_p1_space(m), build_dp0_space(m)
                rule = triangle_rule(T, 4)
                ipp = assemble_l2_identity_matrix(m, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
                ipq = assemble_l2_identity_matrix(m, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
                singular_cache = build_singular_correction_cache(m, 4)
                device_cache = CWK.build_cuda_regular_assembly_cache(m, rule)
                device_singular = CWK.build_cuda_singular_correction_cache(singular_cache, p1, dp0)
                device_image = symmetry == :off ? nothing :
                    CWK.build_cuda_image_singular_correction_cache(m, p1, dp0, 4, eachindex(m.faces), symmetry)
                results = Dict{Any,Any}()
                for s in (0, shift)
                    cpu = assemble_burton_miller_neumann_system_cpu(m, p1, dp0, q, T(k), rule;
                        identity_p1_p1=ipp, identity_p1_dp0=ipq, singular_order=4,
                        singular_cache=singular_cache, symmetry_mode=symmetry, wavenumber_shift=s)
                    gpu = assemble_burton_miller_neumann_system_cuda(m, p1, dp0, cuda.CuArray(q), T(k), rule;
                        device_cache=device_cache, singular_cache=singular_cache,
                        device_singular_cache=device_singular, device_image_singular_cache=device_image,
                        symmetry_mode=symmetry,
                        identity_p1_p1_block=Tuple(CWK.l2_identity_element_matrix(one(T), :p1, :p1, rule)),
                        wavenumber_shift=s)
                    cpu_p = cpu.matrix \ vec(cpu.rhs)
                    gpu_p = Array(gpu.matrix) \ Array(gpu.rhs)
                    matrix_rel = norm(Array(gpu.matrix) - cpu.matrix) / norm(cpu.matrix)
                    pressure_rel = norm(gpu_p - cpu_p) / norm(cpu_p)
                    @info "CPU vs CUDA" symmetry s matrix_rel pressure_rel
                    @test matrix_rel <= 1e-4
                    @test pressure_rel <= 1e-4
                    results[s] = gpu_p
                end
                @test norm(results[shift] - results[0]) / norm(results[0]) > 1e-4
            end
        end
    end
end
