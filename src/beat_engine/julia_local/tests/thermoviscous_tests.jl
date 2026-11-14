using Test, LinearAlgebra, SparseArrays, StaticArrays
if !isdefined(@__MODULE__, :BeatEngineCore)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
end
using .BeatEngineCore
if !isdefined(@__MODULE__, :BeatEngineCoupled)
    include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
end
using .BeatEngineCoupled

function tv_tetrahedron(; quadratic=false, scalar=Float64)
    T = scalar
    vertices = SVector{3,T}[(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    qtet, qface = NTuple{10,Int}[], NTuple{6,Int}[]
    if quadratic
        append!(vertices, [(vertices[a] + vertices[b]) / 2
                          for (a,b) in ((1,2),(2,3),(3,1),(1,4),(4,3),(2,4))])
        push!(qtet, Tuple(1:10)); push!(qface, (1,2,3,5,6,7))
    end
    VolumeMesh(vertices, [(1,2,3,4)], [1], [(1,2,3)], [2],
               Dict{Tuple{Int,Int},String}(), qtet, qface)
end

@testset "thermoviscous surface elements and passivity" begin
    for quadratic in (false, true), T in (Float32, Float64)
        mesh = tv_tetrahedron(; quadratic, scalar=T)
        operator = thermoviscous_surface_operator(mesh, [1])
        n = length(mesh.vertices)
        ones_p = ones(T, n)
        x = T[v[1] for v in mesh.vertices]
        @test issymmetric(operator.stiffness)
        @test norm(operator.stiffness * ones_p) < 64eps(T)
        @test dot(x, operator.stiffness * x) ≈ T(0.5) rtol=64eps(T)
        @test sum(operator.mass) ≈ T(0.5) rtol=64eps(T)
        @test minimum(eigvals(Symmetric(Matrix(operator.stiffness)))) > -64eps(T)
        @test minimum(eigvals(Symmetric(Matrix(operator.mass)))) > -64eps(T)
        A = assemble_fem_dynamic_stiffness(assemble_fem_matrices(mesh)..., T(10))
        @test add_thermoviscous_terms(A, nothing, T(1000), T(343), T(1.21)) === A
        positive = BeatEngineCore.with_phasor_convention("exp(+i omega t)") do
            add_thermoviscous_terms(A, operator, T(1000), T(343), T(1.21))
        end
        negative = BeatEngineCore.with_phasor_convention("exp(-i omega t)") do
            add_thermoviscous_terms(A, operator, T(1000), T(343), T(1.21))
        end
        @test positive ≈ conj(negative)
        @test imag(dot(ones_p, positive * ones_p)) > 0 # thermal loss for constant pressure
        @test imag(dot(x, positive * x)) > 0
        if quadratic
            x2 = x .^ 2
            # Integral over unit right triangle of |grad(x^2)|^2 = 1/3.
            @test dot(x2, operator.stiffness * x2) ≈ T(1/3) rtol=64eps(T)
        end
    end
    @test_throws ErrorException thermoviscous_coefficients(0.0, 343.0, 1.21)
    mesh = tv_tetrahedron()
    wall = [(tag=2, boundary_id="wall")]
    @test prepare_thermoviscous_operator(mesh, wall).area_m2 ≈ 0.5
    @test prepare_thermoviscous_operator(mesh, []) === nothing
    @test_throws ErrorException prepare_thermoviscous_operator(mesh, [(tag=99,)])
    # The same selected face on x=0 is an artificial symmetry cut, not a wall.
    rotated = VolumeMesh([SVector(v[3],v[1],v[2]) for v in mesh.vertices],
        mesh.tetrahedra, mesh.tetra_physical_tags, mesh.boundary_faces,
        mesh.boundary_physical_tags, mesh.physical_names)
    @test prepare_thermoviscous_operator(rotated, wall; symmetry_mode=:x) === nothing
end

# A slit extruded in x: only the y=+/-h plates have losses; x sides are symmetry.
function tv_slit_mesh(nx; length_m=0.2, half_gap=0.002)
    vertices = SVector{3,Float64}[]
    for iz in 0:nx, iy in 0:1, ix in 0:1
        push!(vertices, SVector(ix * 0.01, (2iy-1)*half_gap, iz*length_m/nx))
    end
    node(ix,iy,iz) = 1 + ix + 2iy + 4iz
    tetrahedra = NTuple{4,Int}[]
    for iz in 0:nx-1
        v = [node(0,0,iz), node(1,0,iz), node(0,1,iz), node(1,1,iz),
             node(0,0,iz+1), node(1,0,iz+1), node(0,1,iz+1), node(1,1,iz+1)]
        for t in ((1,2,4,8),(1,4,3,8),(1,3,7,8),(1,7,5,8),(1,5,6,8),(1,6,2,8))
            push!(tetrahedra, ntuple(i -> v[t[i]], 4))
        end
    end
    counts = Dict{NTuple{3,Int},Int}()
    for t in tetrahedra, f in ((t[1],t[2],t[3]),(t[1],t[2],t[4]),(t[1],t[3],t[4]),(t[2],t[3],t[4]))
        key = Tuple(sort(collect(f))); counts[key] = get(counts, key, 0) + 1
    end
    faces = sort!([f for (f,n) in counts if n == 1])
    tags = [all(v -> vertices[v][2] == half_gap, f) ||
            all(v -> vertices[v][2] == -half_gap, f) ? 2 : 3 for f in faces]
    VolumeMesh(vertices, tetrahedra, ones(Int,length(tetrahedra)), faces, tags,
               Dict{Tuple{Int,Int},String}())
end

@testset "slit propagation against analytical thermoviscous equivalent fluid" begin
    # Independent no-slip/isothermal parallel-plate solution (tanh profile),
    # including overlapping-layer physics; exercise only the thin-layer limit here.
    rho, c, h, length_m = 1.21, 343.0, 0.002, 0.2
    mu, conductivity, cp, gamma = 1.84e-5, 0.0257, 1005.0, 1.4
    for frequency in (500.0, 1000.0)
        omega = 2pi * frequency
        sv = sqrt(im * omega * rho / mu)
        st = sqrt(im * omega * rho * cp / conductivity)
        fv, ft = tanh(sv*h)/(sv*h), tanh(st*h)/(st*h)
        k = omega/c * sqrt((1 + (gamma-1)*ft)/(1-fv))
        @test imag(k) < 0
        errors = Float64[]
        for nx in (40, 80)
            mesh = tv_slit_mesh(nx; length_m, half_gap=h)
            wall = prepare_thermoviscous_operator(mesh, [(tag=2,)])
            K, M = assemble_fem_matrices(mesh)
            A = BeatEngineCore.with_phasor_convention("exp(+i omega t)") do
                add_thermoviscous_terms(assemble_fem_dynamic_stiffness(K,M,omega/c), wall, frequency,c,rho)
            end
            reference = [exp(-im*k*v[3]) for v in mesh.vertices]
            ends = findall(v -> v[3] == 0 || v[3] == length_m, mesh.vertices)
            free = setdiff(eachindex(mesh.vertices), ends)
            pressure = copy(reference)
            pressure[free] = A[free,free] \ (-A[free,ends]*reference[ends])
            push!(errors, norm(pressure-reference)/norm(reference))
        end
        @test errors[2] < 0.005
        @test errors[2] < errors[1]
    end
end
