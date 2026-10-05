# Exercise the compiled source driver, not a separately reconstructed drive.
# Like memory_mesh_tests.jl, omit only the CLI entrypoint when loading it.
module AxialSourceTests
using Test, StaticArrays, LinearAlgebra, Base64
include(joinpath(@__DIR__, "..", "BeatEngineCompiledDriver.jl"))

function write_surface(path, mesh)
    open(path, "w") do io
        println(io, "\$MeshFormat\n2.2 0 8\n\$EndMeshFormat\n\$Nodes")
        println(io, length(mesh.vertices))
        for (index, vertex) in enumerate(mesh.vertices)
            println(io, index, " ", join(vertex, " "))
        end
        println(io, "\$EndNodes\n\$Elements\n", length(mesh.faces))
        for (index, face) in enumerate(mesh.faces)
            tag = mesh.physical_tags[index]
            println(io, index, " 2 2 ", tag, " ", tag, " ", join(face, " "))
        end
        println(io, "\$EndElements")
    end
end

function compiled_result(path, mesh, axis, points, frequency, convention, backend)
    write_surface(path, mesh)
    request = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "example-exterior-request.json"))
    system = request["compiled_system"]
    system["contract_version"] = 2
    system["meshes"][1]["file"] = path
    system["meshes"][1]["scale_to_m"] = 1.0
    system["components"][1]["parameters"] = Dict(
        "motion_profile" => "rigid_translation", "motion_axis" => collect(axis),
    )
    request["frequencies_hz"] = [frequency]
    request["solver_options"] = Dict(
        "precision" => backend == :cpu ? "float64" : "float32",
        "bem_backend" => String(backend), "quadrature_order" => 3, "singular_order" => 3,
        "regular_quadrature_mode" => "fixed", "burton_miller_assembly" => "direct_system",
        "phasor_convention" => convention,
    )
    request["outputs"] = [Dict(
        "id" => quantity, "quantity" => quantity, "target_ids" => [],
        "options" => quantity == "exterior_pressure" ? Dict("points_m" => collect.(points)) : Dict(),
    ) for quantity in ("exterior_pressure", "bem_boundary_pressure", "bem_boundary_neumann", "radiation_impedance")]
    result = solve_compiled_result(request)
    @test result["diagnostics"]["bem_backend"] == String(backend)
    @test result["diagnostics"]["burton_miller_assembly"] ==
          "direct_system"
    return decoded_quantities(result)
end

function solve_compiled_result(request)
    mktemp() do _, output
        redirect_stdout(output) do
            solve_request(request; event_mode=true)
        end
        flush(output)
        seekstart(output)
        events = [JSON.parse(line) for line in eachline(output)]
        @test last(events)["type"] == "result"
        only([event["result"] for event in events if event["type"] == "result"])
    end
end

function decoded_quantities(result)
    Dict(item["quantity"] => collect(reinterpret(
        item["values"]["dtype"] == "complex128" ? ComplexF64 : ComplexF32,
        base64decode(item["values"]["content_base64"]),
    )) for item in result["quantities"])
end

function oscillating_sphere(refinements, radius)
    vertices = SVector{3,Float64}[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    faces = [(1,3,5), (3,2,5), (2,4,5), (4,1,5), (3,1,6), (2,3,6), (4,2,6), (1,4,6)]
    for _ in 1:refinements
        midpoints = Dict{Tuple{Int,Int},Int}()
        midpoint(a, b) = get!(midpoints, minmax(a, b)) do
            push!(vertices, normalize(vertices[a] + vertices[b]))
            length(vertices)
        end
        refined = NTuple{3,Int}[]
        for (a,b,c) in faces
            ab,bc,ca = midpoint(a,b), midpoint(b,c), midpoint(c,a)
            append!(refined, [(a,ab,ca), (ab,b,bc), (ca,bc,c), (ab,bc,ca)])
        end
        faces = refined
    end
    BoundaryMesh(radius .* vertices, faces, fill(2, length(faces)))
end

# Linearized Euler: rho*d(v)/dt = -grad(p). With exp(-i*omega*t),
# dp/dr = +i*rho*omega*U*cos(theta) on r=a. The outgoing l=1 solution is
# p = i*rho*omega*U*h1(k*r)/(k*h1'(k*a))*dot(axis,x/r).
# These explicit Hankel expressions do not call the engine's phasor or kernels.
# Positive time uses h1^(2)=conj(h1^(1)) and the opposite Euler sign.
function dipole_pressure(x, axis, radius, k, density, omega, convention)
    h1(z) = -exp(im*z) * (z + im) / z^2
    dh1(z) = exp(im*z) * (-im/z + 2/z^2 + 2im/z^3)
    radial = im * density * omega * h1(k*norm(x)) / (k * dh1(k*radius))
    convention == POSITIVE_TIME_PHASOR && (radial = conj(radial))
    radial * dot(normalize(axis), normalize(x))
end

function rotation_checks(path, backend)
    # An asymmetric closed tetrahedron with only one driven patch. Rotate every
    # vertex, the source axis and all probes, preserving face order/winding.
    mesh = BoundaryMesh(SVector{3,Float64}[(0,0,0), (.23,0,0), (.02,.17,0), (.03,.04,.13)],
        [(1,3,2), (1,2,4), (1,4,3), (2,3,4)], [1,1,1,2])
    axis = SVector(0.,0.,1.)
    points = SVector{3,Float64}[(.4,.3,.5), (-.3,.2,.4), (.1,-.4,-.3)]
    rx = [1. 0. 0.; 0. 0. -1.; 0. 1. 0.]
    ry = [0. 0. 1.; 0. 1. 0.; -1. 0. 0.]
    u = normalize(SVector(1.,2.,3.))
    cross_u = [0. -u[3] u[2]; u[3] 0. -u[1]; -u[2] u[1] 0.]
    angle = 0.73
    generic = cos(angle)*I + (1-cos(angle))*(u*u') + sin(angle)*cross_u
    tolerance = backend == :cpu ? 1e-10 : 8e-4
    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        original = compiled_result(path, mesh, axis, points, 500., convention, backend)
        @test norm(original["exterior_pressure"]) > 1
        for (name, rotation) in (("about x", rx), ("about y", ry), ("generic", generic))
            rotated = BoundaryMesh([SVector{3,Float64}(rotation*v) for v in mesh.vertices],
                mesh.faces, mesh.physical_tags)
            result = compiled_result(path, rotated, rotation*axis, [rotation*x for x in points],
                500., convention, backend)
            errors = Dict(key => norm(result[key]-original[key])/norm(original[key]) for key in keys(original))
            @info "Axial whole-problem rotation" backend convention rotation=name relative_complex_errors=JSON.json(errors)
            for key in keys(original)
                @test result[key] ≈ original[key] rtol=tolerance atol=tolerance
            end
        end
    end
end

function sphere_checks(path, backend)
    radius, k, density = 0.1, 7.0, 1.21 # ka=0.7, as in the monopole reference.
    omega = k * 343.0
    points = SVector{3,Float64}[(.2,.03,.04), (-.08,.25,.06), (.04,-.07,.3), (-.3,-.2,-.1)]
    axes = (SVector(1.,0.,0.), SVector(0.,1.,0.), SVector(0.,0.,1.), normalize(SVector(1.,-2.,3.)))
    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR), axis in axes
        analytic = [dipole_pressure(x, axis, radius, k, density, omega, convention) for x in points]
        coarse_error = Inf
        for refinement in (2, 3) # 128 then 512 planar facets; no frozen baseline.
            mesh = oscillating_sphere(refinement, radius)
            result = compiled_result(path, mesh, axis, points, omega/(2pi), convention, backend)
            field = result["exterior_pressure"]
            error = norm(field-analytic)/norm(analytic)
            gain = dot(analytic, field)/sum(abs2, analytic)
            @info "Analytical oscillating sphere" backend convention axis=Tuple(axis) faces=length(mesh.faces) relative_complex_error=error complex_gain=gain
            if refinement == 2
                coarse_error = error
            else
                # Keep the 8% accuracy bound of the existing exterior oracle.
                # Refinement must also reduce error: facet bias affects z too,
                # whereas a lateral sign error persists as the mesh is refined.
                @test error < 0.08
                @test error < coarse_error / 2
                @test real(gain) > 0.92
            end
            # A flipped sign has error ~2; conjugation fails the phase check.
            @test abs(angle(gain)) < 0.08
            @test real(gain) > 0
            # Pin boundary data independently to Euler and outward facet normals.
            sign = convention == NEGATIVE_TIME_PHASOR ? 1 : -1
            expected_q = [sign*im*density*omega*dot(axis, normalize(cross(
                mesh.vertices[b]-mesh.vertices[a], mesh.vertices[c]-mesh.vertices[a],
            ))) for (a,b,c) in mesh.faces]
            @test result["bem_boundary_neumann"] ≈ expected_q rtol=1e-6 atol=1e-3
        end
    end
end

function builder_checks()
    request = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "example-exterior-request.json"))
    system = request["compiled_system"]
    components = (ports=system["excitation_ports"], items=system["components"])
    function build(axis, mode, T)
        system["components"][1]["parameters"] = Dict(
            "motion_profile" => "rigid_translation", "motion_axis" => axis,
        )
        only(exterior_excitations(request["excitation_port_ids"], components,
            system["boundaries"], Dict("boundary:source" => 2), only(system["regions"]), mode, T))
    end
    for T in (Float32, Float64)
        # Reach both Float64 limits before conversion to solve precision. Direct
        # normalization overflows at floatmax and underflows at nextfloat(0.0).
        for magnitude in (floatmax(Float64), nextfloat(0.0), 1e300, 1e-300)
            axis = build([magnitude, 0, magnitude], :off, T).motion_axis
            @test all(isfinite, axis)
            @test axis ≈ SVector{3,T}(1/sqrt(2), 0, 1/sqrt(2))
            @test norm(axis) ≈ one(T)
        end
        for invalid in ([0.,0,0], [NaN,0,1], [Inf,0,1], [-Inf,0,1],
                        Any[true,0,1], [0,1], [1,2,3,4], ["x",0,1], nothing, (0,0,1))
            @test_throws ErrorException build(invalid, :off, T)
        end
        @test build([0,1,1], :x, T).motion_axis ≈ SVector{3,T}(0, 1/sqrt(2), 1/sqrt(2))
        @test build([0,0,-1], :xy, T).motion_axis == SVector{3,T}(0,0,-1)
        @test build([1,-2,3], :ground, T).motion_axis ≈ SVector{3,T}(1,-2,3)/T(sqrt(14))
        @test_throws ErrorException build([1,0,0], :x, T)
        # Exercise Y independently; an X violation would fail first in xy.
        @test_throws ErrorException build([0,1,0], :xy, T)
        @test_throws ErrorException build([1,0,0], :xy, T)
    end
end

function component_force_checks(path, backend)
    # Each component owns two boundaries with different weights and a different
    # axis. Build the oracle from winding, not the engine's projection helpers.
    mesh = BoundaryMesh(SVector{3,Float64}[(0,0,0), (.23,0,0), (.02,.17,0), (.03,.04,.13)],
        [(1,3,2), (1,2,4), (1,4,3), (2,3,4)], [2,3,4,5])
    write_surface(path, mesh)
    raw_axes = ([2.,-1.,3.], [-1.,4.,-2.])
    axes = normalize.(raw_axes)
    weights = ((0.7, 1.4), (1.3, 0.6))
    normals = [normalize(cross(mesh.vertices[b]-mesh.vertices[a], mesh.vertices[c]-mesh.vertices[a]))
        for (a,b,c) in mesh.faces]
    areas = [norm(cross(mesh.vertices[b]-mesh.vertices[a], mesh.vertices[c]-mesh.vertices[a]))/2
        for (a,b,c) in mesh.faces]
    tolerance = backend == :cpu ? 1e-10 : 8e-4
    for convention in (NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR)
        request = JSON.parsefile(joinpath(@__DIR__, "..", "..", "beat_contract", "example-exterior-request.json"))
        system = request["compiled_system"]
        system["contract_version"] = 2
        system["meshes"][1]["file"] = path
        system["meshes"][1]["scale_to_m"] = 1.0
        boundary_template = only(system["boundaries"])
        system["boundaries"] = [merge(deepcopy(boundary_template), Dict(
            "id" => "boundary:$face", "group" => merge(deepcopy(boundary_template["group"]), Dict("tag" => face+1)),
        )) for face in 1:4]
        component_template = only(system["components"])
        system["components"] = [merge(deepcopy(component_template), Dict(
            "id" => "component:$component", "boundary_ids" => ["boundary:$(2component-1)", "boundary:$(2component)"],
            "parameters" => Dict("motion_profile" => "rigid_translation", "motion_axis" => raw_axes[component],
                "boundary_motion_weights" => Dict("boundary:$(2component-1)" => weights[component][1],
                    "boundary:$(2component)" => weights[component][2])),
        )) for component in 1:2]
        port_template = only(system["excitation_ports"])
        system["excitation_ports"] = [merge(deepcopy(port_template), Dict(
            "id" => "excitation:$component", "component_id" => "component:$component",
        )) for component in 1:2]
        request["frequencies_hz"] = [500.]
        request["solver_options"] = Dict(
            "precision" => backend == :cpu ? "float64" : "float32", "bem_backend" => String(backend),
            "quadrature_order" => 3, "singular_order" => 3, "regular_quadrature_mode" => "fixed",
            "burton_miller_assembly" => "direct_system", "phasor_convention" => convention,
        )
        request["outputs"] = [Dict("id" => quantity, "quantity" => quantity, "target_ids" => [], "options" => Dict())
            for quantity in ("bem_boundary_pressure", "bem_boundary_neumann", "radiation_impedance")]
        forward = nothing
        for order in ([1,2], [2,1])
            request["excitation_port_ids"] = ["excitation:$component" for component in order]
            result = solve_compiled_result(request)
            @test result["excitation_port_ids"] == request["excitation_port_ids"]
            @test result["diagnostics"]["bem_backend"] == String(backend)
            @test result["diagnostics"]["burton_miller_assembly"] ==
                  "direct_system"
            quantities = decoded_quantities(result)
            pressure = quantities["bem_boundary_pressure"]
            neumann = quantities["bem_boundary_neumann"]
            impedance = quantities["radiation_impedance"]
            @test length(pressure) == 2length(mesh.vertices)
            @test length(neumann) == 2length(mesh.faces)
            @test length(impedance) == 2
            for component in 1:2
                column = findfirst(==(component), order)
                own_faces = (2component-1, 2component)
                own_pressure = pressure[(column-1)*length(mesh.vertices)+1:column*length(mesh.vertices)]
                # Euler fixes the signed Neumann column independently of the solver.
                expected_q = zeros(ComplexF64, length(mesh.faces))
                euler_sign = convention == NEGATIVE_TIME_PHASOR ? 1 : -1
                force = 0.0im
                for (face_index, weight) in zip(own_faces, weights[component])
                    projection = dot(normals[face_index], axes[component])
                    expected_q[face_index] = euler_sign*im*1.21*2pi*500*weight*projection
                    a,b,c = mesh.faces[face_index]
                    force += (own_pressure[a]+own_pressure[b]+own_pressure[c])/3 * areas[face_index] * weight * projection
                end
                @test neumann[(column-1)*length(mesh.faces)+1:column*length(mesh.faces)] ≈ expected_q rtol=tolerance atol=tolerance
                @test abs(force) > 1e-6
                # Radiators follow component order even when excitation ports reverse.
                @test impedance[component] ≈ force rtol=tolerance atol=tolerance
            end
            if forward === nothing
                forward = quantities
            else
                @test impedance ≈ forward["radiation_impedance"] rtol=tolerance atol=tolerance
                @test pressure ≈ vcat(forward["bem_boundary_pressure"][5:8], forward["bem_boundary_pressure"][1:4]) rtol=tolerance atol=tolerance
                @test neumann ≈ vcat(forward["bem_boundary_neumann"][5:8], forward["bem_boundary_neumann"][1:4]) rtol=tolerance atol=tolerance
            end
        end
    end
end

@testset "compiled axial source independent physical checks" begin
    @testset "builder axis normalization and reductions" begin
        builder_checks()
    end
    mktempdir() do directory
        path = joinpath(directory, "surface.msh")
        @testset "CPU per-component generalized force and port order" begin
            component_force_checks(path, :cpu)
        end
        @testset "CPU rotation invariance" begin
            rotation_checks(path, :cpu)
        end
        @testset "CPU analytical translating sphere" begin
            sphere_checks(path, :cpu)
        end
        # Use the same functional-device gate as the runtime phasor/pipeline arms.
        if BeatEngineCore.METAL_MODULE !== nothing && BeatEngineCore.METAL_MODULE.functional()
            @testset "Metal per-component generalized force and port order" begin
                component_force_checks(path, :metal)
            end
            @testset "Metal rotation invariance" begin
                rotation_checks(path, :metal)
            end
            @testset "Metal analytical translating sphere" begin
                sphere_checks(path, :metal)
            end
        end
    end
end
end
