include(joinpath(@__DIR__, "..", "compiled_ground_contract.jl"))

@testset "compiled rigid-ground physical radiator contract" begin
    T = Float32
    raised = BoundaryMesh(
        SVector{3,T}[(0, 1, 0), (1, 1, 0), (0, 1, 1)],
        [(1, 2, 3)], [2],
    )
    pressure = ones(Complex{T}, length(raised.vertices))
    excitation = (tags=[2], amplitudes=Complex{T}[1])

    @test validate_compiled_ground_domain!(raised, :ground) === nothing
    @test validate_compiled_ground_domain!(raised, :off) === nothing
    @test validate_compiled_ground_domain!(raised, :ground; min_clearance_m=1.0) === nothing
    @test_throws "clearance" validate_compiled_ground_domain!(raised, :ground; min_clearance_m=1.1)
    for clearance in (-1.0, NaN, Inf)
        @test_throws "finite and non-negative" validate_compiled_ground_domain!(
            raised, :ground; min_clearance_m=clearance,
        )
    end
    local_impedance = exterior_component_impedance(raised, pressure, excitation, :off, T)
    @test local_impedance ≈ Complex{T}(raised.areas[1])
    @test exterior_component_impedance(raised, pressure, excitation, :ground, T) ≈ local_impedance
    @test exterior_component_impedance(raised, pressure, excitation, :x, T) ≈ 2local_impedance
    @test exterior_component_impedance(raised, pressure, excitation, :xy, T) ≈ 4local_impedance

    axial = (tags=[2], amplitudes=T[1], motion_axis=SVector{3,T}(0, 0, 1))
    @test exterior_motion_factor(axial, SVector{3,T}(0, 0, 1), T) == one(T)
    @test exterior_motion_factor(axial, SVector{3,T}(0, 0, -1), T) == -one(T)
    @test exterior_motion_factor(axial, SVector{3,T}(1, 0, 0), T) == zero(T)
    @test exterior_component_impedance(raised, pressure, axial, :off, T) == zero(Complex{T})
    downward = (tags=[2], amplitudes=T[2], motion_axis=SVector{3,T}(0, -1, 0))
    @test exterior_component_impedance(raised, pressure, downward, :off, T) ≈
          Complex{T}(2 * raised.areas[1])

    straddling = BoundaryMesh(
        SVector{3,T}[(0, -0.1, 0), (1, 0.1, 0), (0, 0.1, 1)],
        [(1, 2, 3)], [2],
    )
    @test_throws ErrorException validate_compiled_ground_domain!(straddling, :ground)
    @test validate_compiled_ground_domain!(straddling, :off) === nothing

    coplanar = BoundaryMesh(
        SVector{3,T}[(0, 0, 0), (1, 0, 0), (0, 0, 1)],
        [(1, 2, 3)], [2],
    )
    @test_throws ErrorException validate_compiled_ground_domain!(coplanar, :ground)

    contact_edge = BoundaryMesh(
        SVector{3,T}[(0, 0, 0), (1, 0, 0), (0, 0.1, 1)],
        [(1, 2, 3)], [2],
    )
    @test validate_compiled_ground_domain!(contact_edge, :ground) === nothing
    @test_throws "clearance" validate_compiled_ground_domain!(contact_edge, :ground; min_clearance_m=0.01)

    raised_volume = (vertices=SVector{3,T}[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],)
    @test validate_compiled_ground_volume!(raised_volume, :ground) === nothing
    below_volume = (vertices=SVector{3,T}[(0, -0.1, 0)],)
    @test_throws ErrorException validate_compiled_ground_volume!(below_volume, :ground)
    @test validate_compiled_ground_volume!(below_volume, :off) === nothing
end
