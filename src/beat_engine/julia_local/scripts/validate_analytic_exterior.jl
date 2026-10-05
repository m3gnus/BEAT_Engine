# Standalone closed-form CPU gate; shares assertions with the reference suite.
# Run with --project=src/beat_engine/julia_local.
using Test, StaticArrays, LinearAlgebra
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "compiled_ground_contract.jl"))
include(joinpath(@__DIR__, "..", "tests", "analytical_exterior_tests.jl"))
println("ANALYTIC_EXTERIOR_VALIDATION_OK")
