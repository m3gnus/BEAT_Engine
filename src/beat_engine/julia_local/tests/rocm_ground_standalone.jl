using Test, LinearAlgebra, StaticArrays
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
BeatEngineCore._require_rocm!(rocsolver=true)
include(joinpath(@__DIR__, "rocm_ground_tests.jl"))
