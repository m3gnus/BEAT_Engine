# Requires the source-request near-correction path to execute on ROCm hardware.
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
BeatEngineCore._require_rocm!(rocsolver=true)
withenv("BLAB_TEST_NEAR_SOURCE_BACKEND" => "rocm") do
    include(joinpath(@__DIR__, "near_correction_rule_independence.jl"))
end
BeatEngineCore.amdgpu_module().synchronize()
GC.gc(true)
BeatEngineCore.amdgpu_module().reclaim()
