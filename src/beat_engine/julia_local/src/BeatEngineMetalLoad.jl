# Included in BeatEngineCore so this import belongs to the bundle's dependency
# context. An explicitly selected Metal backend must never precompile a cached
# `nothing` after a failed dependency import. Preserve the original exception.
if BEAT_ACCELERATOR_HINT in ("cuda", "rocm", "none")
    nothing
elseif BEAT_ACCELERATOR_HINT == "metal"
    import Metal
    Metal
else
    # Standalone, unspecified-backend use may still probe optional accelerators.
    try
        @eval import Metal
        Metal
    catch
        nothing
    end
end
