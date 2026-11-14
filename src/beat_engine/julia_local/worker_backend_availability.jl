function worker_backend_availability()
    backends = Dict{String,Any}("cpu" => Dict("available" => true, "reason" => ""))
    for (name, accelerator) in (
        ("cuda", BeatEngineCore.CUDA_MODULE),
        ("rocm", BeatEngineCore.AMDGPU_MODULE),
        ("metal", BeatEngineCore.METAL_MODULE),
    )
        available, reason = try
            if accelerator === nothing
                (false, "Package is not loaded in this Julia environment.")
            else
                functional = accelerator.functional()
                if functional && name == "rocm"
                    functional = accelerator.functional(:rocblas) && accelerator.functional(:rocsolver)
                end
                (functional, functional ? "" : "Runtime/device or required solver libraries are not functional.")
            end
        catch exception
            (false, sprint(showerror, exception))
        end
        backends[name] = Dict("available" => available, "reason" => reason,
            "phasor_conventions" => [NEGATIVE_TIME_PHASOR, POSITIVE_TIME_PHASOR])
    end
    return backends
end

