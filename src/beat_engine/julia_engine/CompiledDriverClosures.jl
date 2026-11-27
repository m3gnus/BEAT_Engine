# Resolve driver closures by captured fields, never compiler-generated numbers.
# Fail on ambiguity rather than silently precompiling an unrelated closure.
function compiled_driver_closure_type(fields, driver::Module=@__MODULE__)
    matches = Any[]
    for name in names(driver; all=true)
        isdefined(driver, name) || continue
        value = getfield(driver, name)
        value isa UnionAll || value isa DataType || continue
        T = Base.unwrap_unionall(value)
        T <: Function && fieldnames(T) == fields && push!(matches, value)
    end
    return only(matches)
end

# Keep the producer and Neumann closure inventory shared with the hardware-free
# CI gate. Exterior field outputs use a plain loop and explicit method signatures.
function compiled_driver_closure_types(driver::Module=@__MODULE__)
    producer = compiled_driver_closure_type((:system, :metal_fused_kwargs, :frequencies_hz,
        :large_panel_rules, :base_rule, :dp0_space, :p1_space, :excitations, :mesh, :density, :sound_speed, :FloatType), driver)
    neumann = compiled_driver_closure_type((:omega, :mesh, :density), driver)
    return (; producer, neumann)
end
