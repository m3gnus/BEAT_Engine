"""Exterior radiation contributions with the operating coupled flux held fixed."""
module BeatEngineInterfaceRadiation
using LinearAlgebra
using ..BeatEngineCore: cuda_module
export interface_radiation_traces

function _cuda_replay_pressure(replay, interface_flux, ranges)
    T = eltype(interface_flux)
    isempty(ranges) && return zeros(T, size(replay.interface_block, 1), 0)
    # Only the small, source-partitioned flux vectors cross to the device. The
    # dense operator and LU factors stay resident for every excitation.
    masked_flux = zeros(T, length(interface_flux), length(ranges))
    for (column, indices) in enumerate(ranges)
        masked_flux[indices, column] .= view(interface_flux, indices)
    end
    cuda = cuda_module()
    device_flux = cuda.CuArray(masked_flux)
    rhs = pressure = nothing
    try
        rhs = replay.interface_block * device_flux
        rhs .*= -one(T)
        pressure = replay.factorization \ rhs
        # Preserve the host trace contract used by field evaluation and output
        # serialization. This is O(boundary nodes * sources), not a dense matrix.
        return Array(pressure)
    finally
        cuda.unsafe_free!(device_flux)
        isnothing(rhs) || cuda.unsafe_free!(rhs)
        isnothing(pressure) || cuda.unsafe_free!(pressure)
    end
end

"""
Recover the complete scattered pressure driven by each interface's spatial flux.
Masking the solved boundary pressure would omit scattering and is not a source
decomposition. Ranges partition the interface flux DOFs. The optional remainder
groups all directly radiating exterior sources, with their original motion.
"""
function interface_radiation_traces(system, solution, ranges; include_other::Bool=false)
    replay = system.interface_radiation_replay
    isnothing(replay) && error("Interface radiation replay was not requested during assembly.")
    T = eltype(solution.bem_pressure)
    device_replay = get(replay, :backend, :cpu) == :cuda
    rhs = device_replay ? nothing : zeros(T, length(solution.bem_pressure), length(ranges))
    flux = zeros(T, length(solution.bem_neumann), length(ranges))
    for (index, indices) in enumerate(ranges)
        values = view(solution.interface_flux, indices)
        if !device_replay
            rhs[:, index] = -(view(replay.interface_block, :, indices) * values)
        end
        flux[:, index] = system.interface_operators.bem_flux[:, indices] * values
    end
    pressure = device_replay ? _cuda_replay_pressure(replay, solution.interface_flux, ranges) :
               replay.factorization \ rhs
    if include_other
        pressure = hcat(pressure, solution.bem_pressure - vec(sum(pressure; dims=2)))
        flux = hcat(flux, solution.bem_neumann - vec(sum(flux; dims=2)))
    end
    scale = max(norm(solution.bem_pressure), norm(pressure), eps(real(T)))
    reconstruction_error = norm(vec(sum(pressure; dims=2)) - solution.bem_pressure) / scale
    return (pressure=pressure, normal_derivative=flux, reconstruction_error=reconstruction_error)
end
end
