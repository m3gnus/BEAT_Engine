"""
    BeatEngineCoupledCondensed

Standalone CPU coupled FEM-BEM solver that eliminates the FEM interior onto the retained
interface before the dense solve.

`BeatEngineCoupled` assembles the entire coupled system densely, FEM volume block included.
That block is a P1 tetrahedral Helmholtz operator with roughly fifteen nonzeros per row, so
the monolithic formulation materializes a matrix that is about 0.06% dense and then takes a
dense LU of it. This solver instead forms the Schur complement

    S = A_ΓΓ - A_ΓI * A_II⁻¹ * A_IΓ

on the retained set `Γ = interface ∪ transducer surfaces`, and solves a system whose order is
`|Γ| + |BEM| + |interface| + 2 * |transducers|` rather than one that carries every interior
FEM vertex.

This is a deliberate fork of the monolithic solver rather than a branch inside it: the two are
expected to diverge further. It duplicates the coupled assembly rather than sharing it, and in
exchange carries no CUDA paths, no validation-diagnostic paths, and no monolithic formulation.
It reuses `BeatEngineCoupled` only for mesh types, cache preparation, and operator assembly —
the physics inputs, not the solve.

Precision is mixed on purpose. `A_II` is factored in `ComplexF64` whatever `T` is, and only the
assembled `S` is demoted to `Complex{T}`. `S` has poles at the eigenvalues of `A_II` — the
cavity modes with a pressure-release interface — and near one of those the loss of digits scales
like `1/η` in the bulk loss factor. `η = 0` is both the default and a shipped configuration, so
the interior solve carries the double-precision margin unconditionally. The interior is sparse,
which makes that margin nearly free.
"""
module BeatEngineCoupledCondensed

using LinearAlgebra, SparseArrays, StaticArrays, Statistics
using ..BeatEngineCore
using ..BeatEngineCoupled

include(joinpath(@__DIR__, "BeatEngineCondensedAssembly.jl"))
include(joinpath(@__DIR__, "BeatEngineMetalCoupledHost.jl"))
include(joinpath(@__DIR__, "BeatEngineMumps.jl"))
using .BeatEngineMumps

export assemble_condensed_regular_operators,
    wavelength_quadrature_order,
    prepare_condensed_coupled_cache,
    release_condensed_coupled_cache!,
    build_condensed_coupled_system,
    release_condensed_coupled_system!,
    solve_condensed_coupled_excitations,
    solve_condensed_coupled_system,
    solve_condensed_coupled_systems,
    assemble_condensed_bem_operators,
    release_condensed_bem_operators!

"""
    _interior_partition(fem_system, interface_operators, retained_vertices)

Split FEM vertices into the eliminated interior and the retained set `Γ`, enforcing the
structural precondition that makes condensation valid: interface loads must have no support on
interior vertices. If they did, eliminating the interior would have to carry a flux contribution
into the retained/flux block, which this assembly does not form.
"""
function _interior_partition(
    fem_system::SparseMatrixCSC,
    interface_operators::InterfaceOperators,
    retained_vertices,
)
    fem_count = size(fem_system, 1)
    retained = Int.(collect(retained_vertices))
    retained_set = Set(retained)
    interior = [vertex for vertex in 1:fem_count if !(vertex in retained_set)]
    nnz(interface_operators.fem_load[interior, :]) == 0 || error(
        "FEM static condensation currently requires interface loads to have support only on interface nodes.",
    )
    return interior, retained
end

"""
    _densify_sparse_columns!(dense, source, columns)

Scatter `source[:, columns]` into `dense`, which must be at least as wide as `columns`.

`source[:, columns]` would build a whole intermediate `SparseMatrixCSC` before densifying it;
this writes the nonzeros straight into a buffer the caller reuses across blocks, which keeps the
Schur sweep from allocating two large arrays per block per task.
"""
function _densify_sparse_columns!(
    dense::AbstractMatrix{ComplexF64},
    source::SparseMatrixCSC{ComplexF64},
    columns,
)
    return BeatEngineCoupled._densify_sparse_columns!(dense, source, columns)
end

"""
    _schur_block_width(requested, retained_count) -> Int

Narrow `requested` until `Γ` splits into at least one block per thread, so the column sweep never
leaves threads idle. Never widens past `requested`, and never returns less than one.

`fld` rather than `cld`: the guarantee is on the block *count*, `cld(retained_count, width)`. Ten
columns across eight threads need a width of one to reach eight blocks -- `cld(10, 8)` is 2, which
yields five.
"""
function _schur_block_width(requested::Int, retained_count::Int)
    return BeatEngineCoupled._resolved_schur_block_size(requested, retained_count)
end

"""
    _coupled_mode(name, bem_backend=:cpu; metal_default=:auto, cpu_default=:off) -> :off | :on | :auto

Resolve a condensed-coupled optimization switch.

- **Unset:** `metal_default` on the Metal backend, `cpu_default` on the CPU backend, `:off`
  elsewhere (CUDA and ROCm keep their established paths unless a switch is set explicitly).
- **`auto`:** use the optimization when the model's structure allows it; otherwise use the
  established path and record why (`coupled_optimization_fallback_reasons`).
- **`1`/`on`/`true`/`yes`:** require it and raise when the structure does not allow it.
- **`0`/`off`/`false`/`no`:** don't use it.

Anything else is an error rather than a silent default. For switches without a structural
precondition `auto` and `on` behave the same.
"""
function _coupled_mode(name::AbstractString, bem_backend::Symbol=:cpu; metal_default::Symbol=:auto,
                       cpu_default::Symbol=:off)
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return bem_backend == :metal ? metal_default : bem_backend == :cpu ? cpu_default : :off
    value in ("1", "on", "true", "yes") && return :on
    value in ("0", "off", "false", "no") && return :off
    value == "auto" && return :auto
    error("Unsupported $name value: $value. Expected 1/on, 0/off or auto.")
end

_coupled_switch(name::AbstractString, bem_backend::Symbol=:cpu; metal_default::Symbol=:auto, cpu_default::Symbol=:off) =
    _coupled_mode(name, bem_backend; metal_default=metal_default, cpu_default=cpu_default) != :off

"""
    _coupled_choice(name, bem_backend, choices, cpu_default, metal_default) -> Symbol

A named choice (`BLAB_COUPLED_FEM_SOLVER`, `BLAB_COUPLED_INTERFACE_MASS_SOLVER`): unset gives the
backend's default, otherwise one of `choices`.
"""
function _coupled_choice(name::AbstractString, bem_backend::Symbol, choices, cpu_default::Symbol, metal_default::Symbol)
    value = lowercase(strip(get(ENV, name, "")))
    isempty(value) && return bem_backend == :metal ? metal_default : cpu_default
    Symbol(value) in choices && return Symbol(value)
    error("Unsupported $name value: $value. Expected $(join(string.(choices), " or ")).")
end

"""
Eliminate transducer-surface FEM vertices together with the interior instead of retaining them.

Each transducer couples to the FEM only through one load column (`fem_surface`, the prescribed
normal-velocity flux of a rigid piston) and one force row (`fem_force`, the same nodal areas times
the surface completion factor). Both are rank one per transducer, so eliminating those vertices
turns them into `|transducers|` extra interior solves (plus as many transpose solves) instead of
retained Schur columns. Exact up to round-off; see `_build_condensation`.
"""
_transducer_condensation_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_TRANSDUCER_CONDENSATION", bem_backend; cpu_default=:auto)
_transducer_condensation_enabled(bem_backend::Symbol=:cpu) = _transducer_condensation_mode(bem_backend) != :off

"""
`BLAB_COUPLED_DENSE_FLOAT64=1`: assemble and factor the dense coupled system in `ComplexF64`
whatever `T` is, keeping the Schur block in double precision. Off by default on every backend
(on Metal, `BLAB_COUPLED_DENSE_REFINEMENT` gives the same accuracy for less time).
"""
_dense_float64_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_DENSE_FLOAT64", bem_backend; metal_default=:off)
_dense_float64_enabled(bem_backend::Symbol=:cpu) = _dense_float64_mode(bem_backend) != :off

"""
`BLAB_COUPLED_DENSE_REFINEMENT=1`: assemble the dense coupled system in `ComplexF64` as
`BLAB_COUPLED_DENSE_FLOAT64` does, but factor a `ComplexF32` copy and recover the double-precision
solution by iterative refinement against the `ComplexF64` matrix (`RefinedDenseLU`). A solve that
stalls or does not reach the Float64 backward error within `DENSE_REFINEMENT_MAX_ITERATIONS` steps is redone with a
`ComplexF64` LU and says why. Takes precedence over `BLAB_COUPLED_DENSE_FLOAT64` for the factorization.
"""
_dense_refinement_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_DENSE_REFINEMENT", bem_backend; cpu_default=:auto)
_dense_refinement_enabled(bem_backend::Symbol=:cpu) = _dense_refinement_mode(bem_backend) != :off
_dense_double_assembly(bem_backend::Symbol=:cpu) =
    _dense_float64_enabled(bem_backend) || _dense_refinement_enabled(bem_backend)

const DENSE_REFINEMENT_MAX_ITERATIONS = 10

"""
    RefinedDenseLU(matrix)

Single-precision LU of a double-precision dense system, solved by iterative refinement:
`x ← x + F32⁻¹ (b - A x)` with the residual in `ComplexF64`. Each step contracts the error by
about `κ(A) eps(Float32)`; the Multi_region_SAWMOD systems (κ ≈ 5e6) converge in two steps.
The stopping test is LAPACK `zcgesv`'s backward error, `‖r‖∞ ≤ ‖x‖∞ ‖A‖∞ eps(Float64) √n` per
column (`‖A‖∞` the operator norm, `opnorm`), which is what a `ComplexF64` LU attains, whatever `κ`.
Convergence is tested before the stall rule, so a solve already at that level is accepted.

The `ComplexF64` LU is used instead, with the reason in `fallback_reason`, when
- an entry is outside the `Float32` range (narrowing would overflow),
- the `Float32` factorization is singular or not finite, or
- a solve stalls (the backward-error ratio fails to halve), produces a non-finite residual, or has
  not converged after `DENSE_REFINEMENT_MAX_ITERATIONS` steps.
The fallback factor is built once and serves every later solve. A matrix that is singular in
`Float64` too throws `SingularException`; non-finite matrices or right-hand sides throw
`ArgumentError`.

`matrix` is kept, not copied: the caller hands over ownership and must not modify it while the
factorization is in use (the coupled builder allocates it per system and never writes it again).
`iterations` is the most refinement steps any solve needed; `backward_error` is the worst accepted
column's ratio to the Float64 attainable backward error in the last solve (≤ 1 unless it came
from the fallback, whose ratio is recorded as computed).
"""
mutable struct RefinedDenseLU
    matrix::Matrix{ComplexF64}
    matrix_norm::Float64
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    fallback::Union{Nothing,LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{LinearAlgebra.BlasInt}}}
    iterations::Int
    fallback_reason::Union{Nothing,String}
    backward_error::Float64
end

function RefinedDenseLU(matrix::Matrix{ComplexF64})
    all(isfinite, matrix) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    refined = RefinedDenseLU(matrix, opnorm(matrix, Inf), nothing, nothing, 0, nothing, NaN)
    if maximum(abs, matrix; init=0.0) > floatmax(Float32)
        _dense_fall_back!(refined, "an entry is outside the Float32 range")
    else
        candidate = lu!(ComplexF32.(matrix); check=false)
        if issuccess(candidate) && all(isfinite, candidate.factors)
            refined.factor = candidate
        else
            _dense_fall_back!(refined, "the Float32 factorization is singular or not finite")
        end
    end
    return refined
end

function _dense_fall_back!(factorization::RefinedDenseLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason
    factorization.factor = nothing
    factorization.fallback = lu(factorization.matrix)
    return factorization
end

Base.size(factorization::RefinedDenseLU, dims...) = size(factorization.matrix, dims...)

# Worst column's backward error, in units of the double-precision attainable one.
function _dense_backward_error_ratio(factorization::RefinedDenseLU, residual, solution)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(size(factorization.matrix, 1))
    return maximum(
        column -> norm(view(residual, :, column), Inf) /
                  max(norm(view(solution, :, column), Inf) * threshold, floatmin(Float64)),
        axes(solution, 2);
        init=0.0,
    )
end

function _dense_residual!(residual, factorization::RefinedDenseLU, target, solution)
    copyto!(residual, target)
    mul!(residual, factorization.matrix, solution, -one(ComplexF64), one(ComplexF64))
    return residual
end

function Base.:\(factorization::RefinedDenseLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = ComplexF64.(rhs)
    residual = similar(target)
    if isnothing(factorization.fallback)
        solution = ComplexF64.(factorization.factor \ ComplexF32.(target))
        previous = Inf
        reason = nothing
        for iteration in 0:DENSE_REFINEMENT_MAX_ITERATIONS
            ratio = _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
            if !isfinite(ratio)
                reason = "non-finite residual after $iteration refinement steps"
                break
            end
            if ratio <= 1
                factorization.backward_error = ratio
                return solution
            end
            iteration == DENSE_REFINEMENT_MAX_ITERATIONS && break
            if ratio > previous / 2
                reason = "refinement stalled at $(ratio)x the Float64 backward error after $iteration steps"
                break
            end
            previous = ratio
            solution .+= ComplexF64.(factorization.factor \ ComplexF32.(residual))
            factorization.iterations = max(factorization.iterations, iteration + 1)
        end
        isnothing(reason) &&
            (reason = "refinement did not reach the Float64 backward error in $(DENSE_REFINEMENT_MAX_ITERATIONS) steps")
        _dense_fall_back!(factorization, reason)
    end
    solution = factorization.fallback \ target
    factorization.backward_error =
        _dense_backward_error_ratio(factorization, _dense_residual!(residual, factorization, target, solution), solution)
    return solution
end

"""
    dense_solver_diagnostics(system) -> Dict{String,Any}

The dense coupled factorization that ran: `dense_solver` (`lu_float32`, `lu_float64`,
`lu_float32_refined`, or `lu_float64_fallback` after a refinement fallback),
`dense_refinement_iterations`, `dense_refinement_fallback_reason` and `dense_refinement_backward_error`
(the last solve's worst column, in units of the Float64 attainable backward error).
"""
function dense_solver_diagnostics(system)
    factorization = hasproperty(system, :factorization) ? system.factorization : nothing
    if factorization isa RefinedDenseLU
        return Dict{String,Any}(
            "dense_solver" => isnothing(factorization.fallback) ? "lu_float32_refined" : "lu_float64_fallback",
            "dense_refinement_iterations" => factorization.iterations,
            "dense_refinement_fallback_reason" => factorization.fallback_reason,
            "dense_refinement_backward_error" => isnan(factorization.backward_error) ? nothing : factorization.backward_error,
        )
    end
    kind = factorization isa LinearAlgebra.LU ? "lu_" * lowercase(string(real(eltype(factorization)))) : nothing
    return Dict{String,Any}(
        "dense_solver" => kind,
        "dense_refinement_iterations" => 0,
        "dense_refinement_fallback_reason" => nothing,
        "dense_refinement_backward_error" => nothing,
    )
end

"""
`BLAB_COUPLED_FEM_FLOAT64=1`: under `precision=float32`, assemble the FEM stiffness, mass and
bulk-loss matrices in `Float64` (once per mesh) and hand the condensation a `ComplexF64` dynamic
stiffness. No effect under `precision=float64`.

`A_II = K - k² M` has a near-constant pressure mode whose eigenvalue scales like `k²`: the air
spring of an enclosed volume. `K` annihilates constants only up to its round-off, and a `Float32`
`K` leaves row sums near `eps(Float32) ‖K‖`, which that mode amplifies by `~1/(k h)²`. On
Multi_region_SAWMOD at 20 Hz this puts a 5e-5 relative error in the transducer mechanical block
and 1.2e-4 in every output, while the `A_II` factorization is already `Float64`. The matrices are
sparse and cached, so the double-precision assembly costs little.
"""
_fem_float64_mode(bem_backend::Symbol=:cpu) = _coupled_mode("BLAB_COUPLED_FEM_FLOAT64", bem_backend)
_fem_float64_enabled(bem_backend::Symbol=:cpu) = _fem_float64_mode(bem_backend) != :off

"""
    _fem_system_float64(store, fem_mesh, prepared, frequency_hz, sound_speed, density)

The `ComplexF64` FEM dynamic stiffness for `BLAB_COUPLED_FEM_FLOAT64`: the same terms as the
`Complex{T}` one in `build_condensed_coupled_system`, from `Float64` matrices assembled on a
widened copy of `fem_mesh`.

- **Geometry:** Float64 arithmetic on the mesh's own (possibly Float32-rounded) coordinates, not
  higher-precision geometry. What matters for the constant mode is that `K` is assembled
  consistently in Float64, so its row sums vanish to Float64 round-off on that geometry.
- **Elements:** P1, exactly as `prepare_coupled_cache` assembles the cached system; the Float64
  stiffness and mass must have the cached stiffness's sparsity pattern, or this refuses.
- **Walls:** wall-impedance boundary masses are widened, not reassembled: they are positive
  boundary terms with no cancellation to protect.
- **Cache:** `store` (the cache's) keeps the matrices across frequencies, keyed by value on what
  they depend on (vertex coordinates, tetrahedra, per-vertex bulk-loss factors, wall matrices), so
  a mesh or loss change made in place is picked up.
"""
function _fem_system_float64(store, fem_mesh::VolumeMesh, prepared, frequency_hz, sound_speed, density)
    walls = [operator.matrix for operator in prepared.wall_impedance_operators]
    cached = !isnothing(store) && haskey(store, :matrices) &&
             store[:vertices] == fem_mesh.vertices && store[:tetrahedra] == fem_mesh.tetrahedra &&
             store[:bulk_loss] == prepared.bulk_loss_factor_by_vertex && store[:walls] == walls
    matrices = cached ? store[:matrices] : nothing
    if isnothing(matrices)
        mesh = VolumeMesh{Float64}(
            SVector{3,Float64}.(fem_mesh.vertices),
            fem_mesh.tetrahedra,
            fem_mesh.tetra_physical_tags,
            fem_mesh.boundary_faces,
            fem_mesh.boundary_physical_tags,
            fem_mesh.physical_names,
            fem_mesh.quadratic_tetrahedra,
            fem_mesh.quadratic_boundary_faces,
        )
        stiffness, mass = assemble_p1_fem_matrices(mesh)
        size(stiffness) == size(prepared.stiffness) &&
            stiffness.colptr == prepared.stiffness.colptr && stiffness.rowval == prepared.stiffness.rowval &&
            mass.colptr == prepared.stiffness.colptr && mass.rowval == prepared.stiffness.rowval ||
            error("Double-precision FEM matrices do not have the cached (P1) FEM system's structure.")
        matrices = (
            stiffness=stiffness,
            mass=mass,
            bulk_loss_mass=spdiagm(0 => Float64.(prepared.bulk_loss_factor_by_vertex)) * mass,
            walls=[SparseMatrixCSC{Float64,Int}(operator.matrix) for operator in prepared.wall_impedance_operators],
        )
        if !isnothing(store)
            store[:vertices] = copy(fem_mesh.vertices)
            store[:tetrahedra] = copy(fem_mesh.tetrahedra)
            store[:bulk_loss] = copy(prepared.bulk_loss_factor_by_vertex)
            store[:walls] = [copy(matrix) for matrix in walls]
            store[:matrices] = matrices
        end
    end
    omega = 2pi * Float64(frequency_hz)
    system = assemble_fem_dynamic_stiffness(
        matrices.stiffness,
        matrices.mass,
        omega / Float64(sound_speed);
        bulk_loss_mass=matrices.bulk_loss_mass,
    )
    for (operator, matrix) in zip(prepared.wall_impedance_operators, matrices.walls)
        admittance = miki_rigid_backed_surface_admittance(
            Float64(frequency_hz),
            Float64(sound_speed),
            Float64(density),
            Float64(operator.thickness_m),
            Float64(operator.flow_resistivity_pa_s_per_m2),
        )
        system -= neumann_scale(Float64(density), omega) * admittance .* matrix
    end
    return system
end

"""
Eliminate the duplicated interface pressures from the dense coupled system.

The continuity rows are `fem_trace * p_FEM - bem_trace * p_BEM = 0` with both traces unit
selections (`assemble_interface_operators`), so once `Γ` is exactly the interface vertex set,
`p_Γ = P p_B` with `P` an index map. Substituting it removes `|Γ|` unknowns and the `|Γ|`
continuity rows. Exact; the dense order drops from `|Γ| + |B| + |I| + 2t` to `|B| + |I| + 2t`.
"""
_interface_pressure_elimination_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_INTERFACE_PRESSURE_ELIMINATION", bem_backend; metal_default=:off)
_interface_pressure_elimination_enabled(bem_backend::Symbol=:cpu) = _interface_pressure_elimination_mode(bem_backend) != :off

"""
Additionally eliminate the interface fluxes (implies pressure elimination).

After pressure elimination the condensed FEM rows read `S P p_B - M_Γ q + E y = g`, with `M_Γ`
the interface boundary mass matrix restricted to `Γ` (geometry only, SPD for a nondegenerate
conforming interface). Then `q = M_Γ⁻¹ (S P p_B + E y - g)`, and substituting it into the BEM rows
leaves `[p_B, y]` only. `M_Γ` is factored once per geometry; `S` is never inverted.
"""
_interface_flux_elimination_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION", bem_backend; cpu_default=:auto)
_interface_flux_elimination_enabled(bem_backend::Symbol=:cpu) = _interface_flux_elimination_mode(bem_backend) != :off

"""
`BLAB_COUPLED_FEM_SOLVER`: `umfpack` or `mumps`; unset is `mumps` on Metal (whose environment ships
MUMPS_seq_jll) and `umfpack` elsewhere.

`mumps` factors the whole FEM block once per frequency with sequential MUMPS in complex-symmetric
LDLᵀ mode and takes the Schur complement on `Γ` from the partial factorization, instead of one
UMFPACK solve per `Γ` column. The analysis is kept in the cache across frequencies. If the
library cannot be loaded or a factorization fails, the condensation falls back to UMFPACK and
says why in `fem_solver_fallback_reason`. See `BeatEngineMumps`.
"""
_fem_solver_selection(bem_backend::Symbol=:cpu) =
    _coupled_choice("BLAB_COUPLED_FEM_SOLVER", bem_backend, (:umfpack, :mumps), :umfpack, :mumps)

"""
    _interface_elimination_request(bem_backend=:cpu) -> (mode, required)

`mode` is `:flux`, `:pressure` or `:none`; `required` is whether the chosen switch was set to `on`
(refuse unsupported structure) rather than `auto` (fall back to `:none`).
"""
function _interface_elimination_request(bem_backend::Symbol=:cpu)
    flux = _interface_flux_elimination_mode(bem_backend)
    flux != :off && return (:flux, flux == :on)
    pressure = _interface_pressure_elimination_mode(bem_backend)
    pressure != :off && return (:pressure, pressure == :on)
    return (:none, false)
end

_interface_elimination_mode(bem_backend::Symbol=:cpu) = first(_interface_elimination_request(bem_backend))

"""
    _interface_elimination_map(interface_map, interface_operators, gamma_fem_vertices)

Check the structural facts the interface elimination relies on and return, for every `Γ` column
(sorted FEM vertex order), its interface degree of freedom and the BEM vertex it equals.
"""
function _interface_elimination_map(interface_map, interface_operators, gamma_fem_vertices)
    fem_vertices = Int.(interface_map.fem_vertex_indices)
    bem_vertices = Int.(interface_map.fem_to_bem_vertex_indices)
    interface_count = length(fem_vertices)
    # A FEM vertex shared by two interfaces would carry two continuity rows onto one pressure;
    # removing both would drop the constraint between the two BEM copies.
    allunique(fem_vertices) || error(
        "Interface elimination requires each FEM interface vertex to belong to one interface.",
    )
    length(gamma_fem_vertices) == interface_count && sort(fem_vertices) == gamma_fem_vertices || error(
        "Interface elimination requires the Schur set to be exactly the interface vertices; " *
        "retained transducer surfaces widen it (set BLAB_COUPLED_TRANSDUCER_CONDENSATION=1).",
    )
    fem_count = size(interface_operators.fem_trace, 2)
    bem_count = size(interface_operators.bem_trace, 2)
    unit = ones(eltype(interface_operators.fem_trace), interface_count)
    interface_operators.fem_trace == sparse(1:interface_count, fem_vertices, unit, interface_count, fem_count) &&
        interface_operators.bem_trace == sparse(1:interface_count, bem_vertices, unit, interface_count, bem_count) ||
        error("Interface elimination requires unit nodal-selection pressure traces.")
    size(interface_operators.fem_load, 2) == interface_count ||
        error("Interface elimination requires one flux column per interface vertex.")
    dof_by_vertex = Dict(vertex => index for (index, vertex) in enumerate(fem_vertices))
    gamma_dof = [dof_by_vertex[vertex] for vertex in gamma_fem_vertices]
    return (gamma_dof=gamma_dof, bem_of_gamma=bem_vertices[gamma_dof])
end

"""
    _interface_mass_factorization(store, interface_operators, gamma_fem_vertices)

Sparse real LU of `M_Γ = fem_load[Γ, :]`, held in `store` across frequencies of one cached sweep.
The load operator is frequency-independent and owned by the cache, so identity is the key.
"""
function _interface_mass_factorization(store, interface_operators, gamma_fem_vertices)
    if !isnothing(store) &&
       get(store, :fem_load, nothing) === interface_operators.fem_load &&
       get(store, :gamma, nothing) == gamma_fem_vertices
        return store[:factorization], true
    end
    mass = SparseMatrixCSC{Float64,Int}(interface_operators.fem_load[gamma_fem_vertices, :])
    size(mass, 1) == size(mass, 2) || error("Interface mass matrix must be square.")
    factorization = lu(mass)
    if !isnothing(store)
        store[:fem_load] = interface_operators.fem_load
        store[:gamma] = copy(gamma_fem_vertices)
        store[:factorization] = factorization
    end
    return factorization, false
end

"""
`BLAB_COUPLED_INTERFACE_MASS_SOLVER`: `lu` or `cholmod`; unset is `cholmod` on Metal and CPU, `lu` elsewhere.

Only read under `BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION`. `M_Γ` is the real P1 boundary mass
matrix of the interface: symmetric positive definite and frequency independent. `cholmod` factors
it once per geometry with sparse Cholesky and applies it to a complex right-hand side as one real
panel `[Re R, Im R]` instead of the UMFPACK per-column complex solves. The factor is checked
against a random panel when built and replaced by the LU factor, with the reason recorded, if
Cholesky fails or the check does not reach `1e-10`.
"""
_interface_mass_solver_selection(bem_backend::Symbol=:cpu) =
    _coupled_choice("BLAB_COUPLED_INTERFACE_MASS_SOLVER", bem_backend, (:lu, :cholmod),
        bem_backend == :cpu ? :cholmod : :lu, :cholmod)

"""
`BLAB_COUPLED_INTERFACE_MASS_OVERLAP=1`: form `W = M_Γ⁻¹ S` and `V = M_Γ⁻¹ E` in the FEM
condensation stage instead of after it. They depend only on the condensation and the interface
geometry, so on Metal they run while the BEM operators assemble; only the products with the
BEM coupling block wait for the BEM stage.
"""
_interface_mass_overlap_mode(bem_backend::Symbol=:cpu) = _coupled_mode("BLAB_COUPLED_INTERFACE_MASS_OVERLAP", bem_backend)
_interface_mass_overlap_enabled(bem_backend::Symbol=:cpu) = _interface_mass_overlap_mode(bem_backend) != :off

"""
`BLAB_COUPLED_INTERFACE_BLOCKS=1`: keep the block structure of independent FEM components
through the flux elimination.

Vertices in different connected components of the FEM matrix graph share no entry of `A_II`,
`A_IΓ` or `A_ΓΓ`, so `S` has no entry between their `Γ` vertices, and the interface mass matrix
(a boundary mass over faces of one region) has none either. `W = M_Γ⁻¹ S` is then block diagonal
and `B_q W` is a sum of per-component products `B_q[:, Γ_k] W_k`, costing `Σ n_k²` instead of
`n²` columns-by-rows. Transducer columns `E` are not split. The mass matrix is checked
structurally when the operator is built and `S` at every frequency. A mass matrix coupling two
components is an error under `on` and falls back to one block under `auto`; `S` coupling two
components of the FEM graph cannot happen and is always an error.
"""
_interface_blocks_mode(bem_backend::Symbol=:cpu) = _coupled_mode("BLAB_COUPLED_INTERFACE_BLOCKS", bem_backend; cpu_default=:auto)
_interface_blocks_enabled(bem_backend::Symbol=:cpu) = _interface_blocks_mode(bem_backend) != :off

"""
`BLAB_COUPLED_DEMAND_RECONSTRUCTION=1`: skip the FEM interior back substitution when no requested
output needs interior pressure. The caller decides (`reconstruct_interior` in
`solve_condensed_coupled_excitations`); skipped interior pressures are `NaN` and the interior
residual is reported as not evaluated.
"""
_demand_reconstruction_mode(bem_backend::Symbol=:cpu) =
    _coupled_mode("BLAB_COUPLED_DEMAND_RECONSTRUCTION", bem_backend; cpu_default=:auto)
_demand_reconstruction_enabled(bem_backend::Symbol=:cpu) = _demand_reconstruction_mode(bem_backend) != :off

_interface_mass_specialized(bem_backend::Symbol=:cpu) =
    _interface_mass_solver_selection(bem_backend) != :lu || _interface_mass_overlap_enabled(bem_backend) ||
    _interface_blocks_enabled(bem_backend)

"""
    _fem_component_labels(fem_system) -> Vector{Int}

Connected-component label (1-based, in order of first vertex) of every FEM vertex in the graph
of the stored entries of `fem_system`, treating each entry as an undirected edge.
"""
function _fem_component_labels(fem_system::SparseMatrixCSC)
    count = size(fem_system, 1)
    size(fem_system, 2) == count || error("FEM system must be square.")
    parent = collect(1:count)
    find(v) = begin
        while parent[v] != v
            parent[v] = parent[parent[v]]
            v = parent[v]
        end
        v
    end
    rows = rowvals(fem_system)
    for column in 1:count, entry in nzrange(fem_system, column)
        a = find(rows[entry]); b = find(column)
        a == b || (parent[max(a, b)] = min(a, b))
    end
    labels = zeros(Int, count)
    next = 0
    for vertex in 1:count
        root = find(vertex)
        labels[root] == 0 && (labels[root] = (next += 1))
        labels[vertex] = labels[root]
    end
    return labels
end

"""
    _interface_mass_operator(store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver)
        -> (operator, cached)

Per-block factors of `M_Γ` for the specialized flux elimination. `gamma_labels` assigns each `Γ`
column (sorted FEM vertex order) to a block; `nothing` is one block. Block `k` holds
`rows` (its `Γ` indices ordered by interface dof) and `dofs = gamma_dof[rows]` (ascending), so
`M_Γ[rows, dofs]` is the symmetric block `M_kk` and `M_Γ X = R` splits into
`X[dofs, :] = M_kk⁻¹ R[rows, :]`. Held in `store` across frequencies; the key is the load
operator's identity, `Γ`, the labels and the solver.
"""
function _interface_mass_operator(store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver::Symbol)
    labels = isnothing(gamma_labels) ? ones(Int, length(gamma_fem_vertices)) : gamma_labels
    if !isnothing(store) &&
       get(store, :operator_fem_load, nothing) === interface_operators.fem_load &&
       get(store, :operator_gamma, nothing) == gamma_fem_vertices &&
       get(store, :operator_labels, nothing) == labels &&
       get(store, :operator_solver, nothing) == solver
        return store[:operator], true
    end
    mass = SparseMatrixCSC{Float64,Int}(interface_operators.fem_load[gamma_fem_vertices, :])
    count = length(gamma_fem_vertices)
    size(mass) == (count, count) || error("Interface mass matrix must be square.")
    length(gamma_dof) == count && sort(gamma_dof) == 1:count ||
        error("Interface dof map must be a permutation of the interface dofs.")
    block_rows = Vector{Int}[]
    for label in sort(unique(labels))
        members = findall(==(label), labels)
        push!(block_rows, members[sortperm(gamma_dof[members])])
    end
    block_of_row = zeros(Int, count)
    block_of_dof = zeros(Int, count)
    for (block, rows) in enumerate(block_rows)
        block_of_row[rows] .= block
        block_of_dof[gamma_dof[rows]] .= block
    end
    # Values, not the stored pattern: an explicitly stored zero couples nothing.
    mass_rows, mass_columns, mass_values = findnz(mass)
    all(iszero(value) || block_of_row[row] == block_of_dof[column]
        for (row, column, value) in zip(mass_rows, mass_columns, mass_values)) ||
        error("Interface mass matrix couples interface vertices of different FEM components.")
    blocks = map(block_rows) do rows
        dofs = gamma_dof[rows]
        block_mass = mass[rows, dofs]
        issymmetric(block_mass) || error("Interface mass block is not symmetric.")
        kind, factor, reason = solver, nothing, nothing
        if solver == :cholmod
            try
                factor = cholesky(Symmetric(block_mass))
                # Deterministic and generic enough for a residual check.
                probe = [sin(0.7 * row + 1.3 * column * row + column) for row in 1:length(rows), column in 1:2]
                check = norm(block_mass * (factor \ probe) - probe) / norm(probe)
                check < 1e-10 || error("residual check $(check)")
            catch exception
                exception isa InterruptException && rethrow()
                kind, factor = :lu, nothing
                reason = "CHOLMOD interface mass factor rejected: " * sprint(showerror, exception)
            end
        end
        isnothing(factor) && (factor = lu(block_mass))
        (
            rows=rows,
            dofs=dofs,
            contiguous=dofs == first(dofs):last(dofs),
            kind=kind,
            factor=factor,
            fallback_reason=reason,
        )
    end
    operator = (count=count, blocks=blocks, solver=solver)
    if !isnothing(store)
        store[:operator_fem_load] = interface_operators.fem_load
        store[:operator_gamma] = copy(gamma_fem_vertices)
        store[:operator_labels] = copy(labels)
        store[:operator_solver] = solver
        store[:operator] = operator
    end
    return operator, false
end

"""
    _interface_mass_operator_or_single_block(store, operators, gamma, gamma_dof, labels, solver, blocks_mode, fallbacks)

`_interface_mass_operator` with the `BLAB_COUPLED_INTERFACE_BLOCKS` policy: under `auto`, a mass
matrix that couples two components is built as one block and the reason is pushed to `fallbacks`;
under `on` the refusal propagates.
"""
function _interface_mass_operator_or_single_block(
    store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver::Symbol, blocks_mode::Symbol,
    fallbacks::Vector{String},
)
    try
        return _interface_mass_operator(store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver)
    catch exception
        (exception isa ErrorException && blocks_mode == :auto && !isnothing(gamma_labels)) || rethrow()
        push!(fallbacks, "interface blocks not used: " * exception.msg)
        return _interface_mass_operator(store, interface_operators, gamma_fem_vertices, gamma_dof, nothing, solver)
    end
end

"""
    _interface_mass_operator_or_unreduced(store, operators, gamma, gamma_dof, labels, solver, blocks_mode, required, fallbacks)

`_interface_mass_operator_or_single_block`, except that a singular mass matrix returns `(nothing, false)` under flux elimination `auto`, with the
reason pushed to `fallbacks`, so the caller solves the unreduced system. Under `on` it raises.
"""
function _interface_mass_operator_or_unreduced(
    store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver::Symbol, blocks_mode::Symbol,
    required::Bool, fallbacks::Vector{String},
)
    try
        return _interface_mass_operator_or_single_block(
            store, interface_operators, gamma_fem_vertices, gamma_dof, gamma_labels, solver, blocks_mode, fallbacks,
        )
    catch exception
        # Only factorization failures: structural refusals (e.g. interface blocks `on`) still propagate.
        (exception isa Union{LinearAlgebra.SingularException,LinearAlgebra.ZeroPivotException} && !required) ||
            rethrow()
        push!(fallbacks, "interface elimination not used: " * sprint(showerror, exception))
        return nothing, false
    end
end

"""
    _mass_block_solve(block, rhs) -> Matrix{ComplexF64}

`M_kk⁻¹ rhs` for a complex right-hand side. The Cholesky path solves the real and imaginary parts
as one real panel.
"""
function _mass_block_solve(block, rhs::AbstractMatrix)
    if block.kind == :cholmod
        rows, columns = size(rhs)
        panel = Matrix{Float64}(undef, rows, 2 * columns)
        @inbounds for column in 1:columns, row in 1:rows
            value = rhs[row, column]
            panel[row, column] = real(value)
            panel[row, columns + column] = imag(value)
        end
        solved = block.factor \ panel
        result = Matrix{ComplexF64}(undef, rows, columns)
        @inbounds for column in 1:columns, row in 1:rows
            result[row, column] = complex(solved[row, column], solved[row, columns + column])
        end
        return result
    end
    return block.factor \ ComplexF64.(rhs)
end

"""
    _interface_mass_apply(operator, rhs) -> Matrix{ComplexF64}

`M_Γ⁻¹ rhs` with `rhs` in `Γ` row order and the result in interface-dof row order.
"""
function _interface_mass_apply(operator, rhs::AbstractMatrix)
    result = zeros(ComplexF64, operator.count, size(rhs, 2))
    for block in operator.blocks
        result[block.dofs, :] = _mass_block_solve(block, rhs[block.rows, :])
    end
    return result
end

"""
    interface_mass_diagnostics(system) -> Dict{String,Any}

What the flux elimination's `M_Γ` solve actually ran, for the result diagnostics:
`interface_mass_solver_requested` (`lu`/`cholmod`), `interface_mass_solver` (`lu`, `cholmod`, or
`mixed` when some blocks fell back), `interface_mass_block_count` and
`interface_mass_fallback_reasons` (one entry per block that fell back). All `nothing`/`0`/empty
when no interface mass solve ran (no elimination, pressure elimination, or an uncondensed system).
"""
function interface_mass_diagnostics(system)
    elimination = hasproperty(system, :interface_elimination) && system.interface_elimination == :flux ?
                  system.interface_elimination_data : nothing
    if !isnothing(elimination) && hasproperty(elimination, :mass_operator)
        operator = elimination.mass_operator
        kinds = unique(block.kind for block in operator.blocks)
        return Dict{String,Any}(
            "interface_mass_solver_requested" => String(operator.solver),
            # No blocks (no interface vertices): nothing was solved.
            "interface_mass_solver" => isempty(kinds) ? nothing : length(kinds) == 1 ? String(only(kinds)) : "mixed",
            "interface_mass_block_count" => length(operator.blocks),
            "interface_mass_fallback_reasons" =>
                String[block.fallback_reason for block in operator.blocks if !isnothing(block.fallback_reason)],
        )
    elseif !isnothing(elimination) && hasproperty(elimination, :mass_factorization)
        return Dict{String,Any}(
            "interface_mass_solver_requested" => "lu",
            "interface_mass_solver" => "lu",
            "interface_mass_block_count" => 1,
            "interface_mass_fallback_reasons" => String[],
        )
    end
    return Dict{String,Any}(
        "interface_mass_solver_requested" => nothing,
        "interface_mass_solver" => nothing,
        "interface_mass_block_count" => 0,
        "interface_mass_fallback_reasons" => String[],
    )
end

"""
    _flux_mass_presolve(operator, schur, motion_columns) -> (schur_blocks, motion_solution, split)

The FEM-only half of the flux elimination: `W_k = M_kk⁻¹ S[rows_k, rows_k]` per block and
`V = M_Γ⁻¹ E`. Refuses an `S` with an entry between blocks.
"""
function _flux_mass_presolve(operator, schur::AbstractMatrix, motion_columns::AbstractMatrix)
    split = Dict{Symbol,Float64}()
    if length(operator.blocks) > 1
        _split_timed!(split, :block_check) do
            block_of_row = zeros(Int, operator.count)
            for (index, block) in enumerate(operator.blocks)
                block_of_row[block.rows] .= index
            end
            @inbounds for column in axes(schur, 2), row in axes(schur, 1)
                block_of_row[row] == block_of_row[column] || iszero(schur[row, column]) ||
                    error("Schur complement couples interface vertices of different FEM components.")
            end
        end
    end
    schur_blocks = map(operator.blocks) do block
        local_schur = _split_timed!(() -> ComplexF64.(schur[block.rows, block.rows]), split, :schur_convert)
        _split_timed!(() -> _mass_block_solve(block, local_schur), split, :mass_solve)
    end
    motion_solution = _split_timed!(() -> _interface_mass_apply(operator, motion_columns), split, :mass_solve)
    return (schur_blocks=schur_blocks, motion_solution=motion_solution, split=split)
end

const _ACCELERATE_ZGEMM = Ref{Ptr{Cvoid}}(C_NULL)
const _ACCELERATE_ZGEMM_LOOKED_UP = Ref(false)
const _ACCELERATE_ZGEMM_LOCK = ReentrantLock()

# Apple Accelerate's ILP64 "new LAPACK" `zgemm` (macOS 13.3+), or C_NULL. The legacy unsuffixed
# entry points are never used: they are not safe under concurrent calls. The lookup and its result
# are guarded by one lock, so concurrent first callers see either nothing yet or the final pointer.
function _accelerate_zgemm()
    lock(_ACCELERATE_ZGEMM_LOCK) do
        if !_ACCELERATE_ZGEMM_LOOKED_UP[]
            if Sys.isapple() && Sys.ARCH === :aarch64
                handle = Base.Libc.Libdl.dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate";
                    throw_error=false)
                if handle !== nothing
                    symbol = Base.Libc.Libdl.dlsym(handle, "zgemm\$NEWLAPACK\$ILP64"; throw_error=false)
                    _ACCELERATE_ZGEMM[] = symbol === nothing ? C_NULL : symbol
                end
            end
            _ACCELERATE_ZGEMM_LOOKED_UP[] = true
        end
        return _ACCELERATE_ZGEMM[]
    end
end

# A package image stores the pointer as C_NULL but keeps the flag, so a fresh process would never
# look it up again. Reset both in every new process and after a precompile workload.
function reset_accelerate_zgemm!()
    lock(_ACCELERATE_ZGEMM_LOCK) do
        _ACCELERATE_ZGEMM[] = C_NULL
        _ACCELERATE_ZGEMM_LOOKED_UP[] = false
    end
    return nothing
end

__init__() = reset_accelerate_zgemm!()

"""
`BLAB_COUPLED_HOST_ZGEMM` (`auto`, `accelerate`, `blas`): the library for the large ComplexF64
interface products of the flux elimination (`B_q W`, `B_q V`). `auto` uses Apple Accelerate's
ILP64 `zgemm` on Apple Silicon when its new-LAPACK entry point exists, and Julia's BLAS
elsewhere. On an M1 Max, at the products' shapes (about 3,100 x k by k x k, k = 300-1,200),
Accelerate measured 2.0-2.5x faster than OpenBLAS with eight threads, agreeing to 1e-15
relative. Only these products change library; Julia's BLAS is untouched.
"""
function _host_zgemm_symbol()
    raw = lowercase(strip(get(ENV, "BLAB_COUPLED_HOST_ZGEMM", "auto")))
    raw in ("auto", "accelerate", "blas") ||
        error("Unsupported BLAB_COUPLED_HOST_ZGEMM value: $raw. Expected auto, accelerate, or blas.")
    raw == "blas" && return C_NULL
    symbol = _accelerate_zgemm()
    raw == "accelerate" && symbol == C_NULL &&
        error("BLAB_COUPLED_HOST_ZGEMM=accelerate, but Accelerate's new-LAPACK zgemm is not available.")
    return symbol
end

"""
`"accelerate"` or `"blas"`: the library `_host_zgemm` selects for ComplexF64 products (a product
it cannot hand to BLAS still falls back to `A * B`). A diagnostic, so it never throws: an invalid
`BLAB_COUPLED_HOST_ZGEMM`, or `accelerate` without the symbol, reports `"unavailable: <reason>"`
even when the solve never reached a product.
"""
function host_zgemm_path()
    try
        return _host_zgemm_symbol() == C_NULL ? "blas" : "accelerate"
    catch exception
        exception isa ErrorException || rethrow()
        return "unavailable: " * exception.msg
    end
end

"""
    _host_zgemm(A, B) -> Matrix{ComplexF64}

`A * B` for ComplexF64 operands with unit row stride and a column stride at least the row count
(so it is a valid BLAS leading dimension), through `_host_zgemm_symbol()`. Anything else (another
element type, a non-unit row stride, a reversed or overlapping column stride) takes Julia's `*`.
"""
function _host_zgemm(A::AbstractMatrix, B::AbstractMatrix)
    symbol = _host_zgemm_symbol()
    (symbol == C_NULL || eltype(A) !== ComplexF64 || eltype(B) !== ComplexF64 ||
     !(A isa StridedMatrix) || !(B isa StridedMatrix) || stride(A, 1) != 1 || stride(B, 1) != 1 ||
     stride(A, 2) < max(1, size(A, 1)) || stride(B, 2) < max(1, size(B, 1))) &&
        return A * B
    m, k = size(A)
    k == size(B, 1) || throw(DimensionMismatch("A has $k columns, B has $(size(B, 1)) rows"))
    n = size(B, 2)
    C = Matrix{ComplexF64}(undef, m, n)
    (m == 0 || n == 0) && return C
    k == 0 && return fill!(C, zero(ComplexF64))
    GC.@preserve A B C ccall(symbol, Cvoid,
        (Ref{UInt8}, Ref{UInt8}, Ref{Int64}, Ref{Int64}, Ref{Int64}, Ref{ComplexF64}, Ptr{ComplexF64},
         Ref{Int64}, Ptr{ComplexF64}, Ref{Int64}, Ref{ComplexF64}, Ptr{ComplexF64}, Ref{Int64}),
        UInt8('N'), UInt8('N'), m, n, k, one(ComplexF64), pointer(A), stride(A, 2),
        pointer(B), stride(B, 2), zero(ComplexF64), pointer(C), max(1, m))
    return C
end

"""
    _flux_block_products!(coupled, rows, columns_of_gamma, operator, schur_blocks, interface_block, split)

`coupled[rows, columns_of_gamma[j]] += (B_q W)[:, j]` for every `Γ` column `j`, one block at a
time: `B_q[:, dofs_k] W_k` lands on the columns of `rows_k`. A contiguous dof range is read through
a strided view, otherwise the coupling columns are copied.
"""
function _flux_block_products!(coupled, rows, columns_of_gamma, operator, schur_blocks, interface_block, split)
    for (block, schur_block) in zip(operator.blocks, schur_blocks)
        coupling_columns = block.contiguous ?
                           view(interface_block, :, first(block.dofs):last(block.dofs)) :
                           interface_block[:, block.dofs]
        block_coupling = _split_timed!(() -> _host_zgemm(coupling_columns, schur_block), split, :product)
        _split_timed!(split, :scatter) do
            for (local_column, column) in enumerate(block.rows)
                @views coupled[rows, columns_of_gamma[column]] .+= block_coupling[:, local_column]
            end
        end
    end
    return coupled
end

"""
    _gamma_motion_columns(condensation, T, scale, transducer_count, transducer_condensation, operators, gamma)

`E`, the transducer motion columns of the condensed FEM rows on `Γ`, as `ComplexF64`: the same
`Complex{T}` block the unmodified layout writes, promoted.
"""
function _gamma_motion_columns(
    condensation,
    ::Type{T},
    normal_derivative_scale,
    transducer_count,
    transducer_condensation,
    resolved_transducer_operators,
    gamma_fem_vertices,
) where {T}
    transducer_count == 0 && return zeros(ComplexF64, length(gamma_fem_vertices), 0)
    transducer_condensation &&
        return ComplexF64.(-normal_derivative_scale .* Complex{T}.(condensation.motion_gamma))
    return ComplexF64.(
        -normal_derivative_scale .* Complex{T}.(
            Matrix(resolved_transducer_operators.fem_surface[gamma_fem_vertices, :])
        ),
    )
end

"""
    _split_timed!(f, split, key) -> f()

Run `f` and add its wall time in seconds to `split[key]`. Diagnostics only.
"""
function _split_timed!(f, split::Dict{Symbol,Float64}, key::Symbol)
    started = time_ns()
    value = f()
    split[key] = get(split, key, 0.0) + (time_ns() - started) / 1.0e9
    return value
end

function _release_factorization_store!(store)
    isnothing(store) && return nothing
    factorization = get(store, :factorization, nothing)
    if !isnothing(factorization)
        finalize(factorization.numeric)
        finalize(factorization.symbolic)
    end
    empty!(store)
    return nothing
end

"""
    _build_condensation(fem_system, interface_operators, retained_vertices)

Factor the FEM interior with UMFPACK and form the dense Schur complement.

`S` is returned unnegated, in the same sign convention as `fem_system`, so it fills exactly the
slot the monolithic formulation fills with the full FEM block.

`A_IΓ` is densified a block of columns at a time rather than all at once: a transducer-heavy `Γ`
makes `interior_count × retained_count` a multi-gigabyte intermediate.

The blocks are swept in parallel. UMFPACK's solve is one triangular pair per right-hand side with
no BLAS-3 step and no internal threading, so the sweep dominates condensation by well over an
order of magnitude relative to the sparse factorization it consumes, and only the task split
scales it. Each block owns a disjoint column slice of the accumulator, so the only shared mutable
state is the factorization, which is split per task rather than locked.

`schur_block_columns` is not a BLAS-3 tile size: the solve is per column whatever the block width,
so a wide block buys no arithmetic and only costs locality. Peak scratch is
`2 * interior_count * schur_block_columns` complex entries *per task*; wide blocks measure no
faster than narrow ones for several times the scratch, so the default stays narrow.

The requested width is an upper bound, not the width used: `_schur_block_width` narrows it until
`Γ` splits into at least one block per thread. Without that, occupancy depends on the size of the
interface: a 500-node interface under a 256-column block yields two blocks, and six of eight
threads idle through the phase that dominates condensation.
"""
function _build_condensation(
    fem_system::SparseMatrixCSC{<:Complex},
    interface_operators::InterfaceOperators{T},
    retained_vertices;
    schur_block_columns::Int=32,
    motion_surface=nothing,
    motion_force=nothing,
    schur_float64::Bool=false,
    fem_solver::Symbol=:umfpack,
    mumps_store=nothing,
) where {T<:AbstractFloat}
    schur_block_columns > 0 || error("Schur block column count must be positive.")
    fem_solver in (:umfpack, :mumps) || error("Unsupported FEM condensation solver: $fem_solver.")
    isnothing(motion_surface) == isnothing(motion_force) ||
        error("Transducer condensation needs both the motion load and the force operator.")
    interior_vertices, retained = _interior_partition(
        fem_system,
        interface_operators,
        retained_vertices,
    )
    interior_count = length(interior_vertices)
    retained_count = length(retained)

    fallback_reason = nothing
    # MUMPS runs in Schur mode (ICNTL(19)), which rejects an empty Schur set (INFOG(1)=-33). That
    # happens when nothing is retained: no interfaces, and transducer surfaces condensed. The
    # UMFPACK path handles it; say why MUMPS was not tried instead of failing into it every frequency.
    if fem_solver == :mumps && interior_count > 0 && retained_count == 0
        fallback_reason = "MUMPS not used: the Schur set is empty (no retained FEM vertices)"
    elseif fem_solver == :mumps && interior_count > 0
        library = mumps_library()
        if library.available
            try
                return _build_mumps_condensation(
                    library,
                    fem_system,
                    interior_vertices,
                    retained;
                    result_type=T,
                    motion_surface=motion_surface,
                    motion_force=motion_force,
                    mumps_store=mumps_store,
                    schur_float64=schur_float64,
                )
            catch exception
                exception isa InterruptException && rethrow()
                fallback_reason = "MUMPS condensation failed: " * sprint(showerror, exception)
                @warn "MUMPS FEM condensation failed; factoring with UMFPACK instead." exception
            end
        else
            fallback_reason = "MUMPS unavailable: " * library.reason
        end
    end

    interior_system = SparseMatrixCSC{ComplexF64,Int}(fem_system[interior_vertices, interior_vertices])
    interior_retained = SparseMatrixCSC{ComplexF64,Int}(fem_system[interior_vertices, retained])
    retained_interior = SparseMatrixCSC{ComplexF64,Int}(fem_system[retained, interior_vertices])

    # UMFPACK performs symbolic and numeric factorization in one call, so the whole cost lands
    # in `factorization_s` and the analysis slot stays zero. If every FEM vertex is retained,
    # condensation is an exact no-op and there is no interior matrix to factor.
    factorization_started = time_ns()
    factorization = interior_count == 0 ? nothing : lu(interior_system)
    factorization_s = (time_ns() - factorization_started) / 1.0e9

    schur_started = time_ns()
    retained_system = Matrix{ComplexF64}(fem_system[retained, retained])
    schur_result = interior_count == 0 ?
                   (schur=retained_system, block_size=0, thread_count=1) :
                   BeatEngineCoupled._blocked_umfpack_schur_complement(
        factorization,
        interior_retained,
        retained_interior,
        retained_system;
        block_size=schur_block_columns,
    )
    # Kept double precision for a double-precision dense system; demoted otherwise.
    schur = schur_float64 ? schur_result.schur : Complex{T}.(schur_result.schur)

    # Transducer-surface vertices eliminated with the interior leave their coupling behind as
    # low-rank terms. With C the motion load columns and R the force columns (FEM x transducers),
    # W = A_II⁻¹ C_I and Z = A_II⁻ᵀ R_I, the condensed blocks are
    #   Γ rows / mechanical columns:  C_Γ - A_ΓI W   (scaled by the Neumann factor by the caller)
    #   mechanical rows / Γ columns:  R_Γ - A_IΓᵀ Z  (transposed, negated by the caller)
    #   mechanical / mechanical:      Rᵢᵀ W          (subtracted, scaled, from Z_m)
    transducer_condensed = !isnothing(motion_surface)
    transducer_started = time_ns()
    motion_fields = if transducer_condensed
        surface = SparseMatrixCSC{ComplexF64,Int}(motion_surface)
        force = SparseMatrixCSC{ComplexF64,Int}(motion_force)
        size(surface, 1) == size(fem_system, 1) && size(force) == size(surface) ||
            error("Transducer motion operators must have one row per FEM vertex.")
        surface_interior = surface[interior_vertices, :]
        force_interior = force[interior_vertices, :]
        motion_gamma = Matrix(surface[retained, :])
        force_gamma = Matrix(force[retained, :])
        motion_solution = zeros(ComplexF64, interior_count, size(surface, 2))
        force_solution = zeros(ComplexF64, interior_count, size(surface, 2))
        if interior_count > 0
            motion_solution = factorization \ Matrix(surface_interior)
            force_solution = transpose(factorization) \ Matrix(force_interior)
            mul!(motion_gamma, retained_interior, motion_solution, -one(ComplexF64), one(ComplexF64))
            mul!(force_gamma, transpose(interior_retained), force_solution, -one(ComplexF64), one(ComplexF64))
        end
        (
            motion_interior=surface_interior,
            motion_solution=motion_solution,
            force_solution=force_solution,
            motion_gamma=motion_gamma,
            force_gamma=force_gamma,
            motion_force_correction=Matrix(transpose(force_interior) * motion_solution),
        )
    else
        (
            motion_interior=nothing,
            motion_solution=nothing,
            force_solution=nothing,
            motion_gamma=nothing,
            force_gamma=nothing,
            motion_force_correction=nothing,
        )
    end
    transducer_solves_s = (time_ns() - transducer_started) / 1.0e9
    schur_extraction_s = (time_ns() - schur_started) / 1.0e9

    return (
        backend=interior_count == 0 ? :cpu_noop : :cpu_umfpack,
        fem_solver_requested=fem_solver,
        fem_solver_fallback_reason=fallback_reason,
        mumps_solver=nothing,
        mumps_owned=false,
        mumps_threads=0,
        factorization=factorization,
        interior_system=interior_system,
        interior_retained=interior_retained,
        retained_interior=retained_interior,
        schur=schur,
        interior_vertices=interior_vertices,
        retained_vertices=retained,
        interior_count=interior_count,
        retained_count=retained_count,
        # The width actually swept, not the one requested.
        schur_block_columns=schur_result.block_size,
        schur_block_size=schur_result.block_size,
        schur_thread_count=schur_result.thread_count,
        transducer_condensed=transducer_condensed,
        motion_fields...,
        timings=(
            analysis_s=0.0,
            factorization_s=factorization_s,
            schur_extraction_s=schur_extraction_s,
            transducer_solves_s=transducer_solves_s,
        ),
    )
end

"""
    _build_mumps_condensation(library, fem_system, interior_vertices, retained; ...)

The `BLAB_COUPLED_FEM_SOLVER=mumps` counterpart of the UMFPACK branch of `_build_condensation`,
returning the same fields.

MUMPS reads the lower triangle of the whole FEM block (SYM=2, so the block must be complex
symmetric; checked entry by entry against the transpose) and returns `S` from the partial LDLᵀ
factorization. `A_II⁻ᵀ = A_II⁻¹` under that symmetry, so the transducer force solves are
ordinary interior solves:
  motion_gamma, force_gamma  one reduction (`ICNTL(26)=1`) of the full `[C R]` columns
  W, Z                       one internal-problem solve (`ICNTL(26)=0`) of the same columns

`mumps_store` (the cache's) keeps the solver and its analysis across frequencies; without it the
solver belongs to this condensation and `_release_condensation!` frees it.
"""
function _build_mumps_condensation(
    library,
    fem_system::SparseMatrixCSC{Complex{S}},
    interior_vertices,
    retained;
    result_type::Type{T}=S,
    motion_surface=nothing,
    motion_force=nothing,
    mumps_store=nothing,
    schur_float64::Bool=false,
) where {S<:AbstractFloat,T<:AbstractFloat}
    threads = mumps_threads()
    owned = isnothing(mumps_store)
    solver = owned ? nothing : get(mumps_store, :solver, nothing)
    if isnothing(solver) || !solver.initialized || solver.threads != threads
        isnothing(solver) || mumps_release!(solver)
        solver = MumpsSchurSolver(library; threads=threads)
        owned || (mumps_store[:solver] = solver)
    end
    try
        analysis_started = time_ns()
        analysis_reused = mumps_analyse!(solver, fem_system, retained)
        analysis_s = (time_ns() - analysis_started) / 1.0e9

        factorization_started = time_ns()
        # Assembly sums the (i, j) and (j, i) contributions separately, so allow round-off.
        schur_double = mumps_factorize!(solver, fem_system; symmetry_tolerance=64 * eps(S))
        factorization_s = (time_ns() - factorization_started) / 1.0e9

        schur_started = time_ns()
        schur = schur_float64 ? schur_double : Complex{T}.(schur_double)
        transducer_condensed = !isnothing(motion_surface)
        transducer_started = time_ns()
        motion_fields = if transducer_condensed
            surface = SparseMatrixCSC{ComplexF64,Int}(motion_surface)
            force = SparseMatrixCSC{ComplexF64,Int}(motion_force)
            size(surface, 1) == size(fem_system, 1) && size(force) == size(surface) ||
                error("Transducer motion operators must have one row per FEM vertex.")
            transducer_count = size(surface, 2)
            columns = hcat(Matrix(surface), Matrix(force))
            reduced = mumps_reduce(solver, columns)
            interior_solution = mumps_interior_solve(solver, columns)
            motion_solution = interior_solution[interior_vertices, 1:transducer_count]
            force_interior = force[interior_vertices, :]
            (
                motion_interior=surface[interior_vertices, :],
                motion_solution=motion_solution,
                force_solution=interior_solution[interior_vertices, (transducer_count+1):end],
                motion_gamma=reduced[:, 1:transducer_count],
                force_gamma=reduced[:, (transducer_count+1):end],
                motion_force_correction=Matrix(transpose(force_interior) * motion_solution),
            )
        else
            (
                motion_interior=nothing,
                motion_solution=nothing,
                force_solution=nothing,
                motion_gamma=nothing,
                force_gamma=nothing,
                motion_force_correction=nothing,
            )
        end
        transducer_solves_s = (time_ns() - transducer_started) / 1.0e9
        schur_extraction_s = (time_ns() - schur_started) / 1.0e9

        return (
            backend=:mumps_seq,
            fem_solver_requested=:mumps,
            fem_solver_fallback_reason=nothing,
            mumps_solver=solver,
            mumps_owned=owned,
            mumps_threads=threads,
            mumps_blas=library.blas,
            # The full FEM block, for the back substitution's residual and coupling matvecs.
            fem_system=fem_system,
            factorization=nothing,
            interior_system=nothing,
            interior_retained=nothing,
            retained_interior=nothing,
            schur=schur,
            interior_vertices=interior_vertices,
            retained_vertices=retained,
            interior_count=length(interior_vertices),
            retained_count=length(retained),
            schur_block_columns=0,
            schur_block_size=0,
            schur_thread_count=threads,
            analysis_reused=analysis_reused,
            factorization_cached=!owned,
            transducer_condensed=transducer_condensed,
            motion_fields...,
            timings=(
                analysis_s=analysis_s,
                factorization_s=factorization_s,
                schur_extraction_s=schur_extraction_s,
                transducer_solves_s=transducer_solves_s,
            ),
        )
    catch
        # A cached solver in an unknown state must not be refactored next frequency.
        mumps_release!(solver)
        owned || delete!(mumps_store, :solver)
        rethrow()
    end
end

function _release_condensation!(condensation)
    isnothing(condensation) && return nothing
    if hasproperty(condensation, :backend) && condensation.backend == :mumps_seq
        condensation.mumps_owned && mumps_release!(condensation.mumps_solver)
        return nothing
    end
    # UMFPACK holds its factors outside the Julia heap; release them with the system rather than
    # waiting for the finalizer, so a frequency sweep does not accumulate them.
    isnothing(condensation.factorization) || finalize(condensation.factorization)
    return nothing
end

"""
    _forward_schur(condensation, fem_rhs) -> (reduced_rhs, interior_rhs)

Reduce a FEM right-hand side onto `Γ`: `g_Γ = f_Γ - A_ΓI * A_II⁻¹ * f_I`. `f_I` is returned
because the backward substitution needs it and nothing else retains it.
"""
function _forward_schur(
    condensation,
    fem_rhs::AbstractMatrix{Complex{T}};
    result_type::Type{<:AbstractFloat}=T,
) where {T<:AbstractFloat}
    R = Complex{result_type}
    interior_rhs = ComplexF64.(fem_rhs[condensation.interior_vertices, :])
    if hasproperty(condensation, :backend) && condensation.backend == :mumps_seq
        reduced = mumps_reduce(condensation.mumps_solver, fem_rhs)
        return R.(reduced), interior_rhs
    end
    retained_rhs = ComplexF64.(fem_rhs[condensation.retained_vertices, :])
    condensation.interior_count == 0 && return R.(retained_rhs), interior_rhs
    mul!(
        retained_rhs,
        condensation.retained_interior,
        condensation.factorization \ interior_rhs,
        -one(ComplexF64),
        one(ComplexF64),
    )
    return R.(retained_rhs), interior_rhs
end

"""
    _backward_schur(condensation, interior_rhs, retained_pressure)
        -> (fem_pressure, interior_residual)

Recover the eliminated interior, `u_I = A_II⁻¹ (f_I - A_IΓ u_Γ)`, and return the full FEM
pressure in mesh vertex order.

`interior_residual` is `‖A_II u_I + A_IΓ u_Γ - f_I‖` per excitation, relative to the largest of
those three terms. It costs one sparse matvec over blocks retained for the backward solve anyway.

It is normalized against the largest term rather than against `f_I` alone because a
voltage-driven transducer forces the system entirely through the electrical rows, leaving `f_I`
identically zero; dividing by `eps` there turns a converged solve into an O(1) reading.

Note what it does *not* measure: it validates the back-substitution, not the conditioning of
`S`. Near an interior resonance the recovered pressure can be visibly wrong while this residual
stays at round-off, so it is not a resonance detector.
"""
function _backward_schur(
    condensation,
    interior_rhs,
    retained_pressure::AbstractMatrix{Complex{T}};
    motion_velocity=nothing,
    motion_scale=nothing,
) where {T<:AbstractFloat}
    retained_double = ComplexF64.(retained_pressure)
    if hasproperty(condensation, :transducer_condensed) && condensation.transducer_condensed
        # Eliminated transducer vertices carry the motion load `s C_I v` on the interior side.
        (isnothing(motion_velocity) || isnothing(motion_scale)) &&
            error("Transducer-condensed back substitution needs the diaphragm velocity.")
        interior_rhs = interior_rhs +
                       ComplexF64(motion_scale) .* (condensation.motion_interior * ComplexF64.(motion_velocity))
    end
    interior_pressure, interior_term, retained_term = if hasproperty(condensation, :backend) &&
                                                         condensation.backend == :mumps_seq
        _mumps_back_substitution(condensation, interior_rhs, retained_double)
    else
        pressure = condensation.interior_count == 0 ?
                   copy(interior_rhs) :
                   condensation.factorization \
                   (interior_rhs - condensation.interior_retained * retained_double)
        (
            pressure,
            condensation.interior_system * pressure,
            condensation.interior_retained * retained_double,
        )
    end
    residual = interior_term + retained_term - interior_rhs

    interior_residual = zeros(T, size(residual, 2))
    for column in axes(residual, 2)
        scale = max(
            norm(view(interior_term, :, column)),
            norm(view(retained_term, :, column)),
            norm(view(interior_rhs, :, column)),
            eps(Float64),
        )
        interior_residual[column] = T(norm(view(residual, :, column)) / scale)
    end

    fem_pressure = zeros(
        Complex{T},
        condensation.interior_count + condensation.retained_count,
        size(retained_pressure, 2),
    )
    fem_pressure[condensation.interior_vertices, :] = Complex{T}.(interior_pressure)
    fem_pressure[condensation.retained_vertices, :] = retained_pressure
    return fem_pressure, interior_residual
end

"""
    _mumps_back_substitution(condensation, interior_rhs, retained_pressure)
        -> (interior_pressure, interior_term, retained_term)

`u_I = A_II⁻¹ (f_I - A_IΓ u_Γ)` with an explicit right-hand side and an internal-problem solve,
not `ICNTL(26)=2`: MUMPS's expansion reuses the forward solution stored by the last reduction and
ignores the right-hand side it is given, and transducer condensation changes `f_I` after the
reduction. The two coupling terms come from the full FEM block, whose interior rows are exactly
`[A_II A_IΓ]`.
"""
function _mumps_back_substitution(condensation, interior_rhs, retained_pressure)
    matrix = condensation.fem_system
    interior = condensation.interior_vertices
    retained = condensation.retained_vertices
    vertex_count = size(matrix, 1)
    excitation_count = size(retained_pressure, 2)
    lifted = zeros(ComplexF64, vertex_count, excitation_count)
    lifted[retained, :] = retained_pressure
    retained_term = (matrix * lifted)[interior, :]
    rhs = zeros(ComplexF64, vertex_count, excitation_count)
    rhs[interior, :] = interior_rhs - retained_term
    interior_pressure = mumps_interior_solve(condensation.mumps_solver, rhs)[interior, :]
    fill!(lifted, zero(ComplexF64))
    lifted[interior, :] = interior_pressure
    interior_term = (matrix * lifted)[interior, :]
    return interior_pressure, interior_term, retained_term
end

"""
    wavelength_quadrature_order(areas, frequency_hz, sound_speed, base_order; ...)

Pick this frequency's regular quadrature order from `kh` against a mesh element-size statistic,
where `h = sqrt(area_stat)` and `kh = 2*pi*f/c * h`.

Owned by this solver rather than shared with the exterior path: this is the only solver that keys
its quadrature caches per order, and the exterior loop holds its device caches at a single rule.

`q1_max` defaults to 0.0, which disables the one-point tier -- `kh <= 0.0` is false for any
positive frequency. That tier is off deliberately. The hypersingular kernel decays like 1/r^3, so
its regular-pair integrand is far less smooth than the single layer's and a centroid rule
approximates it poorly; enabling it needs its own convergence study, not a default.
"""
function wavelength_quadrature_order(
    areas,
    frequency_hz::Real,
    sound_speed::Real,
    base_order::Int;
    mesh_stat::AbstractString="p90",
    q1_max::Real=0.0,
    q2_max::Real=2.0,
)
    q1_cutoff = Float64(q1_max)
    q2_cutoff = Float64(q2_max)
    q1_cutoff >= 0.0 || error("wavelength_kh_q1_max must be non-negative.")
    q2_cutoff > q1_cutoff || error("wavelength_kh_q2_max must exceed wavelength_kh_q1_max.")
    values = collect(Float64.(areas))
    isempty(values) && error("Cannot select wavelength quadrature order from an empty mesh.")
    area = if mesh_stat == "median"
        median(values)
    elseif mesh_stat == "p75"
        quantile(values, 0.75)
    elseif mesh_stat == "p90"
        quantile(values, 0.90)
    elseif mesh_stat == "max"
        maximum(values)
    else
        error("Unsupported wavelength mesh stat: $mesh_stat. Expected median, p75, p90, or max.")
    end
    element_length = sqrt(area)
    kh = Float64(2pi * frequency_hz / sound_speed) * element_length
    order = kh <= q1_cutoff ? 1 : kh <= q2_cutoff ? 2 : base_order
    return (
        order=order,
        base_order=base_order,
        mesh_stat=mesh_stat,
        area=area,
        length=element_length,
        kh=kh,
        q1_max=q1_cutoff,
        q2_max=q2_cutoff,
    )
end

"""
    _condensed_quadrature_bundle(bem_mesh, p1, dp0, order; singular_order, symmetry_mode)

Build the order-dependent half of a coupled cache for one quadrature order.

`BeatEngineCoupled.prepare_coupled_cache` is used unmodified for everything that does not depend
on the regular rule -- the FEM matrices, interface operators, P1/DP0 spaces and singular caches.
This adds only the parts that do, from the same public constructors that cache uses, so a
frequency sweep that changes order rebuilds those and nothing else.

The bundle owns its rule. `assemble_regular_galerkin_operators_cpu` compares `cpu_cache.rule` to
the rule it is handed, and `TriangleRule` defines no `==`, so that comparison is pointer identity.
It is a strict stale-cache guard and it only holds while a rule and the assembly cache built from
it travel together.
"""
function _condensed_quadrature_bundle(
    bem_mesh::BoundaryMesh{T},
    p1::P1Space,
    dp0::DP0Space,
    order::Int;
    singular_order::Int,
    symmetry_mode::Symbol,
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    rule = triangle_rule(T, order)

    assembly_started = time_ns()
    cpu_assembly_cache = bem_backend == :cpu ? build_beat_cpu_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    ) : nothing
    device_cache = bem_backend == :metal ? build_metal_regular_assembly_cache(
        bem_mesh,
        p1,
        dp0,
        rule;
        singular_order=singular_order,
        symmetry_mode=symmetry_mode,
    ) : nothing
    bem_cpu_assembly_cache_s = (time_ns() - assembly_started) / 1.0e9

    identity_started = time_ns()
    # The identity rule is clamped to order 2 and never follows a q1 regular rule down.
    # `l2_identity_element_matrix` integrates a degree-2 polynomial (P1xP1) or degree-1
    # (P1xDP0), and the 3-point order-2 rule is degree-2 exact -- so every order >= 2 gives
    # identical matrices, but the 1-point centroid rule does not: it returns area/9 uniformly
    # where the exact block is area/6 on the diagonal and area/12 off it. That would silently
    # degrade the Burton-Miller 0.5*I term rather than fail.
    identity_rule = order >= 2 ? rule : triangle_rule(T, 2)
    identity_p1_p1 = assemble_l2_identity_matrix(
        bem_mesh, p1, dp0, identity_rule, :p1, :p1; symmetry_mode=symmetry_mode,
    )
    identity_p1_dp0 = assemble_l2_identity_matrix(
        bem_mesh, p1, dp0, identity_rule, :p1, :dp0; symmetry_mode=symmetry_mode,
    )
    bem_identity_cache_s = (time_ns() - identity_started) / 1.0e9

    field_started = time_ns()
    cpu_field_cache = build_field_evaluation_cache(bem_mesh, rule; symmetry_mode=symmetry_mode)
    field_cache = bem_backend == :metal ?
                  build_metal_field_evaluation_cache(cpu_field_cache) : cpu_field_cache
    field_cache_s = (time_ns() - field_started) / 1.0e9

    return (
        order=order,
        rule=rule,
        cpu_assembly_cache=cpu_assembly_cache,
        device_cache=device_cache,
        identity_p1_p1=identity_p1_p1,
        identity_p1_dp0=identity_p1_dp0,
        field_cache=field_cache,
        timings=(
            bem_cpu_assembly_cache_s=bem_cpu_assembly_cache_s,
            bem_identity_cache_s=bem_identity_cache_s,
            field_cache_s=field_cache_s,
        ),
    )
end

"""
    prepare_condensed_coupled_cache(fem_mesh, bem_mesh, interface_map; regular_quadrature_orders, ...)

Wrap an unmodified `BeatEngineCoupled.prepare_coupled_cache` with one quadrature bundle per order
the sweep will select.

Every order is built eagerly. Coupled frequencies are not solved in ascending order -- the
live-plotting order emits both endpoints first and then the interior in van der Corput order -- so
a lazily grown cache would reach peak memory on the second frequency solved rather than at setup,
after the user has already seen a result and believes the run is healthy.
"""
function prepare_condensed_coupled_cache(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap;
    quadrature_order::Int=2,
    regular_quadrature_orders=nothing,
    singular_order::Int=2,
    symmetry_mode::Symbol=:off,
    retained_fem_vertices=interface_map.fem_vertex_indices,
    bulk_loss_factor_by_vertex=zeros(T, length(fem_mesh.vertices)),
    wall_impedances=NamedTuple[],
    bem_backend::Symbol=:cpu,
) where {T<:AbstractFloat}
    # The condensed solver's linear algebra is CPU-only; the BEM operators may
    # come from the CPU or from Metal, which hands them back as host matrices.
    bem_backend in (:cpu, :metal) ||
        error("Condensed coupled BEM backend must be :cpu or :metal; got $bem_backend.")
    base = prepare_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=quadrature_order,
        singular_order=singular_order,
        bem_backend=bem_backend,
        symmetry_mode=symmetry_mode,
        retained_fem_vertices=retained_fem_vertices,
        bulk_loss_factor_by_vertex=bulk_loss_factor_by_vertex,
        wall_impedances=wall_impedances,
    )
    # The base order is always carried, whether or not any frequency selects it: with the q1 tier
    # enabled and a base order of 4, every frequency can select 1 or 2 and leave the base out of
    # the requested set. The base bundle is free anyway -- it aliases what `prepare_coupled_cache`
    # already built -- and `build_condensed_coupled_system` validates requests against
    # `base_quadrature_order`, so the cache has to be able to answer for it regardless.
    requested = isnothing(regular_quadrature_orders) ? Int[] :
                Int.(collect(regular_quadrature_orders))
    all(order -> order >= 1, requested) ||
        error("Condensed coupled regular quadrature orders must be positive.")
    orders = sort(unique(vcat(requested, quadrature_order)))
    bundles = Dict{Int,Any}()
    # The base order's bundle is what `prepare_coupled_cache` already built; reusing those fields
    # keeps a single-order sweep byte-identical to calling the plain cache directly.
    bundles[quadrature_order] = (
        order=quadrature_order,
        rule=base.rule,
        cpu_assembly_cache=base.cpu_assembly_cache,
        device_cache=base.device_cache,
        identity_p1_p1=base.identity_p1_p1,
        identity_p1_dp0=base.identity_p1_dp0,
        field_cache=base.field_cache,
    )
    for order in orders
        order == quadrature_order && continue
        bundles[order] = _condensed_quadrature_bundle(
            bem_mesh,
            base.p1,
            base.dp0,
            order;
            singular_order=singular_order,
            symmetry_mode=base.symmetry_mode,
            bem_backend=bem_backend,
        )
    end
    # Extra bundles are real setup cost, so fold them into the fields the caller already reports
    # rather than leaving a mixed-order sweep looking as cheap as a single-order one. Field names
    # match `prepare_coupled_cache`'s timings exactly, so `cache.timings` reads the same either way.
    extra = [bundle.timings for (order, bundle) in bundles if order != quadrature_order]
    timings = merge(
        base.timings,
        (
            bem_cpu_assembly_cache_s=base.timings.bem_cpu_assembly_cache_s +
                                     sum(t -> t.bem_cpu_assembly_cache_s, extra; init=0.0),
            bem_identity_cache_s=base.timings.bem_identity_cache_s +
                                 sum(t -> t.bem_identity_cache_s, extra; init=0.0),
            field_cache_s=base.timings.field_cache_s +
                          sum(t -> t.field_cache_s, extra; init=0.0),
        ),
    )
    return (
        base=base,
        quadrature_bundles=bundles,
        metal_combined_identity_store=Dict{Int,Any}(),
        # A sweep that assembles the BEM stage on a producer task (the coupled sweep pipeline)
        # reaches this store from two tasks.
        metal_combined_identity_lock=ReentrantLock(),
        base_quadrature_order=quadrature_order,
        singular_order=singular_order,
        timings=timings,
        # Interface mass factorization for BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION.
        interface_mass_store=Dict{Symbol,Any}(),
        # MUMPS solver (and its analysis) for BLAB_COUPLED_FEM_SOLVER=mumps.
        mumps_store=Dict{Symbol,Any}(),
        # Double-precision FEM matrices for BLAB_COUPLED_FEM_FLOAT64.
        fem_float64_store=Dict{Symbol,Any}(),
    )
end

function release_condensed_coupled_cache!(cache)
    if hasproperty(cache, :metal_combined_identity_store)
        lock(cache.metal_combined_identity_lock) do
            for identity in values(cache.metal_combined_identity_store)
                BeatEngineCore.release_metal_coupled_identity_cache!(identity)
            end
            empty!(cache.metal_combined_identity_store)
        end
    end
    # Extra bundles (orders other than the base) own their own device caches
    # under Metal; host bundles are reclaimed by the collector. The base cache
    # still owns whatever `prepare_coupled_cache` allocated.
    if cache.base.bem_backend == :metal
        for (order, bundle) in cache.quadrature_bundles
            order == cache.base_quadrature_order && continue
            release_metal_regular_assembly_cache!(bundle.device_cache)
            release_metal_field_evaluation_cache!(bundle.field_cache)
        end
    end
    hasproperty(cache, :interface_mass_store) && _release_factorization_store!(cache.interface_mass_store)
    hasproperty(cache, :fem_float64_store) && empty!(cache.fem_float64_store)
    if hasproperty(cache, :mumps_store)
        solver = get(cache.mumps_store, :solver, nothing)
        isnothing(solver) || mumps_release!(solver)
        empty!(cache.mumps_store)
    end
    release_coupled_cache!(cache.base)
    return nothing
end

"""
    _stage_overlap_enabled(bem_backend) -> Bool

Whether to run the FEM static condensation concurrently with the BEM operator
assembly inside `build_condensed_coupled_system`.

The two stages are independent: the condensation reads `fem_system`, the
interface operators and the retained vertex list, none of which the BEM assembly
touches. Run in sequence, one processor idles for the other's duration — but
that only costs anything when the two stages use *different* processors. On
Metal the BEM assembly is on the GPU while the condensation is host UMFPACK, so
overlapping hides the shorter stage. On `:cpu` both are host code competing for
the same cores (the Schur complement already saturates them through
`_blocked_umfpack_schur_complement`), so overlapping buys nothing there.
Hence: Metal on, CPU off.

`BLAB_COUPLED_STAGE_OVERLAP` overrides the default: `auto`, `on`, or `off`. A
spawned condensation needs a thread of its own, so a single-threaded Julia
always runs the stages in sequence.
"""
function _stage_overlap_enabled(bem_backend::Symbol)
    requested = lowercase(strip(get(ENV, "BLAB_COUPLED_STAGE_OVERLAP", "auto")))
    requested in ("auto", "on", "off") || error(
        "Unsupported BLAB_COUPLED_STAGE_OVERLAP value: $requested. Expected auto, on, or off.",
    )
    requested == "off" && return false
    Threads.nthreads() > 1 || return false
    requested == "on" && return true
    return bem_backend == :metal
end

"""
    _condensed_bem_assembly_plan(condensed_cache, prepared, T, quadrature_order) -> (plan, identity)

`BLAB_METAL_COUPLED_BEM_ASSEMBLY` for one condensed frequency: `plan.mode` is `:combined` (Metal
assembles A and C directly; `identity` is the cached identity scatter for this quadrature order) or
`:operators` (the four operators; `identity` is `nothing`), with `plan.fallback_reason` when `auto`
could not use the combined path. The build and the sweep pipeline's producer resolve it alike.
"""
function _condensed_bem_assembly_plan(condensed_cache, prepared, ::Type{T}, quadrature_order::Int) where {T}
    plan = if prepared.bem_backend == :metal
        requested = get(ENV, "BLAB_METAL_COUPLED_BEM_ASSEMBLY", "auto")
        # Operators bypass support probes, preserving the diagnostic/reference path.
        reason = lowercase(strip(requested)) == "operators" ? nothing :
                 BeatEngineCore.metal_coupled_combined_support_reason(prepared, T)
        resolve_metal_coupled_bem_assembly(requested, reason)
    else
        (mode=:operators, fallback_reason=nothing)
    end
    if plan.mode == :combined && !hasproperty(condensed_cache, :metal_combined_identity_store)
        # The identity scatter cache lives in the condensed cache; without one there is no owner.
        reason = "the coupled cache has no combined identity store"
        lowercase(strip(get(ENV, "BLAB_METAL_COUPLED_BEM_ASSEMBLY", "auto"))) == "combined" &&
            error("Combined Metal coupled assembly unavailable: $reason")
        plan = (mode=:operators, fallback_reason=reason)
    end
    plan.mode == :combined || return plan, nothing
    # A sweep that assembles the BEM stage on a producer task reaches the store from two tasks.
    identity = lock(condensed_cache.metal_combined_identity_lock) do
        get!(condensed_cache.metal_combined_identity_store, quadrature_order) do
            BeatEngineCore.build_metal_coupled_identity_cache(prepared, T)
        end
    end
    return plan, identity
end

"""
    _assemble_condensed_bem_operators(bem_mesh, prepared, wavenumber, singular_order; combined_identity=nothing)

The BEM stage of one condensed frequency: the four regular Galerkin operators as host matrices.
Metal assembles them on the GPU; the CPU uses this solver's own fork of the regular assembly.
With a `combined_identity` (`_condensed_bem_assembly_plan`), Metal instead assembles the
Burton-Miller A and C directly, returned as `(combined=...,)` device storage.
"""
function _assemble_condensed_bem_operators(bem_mesh, prepared, wavenumber, singular_order::Int; combined_identity=nothing)
    isnothing(combined_identity) || return (combined=BeatEngineCore.assemble_coupled_burton_miller_metal(
        bem_mesh, prepared, wavenumber; identity_cache=combined_identity,
    ),)
    return if prepared.bem_backend == :metal
        # Metal assembles the four operators on the GPU; the condensed algebra
        # below is CPU-only, so bring them down and free the device copies.
        device_operators = assemble_regular_galerkin_operators(
            bem_mesh,
            prepared.p1,
            prepared.dp0,
            wavenumber,
            prepared.rule;
            skip_singular=false,
            singular_order=singular_order,
            backend=:metal,
            device_cache=prepared.device_cache,
            singular_cache=prepared.singular_cache,
            device_singular_cache=prepared.device_singular_cache,
            symmetry_mode=prepared.symmetry_mode,
        )
        # Wraps shared device storage in place (copies it when the storage mode
        # is private); either way the host tuple owns the device buffers, so
        # `device_operators` must not be released separately.
        metal_host_operators(device_operators)
    else
        assemble_condensed_regular_operators(
            bem_mesh,
            prepared.p1,
            prepared.dp0,
            wavenumber,
            prepared.rule;
            skip_singular=false,
            singular_order=singular_order,
            singular_cache=prepared.singular_cache,
            cpu_cache=prepared.cpu_assembly_cache,
            symmetry_mode=prepared.symmetry_mode,
        )
    end
end

"""
    _combine_condensed_bem_operators!(operators, prepared, wavenumber)
        -> (bem_lhs, bem_rhs_operator, bem_interface_block)

The Burton-Miller combination of one frequency's four operators and the interface coupling block
`-C Q`, as fresh host arrays. Frees the operators' Metal buffers: nothing reads them afterwards.
"""
function _combine_condensed_bem_operators!(operators, prepared, wavenumber)
    if hasproperty(operators, :combined)
        # A = 0.5Mpp - D + alpha H and C = S + alpha (adjD + 0.5Mpq); the operator path's
        # right-hand-side operator is -C. Owned host copies, then the device storage is freed.
        try
            host = BeatEngineCore.metal_host_coupled_burton_miller(operators.combined)
            bem_flux = prepared.interface_operators.bem_flux
            size(host.c, 2) == size(bem_flux, 1) || error("Metal combined flux map must have one row per DP0 dof.")
            return copy(host.a), -host.c, host.c * Complex{typeof(wavenumber)}.(bem_flux)
        finally
            BeatEngineCore.release_metal_coupled_burton_miller!(operators.combined)
        end
    end
    bem_lhs, bem_rhs_operator = burton_miller_neumann_matrices(
        operators,
        prepared.identity_p1_p1,
        prepared.identity_p1_dp0,
        wavenumber,
    )
    # `operators` is dead from here on and the matrices above are freshly
    # allocated host arrays, so free the Metal buffers now rather than leaking
    # one operator set per condensed frequency.
    prepared.bem_backend == :metal && release_operator_storage!(operators)
    bem_interface_block = -(bem_rhs_operator * Complex{typeof(wavenumber)}.(prepared.interface_operators.bem_flux))
    return bem_lhs, bem_rhs_operator, bem_interface_block
end

"""
    assemble_condensed_bem_operators(bem_mesh, cache, frequency_hz, sound_speed;
                                     regular_quadrature_order=nothing, singular_order=2)
        -> (bem_lhs, bem_rhs_operator, bem_interface_block, wavenumber, assembly_s)

The BEM stage of `build_condensed_coupled_system` for one frequency, on its own, so a sweep can
assemble frequency i+1's operators on the GPU while the host finishes frequency i. Pass the
result back through `build_condensed_coupled_system(...; bem_operators=() -> result)`. The
operators are exactly the ones the build would assemble itself; release them with
`release_condensed_bem_operators!` if they are never consumed.
"""
function assemble_condensed_bem_operators(
    bem_mesh::BoundaryMesh{T},
    cache,
    frequency_hz::T,
    sound_speed::T;
    regular_quadrature_order::Union{Nothing,Int}=nothing,
    singular_order::Int=2,
) where {T<:AbstractFloat}
    started = time_ns()
    order = isnothing(regular_quadrature_order) ? cache.base_quadrature_order : regular_quadrature_order
    bundle = get(cache.quadrature_bundles, order, nothing)
    isnothing(bundle) && error("Condensed coupled cache holds no quadrature bundle for order $order.")
    cache.singular_order == singular_order ||
        error("Condensed coupled cache singular order does not match the requested singular order.")
    prepared = merge(cache.base, bundle)
    wavenumber = (T(2pi) * frequency_hz) / sound_speed
    _, combined_identity = _condensed_bem_assembly_plan(cache, prepared, T, order)
    operators = _assemble_condensed_bem_operators(
        bem_mesh, prepared, wavenumber, singular_order; combined_identity=combined_identity,
    )
    bem_lhs, bem_rhs_operator, bem_interface_block = _combine_condensed_bem_operators!(operators, prepared, wavenumber)
    return (
        bem_lhs=bem_lhs,
        bem_rhs_operator=bem_rhs_operator,
        bem_interface_block=bem_interface_block,
        wavenumber=wavenumber,
        assembly_s=(time_ns() - started) / 1.0e9,
    )
end

"""Release an `assemble_condensed_bem_operators` result nobody consumed (host arrays only)."""
release_condensed_bem_operators!(produced, bem_backend::Symbol) = nothing

"""
    build_condensed_coupled_system(fem_mesh, bem_mesh, interface_map, frequency_hz, sound_speed, density; ...)

Assemble and factor the interface-condensed coupled system on the CPU.

Mirrors `BeatEngineCoupled.build_coupled_system` in inputs and in the fields callers read, but
always produces the `:fem_interface_condensed` formulation and never the monolithic one.

`cache` is a `prepare_condensed_coupled_cache` result. `regular_quadrature_order` selects which of its
bundles to assemble with, defaulting to the cache's base order.
"""
function build_condensed_coupled_system(
    fem_mesh::VolumeMesh{T},
    bem_mesh::BoundaryMesh{T},
    interface_map::ConformingInterfaceMap,
    frequency_hz::T,
    sound_speed::T,
    density::T;
    quadrature_order::Int=2,
    regular_quadrature_order::Union{Nothing,Int}=nothing,
    singular_order::Int=2,
    cache=nothing,
    bem_backend::Union{Nothing,Symbol}=nothing,
    validation_diagnostics::Bool=false,
    retain_interface_radiation::Bool=false,
    symmetry_mode::Symbol=:off,
    bulk_loss_factor::T=zero(T),
    bulk_loss_factor_by_vertex=nothing,
    wall_impedances=NamedTuple[],
    transducers::AbstractVector{ElectrodynamicTransducer{T}}=ElectrodynamicTransducer{T}[],
    transducer_operators=nothing,
    prescribed_bem_normal_velocity=nothing,
    schur_block_columns::Int=32,
    allow_transducer_condensation::Bool=true,
    bem_operators=nothing,
) where {T<:AbstractFloat}
    # `relative_residual` needs the monolithic coupled matrix, which this formulation never
    # forms. `fem_interior_residual` on each solution is the condensed-appropriate check.
    validation_diagnostics && error(
        "FEM static condensation cannot be combined with full-matrix validation diagnostics.",
    )

    fem_stage_started = time_ns()
    resolved_transducer_operators = isnothing(transducer_operators) ?
                                    assemble_transducer_operators(fem_mesh, bem_mesh, transducers) :
                                    transducer_operators
    transducer_fem_vertices = isempty(transducers) ?
                              Int[] :
                              unique(findnz(resolved_transducer_operators.fem_surface)[1])
    retained_fem_vertices = sort(
        unique(vcat(interface_map.fem_vertex_indices, transducer_fem_vertices)),
    )
    # Optimization switches resolve against the cache's backend (unset: on for Metal, the reductions and refined LU on CPU, off elsewhere);
    # `optimization_fallbacks` records every `auto` switch that could not be used, and why.
    # An omitted backend inherits a supplied cache for existing direct callers.
    # An uncached request must retain the backend selected by its driver.
    bem_backend = isnothing(bem_backend) ?
                  (isnothing(cache) ? :cpu : cache.base.bem_backend) : bem_backend
    bem_backend in (:cpu, :metal) ||
        error("Condensed coupled BEM backend must be :cpu or :metal; got $bem_backend.")
    !isnothing(cache) && cache.base.bem_backend != bem_backend &&
        error("Condensed coupled cache backend does not match the requested BEM backend.")
    optimization_fallbacks = String[]
    # The cache is keyed on the full moving-surface set either way; only the Schur block changes.
    transducer_condensation_mode = isempty(transducers) ? :off : _transducer_condensation_mode(bem_backend)
    transducer_condensation = transducer_condensation_mode == :on ||
                              (transducer_condensation_mode == :auto && allow_transducer_condensation)
    if transducer_condensation_mode == :auto && !allow_transducer_condensation
        push!(optimization_fallbacks,
            "transducer condensation not used: the request needs transducer surfaces retained (speaker ROM)")
    end
    gamma_fem_vertices = transducer_condensation ?
                         sort(unique(Int.(interface_map.fem_vertex_indices))) :
                         retained_fem_vertices
    # Without a cache, build one for exactly the order this frequency selected, so the uncached
    # path honours the selection instead of silently falling back to the base order.
    selected_quadrature_order = isnothing(regular_quadrature_order) ? quadrature_order :
                                regular_quadrature_order
    condensed_cache = isnothing(cache) ? prepare_condensed_coupled_cache(
        fem_mesh,
        bem_mesh,
        interface_map;
        quadrature_order=selected_quadrature_order,
        singular_order=singular_order,
        bem_backend=bem_backend,
        symmetry_mode=symmetry_mode,
        retained_fem_vertices=retained_fem_vertices,
        bulk_loss_factor_by_vertex=(
            isnothing(bulk_loss_factor_by_vertex) ?
            fill(bulk_loss_factor, length(fem_mesh.vertices)) :
            bulk_loss_factor_by_vertex
        ),
        wall_impedances=wall_impedances,
    ) : cache
    if !isnothing(cache)
        # Orders are compared as integers. `TriangleRule` defines no `==`, so comparing rules here
        # would fall back to `===` on their vectors and never hold for a freshly built rule. A
        # cache we built above is correct by construction, so only a supplied one is checked.
        condensed_cache.singular_order == singular_order ||
            error("Condensed coupled cache singular order does not match the requested singular order.")
        condensed_cache.base_quadrature_order == quadrature_order ||
            error("Condensed coupled cache base quadrature order does not match the requested quadrature order.")
    end
    bundle = get(condensed_cache.quadrature_bundles, selected_quadrature_order, nothing)
    isnothing(bundle) && error(
        "Condensed coupled cache holds no quadrature bundle for order $selected_quadrature_order; " *
        "it was built for $(sort(collect(keys(condensed_cache.quadrature_bundles)))).",
    )
    # The selected bundle's fields shadow the base cache's, so everything below reads the cache
    # exactly as it did before per-order quadrature existed.
    prepared = merge(condensed_cache.base, bundle)
    prepared.bem_backend in (:cpu, :metal) ||
        error("Condensed coupled cache must be built for the CPU or Metal BEM backend.")
    prepared.symmetry_mode == BeatEngineCore.normalized_symmetry_mode(symmetry_mode) ||
        error("Coupled cache symmetry mode does not match requested symmetry.")
    prepared.retained_fem_vertices == retained_fem_vertices ||
        error("Coupled cache retained FEM vertices do not match the current moving surfaces.")

    assembly_plan, combined_identity = _condensed_bem_assembly_plan(condensed_cache, prepared, T, selected_quadrature_order)
    assembly_plan.fallback_reason === nothing || push!(optimization_fallbacks,
        "combined Metal BEM assembly not used: " * assembly_plan.fallback_reason)

    omega = T(2pi) * frequency_hz
    wavenumber = omega / sound_speed
    fem_system = assemble_fem_dynamic_stiffness(
        prepared.stiffness,
        prepared.mass,
        wavenumber;
        bulk_loss_mass=prepared.bulk_loss_mass,
    )
    wall_admittances = Complex{T}[
        miki_rigid_backed_surface_admittance(
            frequency_hz,
            sound_speed,
            density,
            operator.thickness_m,
            operator.flow_resistivity_pa_s_per_m2,
        )
        for operator in prepared.wall_impedance_operators
    ]
    for (operator, admittance) in zip(prepared.wall_impedance_operators, wall_admittances)
        fem_system -= neumann_scale(density, omega) * admittance .* operator.matrix
    end
    interface_operators = prepared.interface_operators
    transducer_count = length(transducers)
    size(resolved_transducer_operators.fem_surface, 2) == transducer_count ||
        error("FEM transducer operator count does not match the transducer list.")
    size(resolved_transducer_operators.bem_surface, 2) == transducer_count ||
        error("BEM transducer operator count does not match the transducer list.")
    normal_derivative_scale = neumann_scale(density, omega)
    bem_motion_flux = normal_derivative_scale .* Complex{T}.(
        resolved_transducer_operators.bem_normal_velocity
    )
    resolved_prescribed_bem_velocity = isnothing(prescribed_bem_normal_velocity) ?
                                       spzeros(T, length(bem_mesh.faces), 0) :
                                       T.(prescribed_bem_normal_velocity)
    size(resolved_prescribed_bem_velocity, 1) == length(bem_mesh.faces) ||
        error("Prescribed BEM normal velocity must contain one row per BEM face.")
    bem_prescribed_neumann = normal_derivative_scale .* Complex{T}.(resolved_prescribed_bem_velocity)
    prescribed_bem_count = size(bem_prescribed_neumann, 2)
    if T !== Float64 && _fem_float64_enabled(bem_backend)
        fem_system = _fem_system_float64(
            hasproperty(condensed_cache, :fem_float64_store) ? condensed_cache.fem_float64_store : nothing,
            fem_mesh,
            prepared,
            frequency_hz,
            sound_speed,
            density,
        )
    end
    fem_system_s = (time_ns() - fem_stage_started) / 1.0e9

    # Everything the condensation reads is final here and nothing below writes
    # to it, so on Metal it runs on the host while the BEM operators assemble
    # on the GPU. Started before the BEM stage rather than at its own marker
    # below because the overlap is the whole point; `fem_condensation_s` then
    # spans the concurrent region, and `stage_overlap` in the timings says so.
    stage_overlap = _stage_overlap_enabled(prepared.bem_backend)
    fem_solver = _fem_solver_selection(bem_backend)
    # First use forwards an LP64 BLAS into libblastrampoline; do that here, before the condensation
    # can overlap host BLAS work in the BEM stage.
    fem_solver == :mumps && mumps_library()
    # Scalar type of the dense coupled system. Under double assembly every block formed from
    # double-precision FEM quantities (Schur, transducer motion/force, the mechanical block, the
    # condensed right-hand side) stays Float64 up to the dense matrix; only inputs that are
    # single precision to begin with (BEM operators, traces, transducer parameters) are Float32.
    dense_type = _dense_double_assembly(bem_backend) ? Float64 : T
    condensation_options = (
        fem_solver=fem_solver,
        mumps_store=(!isnothing(cache) && hasproperty(condensed_cache, :mumps_store)) ?
                    condensed_cache.mumps_store : nothing,
        schur_block_columns=schur_block_columns,
        motion_surface=transducer_condensation ? resolved_transducer_operators.fem_surface : nothing,
        motion_force=transducer_condensation ? resolved_transducer_operators.fem_force : nothing,
        schur_float64=dense_type === Float64,
    )
    interface_elimination, elimination_required = _interface_elimination_request(bem_backend)
    elimination_map = nothing
    if interface_elimination != :none
        try
            elimination_map = _interface_elimination_map(interface_map, interface_operators, gamma_fem_vertices)
        catch exception
            (exception isa ErrorException && !elimination_required) || rethrow()
            push!(optimization_fallbacks, "interface elimination not used: " * exception.msg)
            interface_elimination = :none
        end
    end
    elimination_split = Dict{Symbol,Float64}()
    # Specialized flux elimination (BLAB_COUPLED_INTERFACE_MASS_SOLVER / _MASS_OVERLAP / _BLOCKS).
    mass_operator = nothing
    mass_in_fem_stage = false
    interface_mass_factorization_s = 0.0
    interface_mass_cached = false
    if interface_elimination == :flux && _interface_mass_specialized(bem_backend)
        blocks_mode = _interface_blocks_mode(bem_backend)
        gamma_labels = if blocks_mode != :off
            _split_timed!(elimination_split, :components) do
                _fem_component_labels(fem_system)[gamma_fem_vertices]
            end
        else
            nothing
        end
        mass_store = hasproperty(condensed_cache, :interface_mass_store) ?
                     condensed_cache.interface_mass_store : nothing
        mass_started = time_ns()
        mass_solver = _interface_mass_solver_selection(bem_backend)
        mass_operator, interface_mass_cached = _interface_mass_operator_or_unreduced(
            mass_store, interface_operators, gamma_fem_vertices, elimination_map.gamma_dof, gamma_labels, mass_solver,
            blocks_mode, elimination_required, optimization_fallbacks,
        )
        if isnothing(mass_operator)
            interface_elimination = :none
            elimination_map = nothing
        end
        interface_mass_factorization_s = (time_ns() - mass_started) / 1.0e9
        elimination_split[:mass_prep] = interface_mass_factorization_s
        mass_in_fem_stage = interface_elimination == :flux && _interface_mass_overlap_enabled(bem_backend)
    end
    fem_task_s = Ref(0.0)
    fem_stage = () -> begin
        fem_task_started = time_ns()
        stage_condensation = _build_condensation(
            fem_system,
            interface_operators,
            gamma_fem_vertices;
            condensation_options...,
        )
        presolve = mass_in_fem_stage ? _flux_mass_presolve(
            mass_operator,
            stage_condensation.schur,
            _gamma_motion_columns(
                stage_condensation, dense_type, normal_derivative_scale, transducer_count, transducer_condensation,
                resolved_transducer_operators, gamma_fem_vertices,
            ),
        ) : nothing
        fem_task_s[] = (time_ns() - fem_task_started) / 1.0e9
        (stage_condensation, presolve)
    end
    condensation_started = time_ns()
    condensation_task = stage_overlap ? Threads.@spawn(fem_stage()) : nothing

    # The FEM task may be running from here until it is fetched below. A failure in between joins it
    # before propagating, so the caller's cleanup never releases a solver the task is still using.
    local prefetched, operators, bem_operator_s, bem_lhs, bem_rhs_operator, bem_interface_block,
          bem_combine_s, bem_motion_block, bem_prescribed_rhs, interface_radiation_replay, bem_matrix_s
    try
        bem_operator_started = time_ns()
        prefetched = nothing
        operators = if isnothing(bem_operators)
            _assemble_condensed_bem_operators(
                bem_mesh, prepared, wavenumber, singular_order; combined_identity=combined_identity,
            )
        else
            # Assembled and combined ahead by the sweep pipeline (`assemble_condensed_bem_operators`);
            # waiting for it here keeps the FEM task above overlapping whatever is still in flight.
            prefetched = bem_operators()
            prefetched.wavenumber == wavenumber || error(
                "Coupled sweep pipeline delivered BEM operators for k=$(prefetched.wavenumber) where " *
                "k=$(wavenumber) was needed.",
            )
            nothing
        end
        bem_operator_s = (time_ns() - bem_operator_started) / 1.0e9

        bem_matrix_started = time_ns()
        bem_lhs, bem_rhs_operator, bem_interface_block = if isnothing(prefetched)
            _combine_condensed_bem_operators!(operators, prepared, wavenumber)
        else
            (prefetched.bem_lhs, prefetched.bem_rhs_operator, prefetched.bem_interface_block)
        end
        # The part of `bem_matrix_s` the sweep pipeline moves onto its producer.
        bem_combine_s = isnothing(prefetched) ? (time_ns() - bem_matrix_started) / 1.0e9 : 0.0
        bem_motion_block = transducer_count == 0 ? nothing : -(bem_rhs_operator * bem_motion_flux)
        bem_prescribed_rhs = prescribed_bem_count == 0 ?
                             zeros(Complex{T}, length(bem_mesh.vertices), 0) :
                             Complex{T}.(bem_rhs_operator * bem_prescribed_neumann)
        # Replay the frozen operating flux without changing the coupled state.
        # Preserve host matrices before accelerator assembly storage is released.
        interface_radiation_replay = retain_interface_radiation ? (
            factorization=lu(Array(bem_lhs)),
            interface_block=Array(bem_interface_block),
        ) : nothing
        bem_matrix_s = (time_ns() - bem_matrix_started) / 1.0e9
    catch
        if !isnothing(condensation_task)
            try
                wait(condensation_task)
            catch
            end
        end
        rethrow()
    end
    stage_overlap || (condensation_started = time_ns())
    condensation, fem_stage_presolve = if isnothing(condensation_task)
        fem_stage()
    else
        # `fetch` wraps a task failure in a TaskFailedException, which would
        # make the error a caller sees depend on whether the stage happened to
        # be overlapped. Rethrow the original instead.
        try
            fetch(condensation_task)
        catch exception
            exception isa TaskFailedException || rethrow()
            rethrow(exception.task.result)
        end
    end
    fem_condensation_s = (time_ns() - condensation_started) / 1.0e9

    block_assembly_started = time_ns()
    fem_count = length(fem_mesh.vertices)
    bem_count = length(bem_mesh.vertices)
    interface_count = length(interface_map.fem_vertex_indices)
    retained_fem_count = length(gamma_fem_vertices)
    gamma_range = 1:retained_fem_count
    # Unknown and row layouts. `gamma_row_range` holds the condensed FEM rows' right-hand side.
    #   :none      unknowns [p_Γ, p_B, q, y]  rows [Γ, BEM, continuity, mech, elec]
    #   :pressure  unknowns [q, p_B, y]       rows [Γ, BEM, mech, elec]  (|Γ| == |I|)
    #   :flux      unknowns [p_B, y]          rows [BEM, mech, elec]
    if interface_elimination == :none
        gamma_row_range = gamma_range
        bem_range = (retained_fem_count + 1):(retained_fem_count + bem_count)
        flux_range = (retained_fem_count + bem_count + 1):(retained_fem_count + bem_count + interface_count)
        acoustic_system_count = retained_fem_count + bem_count + interface_count
    elseif interface_elimination == :pressure
        gamma_row_range = 1:retained_fem_count
        flux_range = 1:interface_count
        bem_range = (interface_count + 1):(interface_count + bem_count)
        acoustic_system_count = interface_count + bem_count
    else
        gamma_row_range = 1:0
        flux_range = 1:0
        bem_range = 1:bem_count
        acoustic_system_count = bem_count
    end
    mechanical_range = transducer_count == 0 ?
                       (1:0) :
                       ((acoustic_system_count + 1):(acoustic_system_count + transducer_count))
    electrical_range = transducer_count == 0 ?
                       (1:0) :
                       (
        (acoustic_system_count + transducer_count + 1):
        (acoustic_system_count + 2 * transducer_count)
    )
    system_count = acoustic_system_count + 2 * transducer_count
    mechanical_impedance = Complex{T}[
        BeatEngineCoupled.mechanical_impedance(transducer, omega, density, sound_speed)
        for transducer in transducers
    ]
    electrical_impedance = Complex{T}[
        BeatEngineCoupled.electrical_impedance(transducer, omega)
        for transducer in transducers
    ]
    force_factor = T[transducer.bl_n_per_a for transducer in transducers]

    coupled = zeros(Complex{dense_type}, system_count, system_count)
    interface_elimination_s = 0.0
    elimination = nothing
    # The Schur complement takes the slot the full FEM block occupies in the monolithic
    # formulation, and the interface coupling is restricted to Γ.
    if interface_elimination == :none
        coupled[gamma_range, gamma_range] = condensation.schur
        coupled[gamma_range, flux_range] =
            -Complex{T}.(Matrix(interface_operators.fem_load[gamma_fem_vertices, :]))
        coupled[flux_range, gamma_range] =
            Complex{T}.(Matrix(interface_operators.fem_trace[:, gamma_fem_vertices]))
        coupled[bem_range, bem_range] = bem_lhs
        coupled[bem_range, flux_range] = bem_interface_block
        coupled[flux_range, bem_range] = -Complex{T}.(Matrix(interface_operators.bem_trace))
    end
    if transducer_count > 0 && transducer_condensation
        if interface_elimination == :none
            coupled[gamma_range, mechanical_range] =
                -normal_derivative_scale .* Complex{dense_type}.(condensation.motion_gamma)
            coupled[mechanical_range, gamma_range] = -Complex{dense_type}.(transpose(condensation.force_gamma))
        end
        # Formed in double precision so the interior's margin survives the subtraction.
        coupled[mechanical_range, mechanical_range] = Complex{dense_type}.(
            ComplexF64.(Matrix(Diagonal(mechanical_impedance))) -
            ComplexF64(normal_derivative_scale) .* condensation.motion_force_correction
        )
    elseif transducer_count > 0
        if interface_elimination == :none
            coupled[gamma_range, mechanical_range] =
                -normal_derivative_scale .* Complex{T}.(
                    Matrix(resolved_transducer_operators.fem_surface[retained_fem_vertices, :])
                )
            coupled[mechanical_range, gamma_range] =
                -Complex{T}.(
                    transpose(
                        Matrix(resolved_transducer_operators.fem_force[retained_fem_vertices, :]),
                    )
                )
        end
        coupled[mechanical_range, mechanical_range] = Matrix(Diagonal(mechanical_impedance))
    end
    if transducer_count > 0
        coupled[bem_range, mechanical_range] = bem_motion_block
        coupled[mechanical_range, bem_range] =
            Complex{T}.(transpose(Matrix(resolved_transducer_operators.bem_force)))
        coupled[mechanical_range, electrical_range] =
            -Matrix(Diagonal(Complex{T}.(force_factor)))
        coupled[electrical_range, mechanical_range] = Matrix(Diagonal(Complex{T}.(force_factor)))
        coupled[electrical_range, electrical_range] = Matrix(Diagonal(electrical_impedance))
    end
    if !isnothing(fem_stage_presolve)
        # Ran inside the FEM stage, concurrently with the BEM operators when overlapped; not part
        # of `interface_elimination_s`.
        for (key, value) in fem_stage_presolve.split
            elimination_split[Symbol("fem_stage_", key)] = value
        end
    end
    if interface_elimination != :none
        elimination_started = time_ns()
        # The same blocks the unmodified layout writes (at the dense scalar type), promoted for the elimination.
        gamma_mech, mech_gamma, bem_columns = _split_timed!(elimination_split, :pack) do
            local_gamma_mech = _gamma_motion_columns(
                condensation, dense_type, normal_derivative_scale, transducer_count, transducer_condensation,
                resolved_transducer_operators, gamma_fem_vertices,
            )
            local_mech_gamma = if transducer_count == 0
                zeros(ComplexF64, 0, retained_fem_count)
            elseif transducer_condensation
                ComplexF64.(-Complex{dense_type}.(transpose(condensation.force_gamma)))
            else
                ComplexF64.(
                    -Complex{T}.(transpose(Matrix(resolved_transducer_operators.fem_force[gamma_fem_vertices, :]))),
                )
            end
            # Γ column j is the BEM pressure unknown it equals.
            (local_gamma_mech, local_mech_gamma, collect(bem_range)[elimination_map.bem_of_gamma])
        end
        _split_timed!(elimination_split, :bem_block_copy) do
            coupled[bem_range, bem_range] = bem_lhs
        end
        if interface_elimination == :pressure
            coupled[gamma_row_range, flux_range] =
                -Complex{T}.(Matrix(interface_operators.fem_load[gamma_fem_vertices, :]))
            # `+=`: two interface vertices from different FEM domains may share a BEM vertex.
            for (column, target) in enumerate(elimination_map.bem_of_gamma)
                @views coupled[gamma_row_range, bem_columns[column]] .+= condensation.schur[:, column]
            end
            coupled[bem_range, flux_range] = bem_interface_block
            transducer_count > 0 && (coupled[gamma_row_range, mechanical_range] = gamma_mech)
            elimination = (
                bem_of_gamma=elimination_map.bem_of_gamma,
                gamma_dof=elimination_map.gamma_dof,
            )
        elseif !isnothing(mass_operator)
            presolve = if isnothing(fem_stage_presolve)
                stage_presolve = _flux_mass_presolve(mass_operator, condensation.schur, gamma_mech)
                merge!(+, elimination_split, stage_presolve.split)
                stage_presolve
            else
                fem_stage_presolve
            end
            interface_block = _split_timed!(() -> ComplexF64.(bem_interface_block), elimination_split, :block_convert)
            _flux_block_products!(
                coupled, bem_range, bem_columns, mass_operator, presolve.schur_blocks, interface_block,
                elimination_split,
            )
            if transducer_count > 0
                motion_coupling = _split_timed!(
                    () -> _host_zgemm(interface_block, presolve.motion_solution), elimination_split, :product,
                )
                _split_timed!(() -> (coupled[bem_range, mechanical_range] .+= motion_coupling), elimination_split, :scatter)
            end
            elimination = (
                bem_of_gamma=elimination_map.bem_of_gamma,
                gamma_dof=elimination_map.gamma_dof,
                mass_operator=mass_operator,
                schur_blocks=presolve.schur_blocks,
                motion_solution=presolve.motion_solution,
                interface_block=interface_block,
            )
        else
            mass_store = hasproperty(condensed_cache, :interface_mass_store) ?
                         condensed_cache.interface_mass_store : nothing
            mass_started = time_ns()
            mass_factorization, interface_mass_cached =
                _interface_mass_factorization(mass_store, interface_operators, gamma_fem_vertices)
            interface_mass_factorization_s = (time_ns() - mass_started) / 1.0e9
            elimination_split[:mass_prep] = interface_mass_factorization_s
            # q = M_Γ⁻¹ (S P p_B + E y - g), in interface-dof order (the columns of fem_load).
            schur_double = _split_timed!(() -> ComplexF64.(condensation.schur), elimination_split, :schur_convert)
            schur_solution, motion_solution = _split_timed!(elimination_split, :mass_solve) do
                (mass_factorization \ schur_double, mass_factorization \ gamma_mech)
            end
            schur_double = nothing
            interface_block = _split_timed!(() -> ComplexF64.(bem_interface_block), elimination_split, :block_convert)
            schur_coupling, motion_coupling = _split_timed!(elimination_split, :product) do
                (_host_zgemm(interface_block, schur_solution), _host_zgemm(interface_block, motion_solution))
            end
            _split_timed!(elimination_split, :scatter) do
                for column in eachindex(bem_columns)
                    @views coupled[bem_range, bem_columns[column]] .+= schur_coupling[:, column]
                end
                transducer_count > 0 && (coupled[bem_range, mechanical_range] .+= motion_coupling)
            end
            schur_coupling = nothing
            elimination = (
                bem_of_gamma=elimination_map.bem_of_gamma,
                gamma_dof=elimination_map.gamma_dof,
                mass_factorization=mass_factorization,
                schur_solution=schur_solution,
                motion_solution=motion_solution,
                interface_block=interface_block,
            )
        end
        if transducer_count > 0
            _split_timed!(elimination_split, :scatter) do
                for column in eachindex(bem_columns)
                    @views coupled[mechanical_range, bem_columns[column]] .+= mech_gamma[:, column]
                end
            end
        end
        interface_elimination_s = (time_ns() - elimination_started) / 1.0e9
    end
    block_assembly_s = (time_ns() - block_assembly_started) / 1.0e9

    coupled_factorization_started = time_ns()
    factorization = dense_type === Float64 && T !== Float64 && _dense_refinement_enabled(bem_backend) ?
                    RefinedDenseLU(coupled) : lu!(coupled)
    coupled_factorization_s = (time_ns() - coupled_factorization_started) / 1.0e9

    return (
        fem_mesh=fem_mesh,
        bem_mesh=bem_mesh,
        interface_map=interface_map,
        interface_operators=interface_operators,
        transducers=transducers,
        transducer_operators=resolved_transducer_operators,
        density=density,
        bulk_loss_factor=maximum(prepared.bulk_loss_factor_by_vertex; init=zero(T)),
        bulk_loss_factor_by_vertex=prepared.bulk_loss_factor_by_vertex,
        wall_admittances=wall_admittances,
        omega=omega,
        wavenumber=wavenumber,
        field_cache=prepared.field_cache,
        coupled=nothing,
        factorization=factorization,
        formulation=:fem_interface_condensed,
        # The order the assembly actually used, so diagnostics report what ran, not what was asked.
        regular_quadrature_order=selected_quadrature_order,
        condensation=condensation,
        fem_range=1:fem_count,
        gamma_range=gamma_range,
        # Callers key caches on this set, so it keeps meaning interface ∪ transducer surfaces.
        retained_fem_vertices=retained_fem_vertices,
        # The vertices `gamma_range` indexes; without the transducer surfaces when they are condensed.
        gamma_fem_vertices=gamma_fem_vertices,
        transducer_condensation=transducer_condensation,
        # :none, :pressure or :flux; see `_interface_elimination_mode`. Under elimination,
        # `gamma_range` keeps its length but no longer indexes the dense unknowns.
        interface_elimination=interface_elimination,
        interface_elimination_data=elimination,
        gamma_row_range=gamma_row_range,
        dense_scalar_type=dense_type,
        # Element type of the FEM dynamic stiffness the condensation read (BLAB_COUPLED_FEM_FLOAT64).
        fem_scalar_type=real(eltype(fem_system)),
        # `auto` optimizations that could not be used for this model, with the reason.
        optimization_fallback_reasons=optimization_fallbacks,
        bem_range=bem_range,
        flux_range=flux_range,
        mechanical_range=mechanical_range,
        electrical_range=electrical_range,
        bem_lhs=nothing,
        bem_factorization=nothing,
        bem_rhs_operator=nothing,
        interface_radiation_replay=interface_radiation_replay,
        prescribed_bem_rhs=bem_prescribed_rhs,
        prescribed_bem_neumann=bem_prescribed_neumann,
        bem_backend=prepared.bem_backend,
        coupled_bem_assembly=assembly_plan.mode,
        coupled_bem_assembly_fallback_reason=assembly_plan.fallback_reason,
        linear_backend=:cpu,
        symmetry_mode=prepared.symmetry_mode,
        cache=condensed_cache,
        owns_cache=isnothing(cache),
        validation_diagnostics=false,
        scalar_type=T,
        full_system_order=fem_count + bem_count + interface_count + 2 * transducer_count,
        solved_system_order=system_count,
        timings=(
            fem_system_s=fem_system_s,
            bem_operator_s=bem_operator_s,
            bem_matrix_s=bem_matrix_s,
            bem_combine_s=bem_combine_s,
            fem_condensation_s=fem_condensation_s,
            fem_task_s=fem_task_s[],
            # True when `fem_condensation_s` and `bem_operator_s` cover the same
            # wall-clock span and must not be added together.
            stage_overlap=stage_overlap,
            block_assembly_s=block_assembly_s,
            # Both inside `block_assembly_s`.
            interface_elimination_s=interface_elimination_s,
            interface_mass_factorization_s=interface_mass_factorization_s,
            interface_mass_cached=interface_mass_cached,
            # Parts of `interface_elimination_s` by sub-stage, reported as `interface_elim_<key>_s`.
            interface_elimination_split=elimination_split,
            coupled_factorization_s=coupled_factorization_s,
            replay_factorization_s=0.0,
        ),
    )
end

function release_condensed_coupled_system!(system)
    _release_condensation!(system.condensation)
    system.owns_cache && release_condensed_coupled_cache!(system.cache)
    return nothing
end

function _solution_from_parts(
    system,
    fem_pressure,
    bem_pressure,
    interface_flux,
    fem_interior_residual;
    diaphragm_velocity=zeros(Complex{system.scalar_type}, length(system.transducers)),
    voice_coil_current=zeros(Complex{system.scalar_type}, length(system.transducers)),
    prescribed_bem_neumann=zeros(Complex{system.scalar_type}, length(system.bem_mesh.faces)),
    fem_rhs_condensation_s=nothing,
    fem_reconstruction_s=nothing,
    solve_split=nothing,
)
    T = system.scalar_type
    bem_neumann = (
        Complex{T}.(system.interface_operators.bem_flux) * interface_flux +
        neumann_scale(system.density, system.omega) .*
        (Complex{T}.(system.transducer_operators.bem_normal_velocity) * diaphragm_velocity) +
        prescribed_bem_neumann
    )

    pressure_jump = (
        system.interface_operators.fem_trace * fem_pressure -
        system.interface_operators.bem_trace * bem_pressure
    )
    pressure_scale = max(
        norm(system.interface_operators.fem_trace * fem_pressure),
        norm(system.interface_operators.bem_trace * bem_pressure),
        eps(T),
    )
    fem_integrated_flux = zero(Complex{T})
    bem_integrated_flux_along_fem_normal = zero(Complex{T})
    interface_dof = Dict(
        vertex => index
        for (index, vertex) in enumerate(system.interface_map.fem_vertex_indices)
    )
    for local_face_index in eachindex(system.interface_map.fem_face_indices)
        fem_face = system.fem_mesh.boundary_faces[system.interface_map.fem_face_indices[local_face_index]]
        fem_flux_average = sum(interface_flux[interface_dof[vertex]] for vertex in fem_face) / T(3)
        bem_face_index = system.interface_map.bem_face_indices[local_face_index]
        fem_integrated_flux +=
            BeatEngineCoupled._triangle_area(system.fem_mesh.vertices, fem_face) * fem_flux_average
        bem_integrated_flux_along_fem_normal += (
            T(system.interface_map.normal_sign[local_face_index]) *
            system.bem_mesh.areas[bem_face_index] *
            bem_neumann[bem_face_index]
        )
    end
    flux_scale = max(abs(fem_integrated_flux), abs(bem_integrated_flux_along_fem_normal), eps(T))
    return (
        fem_pressure=fem_pressure,
        bem_pressure=bem_pressure,
        interface_flux=interface_flux,
        bem_neumann=bem_neumann,
        diaphragm_velocity=diaphragm_velocity,
        voice_coil_current=voice_coil_current,
        relative_residual=nothing,
        fem_interior_residual=fem_interior_residual,
        fem_rhs_condensation_s=fem_rhs_condensation_s,
        fem_reconstruction_s=fem_reconstruction_s,
        solve_split=solve_split,
        pressure_continuity_error=norm(pressure_jump) / pressure_scale,
        flux_conservation_error=abs(fem_integrated_flux - bem_integrated_flux_along_fem_normal) / flux_scale,
        all_bem_replay_error=nothing,
        interface_map=system.interface_map,
        interface_operators=system.interface_operators,
    )
end

"""
    solve_condensed_coupled_excitations(system, excitations; reconstruct_interior=true)

`reconstruct_interior=false` skips the FEM interior back substitution: interior entries of
`fem_pressure` are `NaN` and `fem_interior_residual` is `NaN` (not evaluated). Retained-vertex
pressures, BEM pressures, interface fluxes, diaphragm velocities and currents are unaffected.
Only for callers whose outputs never read interior pressure.
"""
function solve_condensed_coupled_excitations(system, excitations; reconstruct_interior::Bool=true)
    T = system.scalar_type
    requested = collect(excitations)
    isempty(requested) && error("At least one coupled excitation is required.")
    excitation_count = length(requested)
    fem_rhs = zeros(Complex{T}, length(system.fem_mesh.vertices), excitation_count)
    bem_rhs = zeros(Complex{T}, length(system.bem_mesh.vertices), excitation_count)
    prescribed_bem_neumann = zeros(Complex{T}, length(system.bem_mesh.faces), excitation_count)
    electrical_rhs = zeros(Complex{T}, length(system.transducers), excitation_count)
    for (column, excitation) in enumerate(requested)
        kind = Symbol(excitation.kind)
        amplitude = Complex{T}(excitation.amplitude)
        if kind == :normal_velocity
            fem_boundary_tags = hasproperty(excitation, :fem_boundary_tags) ?
                                Int.(excitation.fem_boundary_tags) :
                                [Int(excitation.radiator_tag)]
            fem_boundary_weights = hasproperty(excitation, :fem_boundary_weights) ?
                                   T.(excitation.fem_boundary_weights) :
                                   ones(T, length(fem_boundary_tags))
            length(fem_boundary_tags) == length(fem_boundary_weights) ||
                error("Prescribed FEM boundary tags and weights must have the same length.")
            all(weight -> isfinite(weight) && weight > zero(T), fem_boundary_weights) ||
                error("Prescribed FEM boundary weights must be finite and greater than zero.")
            for (tag, weight) in zip(fem_boundary_tags, fem_boundary_weights)
                fem_rhs[:, column] .+= assemble_prescribed_velocity_load(
                    system.fem_mesh,
                    tag,
                    system.density,
                    system.omega,
                    amplitude * weight,
                )
            end
            bem_source_index = hasproperty(excitation, :bem_source_index) ?
                               Int(excitation.bem_source_index) :
                               0
            if bem_source_index > 0
                bem_source_index <= size(system.prescribed_bem_rhs, 2) ||
                    error("Prescribed BEM source index $bem_source_index is unavailable.")
                bem_rhs[:, column] .= amplitude .* view(system.prescribed_bem_rhs, :, bem_source_index)
                prescribed_bem_neumann[:, column] .=
                    amplitude .* view(system.prescribed_bem_neumann, :, bem_source_index)
            end
            isempty(fem_boundary_tags) && bem_source_index == 0 && error(
                "A prescribed-velocity excitation must own at least one FEM or BEM moving boundary.",
            )
        elseif kind == :voltage
            transducer_index = Int(excitation.transducer_index)
            1 <= transducer_index <= length(system.transducers) ||
                error("Voltage excitation references invalid transducer index $transducer_index.")
            electrical_rhs[transducer_index, column] = amplitude
        else
            error("Unsupported coupled excitation kind: $kind.")
        end
    end

    elimination_mode = hasproperty(system, :interface_elimination) ? system.interface_elimination : :none
    elimination = elimination_mode == :none ? nothing : system.interface_elimination_data
    rhs_type = Complex{system.dense_scalar_type}
    rhs = zeros(rhs_type, size(system.factorization, 1), excitation_count)
    rhs[system.bem_range, :] = bem_rhs
    rhs[system.electrical_range, :] = electrical_rhs
    # Sub-stage split of the solve (seconds), reported as `solve_<key>_s`.
    solve_split = Dict{Symbol,Float64}()
    rhs_condensation_started = time_ns()
    reduced_rhs, interior_rhs = _forward_schur(system.condensation, fem_rhs; result_type=system.dense_scalar_type)
    fem_rhs_condensation_s = (time_ns() - rhs_condensation_started) / 1.0e9
    flux_rhs_solution = nothing
    if elimination_mode == :flux
        # BEM rows gain B_q M_Γ⁻¹ g from substituting q = M_Γ⁻¹ (S P p_B + E y - g).
        flux_rhs_solution = _split_timed!(solve_split, :flux_rhs_mass) do
            hasproperty(elimination, :mass_operator) ?
                _interface_mass_apply(elimination.mass_operator, reduced_rhs) :
                elimination.mass_factorization \ ComplexF64.(reduced_rhs)
        end
        _split_timed!(solve_split, :flux_rhs_product) do
            rhs[system.bem_range, :] .+= elimination.interface_block * flux_rhs_solution
        end
    else
        rhs[system.gamma_row_range, :] = reduced_rhs
    end
    transducer_condensed = hasproperty(system.condensation, :transducer_condensed) &&
                           system.condensation.transducer_condensed
    if transducer_condensed && system.condensation.interior_count > 0
        # Mechanical rows pick up R_Iᵀ A_II⁻¹ f_I = Zᵀ f_I from the eliminated force vertices.
        rhs[system.mechanical_range, :] .+=
            Complex{system.dense_scalar_type}.(transpose(system.condensation.force_solution) * interior_rhs)
    end
    dense_solution = _split_timed!(() -> system.factorization \ rhs, solve_split, :dense_solve)
    solution = Complex{T}.(dense_solution)
    flux_reconstruction_started = time_ns()
    retained_pressure, interface_flux = if elimination_mode == :none
        solution[system.gamma_range, :], solution[system.flux_range, :]
    elseif elimination_mode == :pressure
        solution[system.bem_range, :][elimination.bem_of_gamma, :], solution[system.flux_range, :]
    else
        bem_gamma = ComplexF64.(dense_solution[system.bem_range, :][elimination.bem_of_gamma, :])
        flux = if hasproperty(elimination, :mass_operator)
            # W is block diagonal: q[dofs_k] = W_k p_B[rows_k] - h[dofs_k] (+ V y below).
            block_flux = -flux_rhs_solution
            for (block, schur_block) in zip(elimination.mass_operator.blocks, elimination.schur_blocks)
                @views block_flux[block.dofs, :] .+= schur_block * bem_gamma[block.rows, :]
            end
            block_flux
        else
            elimination.schur_solution * bem_gamma - flux_rhs_solution
        end
        isempty(system.mechanical_range) ||
            (flux .+= elimination.motion_solution * ComplexF64.(dense_solution[system.mechanical_range, :]))
        Complex{T}.(bem_gamma), Complex{T}.(flux)
    end
    solve_split[:flux_reconstruction] = (time_ns() - flux_reconstruction_started) / 1.0e9
    reconstruction_started = time_ns()
    fem_pressure, fem_interior_residual = if reconstruct_interior
        _backward_schur(
            system.condensation,
            interior_rhs,
            retained_pressure;
            motion_velocity=transducer_condensed ? solution[system.mechanical_range, :] : nothing,
            motion_scale=transducer_condensed ? neumann_scale(system.density, system.omega) : nothing,
        )
    else
        skipped = fill(
            Complex{T}(T(NaN), T(NaN)),
            system.condensation.interior_count + system.condensation.retained_count,
            size(retained_pressure, 2),
        )
        skipped[system.condensation.retained_vertices, :] = retained_pressure
        skipped, fill(T(NaN), size(retained_pressure, 2))
    end
    fem_reconstruction_s = (time_ns() - reconstruction_started) / 1.0e9
    parts_started = time_ns()
    solutions = [
        _solution_from_parts(
            system,
            fem_pressure[:, column],
            solution[system.bem_range, column],
            interface_flux[:, column],
            fem_interior_residual[column];
            diaphragm_velocity=solution[system.mechanical_range, column],
            voice_coil_current=solution[system.electrical_range, column],
            prescribed_bem_neumann=prescribed_bem_neumann[:, column],
            fem_rhs_condensation_s=fem_rhs_condensation_s,
            fem_reconstruction_s=fem_reconstruction_s,
            solve_split=solve_split,
        )
        for column in axes(fem_pressure, 2)
    ]
    # Shared by every solution of this call, so filled after they are built.
    solve_split[:solution_parts] = (time_ns() - parts_started) / 1.0e9
    return solutions
end

function solve_condensed_coupled_systems(system, radiator_tags; radiator_velocities=nothing)
    T = system.scalar_type
    tags = Int.(collect(radiator_tags))
    isempty(tags) && error("At least one radiator tag is required.")
    velocities = isnothing(radiator_velocities) ?
                 fill(Complex{T}(1, 0), length(tags)) :
                 Complex{T}.(collect(radiator_velocities))
    length(velocities) == length(tags) ||
        error("Radiator tags and velocities must have the same length.")
    return solve_condensed_coupled_excitations(
        system,
        [
            (kind=:normal_velocity, radiator_tag=tag, transducer_index=0, amplitude=velocity)
            for (tag, velocity) in zip(tags, velocities)
        ],
    )
end

function solve_condensed_coupled_system(system, radiator_tag::Int; radiator_velocity=ComplexF64(1, 0))
    return only(
        solve_condensed_coupled_systems(
            system,
            [radiator_tag];
            radiator_velocities=[radiator_velocity],
        ),
    )
end

end # module
