# Thermoviscous wall losses: implementation notes

The opt-in `parameters.thermoviscous_wall_losses` boundary field accepts `off`
(also the default when omitted) or `thin_boundary_layer`. It is valid only for
unlined, stationary rigid walls in bounded air. Workers advertise `fem_wall_loss_models: ["thin_boundary_layer"]`;
the Python engine client rejects enabled requests before submission unless the
worker advertises that model and `fem_wall_loss_scopes: ["boundary"]`. Off requests remain compatible with older workers.

The pressure-only Wentzell operator follows equation (31) of Berggren,
Bernland and Noreland, *Acoustic boundary layers as boundary conditions*,
Journal of Computational Physics 371 (2018), 633–650,
<https://doi.org/10.1016/j.jcp.2018.06.005>.

For `exp(+i omega t)`, add

```
(-1 + i)/2 * (delta_v * K_wall + (gamma - 1) * delta_t * k^2 * M_wall)
delta_v = sqrt(2 * mu / (rho * omega))
delta_t = sqrt(2 * conductivity / (rho * cp * omega))
```

`K_wall` integrates products of tangential shape-function gradients; `M_wall`
integrates products of shape functions. The negative-time convention conjugates
these coefficients. Both operators use existing pressure degrees of freedom and
affine boundary geometry, with P1/P2 traces matching the supported volume order.
The thermal surface mass remains consistent even when the standalone volume
mass is blended. Wall-patch boundaries use natural zero conormal flux.

The first model version fixes room-temperature air properties at dynamic
viscosity 1.84e-5 Pa s, thermal conductivity 0.0257 W/(m K), specific heat
1005 J/(kg K), and heat-capacity ratio 1.4. Density and sound speed come from
the request. It assumes stationary no-slip isothermal walls, linear acoustics,
and non-overlapping thin layers. It does not model sharp-edge corrections,
nonlinear port losses, mean flow, or overlapping layers in narrow gaps.
No automatic gap-validity claim is made. Untreated walls retain their existing
boundary condition; patch transitions do not resolve detailed edge layers.
The superseded draft region-level field is rejected rather than silently ignored.

Compilation selects only explicitly enabled boundaries. Enabling losses on
Miki-lined walls, component-referenced boundaries, exterior walls, interfaces,
terminations or moving assignments is rejected. Assembly additionally removes
faces on active x/xy symmetry cuts. Unassigned faces are not automatically
interpreted as treated solid walls. Existing bulk loss remains independent;
empirical factors fitted to wall dissipation can double-count that loss.

The geometry operators are prepared once per request and included in standalone,
monolithic, condensed, Float64-reassembled and accelerator-scattered FEM systems.
Float64 reassembly recomputes tangential stiffness from promoted geometry rather
than promoting a rounded Float32 surface stiffness. The off path performs no
surface assembly. Enabling losses still requires a fresh numerical solve.

Result diagnostics record model version, selected boundary IDs, actual treated
area/face count after symmetry filtering, fixed material properties, and both
layer thicknesses at each frequency. IDs describe candidate wall patches;
a patch can contain symmetry faces that are excluded from the reported area.

## Validation

`thermoviscous_tests.jl` checks P1/P2 polynomial integrals, symmetry, passivity,
phasor conjugacy, disabled-path identity, symmetry selection, and complex slit
propagation against the analytical parallel-plate equivalent-fluid solution.
`thermoviscous_driver_tests.jl` exercises contract validation, boundary exclusions,
coupled voltage sweeps, Float64 reconstruction, monolithic/condensed agreement,
disabled/omitted equivalence and standalone interior phasor conjugacy. Both run
in the standalone CPU reference gate. Worker tests cover negotiation, including
proving incompatible requests never reach the worker.

Real-project accuracy qualification and performance measurements remain separate
from these small numerical regressions. Use `docs/Benchmarking.md` before making
speed or overhead claims; do not infer them from an unchanged unknown count.
