# Large panels

The regular rule has six points per triangle and the singular rule is a fixed
order. Both assume the kernel is smooth across a panel, and at high frequency it
is not: its phase moves by up to `k * h` across a panel with longest edge `h`.
Past about one wavelength per panel the rule no longer resolves it. Blocks on two
far-apart panels are off by several hundred percent at `k * h = 16`, whatever the
aspect ratio. Field evaluation sums over the same six points and has the same
problem.

## The rule

A panel with `k * h` above 6 gets its own quadrature order,

```text
n = ceil(k * h / 2) + 1
```

capped at 24 (`k * h` about 46). The rule is a collapsed Gauss rule with `n` lines
of constant `u`, each thinned in proportion to its length, which keeps the
accuracy of the full `n` by `n` tensor rule on oscillating kernels with about 60%
of its points. Each panel decides from its own size:

- a regular pair uses the test panel's rule on the test side and the trial
  panel's rule on the trial side, so a large panel next to a small one pays for
  one fine rule, not two;
- a touching pair uses a Duffy order of `n + 1`, capped at 16, from the larger
  of its panels, and never below the request's `singular_order`;
- field evaluation uses each panel's rule for its sources.

`BLAB_BEAT_LARGE_PANEL_KH` overrides the limit. `0` turns this off, which is how
to reproduce the fixed-rule numbers.

## How it is applied

The result is a correction added after the existing passes,

```text
operator += I(fine, fine) - I(base, base)
```

so the regular, singular and image kernels are untouched. A frequency with no
panel above the limit has no plan and runs the same code as before, bit for bit,
on every backend. At a frequency with a plan, a pair of two small panels is left
alone.

| Stage | CPU | Metal |
|---|---|---|
| Regular pairs, fused | the SIMD kernels, called with each class of panels' rule | `_metal_large_panel_blocks_kernel!`, into the pair blocks before the gathers |
| Regular pairs, four operators | the SIMD kernels | the CPU pass on the host |
| Touching pairs | the Duffy kernels, with a second order's rules | the same CPU pass on the host |
| Field | a per-frequency source list | the same list on the device |

The subtracted term is the same kernel with negated weights. Image-touching pairs
take the form the CPU assembly already uses: the regular correction reaches them,
then `(Duffy fine - regular fine) - (Duffy base - regular base)` finishes them.

## What it covers

CPU and Metal exterior solves, in compiled and source requests, including retained
field evaluation of pressure. These stay on the fixed rules:

- CUDA and ROCm, so they will disagree with CPU and Metal on large panels until
  they are ported. The correction's Metal kernel is the model to follow.
- The BEM half of coupled FEM-BEM solves, which assembles through its own cache.
- Deploy solves: `deploy_solver.jl` does not receive a large-panel plan.
- Source requests with `near_correction_enabled`: a near correction replaces the
  regular rule pair by pair, and a second correction on top would count those
  pairs twice.
- Pressure gradients from retained fields.

## Diagnostics

Every exterior, coupled and Deploy result reports, per frequency:

| Field | Meaning |
|---|---|
| `large_panel_correction` | `applied`; `not_needed` (no panel above the limit); `disabled` (limit set to `0`); `unsupported_backend` (CUDA, ROCm, the BEM half of a coupled solve); `near_correction` (near corrections are on, which includes every Deploy solve) |
| `large_panel_count` | panels above the limit, corrected or not |
| `large_panel_max_order` | the largest rule order used, `0` when the correction did not run |
| `large_panel_max_kh` | the mesh's largest `k * h` at this frequency |

When panels exceed the limit and the correction does not run, the request emits
one warning, at the first such frequency: a `status` event with
`level: "warning"`, or a line on stderr for a compiled request outside event
mode.

## Close panels

This rule is sized for phase, not for distance. Two panels that nearly touch
without sharing a vertex are still under-integrated, as they always were: the
near-singular `1/r` needs a separate treatment. Measured on 16 random pairs per
gap at `k * h = 16` against partitioned panels, worst block error, median:

| Gap / h | Base rule | Large-panel rule |
|---|---|---|
| 0.3 | 54% | 0.0005% |
| 0.1 | 134% | 0.2% |
| 0.03 | 230% | 3.7% |
| 0.01 | 353% | 6.7% |

The rule was better on all 64 pairs. Below a gap of about 1e-3 of the panel size a
pair of quadrature points can line up, and the sum then depends on where they
fall: over 6,000 random shifts the largest single-layer block was 5 to 9 times the
median under this rule, and 55 to 96 times under the base rule.

## Choosing the numbers

Measured on the R-OSSE fixture from the study that found this (7,348 triangles,
longest edge 52.6 mm, 680 panels above the limit at 16 kHz, `k * h` up to 15.4),
Float32, normalised RMS over 0 to 60 degrees, horizontal / vertical / diagonal:

| Change | Field moves, dB |
|---|---|
| `n = ceil(k*h/2)` against order 12 | 0.12 / 0.04 / 0.09 |
| `n = ceil(k*h/2) + 1` (the rule) against order 12 | 0.03 / 0.01 / 0.01 |
| Duffy order 8 against order 12 | 0.01 / 0.01 / 0.01 |
| Limit 4 against limit 6 | 0.18 / 0.17 / 0.19 |

The limit stays at 6. Lowering it to 4 changes results on meshes that are not badly
off, for 0.2 dB at the top of the band, and takes in far more panels: 3,763 rather
than 680 on this mesh. Panels just under the limit keep an error of up to about 3%
in a far pair's blocks.

## Gates

`tests/large_panel_tests.jl` compares every block with partitioned panels: each
panel cut into 1,024 parts, integrated with the base rule and projected back onto
the original basis, which shares no code with the rule under test. It also holds
fused against four-operator assembly under every symmetry, reduced against
explicit full meshes, touching pairs against order 16, and field sources against
the same partitioned reference. `tests/metal_large_panel_tests.jl` holds Metal to
the CPU on hardware.
