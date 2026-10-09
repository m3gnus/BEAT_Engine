# Power-consistent projected DP0 interface flux

The coupled FEM–BEM interface uses the same facewise constant normal derivative
in the interior weak load and the exterior BEM. Its coefficients are indexed by
interface vertices, but they parameterize DP0 face flux: `q_face = Q q`, with
each row of `Q` averaging three vertex coefficients. BEM normal orientation is
applied separately. These coefficients are not a physical P1 Neumann trace.

For P1 FEM pressure tests, the exact weak load is `G = Q' W Q`, where `W` contains
triangle areas. The local matrix is `area/9 * ones(3,3)`. The previous interface
load used the consistent P1/P1 mass `area/12 * (I + ones(3,3))`, while exterior
radiation still used face averages. Those loads conserve total flux but differ
in pressure-weighted work. Other boundary masses and prescribed source loads
retain their existing exact integrals.

`interface_normal_derivative` keeps its array shape, units and vertex indexing,
and now labels the coefficient representation in metadata. The physical DP0
values are carried by `bem_boundary_neumann`. Field evaluation and interface
radiation replay use that same representation and do not interpolate flux onto
rigid faces sharing mouth vertices.

## Admissible interfaces

Conforming geometry alone is insufficient. Some meshes have nodal coefficient
modes whose mean vanishes on every face: a single triangular interface and
three-colorable triangulations are examples. The projected transfer must have
full column rank and an adequate numerical conditioning margin. Every coupled
cache setup validates the matrix actually stored, in Float64, for all backends
and both full and condensed formulations. Empty interfaces remain valid.

With `L = diag(G*ones)` the normalized matrix `N = L^(-1/2) G L^(-1/2)` is
nonnegative, symmetric and has maximum eigenvalue one. A sparse Cholesky check
of `N - threshold*I` rejects singular or insufficiently conditioned transfers.
The dimensionless threshold is `max(sqrt(eps(Float64)), 1024*eps(precision))`.
At Float32 this bounds the normalized transfer condition number numerically
below 8192; `eps*condition` is approximately 0.001 at that bound. This is a
first-order budget for the transfer solve, not an acoustic error guarantee.

The solver now explicitly rejects some geometrically valid interfaces that
previously returned a nonconservative result. The remedy is an admissible
triangulation or a formulation with a matching higher-order flux trace. No
fallback to the old inconsistent load is performed. Refinement can worsen this
transfer margin, so observable mesh convergence must still be demonstrated.
A true face-local P1 Neumann representation in BEM is a broader alternative
that could remove this restricted projected-basis admissibility requirement.

The weak-load area is taken from the matched BEM face, so work conservation
uses exactly the same area on both sides. When matching tolerates tiny coordinate
differences, the FEM weak load uses that matched area rather than its own area.

Flux-conservation diagnostics compare FEM geometric area with BEM geometric
area, preserving visibility of admitted geometric mismatch. They do not prove
weak-work conservation. Reported interface mean velocity and represented area
remain averages over the FEM geometric surface.

Monolithic CPU/Metal Float32 requests factor the host coupled matrix in Float64
and return Float32 physical traces. This avoids coefficient error from bare
Float32 LU in badly scaled electrodynamic fixtures; GPU operator precision
remains unchanged. Condensed systems already use their Float64 refined solve.
