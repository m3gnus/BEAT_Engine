# BEAT compiled-system wire contract

BEAT owns the numerical wire format. Boundary Lab owns project authoring,
migration, mesh preparation, and translation into that format. Independent clients
can construct JSON requests without importing Boundary Lab models, Qt, or NumPy.

## Authoritative artifacts and versions

- [JSON Schema](../src/beat_engine/beat_contract/system-v1.schema.json): request
  v1 at `urn:beat-engine:system-solve:1`; compiled system v1 and v2 at its
  `#/$defs/compiled_system` fragment.
- [Independent example](../src/beat_engine/beat_contract/example-exterior-request.json):
  a prescribed-velocity exterior request. Mesh filenames are illustrative; supply
  matching Gmsh assets before solving.
- [Conformance corpus](../src/beat_engine/beat_contract/conformance.json): shared
  acceptance/rejection cases for Python and Julia, including a coupled interface.
- [Python validator](../src/beat_engine/beat_contract/__init__.py) and
  [Julia validator](../src/beat_engine/julia_local/src/BeatEngineContract.jl):
  structural validation plus identifier/reference and topology-length checks.

The schema uses the [JSON Schema 2020-12 structure and reference conventions](https://json-schema.org/understanding-json-schema/structuring).
The two lightweight validators implement only the subset used in the shipped
schema. They are not general-purpose JSON Schema implementations. JSON numbers
must additionally be finite; NaN and infinity are never valid JSON extensions.

| Version field | Current value | Meaning |
|---|---:|---|
| Request `schema_version` | 1 | Solve-request envelope |
| Compiled system `contract_version` | 1 or 2 | V2 requires explicit support for source-motion parameters; v1 remains the uniform-normal baseline |
| Frequency result `schema_version` | 2 | Existing typed binary array representation |
| Optional `source_model_version` | Producer-defined positive integer | Advisory producer provenance only |

Compiled-system v1 retains its original uniform-normal source semantics. V2
adds opt-in source-motion fields while retaining existing normal-source requests. The
application does not define the compiled-system version independently. Request
readers reject missing, fractional, boolean, or
unsupported versions before coercing data or opening mesh files. Integral JSON
numbers such as `1.0` are accepted as integers, consistent with JSON Schema.

## Request envelope

| Field | Meaning |
|---|---|
| `compiled_system` | Resolved graph described below |
| `frequencies_hz` | Nonempty array of finite positive frequencies; order is retained |
| `excitation_port_ids` | Nonempty, unique ordered selection of graph input ports |
| `outputs` | Array of output requests with unique `id`, `quantity`, `target_ids`, and `options` |
| `solver_options` | JSON object interpreted by the numerical implementation |
| `cancel_path` | Optional local cancellation marker filename used by the worker |

Output quantities define how to interpret their targets: a target may be a graph
entity or an output domain. The structural validator does not assume every target
is a component ID. Output options include observation points and domain slices.
Backend choice, precision, quadrature, symmetry, retained fields, and transducer
reference voltage remain explicit numerical options. Syntactic acceptance of an
option or component kind does not establish backend support.

The engine contract permits repeated frequencies and retains their sequence;
the application may impose stricter sweep requirements. Excitations remain
independent columns/rows of the response basis, in the requested ID order. Channel
mixing, crossover processing, and display normalization belong to the client.
Voltage-port basis magnitude follows `transducer_reference_voltage_v`; clients
must not assume every physical input is normalized to one volt.

## Compiled graph

The exact required and optional fields are specified in the schema. IDs are
nonempty strings, unique within each entity collection. Names are display labels
and are not identifiers. Collection order is preserved by the application adapter.

| Collection | Numerical content |
|---|---|
| `meshes` | ID, filename, purpose (`bem_surface` or `fem_volume`), positive scale to meters, three-component translation in meters |
| `regions` | Bounded/unbounded air, nonempty mesh references, resolved volume groups, positive sound speed and density, loss model |
| `boundaries` | Owning region, resolved surface group, boundary kind and numerical parameters |
| `interfaces` | Bounded/unbounded boundary references and resolved FEM-to-BEM topology |
| `components` | Kind, nonempty boundary references, numerical component parameters |
| `excitation_ports` | Port ID, component reference, and normal-velocity or voltage input kind |

Resolved groups contain a mesh ID, positive integer physical tag, dimension (2
for surfaces, 3 for volumes), and optional name. Group mesh IDs must belong to
their region. Component and port references must resolve. Boundary material,
component-model, region-loss, and solver-option parameter dictionaries are
engine-interpreted extensibility points. Backend support and parameter semantics
remain validated by the existing solve-plan and numerical code.

`assumptions`, `metadata`, and `source_model_version` are optional provenance.
They do not select formulations or define the authoring-project schema. Existing
metadata used for result interpretation, such as area normalization, is preserved.
Producers may put descriptive additions in metadata; numerical features must use
their defined engine fields/options and satisfy backend capability checks.

### Exterior ideal-velocity-source motion

An `ideal_velocity_source` with a `normal_velocity` port prescribes a unit
1 m/s velocity basis. By default, each owned moving boundary has uniform
outward-normal velocity; the optional positive `boundary_motion_weights` map
multiplies that basis by boundary ID. Legacy requests without either newly
reserved `motion_profile` or `motion_axis` field keep their numerical arithmetic
and byte-identical outputs. Default-path allocation is not identical to v1;
numerical compatibility does not promise identical execution or allocation.

For a rigid piston moving along a fixed global axis, set component parameters
`motion_profile: "rigid_translation"` and `motion_axis: [x, y, z]`, and set
`compiled_system.contract_version` to `2`. The axis
must be finite and nonzero and is normalized by the worker. On each tagged
triangle the signed normal velocity is the boundary weight times
`dot(face_normal, normalized_motion_axis)` m/s. A face tangent to the motion
has zero velocity; a face whose normal opposes it has negative velocity. The
same projection weights the pressure integral reported as
`radiation_impedance` (generalized force per 1 m/s), so its unit remains N·s/m.
Each source component carries its own axis, independent of the observation
frame and of other sources. A component may own several boundaries only when
they share that rigid motion. Sources with different axes must be separate
components and ports; the client mixes their independent complex responses.
The worker never infers, snaps, flips, or takes the absolute value of an axis.
Reversing an axis reverses its Neumann drive and pressure response; the
generalized impedance remains unchanged because the force projection also
reverses. The axis uses the global mesh coordinate frame, after mesh placement.
`motion_axis` without the rigid-translation profile is an error.
A v1 ideal-source request carrying either reserved field is now refused by
design, including `motion_profile: "uniform_normal"` without an axis. This ensures
older v1-only workers reject v2 axial requests even when a caller bypasses
capability negotiation and submits raw JSON.

For a physical x/xy symmetry reduction the axis must lie in every symmetry
plane used (X=0 for x; X=0 and Y=0 for xy). A standalone y reduction is not
supported by the current exterior solver. Rigid-ground imaging is not a physical radiator copy and does not
impose that restriction. Only exterior BEM ideal sources support this profile;
interior/coupled ideal sources reject it. The worker must advertise
`exterior_source_profiles: ["uniform_normal", "rigid_translation"]` before a
client submits the rigid profile. Older workers without this capability are
rejected by negotiation, even though their v1 schema would structurally accept
the open parameters object.

CPU operator assembly and Metal direct/operator assembly consume the same
projected Neumann columns. Deployment boundary solves already accept signed
complex `boundary_neumann` traces: pass the resulting face trace in the deployed
mesh's face order, with its time convention and placement preserved. Deployment
does not interpret compiled component axes. `interface_radiated_pressure` is a
coupled-system output and cannot be requested for an exterior-only source;
coupled electrodynamic transducers retain their existing motion-axis contract.

## Coordinates, paths, topology, and complex values

- Geometry is scaled by `scale_to_m` then translated by `translation_m` in the
  global meter-based coordinate frame. Observation points use that same frame.
  Sound speed is m/s, density is kg/mÂ³, and frequency is Hz.
- V1 mesh paths are worker-local filesystem paths. Relative paths resolve against
  the worker's working directory, not the request JSON file or authoring project.
  Boundary Lab resolves project-relative mesh references before submission.
  Absolute worker-local paths avoid ambiguity. Remote asset transport is not
  defined by this contract.
- Interface indices are **zero-based** positions in the corresponding source
  mesh vertex/surface-triangle arrays, before combined-mesh renumbering. Julia
  performs the conversion to its one-based indices. Reordering a mesh without
  rebuilding its topology map invalidates the request.
- `fem_vertex_indices` and `fem_to_bem_vertex_indices` must have equal lengths.
  `fem_face_indices`, `bem_face_indices`, and `normal_sign` must have equal lengths.
  Indices are nonnegative integers; signs are exactly -1 or +1. Coordinate error
  is nonnegative and measured in meters. Actual index bounds, facet geometry,
  correspondence, interface roles, and supported formulations are checked against
  the mesh and physics by the solver.
- `solver_options.phasor_convention` explicitly selects `exp(+i omega t)` or
  legacy `exp(-i omega t)`. Omitted options retain the legacy convention.
  Workers advertise `phasor_conventions`; clients must negotiate support and
  verify the convention in result diagnostics. Boundary Lab requests positive time. Pressure, current,
  displacement/velocity, and impedance must retain complex values; SPL is not a
  substitute for the response basis.

## Result compatibility and transport

Existing frequency results contain `freq_hz`, ordered `excitation_port_ids`,
`quantities`, and `diagnostics`. Each quantity has an ID, quantity name, unit,
optional target, axis names, numeric array, and metadata. An `excitation` axis
must match the declared excitation count. Result domains define observation
coordinates outside the per-frequency payload.

Result v2 arrays use base64 bytes, numeric `dtype`, explicit `shape`, `order: C`,
and `byte_order: little`. Complex scalars store adjacent real and imaginary
components in the declared complex dtype. Byte count must match shape times
dtype size. Boundary Lab retains result-v1 decimal real/imag decoding for old
workers and historical data; new results continue to use v2. This milestone
does not change result encoding or create a new result archive format.

The existing worker command wraps a request filename and operation. Events remain
`ready`, `status`, `result`, `completed`, `cancelled`, and `failed`. Field-evaluation
operations retain their separate binary-array protocol. The
[worker protocol](BEAT%20Worker%20Protocol.md) defines version/capability negotiation,
event lifecycle, and field-cache lifetime independently of request v1.

## Evolution and validation boundary

Structural records reject unknown fields so misspellings and accidental authoring
fields do not silently reach the solver. New structural fields or changed meanings
require an explicitly supported version and migration strategy. Open metadata and
parameter/option dictionaries permit extensions without changing the surrounding
structure, but do not guarantee that a particular worker understands a new feature.

Python validates before serialization and before reconstructing application
objects. Julia validates at `solve_request` before dispatch, mesh loading, or matrix
assembly. Both use the same schema and conformance corpus. Validation of geometry,
physical feasibility, and available backend implementations is deliberately kept
in the existing numerical/solve-plan layers. Schema validation does not open files
or claim that a structurally valid request can be solved on every backend.

`system_contract.py` is now a Boundary Lab adapter: it maps a fixed list of fields
instead of calling `asdict` on the compiled object. Adding an application dataclass
field therefore cannot silently change the wire format. The `beat_contract`
directory, its schema/examples, and the Julia validator can move with BEAT when
the engine repository is extracted.

## Coupled CUDA assembly options

`solver_options.coupled_bem_assembly` accepts `auto` (default), `combined`, or
`operators`. `coupled_bem_image_fusion` defaults to true;
`coupled_bem_max_registers` defaults to 0, with explicit caps from 32 through 255.
These options affect the coupled CUDA numerical path, independently of the
exterior-only `burton_miller_assembly` option. Diagnostics retain the individual
operators. Unsupported modes/backends fail before frequency assembly. See
[Coupled CUDA Assembly](Coupled%20CUDA%20Assembly.md) for applicability and cache semantics.
`burton_miller_assembly` applies to CUDA and Metal exterior solves; on Metal,
`direct_system` selects the fused assembler described in
[Metal Backend](Metal%20Backend.md), and `coupled_bem_assembly` resolves to
`operators`.
