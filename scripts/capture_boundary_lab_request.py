"""Capture the compiled-system request a Boundary Lab project sends to BEAT.

Run inside a Boundary Lab environment (it imports `blab`) with this package
importable, e.g. from a Boundary Lab checkout:

  PYTHONPATH=<beat_engine checkout>/src .venv/bin/python \\
      <beat_engine checkout>/scripts/capture_boundary_lab_request.py \\
      examples/Multi_region_SAWMOD/Multi_region_SAWMOD.blab.json --out sawmod.json

The request is what `blab project solve` would submit (outputs, excitations,
symmetry, solver options), with the frequencies replaced by --frequencies. Mesh
paths in it are absolute, so it stays tied to that Boundary Lab checkout.

--exterior-mesh captures an exterior-only system instead, seeded from one
surface mesh and its radiator tag the way the GUI seeds one for an exterior
project, e.g. --exterior-mesh <beat checkout>/src/beat_engine/julia_local/test_meshes/sample.msh
"""

import argparse
import json
import tempfile
from pathlib import Path

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.system_contract import system_solve_request_to_dict

DEFAULT_FREQUENCIES = [20.0 * 1000.0 ** (i / 11) for i in range(12)]


def seeded_exterior_project(mesh: Path, tag: int, scale: float, symmetry: str, directory: Path) -> Path:
    """Write the exterior-only project the GUI seeds from a mesh and a radiator tag."""
    from blab.config import RadiatorConfig
    from blab.mesh_inventory import InventoryEntry, inspect_system_meshes
    from blab.physical_model import physical_system_to_dict
    from blab.project.migration import seed_exterior_system

    inventory = inspect_system_meshes(
        (InventoryEntry(name=mesh.stem, source_file=str(mesh.resolve()), scale_factor=scale),)
    )
    system, channels = seed_exterior_system(
        inventory,
        (RadiatorConfig(name=f"{mesh.stem}:radiator", tag=tag, mesh=mesh.stem),),
    )
    path = directory / f"{mesh.stem}.blab.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "symmetry": symmetry,
                "physical_system": physical_system_to_dict(system),
                "component_channel_by_id": channels,
            }
        )
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("project", type=Path, nargs="?", help=".blab.json project")
    what.add_argument("--exterior-mesh", type=Path, help="surface mesh for a seeded exterior-only system")
    parser.add_argument("--tag", type=int, default=2, help="--exterior-mesh radiator physical tag")
    parser.add_argument("--scale", type=float, default=0.001, help="--exterior-mesh scale to metres")
    parser.add_argument("--symmetry", default="off", choices=("off", "x", "xy"), help="--exterior-mesh symmetry")
    parser.add_argument("--frequencies", type=float, nargs="+", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as scratch:
        path = args.project or seeded_exterior_project(
            args.exterior_mesh,
            args.tag,
            args.scale,
            args.symmetry,
            Path(scratch),
        )
        capture(path, args.frequencies, args.out)


def capture(path: Path, frequencies, out: Path):
    project = load_headless_project(path)
    spec = HeadlessSolveSpec(frequencies_hz=tuple(frequencies), raw={"schema_version": 1})
    prepared = prepare_headless_solve(project, spec, backend_id="beat_cpu")
    request = system_solve_request_to_dict(prepared.request)
    out.write_text(json.dumps(request))
    print(
        f"{out}: {prepared.solve_kind}, symmetry {project.symmetry}, "
        f"{len(request['excitation_port_ids'])} excitations, {len(request['frequencies_hz'])} frequencies"
    )


if __name__ == "__main__":
    main()
