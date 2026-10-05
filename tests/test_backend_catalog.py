import json
import subprocess
import sys
import tomllib
from dataclasses import FrozenInstanceError

import pytest

from beat_engine import BackendInfo, backend_catalog, backend_info, engine_paths


def test_catalog_is_immutable_and_resolves_packaged_assets():
    catalog = backend_catalog()
    assert isinstance(catalog, tuple)
    assert {info.backend_id for info in catalog} == {"cpu", "cuda", "rocm", "metal"}
    for info in catalog:
        assert isinstance(info, BackendInfo)
        assert backend_info(info.backend_id) is info
        paths = engine_paths(info.backend_id)
        assert paths.project.name == info.project_directory
        assert (paths.project / "Project.toml").is_file()
        with pytest.raises(FrozenInstanceError):
            info.label = "changed"


@pytest.mark.parametrize("backend", [info.backend_id for info in backend_catalog()])
def test_every_backend_project_declares_its_bundle(backend):
    paths = engine_paths(backend)
    name = f"BeatEngine{backend.capitalize()}Bundle"
    bundle = paths.root / "julia_engine" / name
    project = tomllib.loads((paths.project / "Project.toml").read_text(encoding="utf-8"))
    manifest = tomllib.loads((paths.project / "Manifest.toml").read_text(encoding="utf-8"))
    bundle_project = tomllib.loads((bundle / "Project.toml").read_text(encoding="utf-8"))

    assert project["deps"][name] == bundle_project["uuid"]
    (entry,) = manifest["deps"][name]
    assert entry["uuid"] == bundle_project["uuid"]
    assert (paths.project / entry["path"]).resolve() == bundle.resolve()
    assert (bundle / "src" / f"{name}.jl").is_file()


def test_unknown_backend_is_not_redirected():
    for lookup in (backend_info, engine_paths):
        with pytest.raises(ValueError, match="Unsupported BEAT backend: typo"):
            lookup("typo")


def test_catalog_and_cli_do_not_launch_julia():
    code = """
import subprocess, sys

def no_process(*args, **kwargs):
    raise AssertionError('Backend listing must not start a process')
subprocess.Popen = no_process
from beat_engine import backend_catalog
assert all(info.backend_id for info in backend_catalog())
from beat_engine.__main__ import main
sys.argv = ['beat-engine', 'backends']
main()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert [item["backend_id"] for item in json.loads(result.stdout)] == [i.backend_id for i in backend_catalog()]


def test_cli_backend_choices_follow_catalog(monkeypatch, capsys):
    from beat_engine import __main__ as cli
    from beat_engine import backends

    future = BackendInfo("future", "Future backend", "julia_future", ("linux",))
    monkeypatch.setattr(backends, "_BACKENDS", (*backend_catalog(), future))
    monkeypatch.setattr(sys, "argv", ["beat-engine", "paths", "--backend", "future"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["project"].endswith("julia_future")
