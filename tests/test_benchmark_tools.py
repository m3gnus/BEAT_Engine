import argparse
import array
import base64
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_script("benchmark_worker")
compare = load_script("compare_benchmarks")


@pytest.mark.parametrize("failure", [None, "open", "query"])
def test_windows_reader_uses_current_bytes_and_closes_handles(monkeypatch, failure):
    closed = []
    handle = 2**40 + 123  # Catch accidental truncation of a 64-bit HANDLE.

    def query(process, pointer, size):
        assert process == handle
        pointer._obj.PeakWorkingSetSize = 64 * 2**20
        pointer._obj.WorkingSetSize = 8 * 2**20
        return failure != "query"

    kernel = SimpleNamespace(
        OpenProcess=lambda *args: 0 if failure == "open" else handle,
        K32GetProcessMemoryInfo=query,
        CloseHandle=lambda process: closed.append(process),
    )
    monkeypatch.setattr(benchmark.ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    value = benchmark.windows_working_set_reader(1234)()
    assert value == (8 * 2**20 if failure is None else None)
    assert closed == ([] if failure == "open" else [handle])


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows working-set API")
def test_windows_memory_tracks_touched_allocation():
    memory = benchmark.PeakMemory(os.getpid())
    before = memory.current_mb()
    assert before is not None and before > 0
    with memory:
        allocation = bytearray(32 * 2**20)
        allocation[::4096] = b"x" * (len(allocation) // 4096)
        during = memory.current_mb()
    assert during > before + 16
    assert memory.peak_mb > before + 16
    assert allocation[0] == ord("x")


def wire(values, shape=None):
    floats = array.array("d", (part for value in values for part in (complex(value).real, complex(value).imag)))
    return [base64.b64encode(floats.tobytes()).decode(), shape if shape is not None else [len(values)]]


def run(values, shape=None):
    return {"outputs": {"pressure@100.000": wire(values, shape)}}


def test_windows_report_preserves_unavailable_memory(tmp_path, monkeypatch, capsys):
    monkeypatch.delattr(os, "uname", raising=False)
    monkeypatch.setattr(benchmark.PeakMemory, "_reader", lambda self: lambda: None)
    with benchmark.PeakMemory(os.getpid()) as memory:
        assert memory.current_mb() is None
    assert memory.peak_mb is None
    args = argparse.Namespace(
        label="test", backend="cpu", precision="float32", request=None, mesh="test.msh", threads="2", frequencies=[]
    )
    out = tmp_path / "run.json"
    benchmark.write_run(out, args, "test", 0, 0, 1, 1, [], None, memory.peak_mb)
    record = json.loads(out.read_text())
    assert record["memory_mb"] == {"before_sweep": None, "peak_sweep": None}
    assert record["host"] == benchmark.platform.machine()
    assert "peak unavailable" in capsys.readouterr().out


def benchmark_result(frequency, values=(1, 2)):
    return {
        "freq_hz": frequency,
        "quantities": [
            {
                "id": name,
                "quantity": "exterior_pressure",
                "values": {
                    "dtype": "complex128",
                    "shape": [1],
                    "content_base64": wire([value])[0],
                },
            }
            for name, value in zip(("probe@near", "probe@far"), values)
        ],
    }


def test_benchmark_preserves_same_quantity_outputs_and_close_frequencies():
    outputs = {}
    for index, frequency in enumerate((100.0001, 100.0002, 100.0001)):
        outputs.update(benchmark.outputs_of(benchmark_result(frequency), frequency_index=index))
    assert len(outputs) == 6
    metrics = compare.accuracy({"outputs": outputs}, {"outputs": outputs})
    assert set(metrics) == {"exterior_pressure:probe@near", "exterior_pressure:probe@far"}
    changed = dict(outputs)
    key = next(key for key in outputs if "probe@far" in key)
    changed[key] = wire([4])
    assert compare.accuracy({"outputs": changed}, {"outputs": outputs})["exterior_pressure:probe@far"][0] > 0


def test_benchmark_canonicalizes_float32_frequency_without_rounding_identity():
    frequency = 123.456789
    reported = array.array("f", [frequency])[0]
    actual = benchmark.outputs_of(benchmark_result(reported), requested_frequency=frequency)
    expected = benchmark.outputs_of(benchmark_result(frequency), requested_frequency=frequency)
    assert actual == expected
    with pytest.raises(ValueError, match="frequency"):
        benchmark.outputs_of(benchmark_result(124), requested_frequency=frequency)


def test_benchmark_rejects_duplicate_output_identity():
    result = benchmark_result(100)
    result["quantities"].append(result["quantities"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        benchmark.outputs_of(result)


def test_benchmark_rejects_invalid_output_instead_of_dropping_it():
    result = benchmark_result(100)
    result["quantities"][1]["values"] = None
    with pytest.raises(ValueError, match="Invalid benchmark output"):
        benchmark.outputs_of(result)


def test_benchmark_rejects_missing_sweep_rows(tmp_path):
    args = argparse.Namespace(frequencies=[100, 200])
    with pytest.raises(ValueError, match="zip"):
        benchmark.write_run(tmp_path / "run.json", args, "test", 0, 0, 1, 1, [benchmark_result(100)], None, None)
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("actual", [run([1]), run([1], [2]), run([1, 2], [1, 2]), {"outputs": {}}])
def test_accuracy_rejects_missing_or_truncated_outputs(actual):
    with pytest.raises(ValueError, match="shape|length|quantities"):
        compare.accuracy(actual, run([1, 2]))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_accuracy_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([value]), run([1]))
    with pytest.raises(ValueError, match="non-finite"):
        compare.accuracy(run([1]), run([value]))


def test_accuracy_handles_zero_signals():
    assert compare.accuracy(run([0]), run([1]))["pressure"] == (1.0, math.inf)
    assert compare.accuracy(run([1]), run([0]))["pressure"][0] == math.inf
    assert compare.accuracy(run([0]), run([0]))["pressure"] == (0.0, 0.0)


def test_accuracy_measures_complex_phase():
    relative, db = compare.accuracy(run([1j]), run([1]))["pressure"]
    assert relative == pytest.approx(math.sqrt(2))
    assert db == 0


def test_stage_checks_later_repeats():
    ref = run([1])
    relative, db = compare.stage_accuracy([ref, run([2])], ref)["pressure"]
    assert relative == 1
    assert db == pytest.approx(20 * math.log10(2))
    with pytest.raises(ValueError, match="run 2"):
        compare.stage_accuracy([ref, run([])], ref)


def test_cli_compares_all_repeats_with_unavailable_memory(tmp_path, monkeypatch, capsys):
    for index, value in enumerate([1, 2]):
        record = {
            **run([value]),
            "wall_s": 1.0,
            "commit": "test",
            "backend": "cpu",
            "memory_mb": {"before_sweep": None, "peak_sweep": None},
            "frequencies": [],
        }
        (tmp_path / f"run{index}.json").write_text(json.dumps(record))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    compare.main()
    output = capsys.readouterr().out
    assert "memory  unavailable" in output
    assert "pressure 1.0e+00 rel, 6.021 dB" in output


def test_cli_rejects_invalid_later_repeat(tmp_path, monkeypatch):
    reference = {
        **run([1]),
        "wall_s": 1.0,
        "commit": "test",
        "backend": "cpu",
        "memory_mb": {"before_sweep": None, "peak_sweep": None},
        "frequencies": [],
    }
    (tmp_path / "run0.json").write_text(json.dumps(reference))
    (tmp_path / "run1.json").write_text(json.dumps({**reference, "outputs": {}}))
    monkeypatch.setattr(
        sys, "argv", ["compare", "--reference", str(tmp_path / "run0.json"), "--stage", f"cpu={tmp_path / 'run*.json'}"]
    )
    with pytest.raises(SystemExit) as exc:
        compare.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("kind", ["working_set", "mixed", "legacy"])
def test_cli_labels_memory_and_does_not_mix_measurements(tmp_path, monkeypatch, capsys, kind):
    for index in range(2):
        record = {
            "wall_s": 1.0,
            "commit": "test",
            "backend": "rocm",
            "frequencies": [],
            "memory_mb": {"before_sweep": 100.0, "peak_sweep": 120.0},
        }
        if kind != "legacy":
            record["memory_measurement"] = {
                "metric": "resident_set" if kind == "mixed" and index else "working_set",
                "scope": "worker_process",
                "peak_kind": "sampled_within_sweep",
                "poll_interval_s": 0.02,
                "dedicated_gpu_memory": "not_measured",
            }
        (tmp_path / f"run{index}.json").write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["compare", "--stage", f"rocm={tmp_path / 'run*.json'}"])
    compare.main()
    output = capsys.readouterr().out
    if kind == "mixed":
        assert "measurement methods differ" in output
        assert "peak 120" not in output
    else:
        assert "peak 120 MB" in output
        if kind == "legacy":
            assert "unspecified (legacy report)" in output
        else:
            assert "working_set; worker_process; sampled_within_sweep" in output
            assert "dedicated GPU: not_measured" in output
