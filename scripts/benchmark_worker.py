"""Time one BEAT sweep through the real worker and keep what is needed to compare it.

One invocation = one configuration: a worker is started, a two-frequency warm-up
request absorbs most JIT compilation, then --repeats timed sweeps run in that same
warm worker. Both entry points rebuild their per-request caches for every request.
Sweep 1 matches a fresh worker's sweep, including one-off costs the short warm-up
does not trigger; later sweeps are steady state and can be faster. Compare sweep 1
with sweep 1, or repeats with repeats. The result JSON records wall
time, every per-frequency section timing the worker reports, worker memory, the
effective diagnostics, and the complex outputs (for accuracy against a reference).
See docs/Benchmarking.md for the procedure; compare runs with compare_benchmarks.py.

  python scripts/benchmark_worker.py --request system.json --backend metal --out run.json
  python scripts/benchmark_worker.py --mesh box.msh --backend cpu --out run.json

--request takes a compiled-system request (the path Boundary Lab system solves use);
--mesh builds a source request for the exterior radiator driver. The engine that
runs is whichever `beat_engine` is importable, so benchmark another commit with
PYTHONPATH=<checkout>/src.
"""

import argparse
import array
import base64
import ctypes
import json
import math
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

from beat_engine import EngineWorker, engine_paths

DEFAULT_FREQUENCIES = [20.0 * 1000.0 ** (i / 11) for i in range(12)]  # 12 log-spaced, 20 Hz - 20 kHz
MEMORY_POLL_INTERVAL_S = 0.02


def windows_working_set_reader(pid):
    """Read current resident host bytes; neither lifetime peak nor GPU VRAM."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_uint32]
    kernel.K32GetProcessMemoryInfo.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int

    def read():
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not kernel.K32GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return None
            return counters.WorkingSetSize
        finally:
            kernel.CloseHandle(handle)

    return read


class PeakMemory:
    """Peak memory of one process, polled from a thread.

    macOS: phys_footprint, which includes Metal buffers in unified memory (RSS
    does not). Linux: VmRSS. Windows: current process working set (host RAM,
    not dedicated GPU memory). Peaks are sampled within the sweep, not lifetime.
    """

    def __init__(self, pid: int, interval_s: float = MEMORY_POLL_INTERVAL_S):
        self.pid, self.interval_s, self.peak = pid, interval_s, None
        self._read = self._reader()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _reader(self):
        if sys.platform == "win32":
            return windows_working_set_reader(self.pid)
        if sys.platform == "darwin":
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

            class RUsageV4(ctypes.Structure):
                _fields_ = [("uuid", ctypes.c_uint8 * 16), ("vals", ctypes.c_uint64 * 36)]

            buf = RUsageV4()
            return lambda: buf.vals[7] if libc.proc_pid_rusage(self.pid, 4, ctypes.byref(buf)) == 0 else None
        status = Path(f"/proc/{self.pid}/status")
        if status.exists():

            def read():
                for line in status.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
                return None

            return read
        return lambda: None

    def current_mb(self) -> float | None:
        value = self._read()
        return None if value is None else value / 2**20

    @property
    def peak_mb(self) -> float | None:
        return None if self.peak is None else self.peak / 2**20

    def _sample(self):
        value = self._read()
        if value is not None:
            self.peak = value if self.peak is None else max(self.peak, value)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def source_request(mesh: Path, backend: str, frequencies, scale: float, tag: int):
    return {
        "config": {
            "mesh_file": str(mesh),
            "scale_factor": scale,
            "tag_throat": tag,
            "symmetry": "off",
            "step_size": 5.0,
            "distance": 2.0,
            "quadrature_order": 4,
            "singular_order": 4,
        },
        "frequencies_hz": frequencies,
        "beat_engine_backend": backend,
    }


def system_request(path: Path, backend: str, precision: str, frequencies):
    request = json.loads(path.read_text())
    request["frequencies_hz"] = frequencies
    request.setdefault("solver_options", {}).update({"bem_backend": backend, "precision": precision})
    return request


def outputs_of(result, *, requested_frequency=None, frequency_index=0) -> dict:
    """Complex outputs keyed by quantity, output ID, sweep index and frequency."""
    actual_frequency = float(result["freq_hz"])
    freq = actual_frequency if requested_frequency is None else float(requested_frequency)
    # Workers may report a Float32 frequency. Preserve the requested Float64
    # identity while checking that the returned row belongs to this sweep slot.
    if (
        not math.isfinite(freq)
        or not math.isfinite(actual_frequency)
        or not math.isclose(actual_frequency, freq, rel_tol=2e-6, abs_tol=1e-5)
    ):
        raise ValueError("Returned frequency does not match the requested sweep slot")
    found = {}
    for q in result.get("quantities", []):  # compiled-system results
        values = q.get("values")
        if not isinstance(values, dict) or "content_base64" not in values:
            raise ValueError(f"Invalid benchmark output values: {q.get('id')}")
        name = f"{q['quantity']}:{q['id']}"
        if name in found:
            raise ValueError(f"Duplicate benchmark output: {name}")
        found[name] = values
    for key in ("horizontal_pressure", "vertical_pressure"):  # source-request results
        if isinstance(result.get(key), dict):
            found[key] = result[key]
    packed = {}
    for name, wire in found.items():
        if "content_base64" in wire:
            raw = base64.b64decode(wire["content_base64"])
            if wire["dtype"] == "complex64":
                raw = array.array("d", array.array("f", raw)).tobytes()
            shape = wire["shape"]
        else:  # {"real": [[...]], "imag": [[...]]}
            rows_re, rows_im = wire["real"], wire["imag"]
            floats = array.array("d")
            for row_re, row_im in zip(rows_re, rows_im):
                for re, im in zip(row_re, row_im):
                    floats.extend((re, im))
            raw = floats.tobytes()
            shape = [len(rows_re), len(rows_re[0]) if rows_re else 0]
        packed[f"{name}@{frequency_index}:{freq!r}"] = [base64.b64encode(raw).decode(), shape]
    return packed


def submit(worker, request, path: Path):
    path.write_text(json.dumps(request))
    rows, started = [], time.perf_counter()
    for event in worker.submit(path):
        if event["type"] == "result":
            rows.append(event["result"])
        elif event["type"] == "failed":
            raise SystemExit(f"FAILED {path.name}: {event.get('error')}")
    return time.perf_counter() - started, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--request", type=Path, help="compiled-system request JSON")
    what.add_argument("--mesh", type=Path, help="mesh for a source (exterior radiator) request")
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda", "rocm", "metal"))
    parser.add_argument("--precision", default="float32", help="compiled-system requests only")
    parser.add_argument("--frequencies", type=float, nargs="+", default=DEFAULT_FREQUENCIES)
    parser.add_argument("--scale", type=float, default=0.001, help="--mesh scale factor")
    parser.add_argument("--tag", type=int, default=2, help="--mesh radiator physical tag")
    parser.add_argument("--threads", default="4")
    parser.add_argument("--label", default="", help="free text stored with the run")
    parser.add_argument(
        "--advertise-metal", action="store_true", help="for workers that run Metal but predate advertising it"
    )
    parser.add_argument("--repeats", type=int, default=1, help="timed sweeps in the one warm worker")
    parser.add_argument(
        "--out", type=Path, required=True, help="run JSON; with --repeats N > 1, one file per sweep as <out>_r<i>.json"
    )
    args = parser.parse_args()

    try:
        paths = engine_paths(args.backend)
    except ValueError:  # packages that predate engine_paths("metal")
        cpu = engine_paths("cpu")
        paths = type(cpu)(**{**cpu.__dict__, "project": Path(cpu.root) / f"julia_{args.backend}"})
    if args.request:
        script = paths.system_solver
        build = lambda f: system_request(args.request, args.backend, args.precision, f)  # noqa: E731
    else:
        script = paths.source_solver
        build = lambda f: source_request(args.mesh.resolve(), args.backend, f, args.scale, args.tag)  # noqa: E731

    root = Path(paths.root)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    scratch = args.out.with_suffix(".requests")
    scratch.mkdir(parents=True, exist_ok=True)
    worker = EngineWorker(
        julia_executable="julia", solver_script=script, julia_threads=args.threads, julia_project=paths.project
    )
    try:
        started = time.perf_counter()
        worker.ensure_started()
        startup_s = time.perf_counter() - started
        if args.advertise_metal:
            worker._worker_info.setdefault("backends", {}).setdefault(
                "metal", {"available": True, "reason": "", "phasor_conventions": ["exp(-i omega t)"]}
            )
        freqs = list(args.frequencies)
        warmup_s, _ = submit(
            worker, build([freqs[len(freqs) // 4], freqs[3 * len(freqs) // 4]]), scratch / "warmup.json"
        )
        sweeps = []
        for repeat in range(args.repeats):
            memory = PeakMemory(worker._process.pid)
            before_mb = memory.current_mb()
            with memory:
                wall_s, rows = submit(worker, build(freqs), scratch / f"sweep_{repeat + 1}.json")
            sweeps.append((wall_s, rows, before_mb, memory.peak_mb))
    finally:
        worker.terminate()

    for repeat, (wall_s, rows, before_mb, peak_mb) in enumerate(sweeps, start=1):
        out = args.out if args.repeats == 1 else args.out.with_name(f"{args.out.stem}_r{repeat}.json")
        write_run(out, args, commit, startup_s, warmup_s, repeat, wall_s, rows, before_mb, peak_mb)


def write_run(out, args, commit, startup_s, warmup_s, repeat, wall_s, rows, before_mb, peak_mb):
    frequencies, outputs = [], {}
    for index, (result, requested_frequency) in enumerate(zip(rows, args.frequencies, strict=True)):
        diagnostics = result.get("diagnostics") or {}
        timings = result.get("timings") or diagnostics.get("timings") or {}
        frequencies.append(
            {
                "freq_hz": float(result["freq_hz"]),
                "requested_freq_hz": float(requested_frequency),
                "timings": {k: v for k, v in timings.items() if isinstance(v, (int, float))},
                "diagnostics": {k: v for k, v in diagnostics.items() if isinstance(v, (int, float, str, bool))},
            }
        )
        outputs.update(outputs_of(result, requested_frequency=requested_frequency, frequency_index=index))
    record = {
        "label": args.label,
        "commit": commit,
        "backend": args.backend,
        "precision": args.precision,
        "case": str(args.request or args.mesh),
        "threads": args.threads,
        "host": platform.machine(),
        "startup_s": startup_s,
        "warmup_s": warmup_s,
        "repeat_in_worker": repeat,
        "wall_s": wall_s,
        "memory_mb": {"before_sweep": before_mb, "peak_sweep": peak_mb},
        "memory_measurement": {
            "metric": {"win32": "working_set", "linux": "resident_set", "darwin": "phys_footprint"}.get(
                sys.platform, "unavailable"
            ),
            "scope": "worker_process",
            "peak_kind": "sampled_within_sweep",
            "poll_interval_s": MEMORY_POLL_INTERVAL_S,
            "dedicated_gpu_memory": "not_measured",
        },
        "frequencies": frequencies,
        "outputs": outputs,
        "output_key_version": 2,
    }
    out.write_text(json.dumps(record))
    totals = {}
    for row in frequencies:
        for key, value in row["timings"].items():
            totals[key] = totals.get(key, 0.0) + value
    sections = "  ".join(f"{k}={v:.2f}" for k, v in sorted(totals.items()) if v >= 0.01)
    peak = "unavailable" if peak_mb is None else f"{peak_mb:.0f} MB"
    print(f"{commit} {args.backend} r{repeat}: wall {wall_s:.2f} s  peak {peak}  {sections}")


if __name__ == "__main__":
    main()
