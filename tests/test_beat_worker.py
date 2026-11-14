"""Subprocess transport tests independent of Julia, NumPy, and application models."""

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from beat_engine.worker import WorkerPool, WorkerProcess


@pytest.fixture
def worker_script(tmp_path):
    script = tmp_path / "worker.py"
    script.write_text(
        """
import json
import os
import sys

console_window = 0
if os.name == "nt":
    import ctypes
    console_window = ctypes.windll.kernel32.GetConsoleWindow()

print(json.dumps({"type": "ready"}), flush=True)
for line in sys.stdin:
    submission = json.loads(line)
    operation = submission["operation"]
    if operation == "crash":
        print("synthetic worker crash", file=sys.stderr, flush=True)
        sys.exit(7)
    if operation == "hang":
        print(json.dumps({"type": "status", "message": "waiting"}), flush=True)
        for unused in sys.stdin:
            pass
        continue
    if operation in {"cancelled", "failed"}:
        print(json.dumps({"type": operation}), flush=True)
        continue
    with open(submission["request"], encoding="utf-8") as stream:
        payload = json.load(stream)
    print("worker progress", flush=True)
    print(json.dumps({"type": "result", "payload": payload, "operation": operation,
                      "pid": os.getpid(), "threads": os.environ["JULIA_NUM_THREADS"],
                      "sdk": os.environ.get("TEST_SDK"), "console_window": console_window}), flush=True)
    print(json.dumps({"type": "completed"}), flush=True)
""",
        encoding="utf-8",
    )
    return script


def options(script, **overrides):
    return dict(julia_executable=sys.executable, solver_script=script, julia_threads=2, julia_project=None) | overrides


def result(worker, request, operation="solve"):
    events = list(worker.submit(request, operation=operation))
    assert events[-1]["type"] == "completed"
    return next(event for event in events if event["type"] == "result")


def test_worker_streams_opaque_requests_and_reuses_process(worker_script, tmp_path):
    request = tmp_path / "request.json"
    payload = {"excitation_ids": ["left", "right"], "pressure": {"real": [1, 2], "imag": [-3, 4]}}
    request.write_text(json.dumps(payload))
    environment = dict(os.environ, TEST_SDK="first")
    worker = WorkerProcess(**options(worker_script, environment=environment))
    environment["TEST_SDK"] = "mutated"
    try:
        first = result(worker, request)
        field = result(worker, request, "bem_field")
        assert first["payload"] == field["payload"] == payload
        assert first["pid"] == field["pid"]
        assert first["console_window"] == field["console_window"] == 0
        assert first["threads"] == "2"
        assert first["sdk"] == "first"
        assert field["operation"] == "bem_field"
        assert first["_transport"]["julia_stdout_bytes"] > 0
    finally:
        worker.terminate()


@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
def test_terminal_event_releases_submission_lock(worker_script, tmp_path, terminal):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        assert list(worker.submit(request, operation=terminal)) == [{"type": terminal}]
        assert result(worker, request)["payload"] == {}
    finally:
        worker.terminate()


def test_worker_can_restart_after_process_failure(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        first = result(worker, request)
        with pytest.raises(RuntimeError, match="exited with code 7"):
            list(worker.submit(request, operation="crash"))
        assert result(worker, request)["pid"] != first["pid"]
    finally:
        worker.terminate()


@pytest.fixture
def never_ready_script(tmp_path):
    script = tmp_path / "never_ready.py"
    script.write_text("import time\nwhile True: time.sleep(1)\n", encoding="utf-8")
    return script


@pytest.mark.parametrize("entry", ["ensure_started", "submit"])
def test_terminate_interrupts_never_ready_startup(never_ready_script, entry):
    worker = WorkerProcess(**options(never_ready_script))
    request = never_ready_script.parent / "request.json"
    request.write_text("{}")
    finished = threading.Event()
    outcome = []

    def start_worker():
        try:
            if entry == "ensure_started":
                worker.ensure_started()
            else:
                worker.submit(request)
        except Exception as exc:
            outcome.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=start_worker)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while worker._process is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker._process is not None
        process = worker._process
        worker.terminate()
        assert finished.wait(5)
        assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError), outcome
        assert "startup terminated" in str(outcome[0])
        assert process.poll() is not None
        assert worker._process is None
    finally:
        worker.terminate()
        thread.join(timeout=5)


@pytest.mark.parametrize("entry", ["ensure_started", "submit"])
def test_never_ready_startup_times_out(never_ready_script, entry):
    worker = WorkerProcess(**options(never_ready_script, startup_timeout_s=0.1))
    request = never_ready_script.parent / "request.json"
    request.write_text("{}")
    try:
        with pytest.raises(RuntimeError, match="startup timed out"):
            if entry == "ensure_started":
                worker.ensure_started()
            else:
                worker.submit(request)
        assert worker._process is None
    finally:
        worker.terminate()


def test_closing_unstarted_stream_discards_its_unread_events(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        stream = worker.submit(request)
        first_process = worker._process
        stream.close()
        assert first_process.poll() is not None
        assert result(worker, request)["pid"] != first_process.pid
    finally:
        worker.terminate()


def test_dropped_unstarted_stream_does_not_keep_worker_busy(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        stream = worker.submit(request)
        first_process = worker._process
        del stream
        gc.collect()
        assert first_process.poll() is not None
        assert result(worker, request)["pid"] != first_process.pid
    finally:
        worker.terminate()


def test_early_close_after_nonterminal_event_discards_process(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        stream = worker.submit(request)
        assert next(stream)["type"] == "status"
        first_process = worker._process
        stream.close()
        assert first_process.poll() is not None
        assert result(worker, request)["pid"] != first_process.pid
    finally:
        worker.terminate()


def test_terminal_event_releases_ownership_without_extra_next(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    try:
        stream = worker.submit(request, operation="cancelled")
        assert next(stream) == {"type": "cancelled"}
        assert result(worker, request)["payload"] == {}
        assert list(stream) == []
    finally:
        worker.terminate()


def test_concurrent_submission_waits_for_stream_owner(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    entered = threading.Event()
    finished = threading.Event()
    outcome = []
    thread = None
    try:
        first = worker.submit(request)
        first_process = worker._process

        def submit_second():
            entered.set()
            try:
                outcome.append(result(worker, request))
            except Exception as exc:
                outcome.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=submit_second)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.1)
        first.close()
        assert finished.wait(5)
        assert len(outcome) == 1 and isinstance(outcome[0], dict), outcome
        assert outcome[0]["pid"] != first_process.pid
    finally:
        if thread is not None:
            thread.join(timeout=5)
        worker.terminate()


def test_termination_invalidates_stream_and_unblocks_waiting_submission(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    entered = threading.Event()
    finished = threading.Event()
    outcome = []
    thread = None
    try:
        first = worker.submit(request)
        first_process = worker._process

        def submit_second():
            entered.set()
            try:
                outcome.append(result(worker, request))
            except Exception as exc:
                outcome.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=submit_second)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.1)
        worker.terminate()
        assert first_process.poll() is not None
        with pytest.raises(RuntimeError, match="terminated"):
            next(first)
        assert finished.wait(5)
        assert len(outcome) == 1 and isinstance(outcome[0], dict), outcome
        assert outcome[0]["pid"] != first_process.pid
    finally:
        if thread is not None:
            thread.join(timeout=5)
        worker.terminate()


def test_termination_interrupts_blocked_event_read(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    entered = threading.Event()
    finished = threading.Event()
    outcome = []
    thread = None
    try:
        stream = worker.submit(request, operation="hang")
        assert next(stream)["message"] == "waiting"

        def read_next():
            entered.set()
            try:
                outcome.append(next(stream))
            except Exception as exc:
                outcome.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=read_next)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.1)
        worker.terminate()
        assert finished.wait(5)
        assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError), outcome
        assert "terminated" in str(outcome[0])
        assert result(worker, request)["type"] == "result"
    finally:
        if thread is not None:
            thread.join(timeout=5)
        worker.terminate()


def test_close_interrupts_reader_in_another_thread(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    worker = WorkerProcess(**options(worker_script))
    entered = threading.Event()
    finished = threading.Event()
    outcome = []
    thread = None
    try:
        stream = worker.submit(request, operation="hang")
        assert next(stream)["message"] == "waiting"

        def read_next():
            entered.set()
            try:
                outcome.append(next(stream))
            except Exception as exc:
                outcome.append(exc)
            finally:
                finished.set()

        thread = threading.Thread(target=read_next)
        thread.start()
        assert entered.wait(2)
        assert not finished.wait(0.1)
        stream.close()
        assert finished.wait(5)
        assert len(outcome) == 1 and isinstance(outcome[0], RuntimeError), outcome
        assert "terminated" in str(outcome[0])
        assert result(worker, request)["type"] == "result"
    finally:
        if thread is not None:
            thread.join(timeout=5)
        worker.terminate()


def test_pool_keys_include_runtime_environment_and_shutdown_releases_workers(worker_script, tmp_path):
    request = tmp_path / "request.json"
    request.write_text("{}")
    pool = WorkerPool()
    env = dict(os.environ, TEST_SDK="first")
    first = pool.get_worker(**options(worker_script, environment=env))
    try:
        assert pool.get_worker(**options(worker_script, environment=dict(env))) is first
        changed = pool.get_worker(**options(worker_script, environment=env | {"TEST_SDK": "second"}))
        assert changed is not first
        assert pool.get_worker(**options(worker_script, julia_threads=3, environment=env)) is not first
        assert pool.get_worker(**options(worker_script, environment=env, startup_timeout_s=30)) is not first
        assert result(first, request)["sdk"] == "first"
        assert result(changed, request)["sdk"] == "second"
        processes = [first._process, changed._process]
        pool.shutdown()
        assert all(process.poll() is not None for process in processes)
        assert pool.get_worker(**options(worker_script, environment=env)) is not first
    finally:
        pool.shutdown()


def test_transport_loads_without_any_application_or_third_party_imports():
    script = Path(__file__).resolve().parents[1] / "src/beat_engine/worker.py"
    code = """
import importlib.abc
import importlib.util
import sys

class StandardLibraryOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # Python 3.11's stdlib copy module probes for Jython support.
        # Keep that unavailable via ImportError, as on ordinary CPython.
        if fullname == 'org':
            raise ModuleNotFoundError(fullname, name=fullname)
        if fullname.split('.')[0] not in sys.stdlib_module_names:
            raise AssertionError(f'Non-standard-library dependency: {fullname}')

sys.meta_path.insert(0, StandardLibraryOnly())
spec = importlib.util.spec_from_file_location('standalone_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.WorkerPool
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(script)], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stderr
