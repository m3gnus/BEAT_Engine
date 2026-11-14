"""Compare repeated benchmark_worker.py runs: speed with variance, sections, memory, accuracy.

  python scripts/compare_benchmarks.py --reference ref.json \\
      --stage "main CPU=runs/main_cpu_r*.json" --stage "branch Metal=runs/branch_metal_r*.json"

Each --stage is LABEL=GLOB over the runs of one configuration. Stages are
compared in the order given. With --variance (the run-to-run half-range, in %,
established for this machine and workload), a change is called real when it
exceeds twice that, and "close" otherwise: add runs and use the per-section
timings to locate it. Without --variance, stages with 3+ runs are compared by
whether their min-max ranges overlap. Accuracy is the worst across all runs
against --reference: relative L2 over every output, and the worst dB error over
points within 30 dB of each output's peak (deeper nulls are noise in dB).
"""

import argparse
import array
import base64
import glob
import json
import math
import statistics
from pathlib import Path


def complex_values(entry):
    raw, shape = entry
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("invalid output shape")
    floats = array.array("d")
    floats.frombytes(base64.b64decode(raw, validate=True))
    if len(floats) != 2 * math.prod(shape):
        raise ValueError("output length does not match its shape")
    if not all(math.isfinite(value) for value in floats):
        raise ValueError("output contains non-finite values")
    return [complex(floats[i], floats[i + 1]) for i in range(0, len(floats), 2)]


def accuracy(run, ref):
    """{quantity: (relative L2, worst dB within 30 dB of peak)}."""
    groups = {}
    if not ref["outputs"]:
        raise ValueError("reference has no outputs")
    if run["outputs"].keys() != ref["outputs"].keys():
        raise ValueError("output quantities/frequencies do not match the reference")
    for key, entry in ref["outputs"].items():
        actual = run["outputs"][key]
        if actual[1] != entry[1]:
            raise ValueError(f"{key}: output shape does not match the reference")
        groups.setdefault(key.rsplit("@", 1)[0], []).append((complex_values(actual), complex_values(entry)))
    result = {}
    for quantity, pairs in groups.items():
        num = den = worst = 0.0
        for values, reference in pairs:
            num += sum(abs(a - b) ** 2 for a, b in zip(values, reference))
            den += sum(abs(b) ** 2 for b in reference)
            peak = max((abs(b) for b in reference), default=0.0)
            for a, b in zip(values, reference):
                if abs(b) > peak * 10 ** (-30 / 20):
                    error = abs(20 * math.log10(abs(a) / abs(b))) if abs(a) else math.inf
                    worst = max(worst, error)
        relative = math.sqrt(num / den) if den else (math.inf if num else 0.0)
        result[quantity] = (relative, worst)
    return result


def stage_accuracy(runs, reference):
    """Validate every repeat and report each metric's worst value."""
    worst = {}
    for index, run in enumerate(runs, start=1):
        try:
            metrics = accuracy(run, reference)
        except ValueError as exc:
            raise ValueError(f"run {index}: {exc}") from exc
        for quantity, (relative, db) in metrics.items():
            previous = worst.get(quantity, (0.0, 0.0))
            worst[quantity] = (max(previous[0], relative), max(previous[1], db))
    return worst


def sections(runs):
    """Median over runs of each section's sweep total, in first-seen order."""
    keys = []
    for run in runs:
        for row in run["frequencies"]:
            keys.extend(k for k in row["timings"] if k not in keys)
    totals = {
        k: statistics.median(sum(r["timings"].get(k, 0.0) for r in run["frequencies"]) for run in runs) for k in keys
    }
    return {k: v for k, v in totals.items() if v >= 0.01}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append", required=True, help="LABEL=GLOB")
    parser.add_argument("--reference", type=Path, help="run JSON to measure accuracy against")
    parser.add_argument("--variance", type=float, help="established run-to-run half-range, percent")
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text()) if args.reference else None

    stages = []
    for spec in args.stage:
        label, pattern = spec.split("=", 1)
        runs = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(pattern))]
        if not runs:
            raise SystemExit(f"no runs match {pattern!r}")
        stages.append((label, runs))

    first_median = statistics.median(r["wall_s"] for r in stages[0][1])
    for label, runs in stages:
        walls = [r["wall_s"] for r in runs]
        median = statistics.median(walls)
        half_range = (max(walls) - min(walls)) / 2 / median * 100
        memories = [r["memory_mb"] for r in runs]
        memory_kinds = {
            tuple(
                r.get("memory_measurement", {}).get(key)
                for key in (
                    "metric",
                    "scope",
                    "peak_kind",
                    "poll_interval_s",
                    "dedicated_gpu_memory",
                )
            )
            for r in runs
        }
        print(f"{label} ({runs[0]['commit']}, {runs[0]['backend']}, n={len(runs)})")
        print(
            f"  wall    {median:.2f} s  range {min(walls):.2f}-{max(walls):.2f} (±{half_range:.1f}%)"
            f"  {first_median / median:.2f}x vs {stages[0][0]}"
        )
        if len(memory_kinds) != 1:
            print("  memory  unavailable (measurement methods differ across runs)")
        elif all(m["peak_sweep"] is not None and m["before_sweep"] is not None for m in memories):
            peak = statistics.median(m["peak_sweep"] for m in memories)
            growth = statistics.median(m["peak_sweep"] - m["before_sweep"] for m in memories)
            print(f"  memory  peak {peak:.0f} MB, +{growth:.0f} MB over the pre-sweep worker")
            metric, scope, peak_kind, interval, gpu = next(iter(memory_kinds))
            if metric is None:
                print("          measurement method unspecified (legacy report)")
            else:
                print(f"          {metric}; {scope}; {peak_kind}; polling {interval} s; dedicated GPU: {gpu}")
        else:
            print("  memory  unavailable (not measured for every run)")
        print("  sections " + "  ".join(f"{k.removesuffix('_s')} {v:.2f}" for k, v in sections(runs).items()))
        if reference is not None:
            try:
                metrics = stage_accuracy(runs, reference)
            except ValueError as exc:
                parser.error(f"{label}: {exc}")
            print(
                "  accuracy (worst across runs) "
                + "; ".join(f"{q} {rel:.1e} rel, {db:.3f} dB" for q, (rel, db) in metrics.items())
            )
    print("changes:")
    for (label_a, runs_a), (label_b, runs_b) in zip(stages, stages[1:]):
        a, b = [r["wall_s"] for r in runs_a], [r["wall_s"] for r in runs_b]
        change = statistics.median(a) / statistics.median(b) - 1
        size = f"{change * 100:.1f}% faster" if change > 0 else f"{-change / (1 + change) * 100:.1f}% slower"
        if args.variance is not None:
            real = abs(change) * 100 > 2 * args.variance
            verdict = size if real else f"close ({size}, within 2x ±{args.variance}%): add runs, check sections"
        elif min(len(a), len(b)) >= 3:
            overlap = min(max(a), max(b)) >= max(min(a), min(b))
            verdict = f"within variance ({size})" if overlap else size
        else:
            verdict = f"{size}; no verdict without --variance or 3+ runs per stage"
        print(f"  {label_a} -> {label_b}: {verdict}")


if __name__ == "__main__":
    main()
