# Copyright 2026 The HRX Authors
# Licensed under the MIT License.
# SPDX-License-Identifier: MIT
"""Bounded, best-effort wheel cache measurements for the job summary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def read_json(path):
    if path.stat().st_size > 65536:
        raise ValueError("Metadata exceeds reporting budget")
    return json.loads(path.read_text())


def native_metrics(path):
    # BEP actionSummary.actionsExecuted includes cache hits: report runners separately.
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index >= 100000 or len(line) > 1048576:
                raise ValueError("BEP exceeds reporting budget")
            metrics = json.loads(line).get("buildMetrics")
            if metrics:
                action = metrics.get("actionSummary", {})
                runners = {item["name"]: item.get("count", 0)
                           for item in action.get("runnerCount", [])}
                return dict(cache_hit_processes=sum(int(count) for name, count in runners.items() if "cache hit" in name),
                            executed_processes=sum(int(count) for name, count in runners.items() if "cache hit" not in name and name not in {"internal", "total"}),
                            wall_ms=metrics.get("timingMetrics", {}).get("wallTimeInMs"),
                            actions_executed_including_cache=action.get("actionsExecuted"),
                            runners=runners)
    return "unavailable"


def snapshot_size(snapshot):
    apparent = allocated = count = 0
    for directory, subdirectories, files in os.walk(snapshot, followlinks=False):
        for name in subdirectories + files:
            count += 1
            if count > 100000:
                raise ValueError("Snapshot exceeds reporting budget")
            stat = (Path(directory) / name).lstat()
            apparent += stat.st_size
            allocated += stat.st_blocks * 512
    return dict(apparent_bytes=apparent, allocated_bytes=allocated)


def collect(snapshot, wheel_dir, diagnostics, events, environ):
    result = {"source_commit": environ.get("GITHUB_SHA", "unavailable")}
    for name in ("RESTORE_OUTCOME", "MATCHED_KEY", "ENABLED", "READY", "ATTEMPTED_KEY", "SAVE_OUTCOME"):
        result[name.lower()] = environ.get(name, "unavailable")[:256]
    for name, callback in {
        "provenance": lambda: {key: value for key, value in read_json(snapshot / "manifest.json").items()
                               if key in {"source_commit", "hrx_pin", "bazel_version", "sdk_identity", "compiler_fingerprint", "compiler_contract", "complete"}},
        "snapshot_size": lambda: snapshot_size(snapshot),
        "admission": lambda: read_json(diagnostics / "admit.json"),
        "preparation": lambda: read_json(diagnostics / "prepare.json"),
        "native": lambda: native_metrics(events),
    }.items():
        try:
            result[name] = callback()
        except (OSError, ValueError, KeyError, TypeError) as error:
            result[name] = "unavailable: " + type(error).__name__
    wheels = []
    for wheel in sorted(wheel_dir.glob("*.whl"))[:4]:
        try:
            digest = hashlib.sha256()
            with wheel.open("rb") as stream:
                for block in iter(lambda: stream.read(1048576), b""):
                    digest.update(block)
            wheels.append(dict(name=wheel.name, sha256=digest.hexdigest(), bytes=wheel.stat().st_size))
        except OSError:
            pass
    result["wheels"] = wheels
    durations = {}
    for stage in ("restore", "build", "validation", "save"):
        try:
            value = float((diagnostics / (stage + ".seconds")).read_text())
            durations[stage] = max(0, value)
        except (OSError, ValueError):
            if stage == "save":
                try:
                    durations[stage] = max(0, time.time() - float((diagnostics / "save.started").read_text()))
                except (OSError, ValueError):
                    pass
    result["stage_seconds"] = durations
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = collect(args.snapshot, Path("build/wheel/wheelhouse"), args.diagnostics,
                         Path("build/wheel/metrics/native-build.events.json"), os.environ)
        report = json.dumps(result, indent=2, sort_keys=True).replace("`", "'")[:12000]
        print(report)
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as stream:
                stream.write("\nWheel cache measurements\n```json\n" + report + "\n```\n")
    except Exception as error:
        print(f"Cache measurements unavailable ({type(error).__name__})")


if __name__ == "__main__":
    main()
