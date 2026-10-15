"""Compare one seed with original randomization and added physical changes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from .conditions import digest, sample
from .manifest import atomic_json


def compare(task, seed, output, timeout=1800):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {"task": task, "seed": seed, "cases": [],
               "scope": "Independent adapter; official_random preserves the original physical parameters"}
    for protocol in ("official_random", "physical_random"):
        condition = sample(task, seed, protocol)
        episode = {"seed": seed, "episode_index": 0, "condition": condition,
                   "condition_sha256": digest(condition), "expert_success": False}
        request_path = output / f"{protocol}.request.json"
        result_path = output / f"{protocol}.result.json"
        log_path = output / f"{protocol}.log"
        # Do not mistake a stale result for evidence from this diagnostic run.
        if result_path.exists():
            raise FileExistsError(f"Diagnostic already exists: {result_path}; use a new output directory")
        atomic_json(request_path, {"mode": "expert", "episode": episode, "instruction_type": "unseen"})
        with log_path.open("w") as log:
            child = subprocess.run([sys.executable, "-m", "robotwin_phys.worker",
                                    str(request_path), str(result_path)],
                                   stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "error", "error": "Worker did not produce a result"}
        if child.returncode and result.get("status") != "error":
            result = {"status": "error", "error": f"Worker exited {child.returncode}",
                      "worker_result": result}
        case = {"protocol": protocol, "returncode": child.returncode,
                "status": result["status"], "reason": result.get("reason"),
                "details": result.get("details"), "planning_failure": result.get("planning_failure"),
                "error": result.get("error"), "result_path": str(result_path), "log_path": str(log_path)}
        summary["cases"].append(case)
        atomic_json(output / "summary.json", summary)
        print("EXPERT_DIAGNOSIS_CASE " + json.dumps(case), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--seed", type=int, default=100054)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    summary = compare(args.task, args.seed, args.output, args.timeout)
    # A rejected candidate is an expected outcome; runtime errors require review.
    failed = any(case["status"] not in ("verified", "rejected") for case in summary["cases"])
    print("EXPERT_DIAGNOSIS_ERROR" if failed else "EXPERT_DIAGNOSIS_COMPLETE", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
