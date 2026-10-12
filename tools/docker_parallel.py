#!/usr/bin/env python3
"""Use one independent simulator container per GPU for task-level parallelism.

Only Python's standard library runs on the host. Simulator imports, expert
rollouts, manifest validation and GPU operations all run inside the image.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid

ROOT = Path(__file__).resolve().parents[1]


def registry():
    # This pinned upstream registry is a flat task: integer mapping.
    text = (ROOT / "vendor/RoboTwin/task_config/_eval_step_limit.yml").read_text()
    names = re.findall(r"^([a-z][a-z0-9_]+):\s*[0-9]+\s*$", text, re.MULTILINE)
    if len(names) != 50 or len(set(names)) != 50:
        raise RuntimeError("Unexpected pinned task registry; review parser before scheduling")
    return sorted(names)


def gpu_inventory():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return dict(tuple(part.strip() for part in row.split(",")) for row in rows.splitlines() if row.strip())


def select_gpus(requested, available):
    values = [item for value in requested for item in value.split(",")]
    if values == ["auto"]:
        values = list(available)
    if not values or any(not re.fullmatch(r"[0-9]+|GPU-[a-fA-F0-9-]+", gpu) for gpu in values):
        raise ValueError("Use --gpus auto, --gpus 0, --gpus 0 1, or GPU UUIDs")
    selected = [available.get(value, value) for value in values]
    if len(set(selected)) != len(values) or not set(selected).issubset(available.values()):
        raise ValueError("GPU selection contains unavailable or duplicate physical GPUs")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate"))
    parser.add_argument("--gpus", nargs="+", default=["auto"], help="auto (all available), one index, several indices, or UUIDs; commas accepted")
    parser.add_argument("--tasks", nargs="+", help="freeze: defaults to all 50 tasks")
    parser.add_argument("--protocol", default="physical_random", choices=("clean", "official_random", "physical_random", "physical_only"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--output", help="freeze: manifest directory beneath manifests/")
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--base-port", type=int, default=8765)
    args = parser.parse_args()
    try:
        args.gpus = select_gpus(args.gpus, gpu_inventory())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(f"Cannot select CUDA GPUs: {error}")
    if not 1024 <= args.base_port <= 65535 - len(args.gpus):
        parser.error("Invalid live port range")
    tasks = args.tasks or registry()
    if len(set(tasks)) != len(tasks) or set(tasks) - set(registry()):
        parser.error("Unknown or duplicate tasks")
    target = Path(args.output or f"manifests/{args.protocol}")
    if target.is_absolute() or ".." in target.parts or target.parts[0] != "manifests":
        parser.error("Use a relative output directory under manifests/")
    if args.episodes < 1 or args.max_candidates < args.episodes:
        parser.error("Invalid episode/candidate count")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    folder = ROOT / "reports/docker-parallel" / run_id
    folder.mkdir(parents=True)
    stop = threading.Event()
    running = {}
    lock = threading.Lock()
    results = []

    def worker(slot, gpu):
        jobs = tasks[slot::len(args.gpus)] if args.command == "freeze" else ["validate"]
        report_dir = folder / f"gpu-{gpu}"
        report_dir.mkdir()
        env = {**os.environ, "ROBOTWIN_GPU": gpu, "ROBOTWIN_REPORTS_DIR": str(report_dir),
               "ROBOTWIN_CACHE_DIR": str(ROOT / ".container/cache" / f"gpu-{gpu}")}
        for task in jobs:
            if stop.is_set():
                return
            name = f"robotwin-{run_id}-worker-{slot}"
            env["ROBOTWIN_CONTAINER_NAME"] = name
            base = ["bash", str(ROOT / "tools/docker.sh")]
            if args.command == "validate":
                command = base + ["validate"]
            else:
                manifest = str(target / f"{task}.json")
                if (ROOT / manifest).exists():
                    # Validate existing pools inside the image, including runtime
                    # provenance; do not silently accept a pool with other settings.
                    code = (
                        "import sys; from robotwin_phys.manifest import load; "
                        "from robotwin_phys.provenance import fingerprint; m=load(sys.argv[1]); "
                        "assert m['provenance']==fingerprint(); "
                        "assert m['task']==sys.argv[5] and len(m['episodes'])==int(sys.argv[3]); "
                        "assert (m['protocol'],m['requested_episodes'],m['start_seed'],m['only'],m['without'])"
                        "==(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),[],[])"
                    )
                    command = base + ["run", "python", "-c", code, manifest, args.protocol, str(args.episodes), str(args.seed), task]
                else:
                    command = base + ["run", "robotwin-phys", "freeze", "--task", task,
                        "--protocol", args.protocol, "--episodes", str(args.episodes), "--seed", str(args.seed),
                        "--output", manifest, "--max-candidates", str(args.max_candidates), "--resume",
                        "--port", str(args.base_port + slot)]
                    for flag in ("live", "video"):
                        if getattr(args, flag):
                            command.append(f"--{flag}")
            print(f"GPU {gpu}: {task}; live port {args.base_port + slot}; log {report_dir / (task + '.log')}", flush=True)
            with (report_dir / f"{task}.log").open("w") as log:
                with lock:
                    if stop.is_set():
                        return
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    running[name] = child
                status = child.wait()
            with lock:
                running.pop(name, None)
                results.append({"gpu": gpu, "task": task, "exit_code": status, "log": str(report_dir / f"{task}.log")})
            if status:
                stop.set()
                return

    pool = ThreadPoolExecutor(max_workers=len(args.gpus))
    futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(args.gpus)]
    try:
        for future in futures:
            future.result()
    except (KeyboardInterrupt, Exception):
        stop.set()
        with lock:
            names = list(running)
        for name in names:
            subprocess.run(["docker", "stop", "--time", "10", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    finally:
        pool.shutdown(wait=True)
        expected = len(tasks) if args.command == "freeze" else len(args.gpus)
        passed = len(results) == expected and all(item["exit_code"] == 0 for item in results)
        (folder / "summary.json").write_text(json.dumps({"status": "passed" if passed else "failed",
            "command": args.command, "gpus": args.gpus,
            "episodes_per_task": args.episodes if args.command == "freeze" else None,
            "seed": args.seed if args.command == "freeze" else None, "results": results}, indent=2) + "\n")
        print(f"Parallel run report: {folder}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
