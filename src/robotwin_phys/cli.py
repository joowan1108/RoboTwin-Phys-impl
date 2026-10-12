import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .conditions import PROTOCOLS, FACTORS, ROOT, digest, sample, task_names
from .manifest import atomic_json, seal, load


def episode(task, seed, index, args):
    condition = sample(task, seed, args.protocol, tuple(args.only or ()), tuple(args.without or ()))
    return {"episode_index": index, "seed": seed, "condition": condition, "condition_sha256": digest(condition), "expert_success": False}


def worker(request, folder, label, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    req, result = folder / (label + ".request.json"), folder / (label + ".result.json")
    atomic_json(req, request)
    with (folder / (label + ".log")).open("w") as log:
        child = subprocess.run([sys.executable, "-m", "robotwin_phys.worker", str(req), str(result)],
                               stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    if child.returncode:
        raise RuntimeError(f"Episode worker failed; see {folder / (label + '.log')}. This error is not excluded as infeasible.")
    return json.loads(result.read_text())


def freeze(args, verify=True):
    from .provenance import fingerprint
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Frozen manifests are not overwritten: {out}")
    if args.task not in task_names():
        raise ValueError(f"Unknown task: {args.task}")
    if args.episodes < 1 or args.max_candidates < args.episodes or not 0 <= args.seed < 2**32:
        raise ValueError("Invalid episode count / candidate budget / seed")
    config = {"schema": "robotwin-phys-local-v1", "implementation": "independent-paper-reproduction",
              "task": args.task, "protocol": args.protocol, "start_seed": args.seed,
              "requested_episodes": args.episodes, "instruction_type": args.instruction_type,
              "only": args.only or [], "without": args.without or [], "episodes": [], "rejected": [],
              "status": "building" if verify else "candidates", "provenance": fingerprint() if verify else None}
    folder = out.parent / (out.stem + "-verification")
    start_candidate = 0
    progress = out.with_suffix(".progress.json")
    if progress.exists():
        if not args.resume:
            raise FileExistsError(f"Partial manifest exists: {progress}; use --resume")
        prior = json.loads(progress.read_text())
        if seal(prior)["sha256"] != prior.get("sha256"):
            raise ValueError("Partial manifest was modified")
        for key in config.keys() - {"episodes", "rejected", "status"}:
            if prior[key] != config[key]:
                raise ValueError(f"Cannot resume: {key} changed")
        config["episodes"], config["rejected"] = prior["episodes"], prior["rejected"]
        used = [e["seed"] for e in config["episodes"] + config["rejected"]]
        start_candidate = max(used) - args.seed + 1 if used else 0
        if len(config["episodes"]) == args.episodes:
            config["status"] = "verified" if verify else "candidates"
            atomic_json(out, seal(config))
            return
    for candidate in range(start_candidate, args.max_candidates):
        ep = episode(args.task, args.seed + candidate, len(config["episodes"]), args)
        if verify:
            req = {"mode": "expert", "episode": ep, "instruction_type": args.instruction_type,
                   "live": args.live, "port": args.port, "viewer": args.viewer}
            if args.video:
                req["video"] = str(folder / f"seed-{ep['seed']}.mp4")
            result = worker(req, folder, f"seed-{ep['seed']}", args.timeout)
            if result["status"] == "rejected":
                config["rejected"].append({"seed": ep["seed"], **result})
                atomic_json(out.with_suffix(".progress.json"), seal(config))
                print(f"Rejected seed {ep['seed']}: {result['reason']}", flush=True)
                continue
            if result["status"] != "verified":
                raise RuntimeError(result)
            ep = result["episode"]
        config["episodes"].append(ep)
        print(f"{len(config['episodes'])}/{args.episodes}: seed {ep['seed']}", flush=True)
        atomic_json(out.with_suffix(".progress.json"), seal(config))
        if len(config["episodes"]) == args.episodes:
            config["status"] = "verified" if verify else "candidates"
            atomic_json(out, seal(config))
            print(f"Manifest: {out}", flush=True)
            return
    raise RuntimeError("Candidate budget exhausted. Partial pool is not eligible for evaluation; see progress.json")


def evaluate(args, replay=False):
    from .provenance import fingerprint
    manifest = load(Path(args.manifest).resolve())
    current = fingerprint()
    if current != manifest["provenance"]:
        raise ValueError("Benchmark source/config/runtime changed since verification. Build a new manifest; do not mix results.")
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError("Output contains results; use --resume for the same experiment")
    out.mkdir(parents=True, exist_ok=True)
    identity = None
    if not replay:
        from .rpc import PolicyClient
        client = PolicyClient(args.host, args.policy_port)
        identity = client.identity
        client.close()
    run = {"manifest_sha256": manifest["sha256"], "policy": identity,
           "policy_seed": args.policy_seed, "smoke_max_actions": args.smoke_max_actions,
           "mode": "replay" if replay else "evaluate"}
    meta = out / "run.json"
    if meta.exists() and json.loads(meta.read_text()) != run:
        raise ValueError("Resume identity mismatch; use another output directory")
    atomic_json(meta, run)
    results = []
    for ep in manifest["episodes"]:
        label = f"episode-{ep['episode_index']:04d}"
        saved = out / (label + ".result.json")
        result = json.loads(saved.read_text()) if args.resume and saved.exists() else None
        if result is None or result["status"] not in ("completed", "matched"):
            req = {"mode": "replay" if replay else "evaluate", "episode": ep, "host": args.host,
                   "policy_port": args.policy_port, "policy_seed": args.policy_seed, "policy_identity": identity,
                   "live": args.live, "port": args.port, "viewer": args.viewer, "smoke_max_actions": args.smoke_max_actions}
            if args.video:
                req["video"] = str(out / (label + ".mp4"))
            result = worker(req, out, label, args.timeout)
        results.append(result)
        print(f"{label}: {result['status']} success={result.get('success')}", flush=True)
    eligible = (not replay and len(results) == 100 and all(r.get("benchmark_eligible") for r in results)
                and manifest["protocol"] in ("clean", "official_random", "physical_random")
                and not manifest["only"] and not manifest["without"])
    summary = {**run, "episodes": len(results), "benchmark_eligible": eligible,
               "implementation": "independent-paper-reproduction", "protocol": manifest["protocol"],
               "ablation": {"only": manifest["only"], "without": manifest["without"]},
               "successes": sum(r.get("success", False) for r in results),
               "success_rate": sum(r.get("success", False) for r in results)/len(results) if not replay else None}
    atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description="Independent RoboTwin-Phys reproduction: fixed conditions, expert filtering, live rendering")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("tasks")
    for name in ("candidates", "freeze"):
        q = sub.add_parser(name)
        q.add_argument("--task", required=True)
        q.add_argument("--protocol", choices=PROTOCOLS, default="physical_random")
        q.add_argument("--episodes", type=int, default=100)
        q.add_argument("--seed", type=int, default=100000, help="Official RoboTwin starting seed for run seed=0")
        q.add_argument("--max-candidates", type=int, default=10000)
        group = q.add_mutually_exclusive_group()
        group.add_argument("--only", nargs="+", choices=FACTORS)
        group.add_argument("--without", nargs="+", choices=FACTORS)
        q.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
        q.add_argument("--output", required=True)
        q.add_argument("--resume", action="store_true")
        add_view(q)
    for name in ("evaluate", "replay"):
        q = sub.add_parser(name)
        q.add_argument("--manifest", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--host", default="127.0.0.1")
        q.add_argument("--policy-port", type=int, default=9000)
        q.add_argument("--policy-seed", type=int, default=0)
        q.add_argument("--smoke-max-actions", type=int, help="Debug only; explicitly excludes run from benchmark results")
        q.add_argument("--resume", action="store_true")
        add_view(q)
    args = p.parse_args()
    if getattr(args, "viewer", False) and not os.environ.get("DISPLAY"):
        p.error("--viewer requires DISPLAY. Use --live for browser rendering on a headless machine.")
    if getattr(args, "smoke_max_actions", None) is not None and args.smoke_max_actions < 1:
        p.error("--smoke-max-actions must be positive")
    if args.command == "tasks":
        print("\n".join(task_names()))
    elif args.command in ("candidates", "freeze"):
        freeze(args, verify=args.command == "freeze")
    else:
        evaluate(args, replay=args.command == "replay")


def add_view(q):
    q.add_argument("--live", action="store_true", help="Serve live camera at http://127.0.0.1:8765")
    q.add_argument("--port", type=int, default=8765)
    q.add_argument("--viewer", action="store_true", help="Native SAPIEN desktop viewer")
    q.add_argument("--video", action="store_true", help="Save head camera MP4 at 30fps")
    q.add_argument("--timeout", type=int, default=1800)


if __name__ == "__main__":
    main()
