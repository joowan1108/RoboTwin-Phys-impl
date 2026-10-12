"""Run with each model's own Python environment, using PYTHONPATH=.../src."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import socket
import sys
import traceback
import subprocess
import hashlib

import numpy as np
import yaml
from .conditions import ROOT, digest
from .rpc import RemoteEnvironment, recv, send


def load_adapter(config):
    for path in reversed(config.get("python_paths", [])):
        sys.path.insert(0, str((ROOT / path).resolve()))
    file = (ROOT / config["adapter"]).resolve()
    spec = importlib.util.spec_from_file_location("robotwin_model_adapter", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_paths(value):
    if isinstance(value, str):
        return value.replace("${ROOT}", str(ROOT))
    if isinstance(value, dict):
        return {k: resolve_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_paths(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    config = resolve_paths(yaml.safe_load(Path(args.config).read_text()))
    adapter = (ROOT / config["adapter"]).resolve()
    if config.get("code_revision"):
        sha = subprocess.check_output(["git", "-C", str(adapter.parent), "rev-parse", "HEAD"], text=True).strip()
        if sha != config["code_revision"]:
            raise RuntimeError(f"Policy code revision mismatch: expected {config['code_revision']}, got {sha}")
    if config.get("requires_native_bf16"):
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
            raise SystemExit("This official model configuration needs an Ampere-or-newer GPU. Run this server on the supported GPU and connect via SSH; RTX 2080 Ti is sm_75.")
    if config.get("minimum_vram_gb"):
        import torch
        available = torch.cuda.get_device_properties(0).total_memory / 2**30
        if available < config["minimum_vram_gb"]:
            raise SystemExit(f"Model profile requires {config['minimum_vram_gb']} GiB; this GPU has {available:.1f} GiB. Use the remote server path.")
    for path in config.get("required_paths", []):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing model asset: {path}. See configs/policies and tools/download_models.py")
    if config.get("working_directory"):
        os.chdir(config["working_directory"])
    module = load_adapter(config)
    model = module.get_model(config.get("args", {}))
    identity = {"type": "hello", "name": config["name"], "config_sha256": digest(config),
                "checkpoint": config.get("checkpoint"), "code_revision": config.get("code_revision"),
                "adapter_sha256": hashlib.sha256(adapter.read_bytes()).hexdigest()}
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Policy ready: {config['name']} at {args.host}:{server.getsockname()[1]}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                conn.settimeout(600)
                send(conn, identity)
                env = RemoteEnvironment(conn)
                try:
                    while True:
                        request = recv(conn)
                        kind = request["type"]
                        if kind == "reset":
                            seed = request["seed"]
                            np.random.seed(seed)
                            random.seed(seed)
                            try:
                                import torch
                                torch.manual_seed(seed)
                                torch.cuda.manual_seed_all(seed)
                            except ImportError:
                                pass
                            module.reset_model(model)
                            send(conn, {"type": "done"})
                        elif kind == "step":
                            module.eval(env, model, request["observation"])
                            send(conn, {"type": "done"})
                        else:
                            raise ValueError(f"Unknown command: {kind}")
                except (ConnectionError, BrokenPipeError):
                    pass
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    try:
                        send(conn, {"type": "error", "error": error})
                    except OSError:
                        pass


if __name__ == "__main__":
    main()
