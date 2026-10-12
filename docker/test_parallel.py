"""Check task routing and failure reporting without running a GPU workload."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location("docker_parallel", ROOT / "tools/docker_parallel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tasks_keep_seeds_and_isolate_gpu_reports_cache_and_ports(monkeypatch, tmp_path):
    module = load_launcher()
    module.ROOT = tmp_path
    module.registry = lambda: ["task_a", "task_b", "task_c"]
    module.gpu_inventory = lambda: {"0": "GPU-aaaa", "1": "GPU-bbbb"}
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs["env"]))
        def wait(self): return 0

    module.subprocess = SimpleNamespace(Popen=Process, STDOUT=-2)
    monkeypatch.setattr(sys, "argv", ["launcher", "freeze", "--gpus", "0", "1", "--episodes", "1",
        "--tasks", "task_a", "task_b", "task_c", "--seed", "100000", "--live"])
    module.main()
    by_gpu = {}
    for command, env in calls:
        gpu = env["ROBOTWIN_GPU"]
        by_gpu.setdefault(gpu, []).append(command[command.index("--task") + 1])
        assert command[command.index("--seed") + 1] == "100000"
        assert command[command.index("--port") + 1] == str(8765 + int(gpu))
        assert env["ROBOTWIN_REPORTS_DIR"].endswith(f"gpu-{gpu}")
        assert env["ROBOTWIN_CACHE_DIR"].endswith(f"gpu-{gpu}")
    assert by_gpu == {"0": ["task_a", "task_c"], "1": ["task_b"]}
    summary = json.loads(next(tmp_path.glob("reports/docker-parallel/*/summary.json")).read_text())
    assert summary["status"] == "passed" and len(summary["results"]) == 3


def test_worker_failure_is_reported_as_failure(monkeypatch, tmp_path):
    module = load_launcher()
    module.ROOT = tmp_path
    module.registry = lambda: ["task_a", "task_b"]
    module.gpu_inventory = lambda: {"0": "GPU-aaaa"}

    class Process:
        def __init__(self, *args, **kwargs): pass
        def wait(self): return 1

    module.subprocess = SimpleNamespace(Popen=Process, STDOUT=-2)
    monkeypatch.setattr(sys, "argv", ["launcher", "freeze", "--gpus", "0", "--episodes", "1"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    summary = json.loads(next(tmp_path.glob("reports/docker-parallel/*/summary.json")).read_text())
    assert summary["status"] == "failed"
    assert [r["task"] for r in summary["results"]] == ["task_a"]


def test_gpu_selection_handles_single_gpu_auto_and_explicit_subsets():
    select = load_launcher().select_gpus
    assert select(["auto"], {"0": "GPU-aaaa"}) == ["0"]
    inventory = {"0": "GPU-aaaa", "1": "GPU-bbbb", "2": "GPU-cccc"}
    assert select(["auto"], inventory) == ["0", "1", "2"]
    assert select(["1"], inventory) == ["1"]
    assert select(["0,2"], inventory) == ["0", "2"]
    assert select(["GPU-bbbb"], inventory) == ["GPU-bbbb"]
    for request in (["3"], ["0", "GPU-aaaa"], ["auto", "0"]):
        with pytest.raises(ValueError):
            select(request, inventory)
