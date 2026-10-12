"""Exercise launcher failure/argument handling without a Docker daemon."""
import gzip
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def docker_mock(tmp_path):
    executable = tmp_path / "docker"
    executable.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['DOCKER_CALL_LOG'], 'a') as f:
    f.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'info':
    sys.exit(int(os.environ.get('MOCK_INFO_EXIT', '0')))
if sys.argv[1:3] == ['image', 'save']:
    sys.stdout.buffer.write(b'image-archive-test')
    sys.exit(int(os.environ.get('MOCK_SAVE_EXIT', '0')))
if sys.argv[1] == 'compose':
    print('selected_gpu=' + os.environ['ROBOTWIN_GPU'])
""")
    executable.chmod(0o755)
    # Simulate a server with exactly one physical GPU.
    nvidia = tmp_path / "nvidia-smi"
    nvidia.write_text("#!/bin/sh\necho 0\n")
    nvidia.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = {**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
           "DOCKER_CALL_LOG": str(log), "ROBOTWIN_GPU_MODE": "nvidia"}

    def invoke(*args, **variables):
        result = subprocess.run(["bash", str(ROOT / "tools/docker.sh"), *args],
            env={**env, **variables}, capture_output=True, text=True)
        return result, [json.loads(line) for line in log.read_text().splitlines()]
    return invoke


def test_daemon_denied_stops_before_build(docker_mock):
    result, calls = docker_mock("build", MOCK_INFO_EXIT="1")
    assert result.returncode == 1
    assert calls == [["info"]]
    assert "no host permissions or drivers were changed" in result.stderr


def test_single_gpu_host_auto_selects_gpu_zero(docker_mock):
    result, _ = docker_mock("run", "true", ROBOTWIN_GPU="auto")
    assert result.returncode == 0, result.stderr
    assert "selected_gpu=0" in result.stdout


def test_run_preserves_paths_with_spaces_and_explicit_env_file(docker_mock):
    args = ["robotwin-phys", "replay", "--manifest", "manifests/pool with spaces.json"]
    result, calls = docker_mock("run", *args)
    assert result.returncode == 0, result.stderr
    assert calls[1][0:3] == ["compose", "--env-file", "docker/runtime.env"]
    assert calls[1][-len(args):] == args
    assert calls[1][-len(args)-5:-len(args)] == ["run", "--rm", "--no-deps", "-T", "sim"]


def test_failed_save_does_not_leave_a_valid_looking_archive(docker_mock, tmp_path):
    output = tmp_path / "failed image.tar.gz"
    result, _ = docker_mock("save", str(output), MOCK_SAVE_EXIT="42")
    assert result.returncode != 0
    assert not output.exists()
    assert not list(tmp_path.glob("failed image.tar.gz.*"))


def test_save_exports_selected_image_and_preserves_existing_archive(docker_mock, tmp_path):
    output = tmp_path / "image.tar.gz"
    result, calls = docker_mock("save", str(output), ROBOTWIN_IMAGE="sim:reviewed")
    assert result.returncode == 0, result.stderr
    assert calls[-1] == ["image", "save", "sim:reviewed"]
    assert gzip.decompress(output.read_bytes()) == b"image-archive-test"
    result, _ = docker_mock("save", str(output))
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert gzip.decompress(output.read_bytes()) == b"image-archive-test"
