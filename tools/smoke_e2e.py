#!/usr/bin/env python3
"""Real expert -> new-process replay -> dummy RPC policy -> playable MP4."""
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/verified-smoke-stack-blocks.json"
OUTPUT = ROOT / "runs/verified-smoke"


def cli(*args):
    subprocess.run([sys.executable, "-m", "robotwin_phys.cli", *args], cwd=ROOT, check=True)


def main():
    if not MANIFEST.exists():
        cli("freeze", "--task", "stack_blocks_two", "--episodes", "2", "--seed", "100000", "--max-candidates", "20",
            "--output", str(MANIFEST), "--video", "--live", "--resume")
    cli("replay", "--manifest", str(MANIFEST), "--output", str(OUTPUT / "replay"), "--resume", "--video")
    with (ROOT / "reports/noop-server.log").open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "robotwin_phys.policy_server", "--config",
            str(ROOT / "configs/policies/noop.yaml"), "--port", "9000"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                if server.poll() is not None:
                    raise RuntimeError("Noop server exited; see reports/noop-server.log")
                try:
                    with socket.create_connection(("127.0.0.1", 9000), timeout=.2):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Noop server did not start")
            cli("evaluate", "--manifest", str(MANIFEST), "--output", str(OUTPUT / "policy"),
                "--smoke-max-actions", "3", "--video", "--live", "--resume")
            # Resume must preserve finished results and experiment identity.
            before = (OUTPUT / "policy/episode-0000.result.json").read_bytes()
            cli("evaluate", "--manifest", str(MANIFEST), "--output", str(OUTPUT / "policy"),
                "--smoke-max-actions", "3", "--video", "--resume")
            assert (OUTPUT / "policy/episode-0000.result.json").read_bytes() == before
        finally:
            server.terminate()
            server.wait(timeout=10)
    videos = list(OUTPUT.rglob("*.mp4")) + list(MANIFEST.with_name(MANIFEST.stem+"-verification").glob("*.mp4"))
    media = []
    for video in videos:
        result = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", str(video)], text=True)
        stream = json.loads(result)["streams"][0]
        assert stream["width"] == 320 and stream["height"] == 240 and int(stream["nb_read_frames"]) > 0
        media.append({"path": str(video.relative_to(ROOT)), **stream})
    summary={"expert_episodes":2,"fresh_process_replay":"passed","policy_rpc":"passed",
             "resume":"passed","learned_policy_evaluated":False,"videos":media}
    (ROOT / "reports/e2e-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
