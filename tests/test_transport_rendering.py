import json
import socket
import subprocess
import threading
import urllib.request

import numpy as np
import pytest

from robotwin_phys.rpc import recv, send, PolicyClient, RemoteEnvironment
from robotwin_phys.rendering import LiveView, Recorder


def test_array_transport_and_environment_callbacks():
    a, b = socket.socketpair()
    frame = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    result = []
    def server():
        request = recv(b)
        assert np.array_equal(request["frame"], frame)
        env = RemoteEnvironment(b)
        result.append(env.get_instruction())
        env.take_action(np.ones(14), action_type="qpos")
        send(b, {"type": "done"})
    thread = threading.Thread(target=server)
    thread.start()
    class Env:
        def get_instruction(self): return "Stack blocks"
        def take_action(self, action, action_type):
            assert action_type == "qpos" and np.array_equal(action, np.ones(14))
    client = PolicyClient.__new__(PolicyClient); client.sock = a
    client.request("step", env=Env(), frame=frame)
    thread.join(timeout=3)
    assert result == ["Stack blocks"]
    a.close(); b.close()


def test_live_server_frame_and_status():
    view = LiveView(port=0)
    try:
        view.update(np.full((24, 32, 3), 127, np.uint8), {"seed": 100000})
        status = json.load(urllib.request.urlopen(view.url + "/status", timeout=3))
        assert status["seed"] == 100000
        with urllib.request.urlopen(view.url + "/stream.mjpg", timeout=3) as response:
            assert b"Content-Type: image/jpeg" in response.read(100)
    finally:
        view.close()


def test_mp4_is_decodable(tmp_path):
    path = tmp_path / "test.mp4"
    writer = Recorder(path)
    for i in range(5):
        writer.write(np.full((24, 32, 3), i*40, np.uint8))
    writer.close()
    result = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=width,height,nb_read_frames", "-of", "json", str(path)], text=True)
    stream = json.loads(result)["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (32, 24, 5)
