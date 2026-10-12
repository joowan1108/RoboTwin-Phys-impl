"""Bounded JSON/ndarray RPC. Preserves the official get_model/eval/reset API.

Model environments can have different Python/Torch/CUDA versions. No pickle,
and no simulator code is imported by the policy process. Use SSH for remote hosts.
"""
import base64
import json
import socket
import struct
import numpy as np

MAX_PACKET = 64 * 1024 * 1024
METHODS = {"get_obs", "get_instruction", "take_action", "check_success"}
ATTRIBUTES = {"take_action_cnt", "step_lim", "eval_success", "task_name", "ep_num",
              "current_seed", "episode_seed", "test_num", "eval_video_path", "save_dir"}


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not supported")
        value = np.ascontiguousarray(value)
        return {"__ndarray__": base64.b64encode(value.tobytes()).decode(),
                "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value)}")


def decode(value):
    if "__ndarray__" in value:
        dtype = np.dtype(value["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are forbidden")
        return np.frombuffer(base64.b64decode(value["__ndarray__"], validate=True), dtype=dtype).reshape(value["shape"]).copy()
    return value


def send(sock, value):
    packet = json.dumps(value, default=encode, allow_nan=False, separators=(",", ":")).encode()
    if len(packet) > MAX_PACKET:
        raise ValueError("RPC packet exceeds limit")
    sock.sendall(struct.pack("!I", len(packet)) + packet)


def recv(sock):
    def read(n):
        chunks = bytearray()
        while len(chunks) < n:
            data = sock.recv(n - len(chunks))
            if not data:
                raise ConnectionError("Policy connection closed")
            chunks.extend(data)
        return bytes(chunks)
    length = struct.unpack("!I", read(4))[0]
    if length > MAX_PACKET:
        raise ValueError("RPC packet exceeds limit")
    return json.loads(read(length), object_hook=decode)


class RemoteEnvironment:
    def __init__(self, sock):
        self.sock = sock

    def __getattr__(self, name):
        if name in METHODS:
            def call(*args, **kwargs):
                send(self.sock, {"type": "env_call", "name": name, "args": args, "kwargs": kwargs})
                answer = recv(self.sock)
                if "error" in answer:
                    raise RuntimeError(answer["error"])
                return answer["value"]
            return call
        if name in ATTRIBUTES:
            send(self.sock, {"type": "env_attr", "name": name})
            answer = recv(self.sock)
            if "error" in answer:
                raise AttributeError(answer["error"])
            return answer["value"]
        raise AttributeError(name)


class PolicyClient:
    def __init__(self, host, port, timeout=600):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.identity = recv(self.sock)
        if self.identity.get("type") != "hello":
            raise RuntimeError("Not a RoboTwin-Phys policy server")

    def request(self, command, env=None, **kwargs):
        send(self.sock, {"type": command, **kwargs})
        while True:
            reply = recv(self.sock)
            kind = reply["type"]
            if kind == "done":
                return reply.get("value")
            if kind == "error":
                raise RuntimeError("Policy server: " + reply["error"])
            if env is None:
                raise RuntimeError("Unexpected environment request")
            name = reply["name"]
            try:
                if kind == "env_call" and name in METHODS:
                    if name == "take_action":
                        action = np.asarray(reply["args"][0])
                        action_type = reply.get("kwargs", {}).get("action_type", "qpos")
                        if len(reply["args"]) > 1:
                            action_type = reply["args"][1]
                        dim = 14 if action_type == "qpos" else 16 if action_type == "ee" else 0
                        if action.shape != (dim,) or not np.isfinite(action).all():
                            raise ValueError(f"Invalid {action_type} action: expected finite ({dim},)")
                    value = getattr(env, name)(*reply["args"], **reply["kwargs"])
                elif kind == "env_attr" and name in ATTRIBUTES:
                    value = getattr(env, name)
                    if isinstance(value, __import__('pathlib').Path):
                        value = str(value)
                else:
                    raise ValueError("Disallowed environment operation")
                send(self.sock, {"value": value})
            except Exception as exc:
                send(self.sock, {"error": f"{type(exc).__name__}: {exc}"})
                raise

    def close(self):
        self.sock.close()
