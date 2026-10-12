"""Local MJPEG viewer and checked ffmpeg recording; no extra simulation steps."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
import json
import shutil
import subprocess
import threading
import time

import numpy as np
from PIL import Image


class LiveView:
    def __init__(self, port=8765, host="127.0.0.1"):
        self.jpeg = None
        self.info = {}
        self.condition = threading.Condition()
        self.sequence = 0
        self.closed = False
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/":
                    data = b'''<!doctype html><meta charset="utf-8"><title>RoboTwin-Phys Live</title>
<style>body{background:#111827;color:#eee;font:18px system-ui;margin:30px}img{width:min(100%,960px);image-rendering:auto}pre{white-space:pre-wrap}</style>
<h1>RoboTwin-Phys evaluation</h1><img id="feed" src="/stream.mjpg" onerror="setTimeout(()=>this.src='/stream.mjpg?t='+Date.now(),1500)"><pre id="status">Waiting for an episode...</pre>
<script>let online=false;setInterval(async()=>{try{let s=await(await fetch('/status')).json();document.getElementById('status').textContent=JSON.stringify(s,null,2);if(!online)document.getElementById('feed').src='/stream.mjpg?t='+Date.now();online=true}catch(e){online=false;document.getElementById('status').textContent='Waiting for the next episode...'}},1000)</script>'''
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path == "/status":
                    data = json.dumps(view.info).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path.split("?", 1)[0] == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    last = -1
                    try:
                        while not view.closed:
                            with view.condition:
                                view.condition.wait_for(lambda: view.sequence != last or view.closed, timeout=2)
                                if view.closed:
                                    break
                                last, frame = view.sequence, view.jpeg
                            if frame:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                                 str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self.send_error(404)

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://{host}:{self.server.server_port}"
        print(f"Live rendering: {self.url}", flush=True)

    def update(self, rgb, info):
        buf = BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        with self.condition:
            self.jpeg, self.info = buf.getvalue(), dict(info)
            self.sequence += 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


class Recorder:
    def __init__(self, path, fps=30):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.proc = None
        self.frames = 0
        self.size = None

    def write(self, rgb):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB frame")
        if self.proc is None:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is missing; source tools/activate.sh")
            h, w, _ = rgb.shape
            self.size = rgb.shape
            self.proc = subprocess.Popen([
                ffmpeg, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-threads", "2", "-pix_fmt", "yuv420p", "-crf", "20",
                "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if rgb.shape != self.size:
            raise ValueError("Camera frame size changed during episode")
        self.proc.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.stdin = None
            _, err = self.proc.communicate(timeout=60)
            if self.proc.returncode:
                raise RuntimeError(f"ffmpeg failed: {err.decode()}")
            self.proc = None
