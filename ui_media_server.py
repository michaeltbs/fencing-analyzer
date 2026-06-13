"""
ui_media_server.py — Mini HTTP server for serving large videos + player HTML.

Used by the Streamlit app so browsers can stream mp4 files via HTTP Range
requests. Keeps app.py free of HTTP-server boilerplate.
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


_media_server = None
_media_server_port = None


class MediaRequestHandler(BaseHTTPRequestHandler):
    """Serves video file with range support + Player HTML page."""
    video_path = None
    frame_data_json = None
    metrics_json = None
    player_html = None

    def do_GET(self):
        if self.path == "/video":
            self._serve_video()
        elif self.path == "/data.json":
            self._serve_json(MediaRequestHandler.frame_data_json, "application/json")
        elif self.path == "/metrics.json":
            self._serve_json(MediaRequestHandler.metrics_json, "application/json")
        elif self.path in ("/player", "/"):
            self._serve_player()
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_video(self):
        path = MediaRequestHandler.video_path
        if not path or not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        file_size = path.stat().st_size
        range_header = self.headers.get("Range", "")

        if range_header.startswith("bytes="):
            start, end = 0, file_size - 1
            parts = range_header[6:].split("-")
            if parts[0]:
                start = int(parts[0])
            if parts[1]:
                end = int(parts[1])
            content_length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
        else:
            start, end = 0, file_size - 1
            content_length = file_size
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")

        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _serve_player(self):
        if MediaRequestHandler.player_html is None:
            self.send_response(404)
            self.end_headers()
            return
        data_bytes = MediaRequestHandler.player_html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data_bytes)

    def _serve_json(self, data, mime):
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        data_bytes = data.encode("utf-8") if isinstance(data, str) else data
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "max-age=3600")
        self.end_headers()
        self.wfile.write(data_bytes)

    def log_message(self, format, *args):
        pass  # suppress HTTP server logs


def start_media_server(video_path, frame_data, metrics, player_html=None):
    """Startet Mini-HTTP-Server auf einem freien Port, gibt die URL zurück."""
    global _media_server, _media_server_port

    # Stop old server
    stop_media_server()

    MediaRequestHandler.video_path = Path(video_path)
    MediaRequestHandler.frame_data_json = json.dumps(frame_data)
    MediaRequestHandler.metrics_json = json.dumps(metrics)
    MediaRequestHandler.player_html = player_html

    # Find free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _media_server_port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer(("127.0.0.1", _media_server_port), MediaRequestHandler)
    _media_server = server
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return f"http://127.0.0.1:{_media_server_port}"


def stop_media_server():
    global _media_server
    if _media_server:
        _media_server.shutdown()
        _media_server = None


def get_media_server_port():
    """Return currently active port or None."""
    return _media_server_port
