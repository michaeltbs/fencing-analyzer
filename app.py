"""
Fencing Analyzer — Streamlit App
Fecht-Video-Analyse mit YOLOv8m-Pose + Interaktivem Live-Video-Player

Usage:
  streamlit run C:\\Users\\micha\\Desktop\\fencing_analyzer\\app.py
"""
import streamlit as st
st.set_page_config(page_title="Fecht-Analyzer", layout="wide", page_icon="\U0001F93A")

import os, sys, json, math, time, shutil, subprocess, tempfile, struct, base64, socket, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
from PIL import Image
from ultralytics import YOLO
from streamlit.components.v1 import html as st_html

# --- CONSTS ---
C_GREEN  = "#00ff88"
C_RED    = "#ff4466"
C_BLUE   = "#00ccff"
C_BG     = "#0d1117"
C_CARD   = "#161b22"
C_BORDER = "#30363d"
C_TEXT   = "#c9d1d9"
C_MUTED  = "#8b949e"
C_ACCENT = "#58a6ff"

PISTE_WIDTH_CM = 200
PISTE_WIDTH_PX_FALLBACK = 130

# --- MINI HTTP SERVER für Large-Video-Support ---
_media_server = None
_media_server_port = None

class MediaRequestHandler(BaseHTTPRequestHandler):
    """Serves video file with range support + JSON data for the live player."""
    video_path = None
    frame_data_json = None
    metrics_json = None
    
    def do_GET(self):
        if self.path == "/video":
            self._serve_video()
        elif self.path == "/data.json":
            self._serve_json(MediaRequestHandler.frame_data_json, "application/json")
        elif self.path == "/metrics.json":
            self._serve_json(MediaRequestHandler.metrics_json, "application/json")
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

def start_media_server(video_path, frame_data, metrics):
    """Startet Mini-HTTP-Server auf einem freien Port, gibt die URL zurück."""
    global _media_server, _media_server_port
    
    # Stop old server
    stop_media_server()
    
    MediaRequestHandler.video_path = Path(video_path)
    MediaRequestHandler.frame_data_json = json.dumps(frame_data)
    MediaRequestHandler.metrics_json = json.dumps(metrics)
    
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

# COCO Pose skeleton connections (17 keypoints)
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),       # face
    (5,6),(5,7),(7,9),(6,8),(8,10), # arms
    (5,11),(6,12),(11,12),          # torso
    (11,13),(13,15),(12,14),(14,16) # legs
]

def kpt(kpts, idx):
    if kpts is None or idx >= len(kpts): return None
    x, y = float(kpts[idx][0]), float(kpts[idx][1])
    return (x, y) if x > 0 and y > 0 else None

def midpoint(a, b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2) if a and b else None

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) if a and b else None

def angle_at(p1, p2, p3):
    if not all([p1,p2,p3]): return None
    v1 = (p1[0]-p2[0], p1[1]-p2[1])
    v2 = (p3[0]-p2[0], p3[1]-p2[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1 or n2 < 1: return None
    return math.degrees(math.acos(max(-1, min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)))))


# === ANALYSIS ENGINE ===
@st.cache_resource
def load_model():
    return YOLO("yolov8m-pose.pt")

def extract_clip(video_path, output_path, start_sec=0, duration_sec=15, target_width=640):
    cmd = [
        str(Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/ffmpeg.exe"),
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", str(video_path),
        "-vf", f"scale={target_width}:-2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        "-y", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, timeout=600)
    return output_path.exists()

def analyze_video(video_path, progress_callback=None, model=None):
    """Returns result dict with frames (keypoints per frame), metrics, summary."""
    if model is None:
        model = load_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Video konnte nicht geoeffnet werden")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, first_frame = cap.read()
    w_frame = first_frame.shape[1] if ret else 640
    h_frame = first_frame.shape[0] if ret else 360
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    TRACK_IDS = [0, 1]
    prev_centers = {}
    frame_data = []  # compact: [time, m_kpts_xy, g_kpts_xy]
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        r = results[0]
        current_centers = {}
        current_kpts = {}

        if r.boxes is not None and len(r.boxes.conf) > 0:
            for pi in range(len(r.boxes.conf)):
                x1, y1, x2, y2 = map(float, r.boxes.xyxy[pi].cpu().numpy().tolist())
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                current_centers[pi] = (cx, cy)
                if r.keypoints is not None and pi < len(r.keypoints.xy):
                    current_kpts[pi] = r.keypoints.xy[pi].cpu().numpy().tolist()

        if frame_idx == 0:
            assigned = {tid: tid for tid in TRACK_IDS if tid in current_centers}
            prev_centers = {tid: current_centers.get(tid) for tid in TRACK_IDS}
        else:
            assigned = {}
            used = set()
            for track_id, prev_center in prev_centers.items():
                if prev_center is None:
                    continue
                best_match, best_dist = None, float('inf')
                for pi, center in current_centers.items():
                    if pi in used:
                        continue
                    d = math.hypot(prev_center[0]-center[0], prev_center[1]-center[1])
                    if d < best_dist and d < 400:
                        best_dist, best_match = d, pi
                if best_match is not None:
                    assigned[track_id] = best_match
                    used.add(best_match)
                    prev_centers[track_id] = current_centers[best_match]

        def get_kpts(tid):
            pi = assigned.get(tid)
            return current_kpts.get(pi) if pi is not None else None

        m_kpts = get_kpts(0)
        g_kpts = get_kpts(1)

        # Compact: flatten keypoints to [x,y] * 17
        def flatten_kpts(kpts_obj):
            if kpts_obj is None:
                return None
            flat = []
            for i in range(17):
                x, y = float(kpts_obj[i][0]), float(kpts_obj[i][1])
                if x > 0 and y > 0:
                    flat.extend([round(x), round(y)])
                else:
                    flat.extend([0, 0])
            return flat

        frame_data.append({
            "t": round(frame_idx / fps, 2),
            "m": flatten_kpts(m_kpts),
            "g": flatten_kpts(g_kpts),
        })

        frame_idx += 1
        if progress_callback and frame_idx % 15 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    N = len(frame_data)

    # Kalibrierung
    all_hip_x = []
    for f in frame_data:
        for label in ("m", "g"):
            k = f[label]
            if k:
                hip_x = k[22]  # index 11*2 = 22 (hip_l x)
                if hip_x > 0:
                    all_hip_x.append(hip_x)
    piste_px = (max(all_hip_x) - min(all_hip_x)) if all_hip_x else PISTE_WIDTH_PX_FALLBACK
    px_per_cm = piste_px / PISTE_WIDTH_CM

    # Helper: get keypoint from flat [x,y]*17
    def get_kp(flat, idx):
        if flat is None: return None
        x, y = flat[idx*2], flat[idx*2+1]
        return (x, y) if x > 0 and y > 0 else None

    def get_mid(flat, idx1, idx2):
        a, b = get_kp(flat, idx1), get_kp(flat, idx2)
        return midpoint(a, b)

    # Compute all metrics
    m1_dist = []  # distance
    m2_m_angle, m2_g_angle = [], []
    m3_m_lunge, m3_g_lunge = [], []
    m4_m_path, m4_g_path = [], []
    m5_m_tilt, m5_g_tilt = [], []
    m6_m_acc, m6_g_acc = [], []
    m7_m_steps, m7_g_steps = [], []
    m8_vel_m, m8_vel_g = [], []

    for f in frame_data:
        t = f["t"]
        mk, gk = f["m"], f["g"]

        # M1: Distance
        m_hip = get_mid(mk, 11, 12)
        g_hip = get_mid(gk, 11, 12)
        d_val = dist(m_hip, g_hip)
        if d_val:
            m1_dist.append({"t": t, "px": d_val, "cm": d_val / px_per_cm})

        # M2: Weapon arm angle
        a1 = angle_at(get_kp(mk, 6), get_kp(mk, 8), get_kp(mk, 10))  # R shoulder->elbow->wrist
        if a1 is None: a1 = angle_at(get_kp(mk, 5), get_kp(mk, 7), get_kp(mk, 9))
        m2_m_angle.append({"t": t, "deg": a1 or 0})
        b1 = angle_at(get_kp(gk, 5), get_kp(gk, 7), get_kp(gk, 9))  # L shoulder->elbow->wrist
        if b1 is None: b1 = angle_at(get_kp(gk, 6), get_kp(gk, 8), get_kp(gk, 10))
        m2_g_angle.append({"t": t, "deg": b1 or 0})

        # M3: Lunge depth
        if m_hip and get_kp(mk, 15) and get_kp(mk, 16):
            la, ra = get_kp(mk, 15), get_kp(mk, 16)
            front = la if la[1] < ra[1] else ra
            m3_m_lunge.append({"t": t, "px": max(0, m_hip[1] - front[1])})
        if g_hip and get_kp(gk, 15) and get_kp(gk, 16):
            la, ra = get_kp(gk, 15), get_kp(gk, 16)
            front = la if la[1] < ra[1] else ra
            m3_g_lunge.append({"t": t, "px": max(0, g_hip[1] - front[1])})

        # M4: Movement path
        if m_hip: m4_m_path.append({"t": t, "x": m_hip[0], "y": m_hip[1]})
        if g_hip: m4_g_path.append({"t": t, "x": g_hip[0], "y": g_hip[1]})

        # M5: Torso tilt
        m_sh = get_mid(mk, 5, 6)
        if m_sh and m_hip:
            dx = m_sh[0] - m_hip[0]; dy = m_sh[1] - m_hip[1]
            tilt = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0
            m5_m_tilt.append({"t": t, "deg": tilt * (1 if dx > 0 else -1)})
        g_sh = get_mid(gk, 5, 6)
        if g_sh and g_hip:
            dx = g_sh[0] - g_hip[0]; dy = g_sh[1] - g_hip[1]
            tilt = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0
            m5_g_tilt.append({"t": t, "deg": tilt * (1 if dx > 0 else -1)})

    # M6: Acceleration (needs 2-frame delta)
    for i in range(2, N):
        dt = 2/fps
        def wrist_acc(label, wrist_idx, alt_idx):
            p0 = get_kp(frame_data[i-2][label], wrist_idx)
            p1 = get_kp(frame_data[i-1][label], wrist_idx)
            p2 = get_kp(frame_data[i][label], wrist_idx)
            if all([p0, p1, p2]):
                v1 = dist(p0, p1) * fps if dist(p0, p1) else 0
                v2 = dist(p1, p2) * fps if dist(p1, p2) else 0
                return (v2 - v1) / dt
            return None
        a_m = wrist_acc("m", 10, 9)
        if a_m is not None: m6_m_acc.append({"t": frame_data[i]["t"], "acc": a_m})
        a_g = wrist_acc("g", 9, 10)
        if a_g is not None: m6_g_acc.append({"t": frame_data[i]["t"], "acc": a_g})

    # M7: Steps
    for i in range(1, N):
        p = frame_data[i-1]; c = frame_data[i]
        for label in ("m", "g"):
            p_la = get_kp(p[label], 15); p_ra = get_kp(p[label], 16)
            c_la = get_kp(c[label], 15); c_ra = get_kp(c[label], 16)
            if all([p_la, p_ra, c_la, c_ra]):
                dl = dist(p_la, c_la) or 0
                dr = dist(p_ra, c_ra) or 0
                side = None
                if dl > 12 and dr < 6:
                    side = "links"
                elif dr > 12 and dl < 6:
                    side = "rechts"
                if side:
                    # Check if other foot moved recently
                    other_moved = False
                    for j in range(max(0, i-3), i):
                        pp = frame_data[j][label]
                        op = frame_data[j+1][label]
                        other_key = 16 if side == "links" else 15
                        pp_k = get_kp(pp, other_key); op_k = get_kp(op, other_key)
                        if pp_k and op_k:
                            if (dist(pp_k, op_k) or 0) > 12:
                                other_moved = True
                                break
                    step_type = "ganz" if other_moved else "halb"
                    step_rec = {"t": c["t"], "side": side, "dist": dl if side == "links" else dr, "type": step_type}
                    if label == "m":
                        m7_m_steps.append(step_rec)
                    else:
                        m7_g_steps.append(step_rec)

    # M8: Sync
    for i in range(1, len(m4_m_path)):
        dx = m4_m_path[i]["x"] - m4_m_path[i-1]["x"]
        dy = m4_m_path[i]["y"] - m4_m_path[i-1]["y"]
        m8_vel_m.append(math.hypot(dx, dy) * fps)
    for i in range(1, len(m4_g_path)):
        dx = m4_g_path[i]["x"] - m4_g_path[i-1]["x"]
        dy = m4_g_path[i]["y"] - m4_g_path[i-1]["y"]
        m8_vel_g.append(math.hypot(dx, dy) * fps)

    min_vel = min(len(m8_vel_m), len(m8_vel_g))
    mv = np.array(m8_vel_m[:min_vel]) if m8_vel_m else np.array([])
    gv = np.array(m8_vel_g[:min_vel]) if m8_vel_g else np.array([])

    m8_corr = float(np.corrcoef(mv, gv)[0, 1]) if len(mv) > 10 and np.std(mv) > 0 and np.std(gv) > 0 else 0
    m8_lag = 0
    if len(mv) > 10:
        xcorr = np.correlate(mv - np.mean(mv), gv - np.mean(gv), mode='full')
        m8_lag = int(np.argmax(np.abs(xcorr)) - (len(mv) - 1))

    summary = {
        "video": {"frames": N, "duration_s": round(duration, 1), "fps": fps, "resolution": f"{w_frame}x{h_frame}", "w": w_frame, "h": h_frame},
        "metrik_1_distanz": {
            "avg_px": round(float(np.mean([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "min_px": round(float(np.min([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "max_px": round(float(np.max([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "avg_cm": round(float(np.mean([d["cm"] for d in m1_dist])), 1) if m1_dist else 0,
            "data": [round(d["cm"], 1) for d in m1_dist],
        },
        "metrik_2_winkel": {
            "m_avg": round(float(np.mean([x["deg"] for x in m2_m_angle if x["deg"] > 0])), 1),
            "g_avg": round(float(np.mean([x["deg"] for x in m2_g_angle if x["deg"] > 0])), 1),
            "m_data": [round(x["deg"], 1) for x in m2_m_angle],
            "g_data": [round(x["deg"], 1) for x in m2_g_angle],
        },
        "metrik_3_lunge": {
            "m_max": round(float(np.max([x["px"] for x in m3_m_lunge])), 1) if m3_m_lunge else 0,
            "g_max": round(float(np.max([x["px"] for x in m3_g_lunge])), 1) if m3_g_lunge else 0,
        },
        "metrik_5_haltung": {
            "m_avg": round(float(np.mean([x["deg"] for x in m5_m_tilt])), 1) if m5_m_tilt else 0,
            "g_avg": round(float(np.mean([x["deg"] for x in m5_g_tilt])), 1) if m5_g_tilt else 0,
            "m_data": [round(x["deg"], 1) for x in m5_m_tilt],
            "g_data": [round(x["deg"], 1) for x in m5_g_tilt],
        },
        "metrik_6_acc": {
            "m_max": round(float(np.max([abs(x["acc"]) for x in m6_m_acc])), 1) if m6_m_acc else 0,
            "g_max": round(float(np.max([abs(x["acc"]) for x in m6_g_acc])), 1) if m6_g_acc else 0,
            "m_data": [round(x["acc"], 1) for x in m6_m_acc],
            "g_data": [round(x["acc"], 1) for x in m6_g_acc],
        },
        "metrik_7_schritte": {
            "m_total": len(m7_m_steps), "m_rate": round(len(m7_m_steps)/duration, 2) if duration > 0 else 0,
            "g_total": len(m7_g_steps), "g_rate": round(len(m7_g_steps)/duration, 2) if duration > 0 else 0,
            "m_halb": len([s for s in m7_m_steps if s["type"] == "halb"]),
            "m_ganz": len([s for s in m7_m_steps if s["type"] == "ganz"]),
            "m_data": m7_m_steps,
            "g_data": m7_g_steps,
        },
        "metrik_8_sync": {
            "korrelation": round(m8_corr, 3), "lag_frames": m8_lag, "lag_s": round(m8_lag / fps, 2) if fps > 0 else 0,
            "leader": "Michael" if abs(m8_lag) > 2 and m8_lag > 0 else ("Gegner" if abs(m8_lag) > 2 else "neutral"),
            "m_vel": [round(v, 1) for v in m8_vel_m],
            "g_vel": [round(v, 1) for v in m8_vel_g],
        },
    }

    return {
        "summary": summary,
        "frame_data": frame_data,
        "m1_dist": m1_dist, "m2_m_angle": m2_m_angle, "m2_g_angle": m2_g_angle,
        "m3_m_lunge": m3_m_lunge, "m3_g_lunge": m3_g_lunge,
        "m4_m_path": m4_m_path, "m4_g_path": m4_g_path,
        "m5_m_tilt": m5_m_tilt, "m5_g_tilt": m5_g_tilt,
        "m6_m_acc": m6_m_acc, "m6_g_acc": m6_g_acc,
        "m7_m_steps": m7_m_steps, "m7_g_steps": m7_g_steps,
        "m8_corr": m8_corr, "m8_lag": m8_lag,
        "m8_vel_m": m8_vel_m, "m8_vel_g": m8_vel_g,
    }


# === PLOTLY HELPER ===
def fig_theme(fig, title="", xlabel="", ylabel=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C_TEXT), x=0.02),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        font=dict(color=C_TEXT, size=11),
        xaxis=dict(gridcolor="#21262d", title=xlabel, color=C_MUTED, showline=True, linecolor=C_BORDER),
        yaxis=dict(gridcolor="#21262d", title=ylabel, color=C_MUTED, showline=True, linecolor=C_BORDER),
        margin=dict(l=40, r=16, t=28, b=28),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor=C_BORDER, font=dict(size=10)),
        dragmode=False,
    )
    fig.update_traces(hoverinfo="x+y")
    return fig


# === INTERACTIVE LIVE PLAYER (HTML Component) ===

def build_live_player_html(result, clip_path, mode="auto"):
    """
    Generates a complete self-contained HTML page:
    - Video player with canvas overlay (skeleton, distance, wrist angle)
    - Toggle buttons (skeleton / distance / angle / timestamp)
    - Shared time slider (RangeInput)
    - Full Plotly charts (all 9 metrics synced to slider)
    - Click on chart → jump video to that time
    
    mode: "embed" = base64 video + inline JSON
          "server" = fetch from media server
          "auto" = embed if < 30MB, else server
    """
    s = result["summary"]
    fps = s["video"]["fps"]
    duration = s["video"]["duration_s"]
    vw = s["video"]["w"]
    vh = s["video"]["h"]

    # Determine mode based on file size
    clip_size_mb = clip_path.stat().st_size / (1024 * 1024)
    if mode == "auto":
        mode = "embed" if clip_size_mb < 30 else "server"

    if mode == "embed":
        with open(clip_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("ascii")
        video_url = f"data:video/mp4;base64,{video_b64}"
        frame_json = json.dumps(result["frame_data"])
        dist_data = json.dumps([d["cm"] for d in result["m1_dist"]])
        m_angle_data = json.dumps([d["deg"] for d in result["m2_m_angle"]])
        g_angle_data = json.dumps([d["deg"] for d in result["m2_g_angle"]])
        m_haltung = json.dumps([d["deg"] for d in result["m5_m_tilt"]])
        g_haltung = json.dumps([d["deg"] for d in result["m5_g_tilt"]])
        m_acc = json.dumps([d["acc"] for d in result["m6_m_acc"]])
        g_acc = json.dumps([d["acc"] for d in result["m6_g_acc"]])
        m_steps = json.dumps(result["m7_m_steps"])
        g_steps = json.dumps(result["m7_g_steps"])
        m_vel = json.dumps(result["m8_vel_m"])
        g_vel = json.dumps(result["m8_vel_g"])
        m_path = json.dumps([{"x": p["x"], "y": p["y"]} for p in result["m4_m_path"]])
        g_path = json.dumps([{"x": p["x"], "y": p["y"]} for p in result["m4_g_path"]])
        data_block = f"""
const FRAMES = {frame_json};
const DIST_DATA = {dist_data};
const M_ANGLE = {m_angle_data};
const G_ANGLE = {g_angle_data};
const M_HALTUNG = {m_haltung};
const G_HALTUNG = {g_haltung};
const M_ACC = {m_acc};
const G_ACC = {g_acc};
const M_STEPS = {m_steps};
const G_STEPS = {g_steps};
const M_VEL = {m_vel};
const G_VEL = {g_vel};
const M_PATH = {m_path};
const G_PATH = {g_path};
"""
        video_src_block = f'<source src="{video_url}" type="video/mp4">'
    else:
        # Server mode: use the configured media server URL
        video_url = "/video"
        data_url = "/data.json"
        video_src_block = f'<source src="{video_url}" type="video/mp4">'
        data_block = """
const MEDIA_BASE = '';
let FRAMES = [];
let DIST_DATA, M_ANGLE, G_ANGLE, M_HALTUNG, G_HALTUNG;
let M_ACC, G_ACC, M_STEPS, G_STEPS, M_VEL, G_VEL, M_PATH, G_PATH;

// Load data from server
async function loadServerData() {
    const resp = await fetch(MEDIA_BASE + '/data.json');
    FRAMES = await resp.json();
    const mResp = await fetch(MEDIA_BASE + '/metrics.json');
    const metrics = await mResp.json();
    DIST_DATA = metrics.dist;
    M_ANGLE = metrics.m_angle; G_ANGLE = metrics.g_angle;
    M_HALTUNG = metrics.m_haltung; G_HALTUNG = metrics.g_haltung;
    M_ACC = metrics.m_acc; G_ACC = metrics.g_acc;
    M_STEPS = metrics.m_steps; G_STEPS = metrics.g_steps;
    M_VEL = metrics.m_vel; G_VEL = metrics.g_vel;
    M_PATH = metrics.m_path; G_PATH = metrics.g_path;
    drawFrame();
}
loadServerData();
"""

    html = f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0">
<title>Fecht-Analyzer Live</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:100vh; padding:12px; }}
.player-wrapper {{ max-width:960px; margin:0 auto; position:relative; background:#000; border-radius:10px; overflow:hidden; }}
.player-wrapper video {{ width:100%; display:block; }}
.player-wrapper canvas {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:6px; padding:10px 12px; background:#161b22; border:1px solid #30363d; border-radius:8px; margin-bottom:10px; align-items:center; }}
.toolbar .toggle {{ display:flex; align-items:center; gap:4px; padding:5px 10px; border-radius:6px; cursor:pointer; font-size:12px; font-weight:500; border:1px solid #30363d; background:#21262d; color:#c9d1d9; user-select:none; transition:all .15s; }}
.toolbar .toggle.on {{ border-color:#58a6ff; }}
.toolbar .toggle .dot {{ width:10px;height:10px;border-radius:50%; }}
.toolbar .play-btn {{ padding:5px 16px; border-radius:6px; cursor:pointer; font-size:13px; font-weight:600; border:1px solid #2ea043; background:#238636; color:#fff; }}
.toolbar .play-btn:hover {{ background:#2ea043; }}
.toolbar .time-display {{ font-size:13px; font-weight:500; color:#8b949e; margin-left:auto; font-variant-numeric:tabular-nums; }}
.video-stats {{ display:flex; flex-wrap:wrap; gap:8px; padding:8px 12px; background:#161b22; border:1px solid #30363d; border-top:none; border-radius:0 0 8px 8px; }}
.video-stats .stat {{ text-align:center; padding:2px 12px; }}
.video-stats .stat .val {{ font-size:16px; font-weight:600; }}
.video-stats .stat .lbl {{ font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:.3px; }}
.timebar-wrapper {{ padding:10px 0; }}
input[type="range"] {{ width:100%; height:6px; -webkit-appearance:none; appearance:none; background:#21262d; border-radius:3px; outline:none; }}
input[type="range"]::-webkit-slider-thumb {{ -webkit-appearance:none; width:16px; height:16px; border-radius:50%; background:#58a6ff; cursor:pointer; border:2px solid #0d1117; }}
input[type="range"]::-moz-range-thumb {{ width:16px; height:16px; border-radius:50%; background:#58a6ff; cursor:pointer; border:2px solid #0d1117; }}
.charts-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px; }}
.chart-cell {{ background:#161b22; border:1px solid #30363d; border-radius:6px; padding:6px; min-height:180px; }}
.charts-title {{ font-size:14px; font-weight:600; color:#c9d1d9; margin:12px 0 4px; }}
@media (max-width:700px) {{ .charts-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>

<div class="player-wrapper">
  <video id="vid" preload="auto" playsinline controls muted autoplay>
    {video_src_block}
  </video>
  <canvas id="overlay"></canvas>
</div>

<div class="toolbar">
  <button class="play-btn" id="playBtn">▶ Abspielen</button>
  <div class="toggle on" data-layer="skeleton"><span class="dot" style="background:#00ff88"></span> Skelett</div>
  <div class="toggle on" data-layer="distance"><span class="dot" style="background:#00ccff"></span> Distanz</div>
  <div class="toggle on" data-layer="angle"><span class="dot" style="background:#ffaa00"></span> Winkel</div>
  <div class="toggle on" data-layer="time"><span class="dot" style="background:#8b949e"></span> Zeit</div>
  <div class="time-display" id="timeDisplay">00:00.0 / 00:00.0</div>
</div>

<div class="timebar-wrapper">
  <input type="range" id="timeSlider" min="0" max="{duration}" step="0.1" value="0">
</div>

<div class="video-stats">
  <div class="stat"><div class="val" id="statDist">-</div><div class="lbl">Distanz</div></div>
  <div class="stat"><div class="val" id="statAngle">-</div><div class="lbl">Winkel M/G</div></div>
  <div class="stat"><div class="val" id="statSteps">0</div><div class="lbl">Schritte</div></div>
  <div class="stat"><div class="val" id="statAcc">-</div><div class="lbl">Max Beschl</div></div>
</div>

<div class="charts-title">📊 Metriken</div>
<div class="charts-grid" id="chartsGrid"></div>

<script>
// ===== DATA =====
{data_block}
const DURATION = {duration};
const FPS = {fps};
const VW = {vw};
const VH = {vh};

// ===== VIDEO & CANVAS =====
const vid = document.getElementById('vid');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
canvas.width = VW;
canvas.height = VH;

// ===== TOGGLES =====
const layers = {{ skeleton: true, distance: true, angle: true, time: true }};
document.querySelectorAll('.toggle').forEach(el => {{
  el.addEventListener('click', () => {{
    const key = el.dataset.layer;
    layers[key] = !layers[key];
    el.classList.toggle('on');
    drawFrame();
  }});
}});

// ===== PLAY BUTTON =====
const playBtn = document.getElementById('playBtn');
playBtn.addEventListener('click', () => {{
  if (vid.paused) {{ vid.play(); playBtn.textContent = '⏸ Pause'; }}
  else {{ vid.pause(); playBtn.textContent = '▶ Abspielen'; }}
}});
vid.addEventListener('pause', () => playBtn.textContent = '▶ Abspielen');
vid.addEventListener('play', () => playBtn.textContent = '⏸ Pause');

// ===== SKELETON DEFINITION =====
const SKEL = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

// ===== DRAW FRAME =====
function drawFrame() {{
  const t = vid.currentTime;
  // Find nearest frame
  let fi = 0;
  for (let i = 0; i < FRAMES.length; i++) {{
    if (FRAMES[i].t >= t) {{ fi = i; break; }}
  }}
  if (fi >= FRAMES.length) fi = FRAMES.length - 1;
  const f = FRAMES[fi];
  if (!f) return;

  ctx.clearRect(0, 0, VW, VH);

  // Helper: get keypoint
  function getKP(flat, idx) {{
    if (!flat) return null;
    const x = flat[idx*2], y = flat[idx*2+1];
    return (x > 0 && y > 0) ? [x, y] : null;
  }}

  function drawPerson(flat, color, label) {{
    if (!flat) return;
    // Skeleton
    if (layers.skeleton) {{
      for (const [a,b] of SKEL) {{
        const p1 = getKP(flat, a), p2 = getKP(flat, b);
        if (p1 && p2) {{
          ctx.beginPath(); ctx.moveTo(p1[0],p1[1]); ctx.lineTo(p2[0],p2[1]);
          ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
        }}
      }}
      for (let i = 0; i < 17; i++) {{
        const p = getKP(flat, i);
        if (p) {{
          ctx.beginPath(); ctx.arc(p[0],p[1], 3, 0, Math.PI*2);
          ctx.fillStyle = color; ctx.fill();
          ctx.strokeStyle = '#000'; ctx.lineWidth = 1; ctx.stroke();
        }}
      }}
      // Label
      const nose = getKP(flat, 0);
      if (nose) {{
        ctx.font = 'bold 11px sans-serif'; ctx.fillStyle = color; ctx.textAlign = 'center';
        ctx.fillText(label, nose[0], nose[1] - 14);
      }}
    }}

    // Distance (from Michael)
    if (layers.distance && label === 'Michael') {{
      const mHip = getMid(flat, 11, 12);
      if (mHip && f.g) {{
        const gHip = getMid(f.g, 11, 12);
        if (gHip) {{
          const dx = mHip[0] - gHip[0], dy = mHip[1] - gHip[1];
          const d = Math.hypot(dx, dy);
          ctx.font = 'bold 20px sans-serif'; ctx.fillStyle = '#00ccff'; ctx.textAlign = 'right';
          ctx.fillText(Math.round(d / 0.65) + ' cm', VW - 12, 30);
          // Line between hips
          ctx.beginPath(); ctx.moveTo(mHip[0],mHip[1]); ctx.lineTo(gHip[0],gHip[1]);
          ctx.strokeStyle = 'rgba(0,204,255,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([4,4]); ctx.stroke(); ctx.setLineDash([]);
        }}
      }}
    }}

    // Wrist angle
    if (layers.angle) {{
      // Right arm angle (Michael), Left arm (opponent)
      const s = label === 'Michael' ? [getKP(flat,6),getKP(flat,8),getKP(flat,10)] : [getKP(flat,5),getKP(flat,7),getKP(flat,9)];
      if (s[0] && s[1] && s[2]) {{
        const v1 = [s[0][0]-s[1][0], s[0][1]-s[1][1]];
        const v2 = [s[2][0]-s[1][0], s[2][1]-s[1][1]];
        const n1 = Math.hypot(v1[0],v1[1]), n2 = Math.hypot(v2[0],v2[1]);
        if (n1 > 1 && n2 > 1) {{
          const cosA = Math.max(-1, Math.min(1, (v1[0]*v2[0]+v1[1]*v2[1])/(n1*n2)));
          const deg = Math.round(Math.acos(cosA) * 180 / Math.PI);
          ctx.font = '12px sans-serif'; ctx.fillStyle = '#ffaa00'; ctx.textAlign = 'center';
          ctx.fillText(deg + '°', s[1][0], s[1][1] - 8);
        }}
      }}
    }}
  }}

  function getMid(flat, i1, i2) {{
    const a = getKP(flat, i1), b = getKP(flat, i2);
    return (a && b) ? [(a[0]+b[0])/2, (a[1]+b[1])/2] : null;
  }}

  drawPerson(f.m, '#00ff88', 'Michael');
  drawPerson(f.g, '#ff4466', 'Gegner');

  // Timestamp
  if (layers.time) {{
    const mins = Math.floor(t/60), secs = Math.floor(t%60), ds = Math.floor((t%1)*10);
    const ts = String(mins).padStart(2,'0') + ':' + String(secs).padStart(2,'0') + '.' + ds;
    ctx.font = '13px monospace'; ctx.fillStyle = 'rgba(200,200,200,0.7)'; ctx.textAlign = 'left';
    ctx.fillText(ts, 10, VH - 12);
  }}

  // Update stats
  const mHip = getMid(f.m, 11, 12);
  const gHip = getMid(f.g, 11, 12);
  if (mHip && gHip) {{
    const d = Math.hypot(mHip[0]-gHip[0], mHip[1]-gHip[1]);
    document.getElementById('statDist').textContent = Math.round(d / 0.65) + 'cm';
  }}
  document.getElementById('statAngle').textContent = (M_ANGLE[fi] || 0).toFixed(0) + '/' + (G_ANGLE[fi] || 0).toFixed(0);
  document.getElementById('statAcc').textContent = (M_ACC[fi-2] || 0).toFixed(0);
}}

// ===== TIME UPDATE =====
let isSeeking = false;

vid.addEventListener('timeupdate', () => {{
  document.getElementById('timeSlider').value = vid.currentTime;
  document.getElementById('timeDisplay').textContent = formatTime(vid.currentTime) + ' / ' + formatTime(DURATION);
  if (!isSeeking) drawFrame();
}});

vid.addEventListener('seeking', () => {{ isSeeking = true; drawFrame(); }});
vid.addEventListener('seeked', () => {{ isSeeking = false; }});

// ===== SLIDER =====
document.getElementById('timeSlider').addEventListener('input', (e) => {{
  const t = parseFloat(e.target.value);
  vid.currentTime = t;
  document.getElementById('timeDisplay').textContent = formatTime(t) + ' / ' + formatTime(DURATION);
  drawFrame();
  // Update chart vlines
  chartTimes.forEach(cb => cb(t));
}});

function formatTime(s) {{
  const m = Math.floor(s/60), sec = Math.floor(s%60), ds = Math.floor((s%1)*10);
  return String(m).padStart(2,'0') + ':' + String(sec).padStart(2,'0') + '.' + ds;
}}

// ===== PLOTLY CHARTS =====
const chartTimes = [];

function makeChart(containerId, title, xData, yData, color, yTitle) {{
  const steps = yData.length;
  const times = Array.from({{length: steps}}, (_,i) => i * (DURATION / (steps-1 || 1)));
  const trace = {{
    x: times, y: yData, type: 'scatter', mode: 'lines',
    line: {{color: color, width: 1.5}}, name: title,
  }};
  const layout = {{
    title: {{text: title, font: {{size: 12, color: '#c9d1d9'}}, x: 0.02}},
    paper_bgcolor: '#161b22', plot_bgcolor: '#161b22',
    font: {{color: '#8b949e', size: 10}},
    xaxis: {{gridcolor: '#21262d', showticklabels: false, showline: true, linecolor: '#30363d', zeroline: false}},
    yaxis: {{gridcolor: '#21262d', title: yTitle || '', color: '#8b949e', showline: true, linecolor: '#30363d', zeroline: false}},
    margin: {{l: 40, r: 10, t: 24, b: 20}},
    hovermode: 'x unified',
    dragmode: false,
    shapes: [],
    showlegend: false,
  }};
  const config = {{displayModeBar: false, responsive: true}};
  Plotly.newPlot(containerId, [trace], layout, config);

  // Click handler - jump video
  document.getElementById(containerId).on('plotly_click', (data) => {{
    if (data.points.length > 0) {{
      const t = data.points[0].x;
      vid.currentTime = t;
      document.getElementById('timeSlider').value = t;
      document.getElementById('timeDisplay').textContent = formatTime(t) + ' / ' + formatTime(DURATION);
      drawFrame();
    }}
  }});

  // VLine updater
  const vlineUpdater = (t) => {{
    const update = {{shapes: [{{
      type: 'line', x0: t, x1: t, y0: 0, y1: 1, yref: 'paper',
      line: {{color: '#58a6ff', width: 1.5, dash: 'dot'}}
    }}]}};
    Plotly.relayout(containerId, update);
  }};
  chartTimes.push(vlineUpdater);
}}

// Build charts grid
const chartSpecs = [
  ['ch1', 'Distanz', DIST_DATA, '#00ccff', 'cm'],
  ['ch2', 'Winkel M', M_ANGLE, '#00ff88', '°'],
  ['ch3', 'Winkel G', G_ANGLE, '#ff4466', '°'],
  ['ch4', 'Haltung M', M_HALTUNG, '#00ff88', '°'],
  ['ch5', 'Haltung G', G_HALTUNG, '#ff4466', '°'],
  ['ch6', 'Beschl. M', M_ACC, '#00ff88', 'px/s²'],
  ['ch7', 'Beschl. G', G_ACC, '#ff4466', 'px/s²'],
  ['ch8', 'Sync M', M_VEL, '#00ff88', 'px/s'],
  ['ch9', 'Sync G', G_VEL, '#ff4466', 'px/s'],
];

const grid = document.getElementById('chartsGrid');
chartSpecs.forEach(([id, name, data, color, unit]) => {{
  const cell = document.createElement('div');
  cell.className = 'chart-cell';
  cell.id = id;
  grid.appendChild(cell);
  // Delay rendering for perf
  setTimeout(() => makeChart(id, name, null, data, color, unit), 10);
}});

// ===== INITIAL DRAW =====
vid.addEventListener('loadedmetadata', () => {{
  drawFrame();
  document.getElementById('timeDisplay').textContent = '00:00.0 / ' + formatTime(DURATION);
}});

// Mobile: handle touch play
if ('ontouchstart' in window) {{
  vid.addEventListener('touchstart', () => {{
    if (vid.paused) vid.play();
  }});
}}
</script>
</body>
</html>"""

    return html


# === STREAMLIT UI ===

def main():
    st.markdown("""
    <style>
    .stApp { background: #0d1117; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .stProgress > div > div > div > div { background: #58a6ff; }
    h1, h2, h3 { color: #c9d1d9 !important; }
    .st-emotion-cache-1y4p8pa { max-width: 100%; padding: 1rem; }
    iframe { width: 100% !important; min-height: 800px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🤺 Fecht-Analyzer")
    st.caption("YOLOv8m-Pose | 9 Metriken | Live-Video-Player mit synchronisierten Charts")

    with st.sidebar:
        st.header("Video-Quelle")
        input_mode = st.radio("Quelle", ["Datei-Upload", "Lokaler Pfad"], horizontal=True)
        video_path = None

        if input_mode == "Datei-Upload":
            uploaded = st.file_uploader("Video hochladen", type=["mp4", "mov", "avi", "mkv"])
            if uploaded:
                tmp = Path(tempfile.gettempdir()) / "fencing_upload.mp4"
                with open(tmp, "wb") as f:
                    f.write(uploaded.getbuffer())
                video_path = tmp
        else:
            path_str = st.text_input("Pfad zum Video",
                value="C:\\Users\\micha\\Desktop\\Doha 2026\\Veneis-SUI.mp4")
            if path_str:
                p = Path(path_str.strip('"').strip("'"))
                if p.exists():
                    video_path = p
                    st.success(f"Gefunden: {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
                else:
                    st.error("Datei nicht gefunden")

        if video_path:
            # Nur setzen wenn noch nicht in session_state
            vp_key = str(video_path.resolve())
            if st.session_state.get("video_path_key") != vp_key:
                cap = cv2.VideoCapture(str(video_path))
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                cap.release()
                vid_dur = total_f / vid_fps if vid_fps > 0 else 0
                vid_name = Path(video_path).name
                st.caption(f"\U0001F4F9 {vid_name} — {vid_dur:.0f}s @ {vid_fps:.0f}fps")

                st.session_state["video_path"] = video_path
                st.session_state["video_path_key"] = vp_key
                st.session_state["vid_dur"] = vid_dur
                st.session_state["vid_fps"] = vid_fps
                st.session_state["vid_name"] = vid_name
                st.rerun()
            else:
                st.caption(f"\U0001F4F9 {st.session_state['vid_name']} — {st.session_state['vid_dur']:.0f}s @ {st.session_state['vid_fps']:.0f}fps")

    if st.session_state.get("analysis_running"):
        show_analysis_progress()
    elif "result" in st.session_state:
        display_player(st.session_state["result"], st.session_state["clip_path"])
    elif "video_path" in st.session_state:
        display_video_viewer()
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("\U0001F4C1 Video auswählen und Analyse starten")
            st.markdown("""
            **9 Metriken — Live im Player:**
            1. Distanz  
            2. Waffenarm-Winkel  
            3. Lunge-Tiefe  
            4. Bewegungs-Pfad  
            5. Körperhaltung  
            6. Beschleunigung  
            7. Schritt-Rhythmus  
            8. Synchronisierung  
            9. Heatmap  

            **Toggle:** Skelett, Distanz, Winkel, Zeit — ein/aus per Klick
            """)


WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker_analyze.py")


def run_analysis(video_path, start_sec, clip_duration):
    """Extrahiere Clip + starte Worker-Subprocess. Fragment pollt Ergebnis-Datei."""
    clip_progress = st.progress(0, text="Extrahiere Clip...")
    timestamp = int(time.time())
    clip_path = Path(tempfile.gettempdir()) / f"fencing_clip_{timestamp}.mp4"
    result_path = Path(tempfile.gettempdir()) / f"fencing_result_{timestamp}.json"

    ok = extract_clip(video_path, clip_path, start_sec, clip_duration)
    if not ok:
        st.error("Clip-Extraktion fehlgeschlagen")
        return
    clip_size = clip_path.stat().st_size / 1e6
    clip_progress.progress(100, text=f"Clip bereit ({clip_size:.1f} MB)")

    # Worker im Subprocess starten (complett autark, blockiert UI nicht)
    subprocess.Popen(
        [
            sys.executable,
            WORKER_SCRIPT,
            str(clip_path),
            str(result_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0,
    )

    st.session_state["analysis_result_path"] = str(result_path)
    st.session_state["analysis_running"] = True
    st.rerun()


@st.fragment(run_every=3)
def show_analysis_progress():
    """Pollt alle 3s ob der Worker fertig ist. UI bleibt voll responsiv."""
    result_path = Path(st.session_state.get("analysis_result_path", ""))

    st.subheader("Analysiere Video...")
    st.caption("YOLOv8m-Pose verarbeitet Frames im Hintergrund")

    done_marker = Path(str(result_path) + ".done")
    if not done_marker.exists() and not result_path.exists():
        # Laeuft noch
        st.progress(0.4, text="Frame-Erkennung und Metrik-Berechnung...")
        st.info("""
        **Status:** Worker lauft im separaten Prozess
        - Keypoint-Extraktion pro Frame
        - Metrik-Berechnung (9 Metriken)
        - UI bleibt wahrenddessen voll nutzbar
        """)
        st.markdown("<p style='color:#8b949e; font-size:12px;'>Polling alle 3s - automatische Umschaltung bei Fertigstellung</p>", unsafe_allow_html=True)
        return

    if result_path.exists():
        with open(result_path) as f:
            data = json.load(f)

        if "error" in data:
            st.error(f"Analyse fehlgeschlagen: {data['error']}")
            st.code(data.get("traceback", ""))
            st.session_state["analysis_running"] = False
            return

        st.success("Analyse abgeschlossen! Lade Live-Player...")
        st.session_state["result"] = data
        st.session_state["clip_path"] = result_path.parent / f"fencing_clip_{Path(str(result_path)).stem.split('_')[-1]}.mp4"
        st.session_state["analysis_running"] = False
        st.rerun()


def display_player(result, clip_path):
    """Render the live video player with canvas overlay and synced charts."""
    clip_size_mb = clip_path.stat().st_size / (1024 * 1024)
    
    if clip_size_mb >= 30:
        # Large video: start media server
        s = result["summary"]
        metrics_payload = {
            "dist": [d["cm"] for d in result["m1_dist"]],
            "m_angle": [d["deg"] for d in result["m2_m_angle"]],
            "g_angle": [d["deg"] for d in result["m2_g_angle"]],
            "m_haltung": [d["deg"] for d in result["m5_m_tilt"]],
            "g_haltung": [d["deg"] for d in result["m5_g_tilt"]],
            "m_acc": [d["acc"] for d in result["m6_m_acc"]],
            "g_acc": [d["acc"] for d in result["m6_g_acc"]],
            "m_steps": result["m7_m_steps"],
            "g_steps": result["m7_g_steps"],
            "m_vel": result["m8_vel_m"],
            "g_vel": result["m8_vel_g"],
            "m_path": [{"x": p["x"], "y": p["y"]} for p in result["m4_m_path"]],
            "g_path": [{"x": p["x"], "y": p["y"]} for p in result["m4_g_path"]],
        }
        base_url = start_media_server(clip_path, result["frame_data"], metrics_payload)
        html_content = build_live_player_html(result, clip_path, mode="server")
        # Replace placeholder base URL
        html_content = html_content.replace("const MEDIA_BASE = '';", f"const MEDIA_BASE = '{base_url}';")
    else:
        html_content = build_live_player_html(result, clip_path, mode="embed")
    
    st_html(html_content, height=950, scrolling=False)


def display_video_viewer():
    """Zeigt Video-Player mit dualem Slider (Start/Ende) und Analysieren-Button - alles native Streamlit."""
    vp = st.session_state["video_path"]
    dur = st.session_state["vid_dur"]
    fps = st.session_state["vid_fps"]
    name = st.session_state["vid_name"]

    st.subheader(f"\U0001F3AC {name}")

    # Native Streamlit Video-Komponente
    vid_data = open(vp, "rb").read()
    st.video(vid_data, start_time=0)

    # Timeline-Bereichsauswahl als native Streamlit-UI
    st.markdown("<hr style='border-color:#30363d; margin:6px 0;'>", unsafe_allow_html=True)

    col_s, col_e = st.columns([1, 1])
    with col_s:
        start_val = st.number_input(
            "Start (Sekunden)",
            min_value=0.0,
            max_value=dur,
            value=0.0,
            step=0.5,
            key="vid_start_inp",
            format="%.1f"
        )
    with col_e:
        end_val = st.number_input(
            "Ende (Sekunden)",
            min_value=0.0,
            max_value=dur,
            value=min(30.0, dur),
            step=0.5,
            key="vid_end_inp",
            format="%.1f"
        )

    # Korrektur falls Start > Ende
    if start_val > end_val:
        start_val, end_val = end_val, start_val

    clip_dur = end_val - start_val
    clip_frames = int(clip_dur * fps)

    # Info-Bereich mit 4 Metriken nebeneinander
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Start", f"{start_val:.1f}s")
    c2.metric("Ende", f"{end_val:.1f}s")
    c3.metric("Dauer", f"{clip_dur:.1f}s")
    c4.metric("Frames", f"{clip_frames}")

    st.markdown(f"<p style='color:#8b949e; font-size:13px;'>Gesamt: {dur:.0f}s @ {fps:.0f}fps</p>", unsafe_allow_html=True)

    # Analysieren-Button
    if clip_dur >= 3:
        if st.button("\U0001F3AF Bereich analysieren", type="primary", use_container_width=True):
            run_analysis(st.session_state["video_path"], start_val, clip_dur)
    else:
        st.warning("Bereich muss mindestens 3s lang sein")
        st.button("\U0001F3AF Bereich analysieren", disabled=True, use_container_width=True)


if __name__ == "__main__":
    main()
