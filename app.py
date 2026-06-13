"""
Fencing Analyzer — Streamlit App
Fecht-Video-Analyse mit YOLOv8m-Pose + Interaktivem Live-Video-Player

Usage:
  streamlit run C:\\Users\\micha\\Desktop\\fencing_analyzer\\app.py
"""
import streamlit as st
st.set_page_config(page_title="Fecht-Analyzer", layout="wide", page_icon="\U0001F93A")

# === IMPORTS ===
import os, sys, json, math, time, shutil, subprocess, tempfile, struct, base64, threading, re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import cv2
from PIL import Image
from ultralytics import YOLO
from streamlit.components.v1 import html as st_html

# Report-Generator
from report_generator import generate_report

# Shared UI helpers
from ui_media_server import start_media_server, stop_media_server

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

def _ffmpeg():
    """Return ffmpeg path. Works on Windows dev and Linux Docker."""
    if sys.platform == "win32":
        p = Path.home() / "AppData/Local/hermes/hermes-agent/venv/Scripts/ffmpeg.exe"
        if p.exists():
            return str(p)
    return "ffmpeg"

def extract_clip(video_path, output_path, start_sec=0, duration_sec=15, target_width=640):
    cmd = [
        _ffmpeg(),
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-i", str(video_path),
        "-vf", f"scale={target_width}:-2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an",
        "-y", str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, timeout=1800)
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
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:700px; padding:12px; }}
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
  // O(1) Frame-Suche: direkt per Index
  const fi = Math.min(Math.round(t * FPS), FRAMES.length - 1);
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
  }} else {{
    document.getElementById('statDist').textContent = '- cm';
  }}
  document.getElementById('statAngle').textContent = (M_ANGLE[fi] || 0).toFixed(0) + '° / ' + (G_ANGLE[fi] || 0).toFixed(0) + '°';
  document.getElementById('statSteps').textContent = 'M=' + (M_STEPS ? M_STEPS[fi]?.step || 0 : 0) + ' G=' + (G_STEPS ? G_STEPS[fi]?.step || 0 : 0);
  document.getElementById('statAcc').textContent = 'M=' + (M_ACC ? (M_ACC[fi]?.acc || 0).toFixed(0) : '0') + ' G=' + (G_ACC ? (G_ACC[fi]?.acc || 0).toFixed(0) : '0');
}}

// ===== ANIMATION LOOP (requestAnimationFrame statt timeupdate fur flussiges Skelett) =====
let rafRunning = false;

function animationLoop() {{
  if (vid.paused) {{ rafRunning = false; return; }}
  const t = vid.currentTime;
  document.getElementById('timeSlider').value = t;
  document.getElementById('timeDisplay').textContent = formatTime(t) + ' / ' + formatTime(DURATION);
  drawFrame();
  requestAnimationFrame(animationLoop);
}}

vid.addEventListener('play', () => {{
  if (!rafRunning) {{ rafRunning = true; requestAnimationFrame(animationLoop); }}
}});
// Falls Autoplay ohne 'play'-Event startet
setTimeout(() => {{
  if (!vid.paused && !rafRunning) {{ rafRunning = true; requestAnimationFrame(animationLoop); }}
}}, 500);

// ===== SLIDER =====
document.getElementById('timeSlider').addEventListener('input', (e) => {{
  const t = parseFloat(e.target.value);
  vid.currentTime = t;
  document.getElementById('timeDisplay').textContent = formatTime(t) + ' / ' + formatTime(DURATION);
  drawFrame();
}});

function formatTime(s) {{
  const m = Math.floor(s/60), sec = Math.floor(s%60), ds = Math.floor((s%1)*10);
  return String(m).padStart(2,'0') + ':' + String(sec).padStart(2,'0') + '.' + ds;
}}

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

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.code === 'Space') {{
    e.preventDefault();
    if (vid.paused) {{ vid.play(); }} else {{ vid.pause(); }}
  }}
  if (e.code === 'ArrowRight') {{
    e.preventDefault();
    vid.currentTime = Math.min(vid.currentTime + 1, DURATION);
  }}
  if (e.code === 'ArrowLeft') {{
    e.preventDefault();
    vid.currentTime = Math.max(vid.currentTime - 1, 0);
  }}
}});
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
    
    # GPU-Check
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else ""
    except:
        gpu_available = False
        gpu_name = ""
    gpu_label = f"GPU: {gpu_name}" if gpu_available else "CPU (PyTorch)"
    st.caption(f"YOLOv8m-Pose | 15 Metriken | Live-Video-Player | Tracking v2 | {gpu_label}")

    with st.sidebar:
            st.header("Analyse-Modus")
            analysis_mode = st.radio(
                "Modus wählen",
                ["Schnell-Clip (15-60s)", "Full-Length (komplettes Gefecht)"],
                horizontal=False,
                key="analysis_mode_radio",
            )

            st.divider()
            st.header("Video-Quelle")
            input_mode = st.radio("Quelle", ["Datei-Upload", "Lokaler Pfad", "YouTube-Link"], horizontal=True)
            video_path = None

            if input_mode == "YouTube-Link":
                yt_url = st.text_input("YouTube/Streamable-Link", placeholder="https://www.youtube.com/watch?v=...")
                if yt_url and yt_url.strip():
                    yt_url = yt_url.strip()
                    if not any(x in yt_url.lower() for x in ['youtube.com', 'youtu.be', 'streamable.com', 'vimeo.com']):
                        st.error("Bitte einen gültigen YouTube-, Streamable- oder Vimeo-Link eingeben")
                    else:
                        dl_progress = st.progress(0, text="Lade Video herunter...")
                        try:
                            import subprocess as sp
                            sp.run([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"],
                                   capture_output=True, timeout=60)
                            dl_progress.progress(25, text="yt-dlp bereit...")
                            timestamp = int(time.time())
                            out_path = Path(tempfile.gettempdir()) / f"fencing_yt_{timestamp}.mp4"
                            cmd = [sys.executable, "-m", "yt_dlp",
                            "-f", "best[height<=720]",
                            "--js-runtimes", "deno",
                            "-o", str(out_path), "--no-playlist", "--quiet",
                            yt_url]
                            dl_progress.progress(50, text="YouTube-Download läuft...")
                            result = sp.run(cmd, capture_output=True, text=True, timeout=600)
                            dl_progress.progress(90, text="Verarbeite...")
                            if out_path.exists() and out_path.stat().st_size > 100000:
                                video_path = out_path
                                st.success(f"YouTube-Video geladen: {out_path.stat().st_size/1e6:.0f} MB")
                                dl_progress.progress(100, text="Bereit!")
                            else:
                                err = result.stderr.lower()
                                if "sign in" in err:
                                    st.error("🔒 YouTube fordert Login für dieses Video (privat/altersbeschränkt). Nutze Upload oder lokalen Pfad.")
                                else:
                                    st.error(f"Download fehlgeschlagen. {result.stderr[:200]}")
                                dl_progress.empty()

                        except Exception as e:
                            st.error(f"Fehler beim Download: {str(e)[:200]}")
                            dl_progress.empty()

            elif input_mode == "Datei-Upload":
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

    if st.session_state.get("analysis_running") and st.session_state.get("full_proc"):
        full_show_progress()
    elif st.session_state.get("analysis_running"):
        show_analysis_progress()
    elif st.session_state.get("compare_mode"):
        show_comparison()
    elif "result" in st.session_state:
        display_player(st.session_state["result"], st.session_state["clip_path"])
    elif "video_path" in st.session_state:
        # Mode-aware main view
        if "Full-Length" in st.session_state.get("analysis_mode_radio", ""):
            main_view = st.radio("Ansicht", ["Konfiguration", "DB durchsuchen"],
                                    horizontal=True, key="fl_main_view")
            if main_view == "Konfiguration":
                full_render_main(st.session_state["video_path"])
            else:
                full_browse_bouts()
        else:
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


from ui_full_length import (
    full_render_form,
    full_render_main,
    full_browse_bouts,
    full_show_progress,
    FULL_PROGRESS_FILE,
    FULL_DONE_FILE,
    FULL_ERROR_FILE,
)

def run_analysis(video_path, start_sec, clip_duration, label="Analyse"):
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
    start_time = time.time()
    proc = subprocess.Popen(
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
    st.session_state["analysis_start_time"] = start_time
    st.session_state["analysis_label"] = label
    st.rerun()


@st.fragment(run_every=3)
def show_analysis_progress():
    """Pollt alle 3s ob der Worker fertig ist. UI bleibt voll responsiv."""
    result_path = Path(st.session_state.get("analysis_result_path", ""))

    # Worker-Timeout check (10 min)
    start_time = st.session_state.get("analysis_start_time", 0)
    if start_time and (time.time() - start_time) > 600:
        st.error("Analyse abgebrochen: Worker hat 10 Minuten überschritten. Kürzeren Bereich wählen oder GPU beschleunigen.")
        st.session_state["analysis_running"] = False
        return

    st.subheader("Analysiere Video...")
    st.caption("YOLOv8m-Pose | optimiertes Tracking v2 (Side-Constraint + Velocity-Interpolation + Smoothing)")

    done_marker = Path(str(result_path) + ".done")
    if not done_marker.exists() and not result_path.exists():
        elapsed = int(time.time() - start_time) if start_time else 0
        progress_val = min(0.95, elapsed / 120)
        st.progress(progress_val, text=f"Frame-Erkennung und Metrik-Berechnung... ({elapsed}s)")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Worker läuft** — {elapsed}s")
        with col2:
            est_remaining = max(1, 120 - elapsed)
            st.markdown(f"<p style='color:#8b949e; font-size:13px;'>Geschätzt: ~{est_remaining}s verbleibend</p>", unsafe_allow_html=True)
        return

    if result_path.exists():
        with open(result_path) as f:
            try:
                data = json.load(f)
            except (json.JSONDecodeError, Exception):
                st.error("Fehler beim Lesen der Analyse-Ergebnisse (korrupte Datei). Bitte erneut versuchen.")
                st.session_state["analysis_running"] = False
                return

        if "error" in data:
            st.error(f"Analyse fehlgeschlagen: {data['error']}")
            st.code(data.get("traceback", ""))
            st.session_state["analysis_running"] = False
            return

        # No-person check
        total_keypoints = sum(
            1 for f in data.get("frame_data", [])
            if f.get("m") is not None and any(f["m"][i] > 0 for i in range(0, 34, 2))
        )
        if total_keypoints < 5:
            st.warning("""
            **Keine Fechter erkannt.** Mögliche Ursachen:
            - Andere Kamera-Perspektive (seitlich statt frontal?)
            - Zu weit weg (Fechter zu klein im Bild)
            - Video enthält keinen Fechtkampf
            """)
            st.session_state["analysis_running"] = False
            return

        st.success("Analyse abgeschlossen! Lade Ergebnisse...")
        if st.session_state.get("analyze_second"):
            # This is the second analysis for comparison
            label = st.session_state.get("analysis_label", "Bereich 2")
            st.session_state["compare_results"] = [
                st.session_state.get("compare_first", {}).get("result"),
                data,
            ]
            st.session_state["compare_labels"] = [
                st.session_state.get("compare_first", {}).get("label", "Bereich 1"),
                label,
            ]
            st.session_state["compare_mode"] = True
            st.session_state["analyze_second"] = False
            st.session_state["analysis_running"] = False
        else:
            st.session_state["result"] = data
            st.session_state["clip_path"] = Path(str(result_path).replace("fencing_result_", "fencing_clip_").replace(".json", ".mp4"))
            st.session_state["analysis_running"] = False
        st.rerun()


def display_player(result, clip_path):
    """Zeigt Summary + Link zum Player (separater Tab) + native Plotly-Charts."""
    s = result["summary"]

    # 1. Media-Server starten (serviert Video + Player HTML + Daten)
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
        # Neue Metriken 9-15
        "m_hand_h": [d["px"] for d in result["m9_m_hand_h"]],
        "g_hand_h": [d["px"] for d in result["m9_g_hand_h"]],
        "m_ext": [d["px"] for d in result["m10_m_ext"]],
        "g_ext": [d["px"] for d in result["m10_g_ext"]],
        "m_stance": [d["px"] for d in result["m11_m_stance"]],
        "g_stance": [d["px"] for d in result["m11_g_stance"]],
        "expl": [d["cm_s"] for d in result["m12_expl"]],
        "m_head": [d["px"] for d in result["m13_m_head"]],
        "g_head": [d["px"] for d in result["m13_g_head"]],
        "touches": result["m14_touches"],
        "rhythm": result["m15_rhythm"],
    }
    player_html = build_live_player_html(result, clip_path, mode="server")
    base_url = start_media_server(clip_path, result["frame_data"], metrics_payload, None)
    MediaRequestHandler.player_html = player_html.replace(
        "const MEDIA_BASE = '';",
        "const MEDIA_BASE = '" + base_url + "';"
    )

    # 2. Player-Button + Summary (2 rows)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Distanz ⌀", f'{s.get("dist_avg", 0):.0f} cm')
    col2.metric("Winkel M/G", f'{s.get("m_angle_avg", 0):.0f}° / {s.get("g_angle_avg", 0):.0f}°')
    col3.metric("Schritte M/G", f'{s.get("m_steps", 0)} / {s.get("g_steps", 0)}')
    col4.metric("Korrelation", f'{s.get("correlation", 0):.2f}')

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Handhöhe M/G", f'{s.get("m_hand_h_avg", 0):.0f} / {s.get("g_hand_h_avg", 0):.0f} px')
    col6.metric("Arm-Streck M/G", f'{s.get("m_ext_avg", 0):.0f} / {s.get("g_ext_avg", 0):.0f} px')
    col7.metric("Standbreite M/G", f'{s.get("m_stance_avg", 0):.0f} / {s.get("g_stance_avg", 0):.0f} px')
    col8.metric("Touchés", f'{s.get("touches", 0)} ({s.get("touches_high", 0)} high)')

    # Report Button
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        # Auto-Kommentar
        auto_comment = f"""
Michael hielt durchschnittlich {s.get('dist_avg',0):.0f} cm Distanz • {"höhere" if s.get('m_angle_avg',0) > s.get('g_angle_avg',0) else "niedrigere"} Waffenhand als Gegner ({s.get('m_hand_h_avg',0):.0f} vs {s.get('g_hand_h_avg',0):.0f} px) • {"mehr" if s.get('m_steps',0) > s.get('g_steps',0) else "weniger"} Schritte ({s.get('m_steps',0)} vs {s.get('g_steps',0)}) • Korrelation {s.get('correlation',0):.2f} ({'reaktiv' if abs(s.get('lag_frames',0)) < 5 else f'führt an mit {abs(s.get("lag_frames",0))}f Vorsprung'}) • {s.get('touches_high',0)} high-confidence Touché-Kandidaten
"""
        st.markdown(f"<div style='color:#8b949e; font-size:12px; border:1px solid #30363d; border-radius:6px; padding:6px 10px;'>💬 {auto_comment}</div>", unsafe_allow_html=True)
    with col_r2:
        sub_cols = st.columns(2)
        with sub_cols[0]:
            if st.button("📄 PDF-Report", type="primary", use_container_width=True):
                with st.spinner("Generiere PDF..."):
                    try:
                        vname = st.session_state.get("vid_name", "Gefecht").replace(".mp4", "")
                        pdf_path, pdf_preview = generate_report(result, vname)
                        st.success(f"PDF erstellt: {Path(pdf_path).name}")
                        with open(pdf_path, "rb") as f:
                            st.download_button("📥 Download", f.read(), file_name=Path(pdf_path).name, mime="application/pdf", use_container_width=True)
                    except Exception as e:
                        st.error(f"PDF-Fehler: {e}")
        with sub_cols[1]:
            # Vergleichsmodus: "Zweiten Bereich analysieren"
            if st.button("➕ Zweiten Bereich analysieren", use_container_width=True):
                st.session_state["compare_first"] = {
                    "result": result,
                    "label": st.session_state.get("vid_name", "Bereich 1").replace(".mp4", ""),
                }
                del st.session_state["result"]
                st.session_state["analyze_second"] = True
                st.rerun()
            if st.button("📊 CSV Export", use_container_width=True):
                csv_lines = ["t,m_dist_cm,m_angle_deg,g_angle_deg,m_lunge_px,g_lunge_px,m_haltung_deg,g_haltung_deg,m_acc_gegner_acc,m_steps,g_steps,m_hand_h_px,g_hand_h_px,m_ext_px,g_ext_px,m_stance_px,g_stance_px,expl_cm_s,m_head_px,g_head_px,m_vel_px_s,g_vel_px_s"]
                n_frames = len(result["m1_dist"])
                for i in range(n_frames):
                    t = result["m1_dist"][i]["t"]
                    row = [str(t)]
                    # M1 dist
                    row.append(str(result["m1_dist"][i].get("cm", 0)))
                    # M2 angles
                    row.append(str(result["m2_m_angle"][i].get("deg", 0)) if i < len(result["m2_m_angle"]) else "0")
                    row.append(str(result["m2_g_angle"][i].get("deg", 0)) if i < len(result["m2_g_angle"]) else "0")
                    # M3 lunge
                    row.append(str(result["m3_m_lunge"][i].get("px", 0)) if i < len(result["m3_m_lunge"]) else "0")
                    row.append(str(result["m3_g_lunge"][i].get("px", 0)) if i < len(result["m3_g_lunge"]) else "0")
                    # M5 haltung
                    row.append(str(result["m5_m_tilt"][i].get("deg", 0)) if i < len(result["m5_m_tilt"]) else "0")
                    row.append(str(result["m5_g_tilt"][i].get("deg", 0)) if i < len(result["m5_g_tilt"]) else "0")
                    # M6 acc
                    row.append(str(result["m6_m_acc"][i].get("acc", 0)) if i < len(result["m6_m_acc"]) else "0")
                    row.append(str(result["m6_g_acc"][i].get("acc", 0)) if i < len(result["m6_g_acc"]) else "0")
                    # M7 steps
                    row.append(str(result["m7_m_steps"][i].get("step", 0)) if i < len(result["m7_m_steps"]) else "0")
                    row.append(str(result["m7_g_steps"][i].get("step", 0)) if i < len(result["m7_g_steps"]) else "0")
                    # M9 hand height
                    row.append(str(result["m9_m_hand_h"][i].get("px", 0)) if i < len(result["m9_m_hand_h"]) else "0")
                    row.append(str(result["m9_g_hand_h"][i].get("px", 0)) if i < len(result["m9_g_hand_h"]) else "0")
                    # M10 extension
                    row.append(str(result["m10_m_ext"][i].get("px", 0)) if i < len(result["m10_m_ext"]) else "0")
                    row.append(str(result["m10_g_ext"][i].get("px", 0)) if i < len(result["m10_g_ext"]) else "0")
                    # M11 stance
                    row.append(str(result["m11_m_stance"][i].get("px", 0)) if i < len(result["m11_m_stance"]) else "0")
                    row.append(str(result["m11_g_stance"][i].get("px", 0)) if i < len(result["m11_g_stance"]) else "0")
                    # M12 expl (shifted by 1)
                    row.append(str(result["m12_expl"][i].get("cm_s", 0)) if i < len(result["m12_expl"]) else "0")
                    # M13 head
                    row.append(str(result["m13_m_head"][i].get("px", 0)) if i < len(result["m13_m_head"]) else "0")
                    row.append(str(result["m13_g_head"][i].get("px", 0)) if i < len(result["m13_g_head"]) else "0")
                    # M8 velocities
                    row.append(str(result["m8_vel_m"][i]) if i < len(result["m8_vel_m"]) else "0")
                    row.append(str(result["m8_vel_g"][i]) if i < len(result["m8_vel_g"]) else "0")
                    csv_lines.append(",".join(row))
                csv_text = "\n".join(csv_lines)
                vname = st.session_state.get("vid_name", "Gefecht").replace(".mp4", "")
                st.download_button("📥 CSV herunterladen", csv_text, file_name=f"Fecht-Daten_{vname}.csv", mime="text/csv", use_container_width=True)

    st.markdown(
        f'<a href="{base_url}/player" target="_blank">'
        f'<button style="width:100%;padding:14px;background:#238636;border:1px solid #2ea043;'
        f'color:#fff;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;">'
        f'▶ Player öffnen (neuer Tab)</button></a>',
        unsafe_allow_html=True
    )

    # 3. CHARTS: 9 native Plotly-Charts in 3x3 Grid
    st.markdown("---")
    st.subheader("📊 Metriken")

    N = len(result["m1_dist"])
    times = [d["t"] for d in result["m1_dist"]]

    # 3x3 grid with columns
    def make_plotly(title, data, color, y_title, y_range=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=data, mode="lines", line=dict(color=color, width=1.5), name=title, hovertemplate="%{y:.1f}"))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color=C_TEXT), x=0.02),
            paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
            font=dict(color=C_MUTED, size=10),
            xaxis=dict(gridcolor=C_BORDER, showticklabels=False, zeroline=False, showline=True, linecolor=C_BORDER),
            yaxis=dict(gridcolor=C_BORDER, title=y_title, color=C_TEXT, zeroline=False, showline=True, linecolor=C_BORDER,
                       range=y_range if y_range else None),
            margin=dict(l=35, r=8, t=24, b=18),
            hovermode="x unified",
            showlegend=False,
            height=140,
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        return fig

    c1, c2, c3 = st.columns(3)
    dist_data = [d["cm"] for d in result["m1_dist"]]
    with c1:
        fig = make_plotly("Distanz", dist_data, C_BLUE, "cm", [0, max(dist_data)*1.15 or 200])
        # Add vertical touché lines
        for t in result.get("m14_touches", []):
            color = C_GREEN if t["who"] == "Michael" else C_RED if t["who"] == "Gegner" else "#ffaa00"
            fig.add_vline(x=t["t"], line=dict(color=color, width=1, dash="dot"),
                          annotation_text="🎯" if t["confidence"] == "high" else "",
                          annotation_position="top")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        m_ang = [d["deg"] for d in result["m2_m_angle"]]
        st.plotly_chart(make_plotly("Winkel M", m_ang, C_GREEN, "°", [0, 180]), use_container_width=True)
    with c3:
        g_ang = [d["deg"] for d in result["m2_g_angle"]]
        st.plotly_chart(make_plotly("Winkel G", g_ang, C_RED, "°", [0, 180]), use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        m_halt = [d["deg"] for d in result["m5_m_tilt"]]
        st.plotly_chart(make_plotly("Haltung M", m_halt, C_GREEN, "°"), use_container_width=True)
    with c5:
        g_halt = [d["deg"] for d in result["m5_g_tilt"]]
        st.plotly_chart(make_plotly("Haltung G", g_halt, C_RED, "°"), use_container_width=True)
    with c6:
        m_lunge = [d["px"] for d in result["m3_m_lunge"]]
        st.plotly_chart(make_plotly("Lunge M", m_lunge, C_GREEN, "px"), use_container_width=True)

    c7, c8, c9 = st.columns(3)
    with c7:
        m_acc = [d["acc"] for d in result["m6_m_acc"]]
        st.plotly_chart(make_plotly("Beschl. M", m_acc, C_GREEN, "px/s²"), use_container_width=True)
    with c8:
        g_acc = [d["acc"] for d in result["m6_g_acc"]]
        st.plotly_chart(make_plotly("Beschl. G", g_acc, C_RED, "px/s²"), use_container_width=True)
    with c9:
        sync_m = result.get("m8_vel_m", [0])
        sync_g = result.get("m8_vel_g", [0])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times[:len(sync_m)], y=sync_m, mode="lines", line=dict(color=C_GREEN, width=1), name="Michael", hovertemplate="%{y:.1f}"))
        fig.add_trace(go.Scatter(x=times[:len(sync_g)], y=sync_g, mode="lines", line=dict(color=C_RED, width=1), name="Gegner", hovertemplate="%{y:.1f}"))
        fig.update_layout(
            title=dict(text="Sync M/G", font=dict(size=12, color=C_TEXT), x=0.02),
            paper_bgcolor=C_CARD, plot_bgcolor=C_CARD, font=dict(color=C_MUTED, size=10),
            xaxis=dict(gridcolor=C_BORDER, showticklabels=False, zeroline=False),
            yaxis=dict(gridcolor=C_BORDER, title="px/s", color=C_TEXT, zeroline=False),
            margin=dict(l=35, r=8, t=24, b=18), hovermode="x unified", showlegend=False, height=140,
        )
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True)
        st.plotly_chart(fig, use_container_width=True)

    # Row 4: M9 Handhöhe, M10 Arm-Streckung, M11 Standbreite
    c10, c11, c12 = st.columns(3)
    with c10:
        m_hh = [d["px"] for d in result["m9_m_hand_h"]]
        st.plotly_chart(make_plotly("Handhöhe M", m_hh, C_GREEN, "px"), use_container_width=True)
    with c11:
        m_ext = [d["px"] for d in result["m10_m_ext"]]
        st.plotly_chart(make_plotly("Arm-Streck M", m_ext, C_GREEN, "px"), use_container_width=True)
    with c12:
        m_stance = [d["px"] for d in result["m11_m_stance"]]
        st.plotly_chart(make_plotly("Standbreite M", m_stance, C_GREEN, "px"), use_container_width=True)

    # Row 5: M12 Explosivität, M13 Head-Forward, M15 Rhythmus
    c13, c14, c15 = st.columns(3)
    with c13:
        expl = [d["cm_s"] for d in result["m12_expl"]]
        expl_times = [d["t"] for d in result["m12_expl"]]
        st.plotly_chart(make_plotly("Explosivität", expl, C_ACCENT, "cm/s"), use_container_width=True)
    with c14:
        m_head = [d["px"] for d in result["m13_m_head"]]
        st.plotly_chart(make_plotly("Head Fwd M", m_head, C_GREEN, "px"), use_container_width=True)
    with c15:
        if result["m15_rhythm"]:
            rhy_t = [r["t"] for r in result["m15_rhythm"]]
            rhy_f = [r["freq_hz"] for r in result["m15_rhythm"]]
            st.plotly_chart(make_plotly("Rhythmus Hz", rhy_f, "#ffaa00", "Hz"), use_container_width=True)
        else:
            st.caption("Rhythmus: zu wenig Daten")

    # Row 6: Touché-Timeline
    if result["m14_touches"]:
        st.markdown("---")
        st.subheader("🎯 Touché-Kandidaten")
        high_touches = [t for t in result["m14_touches"] if t["confidence"] == "high"]
        medium_touches = [t for t in result["m14_touches"] if t["confidence"] == "medium"]

        def render_touche_table(touches_list):
            touch_data = []
            for t in touches_list:
                who_emoji = "🟢" if t["who"] == "Michael" else "🔴" if t["who"] == "Gegner" else "🟡"
                touch_data.append({
                    "Zeit": f'{t["t"]:.1f}s',
                    "Wer": f'{who_emoji} {t["who"]}',
                    "Distanz": f'{t["dist_cm"]:.0f} cm',
                    "Ext M": f'{t["ext_m"]:.0f} px',
                    "Ext G": f'{t["ext_g"]:.0f} px',
                    "Löst sich": "✓" if t["resolves"] else "✗",
                })
            st.dataframe(touch_data, use_container_width=True, hide_index=True)

        if high_touches:
            st.markdown(f"**{len(high_touches)} high-confidence Treffer**")
            render_touche_table(high_touches)
        if medium_touches:
            with st.expander(f"⚠️ {len(medium_touches)} medium-confidence Kandidaten (Details)"):
                render_touche_table(medium_touches)


def show_comparison():
    """Zeigt zwei Analysen nebeneinander mit Deltas."""
    results = st.session_state.get("compare_results", [])
    labels = st.session_state.get("compare_labels", ["Bereich 1", "Bereich 2"])
    
    if len(results) < 2:
        st.warning("Nicht genug Analysen für Vergleich")
        st.session_state["compare_mode"] = False
        st.rerun()
    
    r1, r2 = results[0], results[1]
    s1, s2 = r1["summary"], r2["summary"]
    lb1, lb2 = labels[0], labels[1]
    
    st.markdown("---")
    st.subheader("📊 Vergleich")
    st.caption(f"{lb1} vs {lb2}")
    
    # Delta Stats in 4 columns
    def delta_str(v1, v2, unit="", higher_better=""):
        d = v2 - v1
        sign = "+" if d > 0 else ""
        emoji = ""
        if higher_better == "up" and d > 0: emoji = "🟢"
        elif higher_better == "down" and d < 0: emoji = "🟢"
        elif abs(d) > 0: emoji = "🟡"
        return f"{emoji} {sign}{d:.1f}{unit}" if d != 0 else "—"
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        d1, d2 = s1.get("dist_avg", 0), s2.get("dist_avg", 0)
        st.metric(f"Distanz ⌀", f"{lb1}: {d1:.0f} cm → {lb2}: {d2:.0f} cm",
                  delta_str(d1, d2, " cm"))
    with col_b:
        m1, m2 = s1.get("m_steps", 0), s2.get("m_steps", 0)
        st.metric(f"Schritte M", f"{m1} → {m2}",
                  delta_str(m1, m2, "", "up"))
    with col_c:
        c1, c2 = s1.get("correlation", 0), s2.get("correlation", 0)
        st.metric(f"Korrelation", f"{c1:.2f} → {c2:.2f}",
                  delta_str(c1, c2, "", "up"))
    with col_d:
        t1, t2 = s1.get("touches_high", 0), s2.get("touches_high", 0)
        st.metric(f"Touchés (high)", f"{t1} → {t2}",
                  delta_str(t1, t2, "", "up"))
    
    # Overlay charts: Distanz side by side
    st.markdown("---")
    st.subheader("📈 Distanz-Vergleich")
    
    times1 = [d["t"] for d in r1["m1_dist"]]
    dist1 = [d["cm"] for d in r1["m1_dist"]]
    times2 = [d["t"] for d in r2["m1_dist"]]
    dist2 = [d["cm"] for d in r2["m1_dist"]]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times1, y=dist1, mode="lines", line=dict(color=C_GREEN, width=2), name=lb1, hovertemplate="%{y:.0f} cm"))
    fig.add_trace(go.Scatter(x=times2, y=dist2, mode="lines", line=dict(color=C_ACCENT, width=2, dash="dash"), name=lb2, hovertemplate="%{y:.0f} cm"))
    fig.update_layout(
        title=dict(text="Distanz-Verlauf (überlagert)", font=dict(size=14, color=C_TEXT), x=0.02),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        font=dict(color=C_MUTED, size=11),
        xaxis=dict(gridcolor=C_BORDER, title="Zeit (s)", color=C_TEXT),
        yaxis=dict(gridcolor=C_BORDER, title="cm", color=C_TEXT),
        margin=dict(l=40, r=16, t=28, b=28),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor=C_BORDER, font=dict(size=10)),
        height=250,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Step comparison
    st.markdown("---")
    st.subheader(f"Details: {lb1} vs {lb2}")
    
    # All metrics table
    comp_data = []
    metrics_pairs = [
        ("Distanz ⌀", f"{s1.get('dist_avg',0):.0f} cm", f"{s2.get('dist_avg',0):.0f} cm"),
        ("Schritte M", str(s1.get('m_steps', 0)), str(s2.get('m_steps', 0))),
        ("Schritte G", str(s1.get('g_steps', 0)), str(s2.get('g_steps', 0))),
        ("Winkel M", f"{s1.get('m_angle_avg',0):.0f}°", f"{s2.get('m_angle_avg',0):.0f}°"),
        ("Winkel G", f"{s1.get('g_angle_avg',0):.0f}°", f"{s2.get('g_angle_avg',0):.0f}°"),
        ("Beschl M max", f"{s1.get('m_acc_max',0):.0f}", f"{s2.get('m_acc_max',0):.0f}"),
        ("Beschl G max", f"{s1.get('g_acc_max',0):.0f}", f"{s2.get('g_acc_max',0):.0f}"),
        ("Handhöhe M", f"{s1.get('m_hand_h_avg',0):.0f} px", f"{s2.get('m_hand_h_avg',0):.0f} px"),
        ("Arm-Streck M", f"{s1.get('m_ext_avg',0):.0f} px", f"{s2.get('m_ext_avg',0):.0f} px"),
        ("Standbreite M", f"{s1.get('m_stance_avg',0):.0f} px", f"{s2.get('m_stance_avg',0):.0f} px"),
        ("Explosivität max", f"{s1.get('expl_max',0):.0f} cm/s", f"{s2.get('expl_max',0):.0f} cm/s"),
        ("Korrelation", f"{s1.get('correlation',0):.2f}", f"{s2.get('correlation',0):.2f}"),
        ("Touchés high", str(s1.get('touches_high', 0)), str(s2.get('touches_high', 0))),
        ("Rhythmus", f"{s1.get('rhythm_dominant',0):.1f} Hz", f"{s2.get('rhythm_dominant',0):.1f} Hz"),
    ]
    for label, v1, v2 in metrics_pairs:
        comp_data.append({"Metrik": label, lb1: v1, lb2: v2})
    st.dataframe(comp_data, use_container_width=True, hide_index=True)
    
    # Clear button
    if st.button("🔄 Neue Analyse starten", use_container_width=True):
        st.session_state["compare_mode"] = False
        st.session_state["analyze_second"] = False
        for k in ["compare_results", "compare_labels", "compare_first"]:
            st.session_state.pop(k, None)
        st.rerun()


def display_video_viewer():
    """Zeigt Video-Player mit dualem Slider (Start/Ende) und Analysieren-Button - alles native Streamlit."""
    vp = st.session_state["video_path"]
    dur = st.session_state["vid_dur"]
    fps = st.session_state["vid_fps"]
    name = st.session_state["vid_name"]
    
    # Check if this is for comparison mode
    is_compare = st.session_state.get("analyze_second", False)
    label_prompt = "Zweiten Bereich wählen" if is_compare else name
    btn_label = "➕ Zweiten Bereich analysieren" if is_compare else "🔍 Bereich analysieren"

    st.subheader(f"🎬 {label_prompt}")

    # Timeline-Bereichsauswahl als Dual-Slider
    st.markdown("<hr style='border-color:#30363d; margin:6px 0;'>", unsafe_allow_html=True)

    start_val, end_val = st.slider(
        "Bereich auswählen (Sekunden)",
        min_value=0.0,
        max_value=dur,
        value=(0.0, min(dur, 60.0)),
        step=0.5,
        format="%.1fs",
        key="vid_range_slider"
    )

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
        if st.button(btn_label, type="primary", use_container_width=True):
            label = st.session_state.get("analysis_label", "Analyse")
            run_analysis(st.session_state["video_path"], start_val, clip_dur, label)
    else:
        st.warning("Bereich muss mindestens 3s lang sein")
        st.button("\U0001F3AF Bereich analysieren", disabled=True, use_container_width=True)


if __name__ == "__main__":
    main()
