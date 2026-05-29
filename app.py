"""
Fencing Analyzer — Streamlit App
Interaktives Dashboard für Fecht-Video-Analyse mit YOLOv8m-Pose

Usage:
  streamlit run C:\\Users\\micha\\Desktop\\fencing_analyzer\\app.py
"""
import streamlit as st
st.set_page_config(page_title="Fecht-Analyzer", layout="wide", page_icon="\U0001F93A")

import os, sys, json, math, time, shutil, subprocess, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image
from ultralytics import YOLO

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

COCO_KP_NAMES = {
    0:"Nase",1:"Auge L",2:"Auge R",3:"Ohr L",4:"Ohr R",
    5:"Schulter L",6:"Schulter R",7:"Ellbogen L",8:"Ellbogen R",
    9:"Handgelenk L",10:"Handgelenk R",11:"Huefte L",12:"Huefte R",
    13:"Knie L",14:"Knie R",15:"Knoechel L",16:"Knoechel R"
}

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

def fig_theme(fig, title="", xlabel="", ylabel=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=C_TEXT), x=0.02),
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        font=dict(color=C_TEXT, size=11),
        xaxis=dict(gridcolor="#21262d", title=xlabel, color=C_MUTED, showline=True, linecolor=C_BORDER),
        yaxis=dict(gridcolor="#21262d", title=ylabel, color=C_MUTED, showline=True, linecolor=C_BORDER),
        margin=dict(l=40, r=16, t=32, b=32),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(0,0,0,0.6)", bordercolor=C_BORDER),
    )
    return fig

# --- ANALYSIS ENGINE ---
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
    subprocess.run(cmd, capture_output=True, timeout=300)
    return output_path.exists()

def analyze_video(video_path, progress_callback=None):
    model = load_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Video konnte nicht geoeffnet werden")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ret, first_frame = cap.read()
    if ret:
        h_frame, w_frame = first_frame.shape[:2]
    else:
        w_frame, h_frame = 640, 360
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    TRACK_IDS = [0, 1]
    prev_centers = {}
    all_frames = []
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

        rec = {"frame": frame_idx, "time": frame_idx/fps}
        for label, kpts_obj in [("m", m_kpts), ("g", g_kpts)]:
            d = {}
            for idx in range(17):
                d[COCO_KP_NAMES[idx]] = kpt(kpts_obj, idx)
            d["nose"] = kpt(kpts_obj, 0)
            d["shoulder_l"] = kpt(kpts_obj, 5); d["shoulder_r"] = kpt(kpts_obj, 6)
            d["elbow_l"] = kpt(kpts_obj, 7); d["elbow_r"] = kpt(kpts_obj, 8)
            d["wrist_l"] = kpt(kpts_obj, 9); d["wrist_r"] = kpt(kpts_obj, 10)
            d["hip_l"] = kpt(kpts_obj, 11); d["hip_r"] = kpt(kpts_obj, 12)
            d["knee_l"] = kpt(kpts_obj, 13); d["knee_r"] = kpt(kpts_obj, 14)
            d["ankle_l"] = kpt(kpts_obj, 15); d["ankle_r"] = kpt(kpts_obj, 16)
            d["hip_mid"] = midpoint(d["hip_l"], d["hip_r"])
            d["shoulder_mid"] = midpoint(d["shoulder_l"], d["shoulder_r"])
            rec[label] = d

        all_frames.append(rec)
        frame_idx += 1
        if progress_callback and frame_idx % 15 == 0:
            progress_callback(frame_idx / total_frames)

    cap.release()
    N = len(all_frames)

    # Kalibrierung
    all_hip_x = []
    for f in all_frames:
        for label in ("m", "g"):
            if f[label]["hip_mid"]:
                all_hip_x.append(f[label]["hip_mid"][0])
    piste_px = (max(all_hip_x) - min(all_hip_x)) if all_hip_x else PISTE_WIDTH_PX_FALLBACK
    px_per_cm = piste_px / PISTE_WIDTH_CM

    # === METRIK 1: DISTANZ ===
    m1_dist = []
    for f in all_frames:
        d_val = dist(f["m"]["hip_mid"], f["g"]["hip_mid"])
        if d_val:
            m1_dist.append({"t": f["time"], "px": d_val, "cm": d_val / px_per_cm})

    # === METRIK 2: WAFFENARM-WINKEL ===
    m2_m_angle = []
    m2_g_angle = []
    for f in all_frames:
        a = angle_at(f["m"]["shoulder_r"], f["m"]["elbow_r"], f["m"]["wrist_r"])
        if a is None: a = angle_at(f["m"]["shoulder_l"], f["m"]["elbow_l"], f["m"]["wrist_l"])
        m2_m_angle.append({"t": f["time"], "deg": a or 0})
        b = angle_at(f["g"]["shoulder_l"], f["g"]["elbow_l"], f["g"]["wrist_l"])
        if b is None: b = angle_at(f["g"]["shoulder_r"], f["g"]["elbow_r"], f["g"]["wrist_r"])
        m2_g_angle.append({"t": f["time"], "deg": b or 0})

    # === METRIK 3: LUNGE-TIEFE ===
    m3_m_lunge = []
    m3_g_lunge = []
    for f in all_frames:
        m_hip = f["m"]["hip_mid"]
        m_la, m_ra = f["m"]["ankle_l"], f["m"]["ankle_r"]
        if m_hip and m_la and m_ra:
            front = m_la if m_la[1] < m_ra[1] else m_ra
            m3_m_lunge.append({"t": f["time"], "px": max(0, m_hip[1] - front[1])})
        g_hip = f["g"]["hip_mid"]
        g_la, g_ra = f["g"]["ankle_l"], f["g"]["ankle_r"]
        if g_hip and g_la and g_ra:
            front = g_la if g_la[1] < g_ra[1] else g_ra
            m3_g_lunge.append({"t": f["time"], "px": max(0, g_hip[1] - front[1])})

    # === METRIK 4: BEWEGUNGS-PFAD ===
    m4_m_path = [{"t": f["time"], "x": f["m"]["hip_mid"][0], "y": f["m"]["hip_mid"][1]}
                 for f in all_frames if f["m"]["hip_mid"]]
    m4_g_path = [{"t": f["time"], "x": f["g"]["hip_mid"][0], "y": f["g"]["hip_mid"][1]}
                 for f in all_frames if f["g"]["hip_mid"]]

    # === METRIK 5: KOERPERHALTUNG ===
    m5_m_tilt = []
    m5_g_tilt = []
    for f in all_frames:
        ms = f["m"]["shoulder_mid"]; mh = f["m"]["hip_mid"]
        if ms and mh:
            dx, dy = ms[0]-mh[0], ms[1]-mh[1]
            tilt = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0
            m5_m_tilt.append({"t": f["time"], "deg": tilt * (1 if dx > 0 else -1)})
        gs = f["g"]["shoulder_mid"]; gh = f["g"]["hip_mid"]
        if gs and gh:
            dx, dy = gs[0]-gh[0], gs[1]-gh[1]
            tilt = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0
            m5_g_tilt.append({"t": f["time"], "deg": tilt * (1 if dx > 0 else -1)})

    # === METRIK 6: BESCHLEUNIGUNG ===
    m6_m_acc = []
    m6_g_acc = []
    for i in range(2, N):
        dt = 2/fps
        def acc_3(label, wrist_key):
            p0 = all_frames[i-2][label][wrist_key]
            p1 = all_frames[i-1][label][wrist_key]
            p2 = all_frames[i][label][wrist_key]
            if all([p0, p1, p2]):
                v1 = dist(p0, p1) * fps if dist(p0, p1) else 0
                v2 = dist(p1, p2) * fps if dist(p1, p2) else 0
                return (v2 - v1) / dt
            return None
        a_m = acc_3("m", "wrist_r")
        if a_m is not None: m6_m_acc.append({"t": all_frames[i]["time"], "acc": a_m})
        a_g = acc_3("g", "wrist_l")
        if a_g is not None: m6_g_acc.append({"t": all_frames[i]["time"], "acc": a_g})

    # === METRIK 7: SCHRITT-RHYTHMUS ===
    def detect_steps(label):
        steps = []
        for i in range(1, N):
            p = all_frames[i-1][label]; c = all_frames[i][label]
            t = all_frames[i]["time"]
            dl = dist(p["ankle_l"], c["ankle_l"]) or 0
            dr = dist(p["ankle_r"], c["ankle_r"]) or 0
            side = None
            if dl > 12 and dr < 6:
                side = "links"
            elif dr > 12 and dl < 6:
                side = "rechts"
            if side:
                other_recent = False
                for j in range(max(0, i-3), i):
                    pp = all_frames[j][label]
                    op = all_frames[j+1][label]
                    other_key = "ankle_r" if side == "links" else "ankle_l"
                    o_d = dist(pp[other_key], op[other_key]) or 0
                    if o_d > 12:
                        other_recent = True
                        break
                if other_recent:
                    step_type = "ganz"
                else:
                    step_type = "halb"
                steps.append({"t": t, "side": side, "dist": dl if side == "links" else dr, "type": step_type})
        return steps

    m7_m_steps = detect_steps("m")
    m7_g_steps = detect_steps("g")

    # === METRIK 8: SYNCHRONISIERUNG ===
    m_vel = []
    g_vel = []
    for i in range(1, len(m4_m_path)):
        dx = m4_m_path[i]["x"] - m4_m_path[i-1]["x"]
        dy = m4_m_path[i]["y"] - m4_m_path[i-1]["y"]
        m_vel.append(math.hypot(dx, dy) * fps)
    for i in range(1, len(m4_g_path)):
        dx = m4_g_path[i]["x"] - m4_g_path[i-1]["x"]
        dy = m4_g_path[i]["y"] - m4_g_path[i-1]["y"]
        g_vel.append(math.hypot(dx, dy) * fps)

    min_vel = min(len(m_vel), len(g_vel))
    mv = np.array(m_vel[:min_vel]) if m_vel else np.array([])
    gv = np.array(g_vel[:min_vel]) if g_vel else np.array([])

    m8_corr = float(np.corrcoef(mv, gv)[0, 1]) if len(mv) > 10 and np.std(mv) > 0 and np.std(gv) > 0 else 0
    m8_lag = 0
    if len(mv) > 10:
        xcorr = np.correlate(mv - np.mean(mv), gv - np.mean(gv), mode='full')
        m8_lag = int(np.argmax(np.abs(xcorr)) - (len(mv) - 1))

    # === METRIK 9: HEATMAP ===
    m9_m_pos = [{"x": p["x"], "y": p["y"]} for p in m4_m_path]
    m9_g_pos = [{"x": p["x"], "y": p["y"]} for p in m4_g_path]

    summary = {
        "video": {"frames": N, "duration_s": round(duration, 1), "fps": fps, "resolution": f"{w_frame}x{h_frame}"},
        "metrik_1_distanz": {
            "avg_px": round(float(np.mean([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "min_px": round(float(np.min([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "max_px": round(float(np.max([d["px"] for d in m1_dist])), 1) if m1_dist else 0,
            "avg_cm": round(float(np.mean([d["cm"] for d in m1_dist])), 1) if m1_dist else 0,
        },
        "metrik_2_winkel": {
            "m_avg": round(float(np.mean([x["deg"] for x in m2_m_angle if x["deg"] > 0])), 1),
            "g_avg": round(float(np.mean([x["deg"] for x in m2_g_angle if x["deg"] > 0])), 1),
        },
        "metrik_3_lunge": {
            "m_max": round(float(np.max([x["px"] for x in m3_m_lunge])), 1) if m3_m_lunge else 0,
            "g_max": round(float(np.max([x["px"] for x in m3_g_lunge])), 1) if m3_g_lunge else 0,
        },
        "metrik_5_haltung": {
            "m_avg": round(float(np.mean([x["deg"] for x in m5_m_tilt])), 1) if m5_m_tilt else 0,
            "g_avg": round(float(np.mean([x["deg"] for x in m5_g_tilt])), 1) if m5_g_tilt else 0,
        },
        "metrik_6_acc": {
            "m_max": round(float(np.max([abs(x["acc"]) for x in m6_m_acc])), 1) if m6_m_acc else 0,
            "g_max": round(float(np.max([abs(x["acc"]) for x in m6_g_acc])), 1) if m6_g_acc else 0,
        },
        "metrik_7_schritte": {
            "m_total": len(m7_m_steps),
            "m_rate": round(len(m7_m_steps)/duration, 2) if duration > 0 else 0,
            "g_total": len(m7_g_steps),
            "g_rate": round(len(m7_g_steps)/duration, 2) if duration > 0 else 0,
            "m_halb": len([s for s in m7_m_steps if s["type"] == "halb"]),
            "m_ganz": len([s for s in m7_m_steps if s["type"] == "ganz"]),
        },
        "metrik_8_sync": {
            "korrelation": round(m8_corr, 3),
            "lag_frames": m8_lag,
            "lag_s": round(m8_lag / fps, 2) if fps > 0 else 0,
            "leader": "Michael" if abs(m8_lag) > 2 and m8_lag > 0 else ("Gegner" if abs(m8_lag) > 2 else "neutral"),
        },
    }

    return {
        "summary": summary,
        "frames": all_frames,
        "m1_dist": m1_dist,
        "m2_m_angle": m2_m_angle, "m2_g_angle": m2_g_angle,
        "m3_m_lunge": m3_m_lunge, "m3_g_lunge": m3_g_lunge,
        "m4_m_path": m4_m_path, "m4_g_path": m4_g_path,
        "m5_m_tilt": m5_m_tilt, "m5_g_tilt": m5_g_tilt,
        "m6_m_acc": m6_m_acc, "m6_g_acc": m6_g_acc,
        "m7_m_steps": m7_m_steps, "m7_g_steps": m7_g_steps,
        "m8_corr": m8_corr, "m8_lag": m8_lag,
        "m8_vel_m": mv.tolist() if len(mv) > 0 else [],
        "m8_vel_g": gv.tolist() if len(gv) > 0 else [],
        "m9_m_pos": m9_m_pos, "m9_g_pos": m9_g_pos,
    }


# === PLOTLY CHARTS ===

def plot_distanz(data):
    fig = go.Figure()
    ts = [d["t"] for d in data]
    vals = [d["cm"] for d in data]
    fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name="Distanz",
                             line=dict(color=C_BLUE, width=2), fill="tozeroy",
                             fillcolor="rgba(0,204,255,0.1)"))
    avg = np.mean(vals)
    fig.add_hline(y=avg, line_dash="dash", line_color=C_BLUE,
                  annotation_text=f"Mittel: {avg:.0f}cm",
                  annotation_font=dict(size=11, color=C_MUTED))
    return fig_theme(fig, "Distanz (Hueft-zu-Hueft)", ylabel="cm")

def plot_winkel(m_data, g_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d["t"] for d in m_data], y=[d["deg"] for d in m_data],
                             mode="lines", name="Michael", line=dict(color=C_GREEN, width=1.5)))
    fig.add_trace(go.Scatter(x=[d["t"] for d in g_data], y=[d["deg"] for d in g_data],
                             mode="lines", name="Gegner", line=dict(color=C_RED, width=1.5)))
    return fig_theme(fig, "Waffenarm-Winkel", ylabel="Grad")

def plot_lunge(m_data, g_data):
    fig = go.Figure()
    if m_data:
        fig.add_trace(go.Scatter(x=[d["t"] for d in m_data], y=[d["px"] for d in m_data],
                                 mode="lines", name="Michael", line=dict(color=C_GREEN, width=1.5)))
    if g_data:
        fig.add_trace(go.Scatter(x=[d["t"] for d in g_data], y=[d["px"] for d in g_data],
                                 mode="lines", name="Gegner", line=dict(color=C_RED, width=1.5)))
    return fig_theme(fig, "Lunge-Tiefe (Hueft-Knoechel vertikal)", ylabel="px")

def plot_bewegungspfad(m_path, g_path):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["x"] for p in m_path], y=[p["y"] for p in m_path],
                             mode="markers", name="Michael",
                             marker=dict(color=C_GREEN, size=3, opacity=0.5)))
    fig.add_trace(go.Scatter(x=[p["x"] for p in g_path], y=[p["y"] for p in g_path],
                             mode="markers", name="Gegner",
                             marker=dict(color=C_RED, size=3, opacity=0.5)))
    fig.update_yaxes(autorange="reversed", title="Y (Pixel)")
    fig.update_xaxes(title="X (Pixel)")
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig_theme(fig, "Bewegungs-Pfad (Hueft-Position)")

def plot_haltung(m_data, g_data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[d["t"] for d in m_data], y=[d["deg"] for d in m_data],
                             mode="lines", name="Michael", line=dict(color=C_GREEN, width=1.5)))
    fig.add_trace(go.Scatter(x=[d["t"] for d in g_data], y=[d["deg"] for d in g_data],
                             mode="lines", name="Gegner", line=dict(color=C_RED, width=1.5)))
    fig.add_hline(y=0, line_color=C_BORDER, line_width=1,
                  annotation_text="aufrecht", annotation_font=dict(size=10, color=C_MUTED))
    return fig_theme(fig, "Koerperhaltung (Oberkoerper-Neigung)", ylabel="Grad")

def plot_acc(m_data, g_data):
    fig = go.Figure()
    if m_data:
        fig.add_trace(go.Scatter(x=[d["t"] for d in m_data], y=[d["acc"] for d in m_data],
                                 mode="lines", name="Michael", line=dict(color=C_GREEN, width=1.2)))
        vals = [abs(d["acc"]) for d in m_data]
        if vals:
            thresh = np.percentile(vals, 95)
            peaks = [(d["t"], d["acc"]) for d in m_data if abs(d["acc"]) > thresh]
            for t, v in peaks[:20]:
                fig.add_annotation(x=t, y=v, text=f"{abs(v):.0f}", showarrow=True,
                                   arrowhead=1, arrowsize=1, ax=0, ay=-20,
                                   font=dict(size=8, color=C_GREEN), bgcolor="rgba(0,0,0,0.6)")
    if g_data:
        fig.add_trace(go.Scatter(x=[d["t"] for d in g_data], y=[d["acc"] for d in g_data],
                                 mode="lines", name="Gegner", line=dict(color=C_RED, width=1.2)))
    return fig_theme(fig, "Waffenhand-Beschleunigung", ylabel="px/s\u00b2")

def plot_schritte(m_steps, g_steps, duration):
    fig = go.Figure()
    times = np.linspace(0, duration, 200)
    m_cum = np.zeros_like(times)
    g_cum = np.zeros_like(times)
    for s in m_steps:
        m_cum[times >= s["t"]] += 1
    for s in g_steps:
        g_cum[times >= s["t"]] += 1
    fig.add_trace(go.Scatter(x=times, y=m_cum, mode="lines", name=f"Michael ({len(m_steps)})",
                             line=dict(color=C_GREEN, width=2.5, shape="hv")))
    fig.add_trace(go.Scatter(x=times, y=g_cum, mode="lines", name=f"Gegner ({len(g_steps)})",
                             line=dict(color=C_RED, width=2.5, shape="hv")))
    for s in m_steps:
        color = {"halb": C_BLUE, "ganz": C_GREEN}.get(s["type"], C_GREEN)
        fig.add_vline(x=s["t"], line_color=color, line_width=1, opacity=0.3)
    return fig_theme(fig, "Schritt-Rhythmus", ylabel="Schritte (kumulativ)")

def plot_sync(vel_m, vel_g, frames):
    fig = go.Figure()
    ts = np.linspace(0, len(vel_m)/30, len(vel_m)) if vel_m else []
    if len(ts) > 0:
        fig.add_trace(go.Scatter(x=ts, y=vel_m, mode="lines", name="Michael",
                                 line=dict(color=C_GREEN, width=1.5), opacity=0.8))
        fig.add_trace(go.Scatter(x=ts, y=vel_g, mode="lines", name="Gegner",
                                 line=dict(color=C_RED, width=1.5), opacity=0.8))
    return fig_theme(fig, "Reaktions-Synchronisierung (Hueft-Geschwindigkeit)", ylabel="px/s")

def plot_heatmap(m_pos, g_pos):
    fig = go.Figure()
    if m_pos and g_pos:
        all_x = [p["x"] for p in m_pos] + [p["x"] for p in g_pos]
        all_y = [p["y"] for p in m_pos] + [p["y"] for p in g_pos]
        fig.add_trace(go.Histogram2d(
            x=all_x, y=all_y, colorscale="Plasma", nbinsx=30, nbinsy=20,
            opacity=0.7, showscale=True, colorbar=dict(title="Dichte", titleside="right")
        ))
    fig.add_trace(go.Scatter(x=[p["x"] for p in m_pos], y=[p["y"] for p in m_pos],
                             mode="markers", name="Michael",
                             marker=dict(color=C_GREEN, size=4, opacity=0.3)))
    fig.add_trace(go.Scatter(x=[p["x"] for p in g_pos], y=[p["y"] for p in g_pos],
                             mode="markers", name="Gegner",
                             marker=dict(color=C_RED, size=4, opacity=0.3)))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
    return fig_theme(fig, "Positions-Heatmap (Piste)", xlabel="X (Pixel)", ylabel="Y (Pixel)")


# === STREAMLIT UI ===

def main():
    st.markdown("""
    <style>
    .stApp { background: #0d1117; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
    .stProgress > div > div > div > div { background: #58a6ff; }
    h1, h2, h3 { color: #c9d1d9 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.title("Fecht-Analyzer")
    st.caption("YOLOv8m-Pose | 9 Metriken | Interaktives Dashboard")

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
            st.divider()
            st.header("Clip-Einstellungen")
            start_sec = st.number_input("Startzeit (s)", min_value=0, value=0, step=5)
            clip_duration = st.number_input("Dauer (s)", min_value=5, value=15, max_value=120, step=5)
            cap = cv2.VideoCapture(str(video_path))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            cap.release()
            st.caption(f"{total_f} Frames @ {vid_fps:.0f}fps = {total_f/vid_fps:.0f}s Gesamt")
            analyze_btn = st.button("Analyse starten", type="primary", use_container_width=True)
        else:
            analyze_btn = False

    if video_path and analyze_btn:
        run_analysis(video_path, start_sec, clip_duration)
    elif "result" in st.session_state:
        display_dashboard(st.session_state["result"])
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("Video auswaehlen und Analyse starten")
            st.markdown("""
            **Verfuegbare Metriken:**
            1. Distanz | 2. Waffenarm-Winkel | 3. Lunge-Tiefe  
            4. Bewegungs-Pfad | 5. Koerperhaltung | 6. Beschleunigung  
            7. Schritte (halb/ganz) | 8. Synchronisierung | 9. Heatmap
            """)


def run_analysis(video_path, start_sec, clip_duration):
    st.session_state.pop("result", None)
    clip_progress = st.progress(0, text="Extrahiere Clip...")
    clip_path = Path(tempfile.gettempdir()) / f"fencing_clip_{int(time.time())}.mp4"

    ok = extract_clip(video_path, clip_path, start_sec, clip_duration)
    if not ok:
        st.error("Clip-Extraktion fehlgeschlagen")
        return
    clip_progress.progress(100, text="Clip bereit")

    status = st.status("Analysiere Video...", expanded=True)
    status.write("YOLOv8m-Pose geladen")
    progress_bar = st.progress(0)

    def on_progress(pct):
        progress_bar.progress(min(pct, 1.0))

    try:
        status.write("Extrahiere Keypoints & Tracke Fechter...")
        result = analyze_video(clip_path, progress_callback=on_progress)
        progress_bar.progress(1.0)
        status.write("Analyse abgeschlossen")
        status.update(state="complete")
        st.session_state["result"] = result
        st.session_state["clip_path"] = clip_path
        json_path = clip_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(result["summary"], f, indent=2)
        st.rerun()
    except Exception as e:
        status.update(state="error")
        st.error(f"Analyse fehlgeschlagen: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_dashboard(result):
    s = result["summary"]

    # Key Stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Distanz", f'{s["metrik_1_distanz"]["avg_cm"]}cm')
    with col2:
        st.metric("Schritte/s", f'{s["metrik_7_schritte"]["m_rate"]+s["metrik_7_schritte"]["g_rate"]:.2f}')
    with col3:
        st.metric("Winkel M/G", f'{s["metrik_2_winkel"]["m_avg"]:.0f}/{s["metrik_2_winkel"]["g_avg"]:.0f}')
    with col4:
        st.metric("Korrelation", f'{s["metrik_8_sync"]["korrelation"]:.2f}')
    with col5:
        acc_max = max(s["metrik_6_acc"]["m_max"], s["metrik_6_acc"]["g_max"])
        st.metric("Max Beschl.", f'{acc_max:.0f}')

    st.divider()

    # Schritt-Detail
    with st.expander("Schritt-Detail (Michael)", expanded=True):
        m_steps = result.get("m7_m_steps", [])
        g_steps = result.get("m7_g_steps", [])
        c1, c2, c3 = st.columns(3)
        with c1:
            m_halb = len([s for s in m_steps if s["type"] == "halb"])
            m_ganz = len([s for s in m_steps if s["type"] == "ganz"])
            st.metric("Halbe Schritte", m_halb, help="Nur vorderer Fuss")
            st.metric("Ganze Schritte", m_ganz, help="Beide Fuesse nacheinander")
        with c2:
            g_halb = len([s for s in g_steps if s["type"] == "halb"])
            g_ganz = len([s for s in g_steps if s["type"] == "ganz"])
            st.metric("Gegner Halbe", g_halb)
            st.metric("Gegner Ganze", g_ganz)
        with c3:
            st.markdown("**Farbcode:**")
            st.markdown('<span style="color:#58a6ff">Blau</span> = Halb', unsafe_allow_html=True)
            st.markdown('<span style="color:#00ff88">Gruen</span> = Ganz', unsafe_allow_html=True)

        # Tabelle
        if m_steps:
            html_rows = ""
            for step in m_steps:
                color = {"halb": "rgba(88,166,255,0.08)", "ganz": "rgba(46,160,67,0.08)"}.get(step["type"], "")
                icon = "\u279c" if step["side"] == "rechts" else "\u2b05"
                html_rows += f"""<tr style="background:{color}">
                    <td>{step['t']:.1f}s</td>
                    <td>{icon} {step['side']}</td>
                    <td><b>{step['type'].upper()}</b></td>
                    <td>{step['dist']:.0f}px</td></tr>"""
            st.markdown(f"""
            <table style="width:100%;border-collapse:collapse;font-size:13px;color:#c9d1d9;">
            <thead><tr style="background:#21262d;">
                <th style="padding:6px 10px;border-bottom:2px solid #30363d;">Zeit</th>
                <th style="padding:6px 10px;border-bottom:2px solid #30363d;">Seite</th>
                <th style="padding:6px 10px;border-bottom:2px solid #30363d;">Typ</th>
                <th style="padding:6px 10px;border-bottom:2px solid #30363d;">Distanz</th>
            </tr></thead><tbody>{html_rows}</tbody></table>""", unsafe_allow_html=True)
        else:
            st.info("Keine Schritte erkannt")

    # Toggles
    st.divider()
    st.subheader("Metriken-Dashboard")
    metric_names = [
        ("1_dist", "Distanz", True), ("2_winkel", "Winkel", True),
        ("3_lunge", "Lunge", True), ("4_pfad", "Pfad", True),
        ("5_haltung", "Haltung", True), ("6_acc", "Beschl.", True),
        ("7_schritte", "Schritte", True), ("8_sync", "Sync", True),
        ("9_heatmap", "Heatmap", True),
    ]
    cols = st.columns(9)
    toggles = {}
    for i, (key, label, _) in enumerate(metric_names):
        with cols[i]:
            toggles[key] = st.checkbox(label, value=True, key=f"t_{key}")

    # Charts
    c_left, c_right = st.columns(2)
    chart_specs = [
        ("1_dist", plot_distanz, result["m1_dist"]),
        ("2_winkel", plot_winkel, result["m2_m_angle"], result["m2_g_angle"]),
        ("3_lunge", plot_lunge, result["m3_m_lunge"], result["m3_g_lunge"]),
        ("4_pfad", plot_bewegungspfad, result["m4_m_path"], result["m4_g_path"]),
        ("5_haltung", plot_haltung, result["m5_m_tilt"], result["m5_g_tilt"]),
        ("6_acc", plot_acc, result["m6_m_acc"], result["m6_g_acc"]),
        ("7_schritte", plot_schritte, result["m7_m_steps"], result["m7_g_steps"], s["video"]["duration_s"]),
        ("8_sync", plot_sync, result["m8_vel_m"], result["m8_vel_g"], result["frames"]),
        ("9_heatmap", plot_heatmap, result["m9_m_pos"], result["m9_g_pos"]),
    ]

    for i, spec in enumerate(chart_specs):
        key = spec[0]
        with c_left if i % 2 == 0 else c_right:
            if toggles.get(key, True):
                try:
                    fig = spec[1](*spec[2:])
                    st.plotly_chart(fig, use_container_width=True, key=f"c_{key}")
                except Exception as e:
                    st.error(f"Chart {key}: {e}")

    # Export
    st.divider()
    st.subheader("Export")
    ec1, ec2, ec3, ec4 = st.columns(4)
    json_bytes = json.dumps(s, indent=2).encode("utf-8")
    ec1.download_button("JSON", data=json_bytes, file_name="fencing_analysis.json", mime="application/json")
    m_steps = result.get("m7_m_steps", [])
    if m_steps:
        df_steps = pd.DataFrame(m_steps)
        csv = df_steps.to_csv(index=False).encode("utf-8")
        ec2.download_button("Schritte-CSV", data=csv, file_name="steps.csv", mime="text/csv")
    if result.get("m1_dist"):
        csv_d = pd.DataFrame(result["m1_dist"]).to_csv(index=False).encode("utf-8")
        ec3.download_button("Distanz-CSV", data=csv_d, file_name="distance.csv", mime="text/csv")

    summary_text = f"""Fecht-Analyse Ergebnis
========================
Dauer: {s['video']['duration_s']}s | Frames: {s['video']['frames']} | {s['video']['fps']}fps

1. DISTANZ: {s['metrik_1_distanz']['avg_cm']}cm Oe
2. WINKEL: Michael {s['metrik_2_winkel']['m_avg']} | Gegner {s['metrik_2_winkel']['g_avg']}
3. LUNGE: Michael max {s['metrik_3_lunge']['m_max']}px | Gegner max {s['metrik_3_lunge']['g_max']}px
5. HALTUNG: Michael {s['metrik_5_haltung']['m_avg']} | Gegner {s['metrik_5_haltung']['g_avg']}
6. BESCHL.: Michael max {s['metrik_6_acc']['m_max']} | Gegner max {s['metrik_6_acc']['g_max']}
7. SCHRITTE: Michael {s['metrik_7_schritte']['m_total']} ({s['metrik_7_schritte']['m_rate']}/s) | Gegner {s['metrik_7_schritte']['g_total']}
   Halb: {s['metrik_7_schritte']['m_halb']} | Ganz: {s['metrik_7_schritte']['m_ganz']}
8. SYNCHRONISIERUNG: Korr {s['metrik_8_sync']['korrelation']} | Lag {s['metrik_8_sync']['lag_s']}s
"""
    ec4.download_button("Report-TXT", data=summary_text.encode("utf-8"), file_name="fencing_report.txt")


if __name__ == "__main__":
    main()
