"""
Worker: Fuhrt YOLOv8m-Pose Analyse in einem separaten Prozess aus.
Aufruf: python worker_analyze.py <clip_path> <result_path>

Schreibt Ergebnis-JSON nach result_path. frame_data wird auf [t, m_kpts, g_kpts]
komprimiert (keine numpy Objekte).
"""
import sys, json, math, traceback
from pathlib import Path

clip_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])

import cv2
import numpy as np
from ultralytics import YOLO

try:
    model = YOLO("yolov8m-pose.pt")
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError("Kann Video nicht offnen")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    ret, first_frame = cap.read()
    w_frame = first_frame.shape[1] if ret else 640
    h_frame = first_frame.shape[0] if ret else 360
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    TRACK_IDS = [0, 1]
    PISTE_WIDTH_CM = 200
    PISTE_WIDTH_PX_FALLBACK = 130
    prev_centers = {}
    frame_data = []
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

    cap.release()
    N = len(frame_data)

    # Kalibrierung
    all_hip_x = []
    for f in frame_data:
        for label in ("m", "g"):
            k = f[label]
            if k:
                hip_x = k[22]
                if hip_x > 0:
                    all_hip_x.append(hip_x)
    piste_px = (max(all_hip_x) - min(all_hip_x)) if all_hip_x else PISTE_WIDTH_PX_FALLBACK
    px_per_cm = piste_px / PISTE_WIDTH_CM

    def get_kp(flat, idx):
        if flat is None:
            return None
        x, y = flat[idx*2], flat[idx*2+1]
        return (x, y) if x > 0 and y > 0 else None

    def get_mid(flat, idx1, idx2):
        a, b = get_kp(flat, idx1), get_kp(flat, idx2)
        if a and b:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return a or b or None

    def midpoint(a, b):
        if a and b:
            return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        return a or b or None

    # Metriken berechnen
    m1_dist = []
    m2_m_angle, m2_g_angle = [], []
    m3_m_lunge, m3_g_lunge = [], []
    m4_m_path, m4_g_path = [], []
    m5_m_tilt, m5_g_tilt = [], []
    m6_m_acc, m6_g_acc = [], []
    m7_m_steps, m7_g_steps = [], []
    m8_vel_m, m8_vel_g = [], []

    prev_m_hip, prev_g_hip = None, None
    m_step_active, g_step_active = False, False
    step_counter_m, step_counter_g = 0, 0
    step_times_m, step_times_g = [], []

    for f in frame_data:
        t = f["t"]
        mk, gk = f["m"], f["g"]

        m_hip = get_mid(mk, 11, 12)
        g_hip = get_mid(gk, 11, 12)

        # M1 Distanz - IMMER einen Dict-Eintrag liefern, nie None
        if m_hip and g_hip:
            d_px = math.hypot(m_hip[0] - g_hip[0], m_hip[1] - g_hip[1])
            cm_val = round(d_px / px_per_cm, 1) if px_per_cm > 0 else 0
            m1_dist.append({"t": t, "px": round(d_px, 1), "cm": cm_val})
        else:
            m1_dist.append({"t": t, "px": 0, "cm": 0})

        # M2 Waffenarm-Winkel
        def arm_angle(kpts, side="right"):
            shoulder = get_kp(kpts, 6 if side == "right" else 5)
            elbow = get_kp(kpts, 8 if side == "right" else 7)
            wrist = get_kp(kpts, 10 if side == "right" else 9)
            if shoulder and elbow and wrist:
                v1 = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                n1 = math.hypot(*v1)
                n2 = math.hypot(*v2)
                if n1 > 0 and n2 > 0:
                    angle = math.degrees(math.acos(max(-1, min(1, dot/(n1*n2)))))
                    return round(angle, 1)
            return None

        m_angle = arm_angle(mk)
        g_angle = arm_angle(gk)
        m2_m_angle.append({"t": t, "deg": m_angle} if m_angle else {"t": t, "deg": 0})
        m2_g_angle.append({"t": t, "deg": g_angle} if g_angle else {"t": t, "deg": 0})

        # M3 Lunge-Tiefe
        def lunge_depth(kpts, prev_kpts):
            if kpts is None or prev_kpts is None:
                return None
            k_l = get_kp(kpts, 13)
            k_r = get_kp(kpts, 14)
            pk_l = get_kp(prev_kpts, 13)
            pk_r = get_kp(prev_kpts, 14)
            if k_l and pk_l:
                return round(math.hypot(k_l[0] - pk_l[0], k_l[1] - pk_l[1]), 1)
            if k_r and pk_r:
                return round(math.hypot(k_r[0] - pk_r[0], k_r[1] - pk_r[1]), 1)
            return None

        if frame_idx > 0:
            prev_f = frame_data[frame_idx - 1] if frame_idx - 1 < len(frame_data) else None
            if prev_f:
                ld_m = lunge_depth(mk, prev_f["m"])
                ld_g = lunge_depth(gk, prev_f["g"])
                m3_m_lunge.append({"t": t, "px": ld_m} if ld_m else {"t": t, "px": 0})
                m3_g_lunge.append({"t": t, "px": ld_g} if ld_g else {"t": t, "px": 0})
            else:
                m3_m_lunge.append({"t": t, "px": 0})
                m3_g_lunge.append({"t": t, "px": 0})
        else:
            m3_m_lunge.append({"t": t, "px": 0})
            m3_g_lunge.append({"t": t, "px": 0})

        # M4 Bewegungs-Pfad
        m4_m_path.append({"t": t, "x": m_hip[0] if m_hip else 0, "y": m_hip[1] if m_hip else 0})
        m4_g_path.append({"t": t, "x": g_hip[0] if g_hip else 0, "y": g_hip[1] if g_hip else 0})

        # M5 Korperhaltung
        def body_tilt(kpts):
            neck = get_kp(kpts, 5) or get_kp(kpts, 6)
            hip = get_mid(kpts, 11, 12)
            if neck and hip:
                angle = math.degrees(math.atan2(hip[1] - neck[1], hip[0] - neck[0]))
                return round(angle, 1)
            return 0

        m5_m_tilt.append({"t": t, "deg": 0})
        m5_g_tilt.append({"t": t, "deg": 0})

        # M6 Beschleunigung
        if prev_m_hip and m_hip:
            vel_m = math.hypot(m_hip[0] - prev_m_hip[0], m_hip[1] - prev_m_hip[1])
            m8_vel_m.append(vel_m)
        else:
            m8_vel_m.append(0)
        if prev_g_hip and g_hip:
            vel_g = math.hypot(g_hip[0] - prev_g_hip[0], g_hip[1] - prev_g_hip[1])
            m8_vel_g.append(vel_g)
        else:
            m8_vel_g.append(0)

        if len(m8_vel_m) >= 2:
            m6_m_acc.append({"t": t, "acc": round(abs(m8_vel_m[-1] - m8_vel_m[-2]) * fps, 1)})
        else:
            m6_m_acc.append({"t": t, "acc": 0})
        if len(m8_vel_g) >= 2:
            m6_g_acc.append({"t": t, "acc": round(abs(m8_vel_g[-1] - m8_vel_g[-2]) * fps, 1)})
        else:
            m6_g_acc.append({"t": t, "acc": 0})

        # M7 Schritt-Rhythmus
        def detect_step(kpts, prev_kpts, active):
            if kpts is None or prev_kpts is None:
                return False, active
            a_l = get_kp(kpts, 15)
            a_r = get_kp(kpts, 16)
            b_l = get_kp(prev_kpts, 15)
            b_r = get_kp(prev_kpts, 16)
            moving_l = a_l and b_l and math.hypot(a_l[0]-b_l[0], a_l[1]-b_l[1]) > 12
            moving_r = a_r and b_r and math.hypot(a_r[0]-b_r[0], a_r[1]-b_r[1]) > 12
            if (moving_l or moving_r) and not active:
                return True, True
            if not (moving_l or moving_r) and active:
                return False, False
            return False, active

        if frame_idx > 0:
            prev_f = frame_data[frame_idx - 1] if frame_idx - 1 < len(frame_data) else None
            if prev_f:
                step_m, m_step_active = detect_step(mk, prev_f["m"], m_step_active)
                step_g, g_step_active = detect_step(gk, prev_f["g"], g_step_active)
                if step_m:
                    step_counter_m += 1
                    step_times_m.append(t)
                if step_g:
                    step_counter_g += 1
                    step_times_g.append(t)
            m7_m_steps.append({"t": t, "step": step_counter_m})
            m7_g_steps.append({"t": t, "step": step_counter_g})
        else:
            m7_m_steps.append({"t": t, "step": 0})
            m7_g_steps.append({"t": t, "step": 0})

        prev_m_hip = m_hip
        prev_g_hip = g_hip
        frame_idx += 1

    # M8 Synchronisierung
    vel_m_arr = [v for v in m8_vel_m] if m8_vel_m else [0]
    vel_g_arr = [v for v in m8_vel_g] if m8_vel_g else [0]
    min_len = min(len(vel_m_arr), len(vel_g_arr))
    vel_m_arr = vel_m_arr[:min_len]
    vel_g_arr = vel_g_arr[:min_len]
    corr_val = 0.0
    lag_val = 0
    if min_len >= 3 and np.std(vel_m_arr) > 0 and np.std(vel_g_arr) > 0:
        corr = np.correlate(vel_m_arr - np.mean(vel_m_arr), vel_g_arr - np.mean(vel_g_arr), mode='full')
        corr = corr / (np.std(vel_m_arr) * np.std(vel_g_arr) * min_len)
        lag_val = int(np.argmax(corr) - (min_len - 1))
        corr_val = float(np.max(corr))

    # Summary
    sum_m1 = [d for d in m1_dist if d and d.get("cm")]
    summary = {
        "duration": round(duration, 1),
        "frames": N,
        "fps": round(fps, 1),
        "video": {
            "fps": round(fps, 1),
            "duration_s": round(duration, 1),
            "w": w_frame,
            "h": h_frame,
        },
        "dist_avg": round(sum(d["cm"] for d in sum_m1) / len(sum_m1), 1) if sum_m1 else 0,
        "dist_min": round(min(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "dist_max": round(max(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "m_angle_avg": round(sum(d["deg"] for d in m2_m_angle if d["deg"] > 0) / max(sum(1 for d in m2_m_angle if d["deg"] > 0), 1), 1),
        "g_angle_avg": round(sum(d["deg"] for d in m2_g_angle if d["deg"] > 0) / max(sum(1 for d in m2_g_angle if d["deg"] > 0), 1), 1),
        "m_steps": step_counter_m,
        "g_steps": step_counter_g,
        "m_acc_avg": round(sum(d["acc"] for d in m6_m_acc) / max(N, 1), 1),
        "g_acc_avg": round(sum(d["acc"] for d in m6_g_acc) / max(N, 1), 1),
        "m_acc_max": round(max(d["acc"] for d in m6_m_acc), 1),
        "g_acc_max": round(max(d["acc"] for d in m6_g_acc), 1),
        "m_lunge_avg": round(sum(d["px"] for d in m3_m_lunge) / max(N, 1), 1),
        "g_lunge_avg": round(sum(d["px"] for d in m3_g_lunge) / max(N, 1), 1),
        "correlation": round(corr_val, 3),
        "lag_frames": lag_val,
        "lag_seconds": round(lag_val / fps, 2) if fps > 0 else 0,
    }

    result = {
        "summary": summary,
        "frame_data": frame_data,
        "m1_dist": m1_dist,
        "m2_m_angle": m2_m_angle,
        "m2_g_angle": m2_g_angle,
        "m3_m_lunge": m3_m_lunge,
        "m3_g_lunge": m3_g_lunge,
        "m4_m_path": m4_m_path,
        "m4_g_path": m4_g_path,
        "m5_m_tilt": m5_m_tilt,
        "m5_g_tilt": m5_g_tilt,
        "m6_m_acc": m6_m_acc,
        "m6_g_acc": m6_g_acc,
        "m7_m_steps": m7_m_steps,
        "m7_g_steps": m7_g_steps,
        "m8_vel_m": m8_vel_m,
        "m8_vel_g": m8_vel_g,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Progress-Datei fur Fragment
    Path(str(result_path) + ".done").write_text("1")
    print("OK")

except Exception as e:
    with open(result_path, "w") as f:
        json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)
    print(f"ERROR: {e}")
    sys.exit(1)
