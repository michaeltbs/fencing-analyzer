"""
Worker: Fuhrt YOLOv8m-Pose Analyse in einem separaten Prozess aus.
Aufruf: python worker_analyze.py <clip_path> <result_path>

Schreibt Ergebnis-JSON nach result_path. frame_data wird auf [t, m_kpts, g_kpts]
komprimiert (keine numpy Objekte).

Metriken: 1-8 (Basis) + 9-15 (Neu: Handhöhe, Arm-Streckung, Standbreite,
Distanz-Explosivität, Head-Forward, Treffer-Kandidat, Rhythmus-Pattern)
"""
import sys, json, math, traceback, struct
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
    all_hip_y = []
    for f in frame_data:
        for label in ("m", "g"):
            k = f[label]
            if k:
                hip_lx = k[22]
                hip_ly = k[23]
                hip_rx = k[24]
                hip_ry = k[25]
                if hip_lx > 0: all_hip_x.append(hip_lx)
                if hip_rx > 0: all_hip_x.append(hip_rx)
                if hip_ly > 0: all_hip_y.append(hip_ly)
                if hip_ry > 0: all_hip_y.append(hip_ry)
    
    # Determine orientation: if hip x-range > y-range → horizontal (side view), else → vertical (front/back)
    x_range = max(all_hip_x) - min(all_hip_x) if all_hip_x else 0
    y_range = max(all_hip_y) - min(all_hip_y) if all_hip_y else 0
    
    if x_range > y_range * 1.5:
        orientation = "seitlich"
        piste_px = x_range if x_range > 0 else PISTE_WIDTH_PX_FALLBACK
    else:
        orientation = "frontal"
        # For frontal view: estimate from shoulder width (typically ~40cm real)
        all_shoulder_dists = []
        for f in frame_data:
            for label in ("m", "g"):
                k = f[label]
                if k:
                    ls = get_kp(k, 5); rs = get_kp(k, 6)
                    if ls and rs:
                        all_shoulder_dists.append(math.hypot(ls[0]-rs[0], ls[1]-rs[1]))
        shoulder_px = np.median(all_shoulder_dists) if all_shoulder_dists else 40
        piste_px = shoulder_px * (PISTE_WIDTH_CM / 40)  # ~200/40 = 5x shoulder width
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

    # === Metriken berechnen ===
    m1_dist = []
    m2_m_angle, m2_g_angle = [], []
    m3_m_lunge, m3_g_lunge = [], []
    m4_m_path, m4_g_path = [], []
    m5_m_tilt, m5_g_tilt = [], []
    m6_m_acc, m6_g_acc = [], []
    m7_m_steps, m7_g_steps = [], []
    m8_vel_m, m8_vel_g = [], []

    # Neue Metriken 9-15
    m9_m_hand_h, m9_g_hand_h = [], []
    m10_m_ext, m10_g_ext = [], []
    m11_m_stance, m11_g_stance = [], []
    m13_m_head, m13_g_head = [], []

    # M16: Pressure Index — kumulativer Vorteil im Vor-Rückzug
    m16_pressure = []
    m16_m_advance = 0  # positive cumulative advance (Michael)
    m16_g_advance = 0  # negative cumulative advance (Gegner)

    prev_m_hip, prev_g_hip = None, None
    m_step_active, g_step_active = False, False
    step_counter_m, step_counter_g = 0, 0
    step_times_m, step_times_g = [], []

    # Für Treffer-Kandidaten (Post-Loop)
    m_ext_max, g_ext_max = 0, 0
    m_ext_series, g_ext_series = [], []

    for f in frame_data:
        t = f["t"]
        mk, gk = f["m"], f["g"]

        m_hip = get_mid(mk, 11, 12)
        g_hip = get_mid(gk, 11, 12)

        # --- M1 Distanz ---
        if m_hip and g_hip:
            d_px = math.hypot(m_hip[0] - g_hip[0], m_hip[1] - g_hip[1])
            cm_val = round(d_px / px_per_cm, 1) if px_per_cm > 0 else 0
            m1_dist.append({"t": t, "px": round(d_px, 1), "cm": cm_val})
        else:
            m1_dist.append({"t": t, "px": 0, "cm": 0})

        # --- M2 Waffenarm-Winkel ---
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

        # --- M3 Lunge-Tiefe ---
        def lunge_depth(kpts, prev_kpts):
            if kpts is None or prev_kpts is None:
                return None
            k_l = get_kp(kpts, 15)
            k_r = get_kp(kpts, 16)
            if not k_l or not k_r:
                return None
            front = k_l if k_l[1] > k_r[1] else k_r
            hip = get_mid(kpts, 11, 12)
            if hip:
                return max(0, round(hip[1] - front[1], 1))
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

        # --- M4 Bewegungs-Pfad ---
        m4_m_path.append({"t": t, "x": m_hip[0] if m_hip else 0, "y": m_hip[1] if m_hip else 0})
        m4_g_path.append({"t": t, "x": g_hip[0] if g_hip else 0, "y": g_hip[1] if g_hip else 0})

        # --- M5 Korperhaltung ---
        def body_tilt(kpts):
            neck = get_mid(kpts, 5, 6)
            hip = get_mid(kpts, 11, 12)
            if neck and hip:
                dx = neck[0] - hip[0]
                dy = neck[1] - hip[1]
                angle = math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0
                sign = -1 if dx < 0 else 1
                return round(angle * sign, 1)
            return 0

        m5_m_tilt.append({"t": t, "deg": body_tilt(mk) if mk else 0})
        m5_g_tilt.append({"t": t, "deg": body_tilt(gk) if gk else 0})

        # --- M6 Beschleunigung + M8 Geschwindigkeit ---
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

        # --- M7 Schritt-Rhythmus ---
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

        # --- M9: Waffenhand-Höhe (relativ zu Schulter, negativ = Hand unter Schulter) ---
        def hand_height(kpts):
            sh = get_kp(kpts, 6) or get_kp(kpts, 5)  # rechte Schulter, fallback linke
            wr = get_kp(kpts, 10) or get_kp(kpts, 9)  # rechtes Handgelenk, fallback linkes
            if sh and wr:
                return round(sh[1] - wr[1], 1)  # positiv = Hand über Schulter (hohe Guard)
            return 0
        m9_m_hand_h.append({"t": t, "px": hand_height(mk) if mk else 0})
        m9_g_hand_h.append({"t": t, "px": hand_height(gk) if gk else 0})

        # --- M10: Arm-Streckung (Schulter→Handgelenk Distanz) ---
        def arm_extension(kpts):
            sh = get_kp(kpts, 6) or get_kp(kpts, 5)
            wr = get_kp(kpts, 10) or get_kp(kpts, 9)
            if sh and wr:
                return round(math.hypot(wr[0]-sh[0], wr[1]-sh[1]), 1)
            return 0
        ext_m = arm_extension(mk) if mk else 0
        ext_g = arm_extension(gk) if gk else 0
        m10_m_ext.append({"t": t, "px": ext_m})
        m10_g_ext.append({"t": t, "px": ext_g})
        m_ext_series.append(ext_m)
        g_ext_series.append(ext_g)
        m_ext_max = max(m_ext_max, ext_m)
        g_ext_max = max(g_ext_max, ext_g)

        # --- M11: Standbreite (Knöchel-Distanz) ---
        def stance_width(kpts):
            la = get_kp(kpts, 15)
            ra = get_kp(kpts, 16)
            if la and ra:
                return round(math.hypot(ra[0]-la[0], ra[1]-la[1]), 1)
            return 0
        m11_m_stance.append({"t": t, "px": stance_width(mk) if mk else 0})
        m11_g_stance.append({"t": t, "px": stance_width(gk) if gk else 0})

        # --- M13: Head-Forward-Index (Nase.x − Hüftmitte.x, positiv = Kopf vor Körper) ---
        def head_forward(kpts):
            nose = get_kp(kpts, 0)
            hip = get_mid(kpts, 11, 12)
            if nose and hip:
                return round(nose[0] - hip[0], 1)
            return 0
        m13_m_head.append({"t": t, "px": head_forward(mk) if mk else 0})
        m13_g_head.append({"t": t, "px": head_forward(gk) if gk else 0})

        prev_m_hip = m_hip
        prev_g_hip = g_hip

        # --- M16: Pressure Index (wer treibt das Gefecht?) ---
        # Bei Seitenansicht: x-Achse = Pisten-Richtung. Michael auf linker Seite → x steigt = Vorwärts
        # Pressure = kumulativer Vorteil: Michael vor + Gegner zurück
        if m_hip and g_hip and prev_m_hip and prev_g_hip:
            # Orientierung: wer ist rechts von wem?
            m_adv_dir = 1 if g_hip[0] > m_hip[0] else -1  # 1: Gegner ist rechts → Michael vor = x+
            g_adv_dir = -m_adv_dir  # Gegner-Vorwärts ist entgegengesetzt
            
            # Michael: positive Bewegung in Richtung Gegner
            m_move = (m_hip[0] - prev_m_hip[0]) * m_adv_dir
            m16_m_advance += max(0, m_move)
            m16_g_advance += max(0, -m_move)  # Michael-Rückzug = Gegner gewinnt Raum
            
            # Gegner: seine Vorwärts-Bewegung in seine Richtung
            g_move = (g_hip[0] - prev_g_hip[0]) * g_adv_dir
            m16_g_advance += max(0, g_move)
            m16_m_advance += max(0, -g_move)  # Gegner-Rückzug = Michael gewinnt Raum
            
            # Netto-Druck: positiv = Michael dominiert
            net = m16_m_advance - m16_g_advance
            m16_pressure.append({"t": t, "net_px": round(net, 1)})
        else:
            if m16_pressure:
                m16_pressure.append({"t": t, "net_px": m16_pressure[-1]["net_px"]})
            else:
                m16_pressure.append({"t": t, "net_px": 0})

    # === M8 Synchronisierung (Post-Loop) ===
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

    # === M12: Distanz-Explosivität (erste Ableitung von Distanz) ===
    m12_expl = []
    for i in range(1, len(m1_dist)):
        dd = abs(m1_dist[i]["cm"] - m1_dist[i-1]["cm"])
        dt = m1_dist[i]["t"] - m1_dist[i-1]["t"]
        if dt > 0:
            m12_expl.append({"t": m1_dist[i]["t"], "cm_s": round(dd/dt, 1)})
        else:
            m12_expl.append({"t": m1_dist[i]["t"], "cm_s": 0})

    # === M14: Treffer-Kandidaten (Arm-Streckung >90% max AND Distanz <15%-Perzentil, min 3 Frames) ===
    dist_cm = [d["cm"] for d in m1_dist if d["cm"] > 0]
    dist_p15 = float(np.percentile(dist_cm, 15)) if dist_cm else 0
    m14_touches = []

    # Pre-compute candidate frames: (frame_idx, reason)
    candidate_flags = []
    for i in range(1, len(m1_dist)):
        ext_m = m10_m_ext[i]["px"] if i < len(m10_m_ext) else 0
        ext_g = m10_g_ext[i]["px"] if i < len(m10_g_ext) else 0
        d_val = m1_dist[i]["cm"]

        m_stretched = ext_m > 0.9 * m_ext_max and m_ext_max > 10
        g_stretched = ext_g > 0.9 * g_ext_max and g_ext_max > 10
        close_enough = d_val < dist_p15 and dist_p15 > 0

        if (m_stretched or g_stretched) and close_enough:
            who = []
            if m_stretched: who.append("Michael")
            if g_stretched: who.append("Gegner")
            t_val = m1_dist[i]["t"]
            candidate_flags.append({"i": i, "t": t_val, "who": " + ".join(who), "ext_m": ext_m, "ext_g": ext_g, "dist": d_val})
        else:
            candidate_flags.append(None)

    # Merge consecutive candidate frames into episodes (min 3 frames = ~100ms)
    episodes = []
    current = []
    for idx, cf in enumerate(candidate_flags):
        if cf is not None:
            current.append(cf)
        else:
            if len(current) >= 3:
                episodes.append(current)
            current = []
    if len(current) >= 3:
        episodes.append(current)

    for ep in episodes:
        mid = ep[len(ep)//2]
        i = mid["i"]
        d_val = mid["dist"]
        # Check resolve: distance rises >15% within next 10 frames
        future_dist = [m1_dist[j]["cm"] for j in range(i, min(i+10, len(m1_dist))) if m1_dist[j]["cm"] > 0]
        resolves = any(fd > d_val * 1.15 for fd in future_dist) if future_dist else False
        t_val = mid["t"]
        who = mid["who"]
        m14_touches.append({
            "t": t_val,
            "who": who,
            "ext_m": mid["ext_m"],
            "ext_g": mid["ext_g"],
            "dist_cm": d_val,
            "resolves": resolves,
            "confidence": "high" if resolves else "medium",
        })

    # === M15: Rhythmus-Pattern (FFT über Distanz, 5s-Fenster) ===
    m15_rhythm = []
    window_s = 5.0
    window_frames = int(window_s * fps)
    dist_arr = np.array(dist_cm) if dist_cm else np.array([0])

    for i in range(0, len(dist_arr), int(fps * 0.5)):  # alle 0.5s
        win = dist_arr[i:i+window_frames]
        if len(win) < window_frames // 2:
            break
        win_centered = win - np.mean(win)
        fft = np.abs(np.fft.rfft(win_centered))
        freqs = np.fft.rfftfreq(len(win), d=1.0/fps)
        # Nur relevante Fecht-Frequenzen (0.3-4 Hz)
        mask = (freqs >= 0.3) & (freqs <= 4.0)
        if mask.sum() > 0:
            peak_idx = int(np.argmax(fft[mask]))
            peak_freq = freqs[mask][peak_idx]
            peak_power = fft[mask][peak_idx]
            t_center = i / fps
            m15_rhythm.append({
                "t": round(t_center, 1),
                "freq_hz": round(peak_freq, 2),
                "power": round(float(peak_power), 1),
            })

    # === Summary ===
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
        # Neue Summaries
        "m_hand_h_avg": round(sum(d["px"] for d in m9_m_hand_h) / max(N, 1), 1),
        "g_hand_h_avg": round(sum(d["px"] for d in m9_g_hand_h) / max(N, 1), 1),
        "m_ext_avg": round(sum(d["px"] for d in m10_m_ext) / max(N, 1), 1),
        "g_ext_avg": round(sum(d["px"] for d in m10_g_ext) / max(N, 1), 1),
        "m_stance_avg": round(sum(d["px"] for d in m11_m_stance) / max(N, 1), 1),
        "g_stance_avg": round(sum(d["px"] for d in m11_g_stance) / max(N, 1), 1),
        "expl_max": round(max(d["cm_s"] for d in m12_expl), 1) if m12_expl else 0,
        "expl_avg": round(sum(d["cm_s"] for d in m12_expl) / max(len(m12_expl), 1), 1) if m12_expl else 0,
        "touches": len(m14_touches),
        "touches_high": len([t for t in m14_touches if t["confidence"] == "high"]),
        "rhythm_dominant": round(max(r["freq_hz"] for r in m15_rhythm), 2) if m15_rhythm else 0,
        # Kalibrierung
        "orientation": orientation,
        "piste_px": round(piste_px, 0),
        "px_per_cm": round(px_per_cm, 2),
        "m16_pressure_net": round(m16_pressure[-1]["net_px"] if m16_pressure else 0, 1),
        "m16_pressure_max": round(max(abs(d["net_px"]) for d in m16_pressure), 1) if m16_pressure else 0,
        "m16_pressure_leader": "Michael" if (m16_pressure and m16_pressure[-1]["net_px"] > 0) else ("Gegner" if (m16_pressure and m16_pressure[-1]["net_px"] < 0) else "neutral"),
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
        "m9_m_hand_h": m9_m_hand_h,
        "m9_g_hand_h": m9_g_hand_h,
        "m10_m_ext": m10_m_ext,
        "m10_g_ext": m10_g_ext,
        "m11_m_stance": m11_m_stance,
        "m11_g_stance": m11_g_stance,
        "m12_expl": m12_expl,
        "m13_m_head": m13_m_head,
        "m13_g_head": m13_g_head,
        "m14_touches": m14_touches,
        "m15_rhythm": m15_rhythm,
        "m16_pressure": m16_pressure,
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
