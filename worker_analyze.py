"""
Worker: Fuhrt YOLOv8m-Pose Analyse in einem separaten Prozess aus.
Aufruf: python worker_analyze.py <clip_path> <result_path>

Schreibt Ergebnis-JSON nach result_path. frame_data wird auf [t, m_kpts, g_kpts]
komprimiert (keine numpy Objekte).

Metriken: 1-8 (Basis) + 9-15 (Handhöhe, Arm-Streckung, Standbreite,
Distanz-Explosivität, Head-Forward, Treffer-Kandidat, Rhythmus-Pattern)

Verbessertes Tracking v2:
- Side-Constraint (linker Fechter = Track 0, rechter = Track 1)
- ByteTrack optimiert (min_match=0.55, activation=0.3, min_consecutive=2)
- Velocity-Interpolation bei verlorenen Fechtern (bis 15 Frames)
- Keypoint-Smoothing (Moving Average, 3 Frames)
"""

import sys, json, math, traceback, struct
from pathlib import Path
from collections import deque

clip_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# === Keypoint Smoothing Buffer ===
KPT_SMOOTH_WINDOW = 3


def smooth_kpts(raw_list, window=KPT_SMOOTH_WINDOW):
    """Moving Average uber window Frames fur Keypoints (17x2)."""
    if not raw_list:
        return raw_list
    smoothed = []
    for i in range(len(raw_list)):
        kpt = raw_list[i]
        if kpt is None:
            smoothed.append(None)
            continue
        half = window // 2
        start = max(0, i - half)
        end = min(len(raw_list), i + half + 1)
        valid = []
        for j in range(start, end):
            nj = raw_list[j]
            if nj is not None:
                valid.append(nj)
        if not valid:
            smoothed.append(kpt)
        else:
            smoothed.append(np.mean(valid, axis=0).tolist())
    return smoothed


class VelocityInterpolator:
    """Interpoliert fehlende Keypoints uber bis zu max_gap Frames mit konstanter Geschwindigkeit."""

    def __init__(self, max_gap=15):
        self.max_gap = max_gap
        self.history = deque(maxlen=5)
        self.history_t = deque(maxlen=5)
        self.consecutive_lost = 0
        self.last_valid = None
        self.last_valid_t = 0.0

    def predict(self, t):
        if self.consecutive_lost > self.max_gap:
            return None
        if self.consecutive_lost <= 0:
            return None
        if len(self.history) < 2:
            return self.last_valid

        old = self.history[-2]
        new = self.history[-1]
        dt_h = max(0.001, self.history_t[-1] - self.history_t[-2])

        velocity = []
        for i in range(17):
            ox, oy = old[i * 2], old[i * 2 + 1]
            nx, ny = new[i * 2], new[i * 2 + 1]
            if ox > 0 and oy > 0 and nx > 0 and ny > 0:
                velocity.append((nx - ox) / dt_h)
                velocity.append((ny - oy) / dt_h)
            else:
                velocity.append(0.0)
                velocity.append(0.0)

        dt = t - self.last_valid_t
        predicted = []
        for i in range(17):
            px = self.last_valid[i * 2] + velocity[i * 2] * dt
            py = self.last_valid[i * 2 + 1] + velocity[i * 2 + 1] * dt
            px = max(10, min(3000, px))
            py = max(10, min(3000, py))
            predicted.append(round(px))
            predicted.append(round(py))
        return predicted

    def update(self, flat_kpts, t):
        if flat_kpts is not None and any(v > 0 for v in flat_kpts):
            self.history.append(flat_kpts)
            self.history_t.append(t)
            self.last_valid = flat_kpts
            self.last_valid_t = t
            self.consecutive_lost = 0
        else:
            self.consecutive_lost += 1


def get_kp(flat, idx):
    if flat is None:
        return None
    x, y = flat[idx * 2], flat[idx * 2 + 1]
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

    # --- ByteTrack optimiert ---
    byte_track = sv.ByteTrack(
        minimum_matching_threshold=0.55,
        track_activation_threshold=0.3,
    )
    byte_track.minimum_consecutive_frames = 2

    # --- Side-Constraint Mapping ---
    bt_to_track = {}
    bt_x_history = {}
    side_assigned = False

    frame_data = []
    frame_idx = 0
    raw_m_kpts_list = []
    raw_g_kpts_list = []
    m_interp = VelocityInterpolator(max_gap=15)
    g_interp = VelocityInterpolator(max_gap=15)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        r = results[0]
        t_sec = frame_idx / fps

        detections = sv.Detections.from_ultralytics(r)
        current_kpts = {}

        if len(detections) > 0 and r.keypoints is not None:
            detections = byte_track.update_with_detections(detections)

            bt_positions = {}
            for pi in range(len(detections)):
                tid = int(detections.tracker_id[pi]) if detections.tracker_id is not None else pi
                x1, y1, x2, y2 = map(float, detections.xyxy[pi].tolist())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                if tid not in bt_x_history:
                    bt_x_history[tid] = deque(maxlen=30)
                bt_x_history[tid].append(cx)
                avg_x = np.mean(bt_x_history[tid]) if bt_x_history[tid] else cx

                kpts_data = None
                if r.keypoints.xy is not None and pi < len(r.keypoints.xy):
                    kpts_data = r.keypoints.xy[pi].cpu().numpy().tolist()
                bt_positions[tid] = (avg_x, cy, pi, kpts_data)

            # Side-Assignment erster Frame mit 2+ Detections
            if not side_assigned and len(bt_positions) >= 2:
                sorted_bt = sorted(bt_positions.items(), key=lambda x: x[1][0])
                bt_to_track = {
                    sorted_bt[0][0]: 0,
                    sorted_bt[1][0]: 1,
                }
                side_assigned = True

            if side_assigned:
                for bt_id, (avg_x, cy, pi, kpts_data) in bt_positions.items():
                    mapped = bt_to_track.get(bt_id)
                    if mapped is not None and mapped in TRACK_IDS and kpts_data:
                        current_kpts[mapped] = kpts_data
                    elif bt_id not in bt_to_track and kpts_data:
                        left_occupied = False
                        right_occupied = False
                        for existing_bt, existing_track in bt_to_track.items():
                            if existing_bt in bt_positions:
                                if existing_track == 0:
                                    left_occupied = True
                                else:
                                    right_occupied = True

                        all_active = [(bid, bt_positions[bid][0]) for bid in bt_positions
                                      if bid in bt_to_track or bid == bt_id]
                        all_active_sorted = sorted(all_active, key=lambda x: x[1])

                        for rank, (bid, _) in enumerate(all_active_sorted):
                            if bid == bt_id and rank < 2:
                                target_track = 0 if not left_occupied else (1 if not right_occupied else None)
                                if target_track is not None:
                                    bt_to_track[bt_id] = target_track
                                    if kpts_data:
                                        current_kpts[target_track] = kpts_data
                                    break
            else:
                for pi in range(len(detections)):
                    tid = int(detections.tracker_id[pi]) if detections.tracker_id is not None else pi
                    if r.keypoints.xy is not None and pi < len(r.keypoints.xy):
                        current_kpts[pi % 2] = r.keypoints.xy[pi].cpu().numpy().tolist()

        m_kpts_raw = current_kpts.get(0) if current_kpts else None
        g_kpts_raw = current_kpts.get(1) if current_kpts else None

        flat_m = flatten_kpts(m_kpts_raw)
        flat_g = flatten_kpts(g_kpts_raw)

        m_interp.update(flat_m, t_sec)
        g_interp.update(flat_g, t_sec)

        if flat_m is None:
            flat_m = m_interp.predict(t_sec)
        if flat_g is None:
            flat_g = g_interp.predict(t_sec)

        raw_m_kpts_list.append(flat_m)
        raw_g_kpts_list.append(flat_g)

        frame_data.append({
            "t": round(t_sec, 2),
            "m": flat_m,
            "g": flat_g,
        })

        frame_idx += 1

    cap.release()

    # Keypoint Smoothing (Post-Processing)
    smoothed_m = smooth_kpts(raw_m_kpts_list, KPT_SMOOTH_WINDOW)
    smoothed_g = smooth_kpts(raw_g_kpts_list, KPT_SMOOTH_WINDOW)
    for i in range(len(frame_data)):
        if smoothed_m[i] is not None:
            frame_data[i]["m"] = smoothed_m[i]
        if smoothed_g[i] is not None:
            frame_data[i]["g"] = smoothed_g[i]

    N = len(frame_data)

    # === Kalibrierung ===
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
                if hip_lx > 0:
                    all_hip_x.append(hip_lx)
                if hip_rx > 0:
                    all_hip_x.append(hip_rx)
                if hip_ly > 0:
                    all_hip_y.append(hip_ly)
                if hip_ry > 0:
                    all_hip_y.append(hip_ry)

    x_range = max(all_hip_x) - min(all_hip_x) if all_hip_x else 0
    y_range = max(all_hip_y) - min(all_hip_y) if all_hip_y else 0

    if x_range > y_range * 1.5:
        orientation = "seitlich"
        piste_px = x_range if x_range > 0 else PISTE_WIDTH_PX_FALLBACK
    else:
        orientation = "frontal"
        all_shoulder_dists = []
        for f in frame_data:
            for label in ("m", "g"):
                k = f[label]
                if k:
                    ls = get_kp(k, 5)
                    rs = get_kp(k, 6)
                    if ls and rs:
                        all_shoulder_dists.append(math.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        shoulder_px = np.median(all_shoulder_dists) if all_shoulder_dists else 40
        piste_px = shoulder_px * (PISTE_WIDTH_CM / 40)
    px_per_cm = piste_px / PISTE_WIDTH_CM

    # === Metriken berechnen ===
    m1_dist = []
    m2_m_angle, m2_g_angle = [], []
    m3_m_lunge, m3_g_lunge = [], []
    m4_m_path, m4_g_path = [], []
    m5_m_tilt, m5_g_tilt = [], []
    m6_m_acc, m6_g_acc = [], []
    m7_m_steps, m7_g_steps = [], []
    m8_vel_m, m8_vel_g = [], []

    m9_m_hand_h, m9_g_hand_h = [], []
    m10_m_ext, m10_g_ext = [], []
    m11_m_stance, m11_g_stance = [], []
    m13_m_head, m13_g_head = [], []

    m16_pressure = []
    m16_m_advance = 0
    m16_g_advance = 0

    prev_m_hip, prev_g_hip = None, None
    m_step_active, g_step_active = False, False
    step_counter_m, step_counter_g = 0, 0
    step_times_m, step_times_g = [], []

    m_ext_max, g_ext_max = 0, 0
    m_ext_series, g_ext_series = [], []

    for f in frame_data:
        t = f["t"]
        mk, gk = f["m"], f["g"]

        m_hip = get_mid(mk, 11, 12)
        g_hip = get_mid(gk, 11, 12)

        # M1 Distanz
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
                dot = v1[0] * v2[0] + v1[1] * v2[1]
                n1 = math.hypot(*v1)
                n2 = math.hypot(*v2)
                if n1 > 0 and n2 > 0:
                    angle = math.degrees(math.acos(max(-1, min(1, dot / (n1 * n2)))))
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

        # M4 Bewegungs-Pfad
        m4_m_path.append({"t": t, "x": m_hip[0] if m_hip else 0, "y": m_hip[1] if m_hip else 0})
        m4_g_path.append({"t": t, "x": g_hip[0] if g_hip else 0, "y": g_hip[1] if g_hip else 0})

        # M5 Korperhaltung
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

        # M6 Beschleunigung + M8 Geschwindigkeit
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
            moving_l = a_l and b_l and math.hypot(a_l[0] - b_l[0], a_l[1] - b_l[1]) > 12
            moving_r = a_r and b_r and math.hypot(a_r[0] - b_r[0], a_r[1] - b_r[1]) > 12
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

        # M9: Waffenhand-Hohe
        def hand_height(kpts):
            sh = get_kp(kpts, 6) or get_kp(kpts, 5)
            wr = get_kp(kpts, 10) or get_kp(kpts, 9)
            if sh and wr:
                return round(sh[1] - wr[1], 1)
            return 0

        m9_m_hand_h.append({"t": t, "px": hand_height(mk) if mk else 0})
        m9_g_hand_h.append({"t": t, "px": hand_height(gk) if gk else 0})

        # M10: Arm-Streckung
        def arm_extension(kpts):
            sh = get_kp(kpts, 6) or get_kp(kpts, 5)
            wr = get_kp(kpts, 10) or get_kp(kpts, 9)
            if sh and wr:
                return round(math.hypot(wr[0] - sh[0], wr[1] - sh[1]), 1)
            return 0

        ext_m = arm_extension(mk) if mk else 0
        ext_g = arm_extension(gk) if gk else 0
        m10_m_ext.append({"t": t, "px": ext_m})
        m10_g_ext.append({"t": t, "px": ext_g})
        m_ext_series.append(ext_m)
        g_ext_series.append(ext_g)
        m_ext_max = max(m_ext_max, ext_m)
        g_ext_max = max(g_ext_max, ext_g)

        # M11: Standbreite
        def stance_width(kpts):
            la = get_kp(kpts, 15)
            ra = get_kp(kpts, 16)
            if la and ra:
                return round(math.hypot(ra[0] - la[0], ra[1] - la[1]), 1)
            return 0

        m11_m_stance.append({"t": t, "px": stance_width(mk) if mk else 0})
        m11_g_stance.append({"t": t, "px": stance_width(gk) if gk else 0})

        # M13: Head-Forward-Index
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

        # M16: Pressure Index
        if m_hip and g_hip and prev_m_hip and prev_g_hip:
            m_adv_dir = 1 if g_hip[0] > m_hip[0] else -1
            g_adv_dir = -m_adv_dir

            m_move = (m_hip[0] - prev_m_hip[0]) * m_adv_dir
            m16_m_advance += max(0, m_move)
            m16_g_advance += max(0, -m_move)

            g_move = (g_hip[0] - prev_g_hip[0]) * g_adv_dir
            m16_g_advance += max(0, g_move)
            m16_m_advance += max(0, -g_move)

            net = m16_m_advance - m16_g_advance
            m16_pressure.append({"t": t, "net_px": round(net, 1)})
        else:
            if m16_pressure:
                m16_pressure.append({"t": t, "net_px": m16_pressure[-1]["net_px"]})
            else:
                m16_pressure.append({"t": t, "net_px": 0})

    # M8 Synchronisierung (Post-Loop)
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

    # M12: Distanz-Explosivitat
    m12_expl = []
    for i in range(1, len(m1_dist)):
        dd = abs(m1_dist[i]["cm"] - m1_dist[i - 1]["cm"])
        dt = m1_dist[i]["t"] - m1_dist[i - 1]["t"]
        if dt > 0:
            m12_expl.append({"t": m1_dist[i]["t"], "cm_s": round(dd / dt, 1)})
        else:
            m12_expl.append({"t": m1_dist[i]["t"], "cm_s": 0})

    # M14: Treffer-Kandidaten
    dist_cm = [d["cm"] for d in m1_dist if d["cm"] > 0]
    dist_p15 = float(np.percentile(dist_cm, 15)) if dist_cm else 0
    m14_touches = []

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
            if m_stretched:
                who.append("Michael")
            if g_stretched:
                who.append("Gegner")
            t_val = m1_dist[i]["t"]
            candidate_flags.append({"i": i, "t": t_val, "who": " + ".join(who), "ext_m": ext_m, "ext_g": ext_g, "dist": d_val})
        else:
            candidate_flags.append(None)

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
        mid = ep[len(ep) // 2]
        i = mid["i"]
        d_val = mid["dist"]
        future_dist = [m1_dist[j]["cm"] for j in range(i, min(i + 10, len(m1_dist))) if m1_dist[j]["cm"] > 0]
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

    # M15: Rhythmus-Pattern (FFT)
    m15_rhythm = []
    window_s = 5.0
    window_frames = int(window_s * fps)
    dist_arr = np.array(dist_cm) if dist_cm else np.array([0])

    for i in range(0, len(dist_arr), int(fps * 0.5)):
        win = dist_arr[i:i + window_frames]
        if len(win) < window_frames // 2:
            break
        win_centered = win - np.mean(win)
        fft = np.abs(np.fft.rfft(win_centered))
        freqs = np.fft.rfftfreq(len(win), d=1.0 / fps)
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

    # Summary
    sum_m1 = [d for d in m1_dist if d and d.get("cm")]
    summary = {
        "duration": round(duration, 1),
        "frames": N,
        "fps": round(fps, 1),
        "video": {"fps": round(fps, 1), "duration_s": round(duration, 1), "w": w_frame, "h": h_frame},
        "dist_avg": round(sum(d["cm"] for d in sum_m1) / len(sum_m1), 1) if sum_m1 else 0,
        "dist_min": round(min(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "dist_max": round(max(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "m_angle_avg": round(sum(d["deg"] for d in m2_m_angle if d["deg"] > 0) / max(sum(1 for d in m2_m_angle if d["deg"] > 0), 1), 1),
        "g_angle_avg": round(sum(d["deg"] for d in m2_g_angle if d["deg"] > 0) / max(sum(1 for d in m2_g_angle if d["deg"] > 0), 1), 1),
        "m_steps": step_counter_m, "g_steps": step_counter_g,
        "m_acc_avg": round(sum(d["acc"] for d in m6_m_acc) / max(N, 1), 1),
        "g_acc_avg": round(sum(d["acc"] for d in m6_g_acc) / max(N, 1), 1),
        "m_acc_max": round(max(d["acc"] for d in m6_m_acc), 1),
        "g_acc_max": round(max(d["acc"] for d in m6_g_acc), 1),
        "m_lunge_avg": round(sum(d["px"] for d in m3_m_lunge) / max(N, 1), 1),
        "g_lunge_avg": round(sum(d["px"] for d in m3_g_lunge) / max(N, 1), 1),
        "correlation": round(corr_val, 3), "lag_frames": lag_val,
        "lag_seconds": round(lag_val / fps, 2) if fps > 0 else 0,
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
        "m2_m_angle": m2_m_angle, "m2_g_angle": m2_g_angle,
        "m3_m_lunge": m3_m_lunge, "m3_g_lunge": m3_g_lunge,
        "m4_m_path": m4_m_path, "m4_g_path": m4_g_path,
        "m5_m_tilt": m5_m_tilt, "m5_g_tilt": m5_g_tilt,
        "m6_m_acc": m6_m_acc, "m6_g_acc": m6_g_acc,
        "m7_m_steps": m7_m_steps, "m7_g_steps": m7_g_steps,
        "m8_vel_m": m8_vel_m, "m8_vel_g": m8_vel_g,
        "m9_m_hand_h": m9_m_hand_h, "m9_g_hand_h": m9_g_hand_h,
        "m10_m_ext": m10_m_ext, "m10_g_ext": m10_g_ext,
        "m11_m_stance": m11_m_stance, "m11_g_stance": m11_g_stance,
        "m12_expl": m12_expl,
        "m13_m_head": m13_m_head, "m13_g_head": m13_g_head,
        "m14_touches": m14_touches,
        "m15_rhythm": m15_rhythm,
        "m16_pressure": m16_pressure,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    Path(str(result_path) + ".done").write_text("1")
    print("OK")

except Exception as e:
    with open(result_path, "w") as f:
        json.dump({"error": str(e), "traceback": traceback.format_exc()}, f)