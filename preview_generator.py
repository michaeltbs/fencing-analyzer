"""
Preview-Video-Generator mit Supervision Annotatoren
Erzeugt ein MP4 mit eingezeichneten Skeletten (Edges + Vertices)
aus den Analyse-Ergebnissen.

Aufruf: python preview_generator.py <clip_path> <result_json_path> <output_path>

Optional: --smooth-frame-window N (gleicht Keypoints über N Frames)
"""

import sys, json, math
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# COCO Pose skeleton edges (17 keypoints)
COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # face
    (5, 6),                                 # shoulders
    (5, 7), (7, 9),                         # left arm
    (6, 8), (8, 10),                        # right arm
    (5, 11), (6, 12),                       # shoulders → hips
    (11, 12),                               # hips
    (11, 13), (13, 15),                     # left leg
    (12, 14), (14, 16),                     # right leg
]

C_GREEN  = sv.Color(r=0x00, g=0xff, b=0x88)
C_RED    = sv.Color(r=0xff, g=0x44, b=0x66)


def unflatten_kpts(flat, w, h):
    """
    Wandelt flaches [x0,y0, x1,y1, ...] Array in supervision KeyPoints.
    flat: list of 34 floats (x0,y0,...x16,y16), zeros = not detected.
    """
    if flat is None:
        return None
    xy = []
    for i in range(17):
        x, y = float(flat[i*2]), float(flat[i*2+1])
        if x > 0 and y > 0:
            xy.append([x, y])
        else:
            xy.append([0.0, 0.0])
    return np.array([xy], dtype=np.float32)  # (1, 17, 2)


def smooth_window(keypoints_list, window=3):
    """Gleicht Keypoints über `window` Frames (moving average)."""
    if len(keypoints_list) < window:
        return keypoints_list
    smoothed = []
    half_w = window // 2
    for i in range(len(keypoints_list)):
        start = max(0, i - half_w)
        end = min(len(keypoints_list), i + half_w + 1)
        valid = [k for k in keypoints_list[start:end] if k is not None]
        if not valid:
            smoothed.append(None)
        else:
            smoothed.append(np.mean(valid, axis=0))
    return smoothed


def generate_preview(clip_path, result_path, output_path, smooth_win=3):
    """Erzeugt annotiertes Preview-MP4."""
    with open(result_path) as f:
        result = json.load(f)

    if "error" in result:
        raise ValueError(f"Analyse-Fehler: {result['error']}")

    frame_data = result["frame_data"]
    vinfo = result.get("summary", {}).get("video", {})
    fps = result.get("summary", {}).get("fps", 30.0)
    w = vinfo.get("w", 640)
    h = vinfo.get("h", 360)

    # Annotatoren
    edge_m = sv.EdgeAnnotator(color=C_GREEN, thickness=2, edges=COCO_EDGES)
    edge_g = sv.EdgeAnnotator(color=C_RED, thickness=2, edges=COCO_EDGES)
    vert_m = sv.VertexAnnotator(color=C_GREEN, radius=4)
    vert_g = sv.VertexAnnotator(color=C_RED, radius=4)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise ValueError("Kann Video nicht offnen")

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Keypoints vorbereiten
    m_raw_list = []
    g_raw_list = []
    for f in frame_data:
        m_kpts = unflatten_kpts(f.get("m"), w, h)
        g_kpts = unflatten_kpts(f.get("g"), w, h)
        m_raw_list.append(m_kpts)
        g_raw_list.append(g_kpts)

    if smooth_win > 1:
        m_raw_list = smooth_window(m_raw_list, smooth_win)
        g_raw_list = smooth_window(g_raw_list, smooth_win)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Michael (grun)
        m_kpts_arr = m_raw_list[frame_idx] if frame_idx < len(m_raw_list) else None
        if m_kpts_arr is not None:
            kp_m = sv.KeyPoints(xy=m_kpts_arr)
            frame = edge_m.annotate(frame, kp_m)
            frame = vert_m.annotate(frame, kp_m)

        # Gegner (rot)
        g_kpts_arr = g_raw_list[frame_idx] if frame_idx < len(g_raw_list) else None
        if g_kpts_arr is not None:
            kp_g = sv.KeyPoints(xy=g_kpts_arr)
            frame = edge_g.annotate(frame, kp_g)
            frame = vert_g.annotate(frame, kp_g)

        out.write(frame)
        frame_idx += 1

        # Fortschritt
        if frame_idx % 100 == 0:
            print(f"  Preview: Frame {frame_idx}/{len(frame_data)}")

    cap.release()
    out.release()
    print(f"OK: Preview saved to {output_path}")
    return output_path


if __name__ == "__main__":
    clip = Path(sys.argv[1])
    result = Path(sys.argv[2])
    out = Path(sys.argv[3])
    smooth = 3
    if len(sys.argv) > 4:
        smooth = int(sys.argv[4])
    generate_preview(clip, result, out, smooth_win=smooth)