"""
mock_worker_chunk.py — Test helper that simulates worker_chunk_analyze.py.

Reads arguments exactly like the real worker, validates it received a
reasonable frame range, and writes a minimal result JSON.

Usage (called by scheduler.py integration test):
    python tests/mock_worker_chunk.py <video> <result.json>
        [--start-frame N] [--end-frame N] [--time-offset SEC]
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("Usage: mock_worker_chunk.py <video> <result.json> "
              "[--start-frame N] [--end-frame N] [--time-offset SEC]",
              file=sys.stderr)
        sys.exit(1)

    video_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    args = sys.argv[3:]

    start_frame = None
    end_frame = None
    time_offset = 0.0

    i = 0
    while i < len(args):
        if args[i] == "--start-frame" and i + 1 < len(args):
            start_frame = int(args[i + 1])
            i += 2
        elif args[i] == "--end-frame" and i + 1 < len(args):
            end_frame = int(args[i + 1])
            i += 2
        elif args[i] == "--time-offset" and i + 1 < len(args):
            time_offset = float(args[i + 1])
            i += 2
        else:
            i += 1

    result_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = 0
    if start_frame is not None and end_frame is not None:
        n_frames = max(0, end_frame - start_frame)

    data = {
        "summary": {
            "fps": 30.0,
            "duration": (end_frame - start_frame) / 30.0 if end_frame else 0,
            "duration_start_s": time_offset,
            "frames": n_frames,
            "dist_avg": 150.0,
            "dist_min": 80,
            "dist_max": 220,
            "m_angle_avg": 95.0,
            "g_angle_avg": 90.0,
            "m_steps": n_frames // 10,
            "g_steps": n_frames // 10,
            "touches": n_frames // 100,
            "touches_high": n_frames // 200,
        },
        "frame_data": [],
        "m1_dist": [],
        "m14_touches": [
            {"t": time_offset + i * 5, "who": "M", "confidence": "high"}
            for i in range(n_frames // 100)
        ],
    }

    # Add per-frame data
    for f in range(n_frames):
        t = time_offset + f / 30.0
        data["frame_data"].append({"t": round(t, 2), "frame": start_frame + f})
        data["m1_dist"].append({"t": round(t, 2), "v": 150.0})

    with open(result_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"Mock worker: {video_path.name} frames [{start_frame}:{end_frame}] "
          f"offset={time_offset:.1f} -> {result_path}")


if __name__ == "__main__":
    main()
