"""
worker_chunk_analyze.py — Wraps worker_analyze.py for chunked full-length analysis.

Usage:
  python worker_chunk_analyze.py <clip_path> <result_path> [--start-frame N] [--end-frame N]

Differences from worker_analyze.py:
  - Optional frame-range clipping (read only portion of source)
  - Time-offset tagging in result (so per-frame t reflects source-video time,
    not chunk-local time)
  - Marker file uses same .done convention

The output JSON is identical in schema to worker_analyze.py so existing
downstream (player, reports) doesn't need to change.
"""
import json
import logging
import subprocess
import sys
import time
import traceback
from pathlib import Path

from video_utils import extract_subclip, probe_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python worker_chunk_analyze.py <clip_path> <result_path> "
            "[--start-frame N] [--end-frame N] [--time-offset SECONDS]"
        )
        sys.exit(1)

    clip_path = Path(sys.argv[1])
    result_path = Path(sys.argv[2])
    start_frame = None
    end_frame = None
    time_offset = 0.0

    args = sys.argv[3:]
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

    # If we have a frame range, extract that subclip first using ffmpeg.
    work_clip = clip_path
    cleanup_clip = None
    if start_frame is not None or end_frame is not None:
        out = result_path.with_suffix(
            f".chunk_{start_frame or 0}_{end_frame or 'end'}.mp4"
        )
        info = probe_video(clip_path)
        work_clip = extract_subclip(clip_path, out, start_frame, end_frame, info["fps"])
        cleanup_clip = work_clip

    # Call worker_analyze.py as subprocess so chunking is fully isolated
    # (one YOLO model load per chunk, no shared state).
    t0 = time.time()
    cmd = [
        sys.executable, "worker_analyze.py", str(work_clip), str(result_path)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        logger.error("worker_analyze.py failed: %s", proc.stderr)
        with open(result_path, "w") as f:
            json.dump({
                "error": f"worker_analyze.py failed: {proc.stderr}",
                "returncode": proc.returncode,
            }, f)
        sys.exit(1)

    # If we extracted a subclip, post-process: shift t values by time_offset
    if time_offset > 0 and result_path.exists():
        with open(result_path) as f:
            data = json.load(f)
        _shift_times(data, time_offset)
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    if cleanup_clip and cleanup_clip.exists():
        try:
            cleanup_clip.unlink()
        except OSError:
            pass

    logger.info("Chunk done in %.1fs -> %s", elapsed, result_path)


def _shift_times(data, offset):
    """Shift t values in all per-frame arrays by `offset` seconds."""
    def shift(arr):
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict) and "t" in item:
                    item["t"] = round(item["t"] + offset, 2)
                elif isinstance(item, (int, float)) and "t" not in str(type(item)):
                    pass  # plain number in m8_vel_*, skip

    for key in ["frame_data", "m1_dist", "m2_m_angle", "m2_g_angle",
                "m3_m_lunge", "m3_g_lunge", "m4_m_path", "m4_g_path",
                "m5_m_tilt", "m5_g_tilt", "m6_m_acc", "m6_g_acc",
                "m7_m_steps", "m7_g_steps", "m9_m_hand_h", "m9_g_hand_h",
                "m10_m_ext", "m10_g_ext", "m11_m_stance", "m11_g_stance",
                "m12_expl", "m13_m_head", "m13_g_head", "m14_touches",
                "m15_rhythm", "m16_pressure"]:
        if key in data:
            shift(data[key])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in worker_chunk_analyze")
        sys.exit(1)
