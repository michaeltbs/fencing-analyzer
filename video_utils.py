"""
video_utils.py — Shared video helpers (ffprobe/ffmpeg).

Avoids duplicating ffprobe parsing across pause_detector, worker_chunk_analyze,
and preview_generator.
"""
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def probe_video(path: str | Path) -> dict:
    """
    Probe a video file with ffprobe.

    Returns:
        dict with keys: fps, duration_s, total_frames, width, height
    """
    path = Path(path)
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration,width,height",
        "-of", "default=noprint_wrappers=1", str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")

    out = {"fps": 30.0, "duration_s": 0.0, "total_frames": 0, "width": 0, "height": 0}
    for line in result.stdout.split("\n"):
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        value = value.strip()
        if key == "r_frame_rate":
            try:
                num, den = value.split("/")
                out["fps"] = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
        elif key == "duration":
            try:
                out["duration_s"] = float(value)
            except ValueError:
                pass
        elif key == "width":
            out["width"] = int(value)
        elif key == "height":
            out["height"] = int(value)

    if out["duration_s"] and out["fps"]:
        out["total_frames"] = int(out["duration_s"] * out["fps"])

    return out


def extract_subclip(
    src: str | Path,
    dst: str | Path,
    start_frame: int,
    end_frame: Optional[int] = None,
    fps: Optional[float] = None,
) -> Path:
    """
    Extract a subclip [start_frame, end_frame) using ffmpeg -ss/-t.
    If fps is not provided, it will be probed.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fps is None:
        fps = probe_video(src)["fps"]

    ss = start_frame / fps
    duration = None
    if end_frame is not None:
        duration = (end_frame - start_frame) / fps

    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(ss), "-i", str(src)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-c", "copy", str(dst)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg subclip failed: {result.stderr}")

    return dst


def has_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0


def has_ffprobe() -> bool:
    """Check if ffprobe is available."""
    return subprocess.run(["ffprobe", "-version"], capture_output=True).returncode == 0


__all__ = ["probe_video", "extract_subclip", "has_ffmpeg", "has_ffprobe"]
