"""
test_worker_frame_range.py — Performance benchmark for frame-range chunking.

Measures wall-clock time for reading a 30s synthetic video with worker_analyze.py
when called on the full video vs a 10s subclip extracted via ffmpeg.

This is an integration benchmark; only runs if --runslow is passed to pytest.
"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

from video_utils import extract_subclip, probe_video, has_ffmpeg


@pytest.fixture(scope="module")
def clip_30s(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench") / "bench_30s.mp4"
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "testsrc=duration=30:size=640x480:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=30",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


def _measure_read_time(video_path, result_json, start_frame=None, end_frame=None):
    """Run worker_analyze.py and return wall-clock duration."""
    # If frame range requested, extract subclip first.
    work_path = video_path
    cleanup = None
    if start_frame is not None or end_frame is not None:
        info = probe_video(video_path)
        sub = result_json.with_suffix(
            f".sub_{start_frame or 0}_{end_frame or 'end'}.mp4"
        )
        work_path = extract_subclip(video_path, sub, start_frame, end_frame, info["fps"])
        cleanup = work_path

    cmd = [sys.executable, "worker_analyze.py", str(work_path), str(result_json)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if cleanup and cleanup.exists():
        try:
            cleanup.unlink()
        except OSError:
            pass

    assert proc.returncode == 0, f"worker_analyze.py failed: {proc.stderr[:500]}"
    assert result_json.exists()
    return elapsed


@pytest.mark.slow
@pytest.mark.skipif(not has_ffmpeg(), reason="ffmpeg required")
def test_frame_range_faster_than_full_read(clip_30s, tmp_path):
    result_full = tmp_path / "full.json"
    result_range = tmp_path / "range.json"

    t_full = _measure_read_time(clip_30s, result_full)
    # 10s from 10s to 20s => frames 300-600
    t_range = _measure_read_time(clip_30s, result_range, start_frame=300, end_frame=600)

    speedup = t_full / max(t_range, 0.001)
    print(f"\nFull read: {t_full:.2f}s | Range read: {t_range:.2f}s | Speedup: {speedup:.1f}x")
    assert speedup >= 2.0, f"expected at least 2x speedup, got {speedup:.1f}x"


# Provide has_ffmpeg here to avoid extra import
def has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False
