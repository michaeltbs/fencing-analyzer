"""
test_video_pipeline.py — Unit tests for pause detection + video utils.

Tests:
  - video_utils.probe_video works on a synthetic mp4
  - pause_detector finds active segments in a synthetic 10s video
  - scheduler frame-range calculation is consistent
"""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_utils import probe_video, has_ffmpeg
from pause_detector import PauseDetector


FIXTURE = Path(__file__).parent / "fixture_10s.mp4"


@pytest.fixture(scope="module")
def fixture_video():
    if not FIXTURE.exists():
        pytest.skip("fixture_10s.mp4 not generated; run 'make test-fixture'")
    return FIXTURE


def test_ffmpeg_available():
    assert has_ffmpeg(), "ffmpeg not installed — video pipeline tests need it"


def test_probe_video(fixture_video):
    info = probe_video(fixture_video)
    assert info["fps"] == 30.0
    assert info["duration_s"] == pytest.approx(10.0, abs=0.5)
    assert info["total_frames"] == pytest.approx(300, abs=10)
    assert info["width"] == 320
    assert info["height"] == 240


def test_pause_detector_fast(fixture_video):
    det = PauseDetector(str(fixture_video), verbose=False)
    det.scan_motion(mode="fast")
    segments = det.find_bout_segments()

    # Synthetic testsrc has constant motion → at least one active segment
    assert len(segments) > 0
    assert any(typ == "active" for typ, _, _ in segments)
    assert det.fps == 30.0
    assert det.duration_s == pytest.approx(10.0, abs=0.5)


def test_pause_detector_profile(fixture_video, tmp_path):
    det = PauseDetector(str(fixture_video), verbose=False)
    det.scan_motion(mode="fast")
    profile_path = tmp_path / "motion_profile.json"
    det.save_profile(profile_path)
    assert profile_path.exists()
    assert profile_path.stat().st_size > 50


# === Scheduler frame-range sanity ===

def test_scheduler_frame_range_math():
    fps = 30.0
    seg_start = 5.5
    seg_end = 12.3
    start_frame = int(seg_start * fps)
    end_frame = int(seg_end * fps)
    assert start_frame == 165
    assert end_frame == 369
    # Reverse check
    assert start_frame / fps == pytest.approx(5.5, abs=1 / fps)
    assert end_frame / fps == pytest.approx(12.3, abs=1 / fps)


# === Clean helpers ===

def test_extract_subclip(fixture_video, tmp_path):
    from video_utils import extract_subclip
    out = tmp_path / "sub.mp4"
    result = extract_subclip(fixture_video, out, start_frame=60, end_frame=120)
    assert result.exists()
    info = probe_video(result)
    assert info["duration_s"] == pytest.approx(2.0, abs=0.2)
    assert info["fps"] == 30.0
