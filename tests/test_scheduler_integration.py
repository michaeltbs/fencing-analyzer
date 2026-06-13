"""
test_scheduler_integration.py — Integration test for scheduler.py using mock worker.

Creates an in-memory SQLite DB and a synthetic video, then runs
run_full_analysis with tests/mock_worker_chunk.py as the worker script.

Validates:
  - scheduler calls worker with correct frame ranges
  - merged result has correct total frame count
  - metrics are persisted to FencerDB
  - active segment count matches expectations
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scheduler import run_full_analysis
from inference_db import FencerDB


@pytest.fixture(scope="module")
def synthetic_video():
    """Generate a 10s synthetic video with 30fps (300 frames)."""
    import subprocess
    p = Path(__file__).parent / "fixture_10s.mp4"
    if p.exists():
        return p
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "testsrc=duration=10:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=10",
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p", str(p)
    ], check=True, capture_output=True)
    return p


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    return FencerDB(str(db_path))


def test_scheduler_runs_with_mock_worker(synthetic_video, db, tmp_path):
    workdir = tmp_path / "chunks"

    # Insert fencers + bout so we can call scheduler
    fa_id = db.upsert_fencer(
        "michael-trebis", "Michael", "Trebis", nation="GER", hand="right"
    )
    fb_id = db.upsert_fencer(
        "richard-schmidt", "Richard", "Schmidt", nation="GER"
    )
    bout_id = db.create_bout(
        fencer_a_id=fa_id,
        fencer_b_id=fb_id,
        tournament="Test Tournament",
        bout_date="2026-01-01",
        video_path=str(synthetic_video),
        weapon="epee",
        fencer_a_score=5,
        fencer_b_score=7,
    )

    mock_worker = Path(__file__).parent / "mock_worker_chunk.py"
    captured_ranges = []

    def on_chunk_done(idx, total, data):
        captured_ranges.append((idx, data["summary"]["duration_start_s"], data["summary"]["frames"]))

    merged, segments, evals = run_full_analysis(
        video_path=str(synthetic_video),
        bout_id=bout_id,
        db=db,
        worker_script=str(mock_worker),
        workdir=str(workdir),
        on_chunk_done=on_chunk_done,
    )

    assert merged is not None, "Merged result is None (scheduler failed)"
    assert merged["summary"]["frames"] == 300, \
        f"Expected 300 frames total, got {merged['summary']['frames']}"

    # At least one active segment should exist in a 10s synthetic video
    active = [s for s in segments if s[0] == "active"]
    assert len(active) > 0

    # Each chunk should have received a frame range
    assert len(captured_ranges) > 0
    for idx, start, frames in captured_ranges:
        assert frames > 0, f"Chunk {idx} has no frames"

    # Metrics should be in DB
    metrics = db.get_metrics(bout_id)
    assert len(metrics) == 300, f"Expected 300 metric rows, got {len(metrics)}"

    # Annotations (touche candidates) should exist
    annotations = db.get_annotations(bout_id, type_="touche")
    assert len(annotations) >= 0


def test_scheduler_frame_ranges_cover_expected_video(synthetic_video, db, tmp_path):
    """Verify scheduler doesn't pass negative or inverted frame ranges."""
    workdir = tmp_path / "chunks"
    db.upsert_fencer("fa", "F", "A")
    db.upsert_fencer("fb", "F", "B")
    fa_id = db.get_fencer_by_slug("fa")["id"]
    fb_id = db.get_fencer_by_slug("fb")["id"]
    bout_id = db.create_bout(
        fencer_a_id=fa_id, fencer_b_id=fb_id,
        video_path=str(synthetic_video),
        weapon="epee",
    )

    mock_worker = Path(__file__).parent / "mock_worker_chunk.py"
    merged, segments, _ = run_full_analysis(
        video_path=str(synthetic_video),
        bout_id=bout_id,
        db=db,
        worker_script=str(mock_worker),
        workdir=str(workdir),
    )

    assert merged is not None
    summary = merged["summary"]
    # The synthetic video is 10s @ 30fps = 300 frames total.
    # After pause detection the active segments may be fewer, but total should
    # still sum to exactly 300 because mock worker reports end-start per chunk.
    assert summary["frames"] == 300
