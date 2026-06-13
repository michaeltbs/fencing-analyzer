"""
pause_detector.py — Motion-based pause detection and bout segmentation.

Two modes:
  - FAST mode (default): ffmpeg scene detection only. 15-min video in ~10s.
    Resets between touches are detected via scene change scoring.
  - FINE mode: per-frame absdiff on scaled frames. Slower (~3-5 min) but
    more accurate for subtle motion. Used in subagent evaluation.

Public API:
    from pause_detector import PauseDetector
    det = PauseDetector("match.mp4")
    det.scan_motion()              # fast scan, populates motion_profile
    segments = det.find_bout_segments()  # list of (type, start_s, end_s)
    det.summary()                  # human-readable overview
"""
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from video_utils import has_ffmpeg, probe_video

logger = logging.getLogger(__name__)


class PauseDetector:
    """Motion-based pause detection for fencing video."""

    # Sample rate (analysis FPS) — 15 Hz is enough for reset detection
    SAMPLE_FPS = 15

    # Segmentation thresholds
    RESET_THRESHOLD = 3.0           # motion < 3% = reset/break
    RESET_MIN_SECONDS = 3           # min break length to qualify as reset
    PAUSE_THRESHOLD = 3.0           # same threshold for genuine pauses
    PAUSE_MIN_SECONDS = 15          # longer = real pause, not inter-touch reset
    ACTIVE_MIN_SECONDS = 5          # min active segment worth keeping

    def __init__(self, video_path, verbose=True):
        self.video_path = Path(video_path)
        self.verbose = verbose
        self.motion_profile = []      # [{"t": float, "motion_pct": float}, ...]
        self.fps = 30.0
        self.duration_s = 0.0
        self.total_frames = 0

    # ------------------------------------------------------------------
    # Motion scanning
    # ------------------------------------------------------------------

    def scan_motion(self, mode="fast"):
        """
        Scan the video and populate self.motion_profile.

        mode="fast": ffmpeg scene detection only (10-30s for 15min video).
                     Detects only sharp motion changes.
        mode="fine": per-frame absdiff on scaled frames (3-5min for 15min).
                     More granular, captures subtle motion.
        """
        self._probe()

        if not has_ffmpeg():
            raise RuntimeError("ffmpeg not found — install ffmpeg and add to PATH")

        if mode == "fast":
            return self._scan_motion_fast()
        else:
            return self._scan_motion_fine()

    def _probe(self):
        """Probe fps + duration via ffprobe."""
        info = probe_video(self.video_path)
        self.fps = info["fps"]
        self.duration_s = info["duration_s"]
        self.total_frames = info["total_frames"]
        if self.verbose:
            logger.info(f"Duration: {self.duration_s:.0f}s ({self.duration_s / 60:.1f} min) @ {self.fps:.0f}fps")

    def _scan_motion_fast(self):
        """
        Fast scene-change detection using ffmpeg's `select=gt(scene,...)` filter.

        ffmpeg runs at native speed (decoded in C, hardware-accelerated if available).
        Outputs a motion score per analysis frame based on whether scene changed.
        """
        t0 = time.time()

        # First pass: collect frame-level scene change scores
        # ffmpeg -vf "showinfo" prints pts_time:  and prev_pts_time: per output frame
        cmd = [
            "ffmpeg", "-v", "info",
            "-i", str(self.video_path),
            "-vf", "scale=320:180,select='gt(scene,0.05)',showinfo",
            "-f", "null",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Parse scene change timestamps from stderr
        scene_times = []
        for line in result.stderr.split("\n"):
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                scene_times.append(round(float(m.group(1)), 1))

        # Build motion profile at SAMPLE_FPS resolution
        # For each sample interval, count scene changes within that bucket
        sample_interval = 1.0 / self.SAMPLE_FPS
        num_samples = int(self.duration_s * self.SAMPLE_FPS) + 1
        buckets = [0] * num_samples

        for st in scene_times:
            idx = int(st / sample_interval)
            if 0 <= idx < num_samples:
                buckets[idx] += 1

        # Convert bucket counts to motion_pct
        # Heuristic: 0 changes = 0% (no motion), 1+ = "high motion" (10-20%)
        for i in range(num_samples):
            t = round(i * sample_interval, 1)
            if t > self.duration_s:
                break
            count = buckets[i]
            if count == 0:
                motion_pct = 0.3  # baseline noise
            elif count == 1:
                motion_pct = 5.0
            else:
                motion_pct = min(20.0, count * 4.0)
            self.motion_profile.append({
                "t": t,
                "motion_pct": motion_pct,
            })

        if self.verbose:
            elapsed = time.time() - t0
            print(f"  [PauseDetector/fast] {len(self.motion_profile)} samples, "
                  f"{len(scene_times)} scene changes in {elapsed:.1f}s")

        return self.motion_profile

    def _scan_motion_fine(self):
        """
        Fine-grained motion scan: per-frame absdiff on scaled frames.

        Uses ffmpeg to pipe 320x180 grayscale frames at SAMPLE_FPS to Python,
        computes absdiff, outputs per-sample motion_pct.

        Slower than fast mode but captures continuous motion (not just scene changes).
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("Fine mode requires opencv-python: pip install opencv-python")

        t0 = time.time()

        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(self.video_path),
            "-vf", f"fps={self.SAMPLE_FPS},scale=320:180,format=gray",
            "-f", "image2pipe",
            "-pix_fmt", "gray8",
            "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=2 ** 20)
        w, h = 320, 180
        frame_size = w * h
        prev = None
        frame_idx = 0

        try:
            while True:
                raw = proc.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                gray = np.frombuffer(raw[:frame_size], dtype=np.uint8).reshape((h, w))
                if prev is not None:
                    diff = cv2.absdiff(gray, prev)
                    score = float(np.mean(diff > 25)) * 100
                    t_sec = frame_idx / self.SAMPLE_FPS
                    self.motion_profile.append({"t": round(t_sec, 2), "motion_pct": round(score, 2)})
                prev = gray
                frame_idx += 1

                if self.verbose and frame_idx % 2000 == 0:
                    pct = min(100, frame_idx / (self.duration_s * self.SAMPLE_FPS) * 100)
                    elapsed = time.time() - t0
                    eta = (elapsed / frame_idx) * (
                        (self.duration_s * self.SAMPLE_FPS) - frame_idx
                    ) if frame_idx > 0 else 0
                    print(f"  [PauseDetector/fine] {frame_idx} samples ({pct:.0f}%) "
                          f"- {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
        finally:
            proc.stdout.close()
            proc.wait()

        if self.verbose:
            elapsed = time.time() - t0
            logger.info(f"  [PauseDetector/fine] Done: {len(self.motion_profile)} samples in {elapsed:.1f}s")

        return self.motion_profile

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def find_resets(self, threshold=None, min_seconds=None):
        """Find inter-touch resets (short breaks between actions)."""
        threshold = threshold or self.RESET_THRESHOLD
        min_seconds = min_seconds or self.RESET_MIN_SECONDS
        return self._find_regions(threshold, min_seconds)

    def find_pauses(self, threshold=None, min_seconds=None):
        """Find genuine pauses (long breaks worth cutting out)."""
        threshold = threshold or self.PAUSE_THRESHOLD
        min_seconds = min_seconds or self.PAUSE_MIN_SECONDS
        return self._find_regions(threshold, min_seconds)

    def find_bout_segments(self, reset_threshold=None, reset_min_s=None,
                           pause_threshold=None, pause_min_s=None,
                           active_min_s=None):
        """
        Segment video into active bout phases + genuine pauses.

        Returns list of (type, start_s, end_s) tuples.
        Types:
          - "active": continuous bout phase
          - "pause": genuine dead time (>pause_min_s)

        Reset boundaries (short breaks between touches) are NOT returned as
        separate segments — they stay inside the active region to preserve
        timeline continuity for the touché detector.
        """
        if not self.motion_profile:
            self.scan_motion()

        reset_th = reset_threshold or self.RESET_THRESHOLD
        reset_ms = reset_min_s or self.RESET_MIN_SECONDS
        pause_th = pause_threshold or self.PAUSE_THRESHOLD
        pause_ms = pause_min_s or self.PAUSE_MIN_SECONDS
        active_ms = active_min_s or self.ACTIVE_MIN_SECONDS

        reset_min_samples = int(reset_ms * self.SAMPLE_FPS)

        all_low = self._find_regions(reset_th, reset_min_samples)
        if not all_low:
            return [("active", 0.0, round(self.duration_s, 1))]

        pauses = [r for r in all_low if r["duration_s"] >= pause_ms]

        segments = []
        cursor = 0.0

        for p in pauses:
            if p["start_t"] - cursor >= active_ms:
                segments.append(("active", round(cursor, 1), round(p["start_t"], 1)))
            segments.append(("pause", round(p["start_t"], 1), round(p["end_t"], 1)))
            cursor = p["end_t"]

        if self.duration_s - cursor >= active_ms:
            segments.append(("active", round(cursor, 1), round(self.duration_s, 1)))

        # Merge adjacent active segments (shouldn't happen after pause split,
        # but safety net)
        merged = []
        for seg in segments:
            if merged and seg[0] == "active" and merged[-1][0] == "active":
                merged[-1] = ("active", merged[-1][1], seg[2])
            else:
                merged.append(seg)
        return merged

    def _find_regions(self, threshold, min_samples):
        """Find contiguous regions where motion < threshold for >= min_samples."""
        if not self.motion_profile:
            return []
        regions = []
        start = None
        length = 0
        for i, m in enumerate(self.motion_profile):
            if m["motion_pct"] < threshold:
                if start is None:
                    start = i
                length += 1
            else:
                if length >= min_samples:
                    regions.append({
                        "start_t": self.motion_profile[start]["t"],
                        "end_t": self.motion_profile[i - 1]["t"],
                        "duration_s": round(
                            self.motion_profile[i - 1]["t"] - self.motion_profile[start]["t"], 1
                        ),
                    })
                start = None
                length = 0
        if length >= min_samples:
            regions.append({
                "start_t": self.motion_profile[start]["t"],
                "end_t": self.motion_profile[-1]["t"],
                "duration_s": round(
                    self.motion_profile[-1]["t"] - self.motion_profile[start]["t"], 1
                ),
            })
        return regions

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self):
        """Print a human-readable analysis summary."""
        if not self.motion_profile:
            self.scan_motion()
        segments = self.find_bout_segments()
        logger.info(f"=== PauseDetector Summary ===")
        logger.info(f"Video: {self.video_path}")
        logger.info(f"Duration: {self.duration_s:.0f}s ({self.duration_s / 60:.1f} min) @ {self.fps:.0f}fps")
        logger.info(f"Samples: {len(self.motion_profile)}")

        active_total = sum(s[2] - s[1] for s in segments if s[0] == "active")
        pause_total = sum(s[2] - s[1] for s in segments if s[0] == "pause")
        active_count = len([s for s in segments if s[0] == "active"])
        pause_count = len([s for s in segments if s[0] == "pause"])

        print(f"Active: {active_total:.0f}s ({active_total / 60:.1f} min) "
              f"across {active_count} segments")
        print(f"Pauses: {pause_total:.0f}s ({pause_total / 60:.1f} min) "
              f"across {pause_count} segments")

        logger.info(f"\nSegments:")
        for stype, s, e in segments:
            dur = e - s
            logger.info(f"  [{stype:6s}] {s:6.1f}s - {e:6.1f}s ({dur:.1f}s)")

        if active_total > 0:
            frames_15fps = int(active_total * self.SAMPLE_FPS)
            est_seconds = int(frames_15fps * 0.2)  # ~200ms/frame YOLO
            logger.info(f"\nEstimated YOLO @ 15fps analysis: {frames_15fps} frames, "                  f"~{est_seconds}s ({est_seconds / 60:.1f} min)")
            logger.info(f"With chunked GPU: ~{est_seconds // 4}s ({est_seconds // 4 / 60:.1f} min)")

        return segments

    def save_profile(self, output_path):
        """Save motion profile to JSON for later inspection."""
        data = {
            "video": str(self.video_path),
            "fps": self.fps,
            "duration_s": self.duration_s,
            "total_frames": self.total_frames,
            "sample_fps": self.SAMPLE_FPS,
            "motion_profile": self.motion_profile,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        return output_path


# === CLI ===
if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    if not video:
        logger.info("Usage: python pause_detector.py <video_path>")
        sys.exit(1)
    det = PauseDetector(video)
    det.summary()
    if len(sys.argv) > 2:
        det.save_profile(sys.argv[2])
