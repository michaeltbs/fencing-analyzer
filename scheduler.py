"""
scheduler.py — Orchestrates chunked full-length fencing analysis.

Workflow:
  1. Pause detection on source video
  2. Split into active segments (skip long pauses)
  3. Each active segment -> chunk of source frames
  4. For each chunk: spawn worker_chunk_analyze.py subprocess
  5. Merge chunk results (frame_data concat, metric arrays concat)
  6. Insert metrics into FencerDB
  7. Add annotations from touché candidates
  8. (Optional) Subagent evaluation loop

Public API:
    from scheduler import run_full_analysis
    run_full_analysis(
        video_path, bout_id, db,
        max_parallel=1,        # GPU-bound: 1 (most desktops have 1 GPU)
        on_chunk_done=None,    # callback(chunk_idx, total, result)
        use_subagent_eval=False
    )
"""
import json
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable, List, Dict, Optional

import logging

logger = logging.getLogger(__name__)



CHUNKS_DIR_NAME = ".chunks"


def run_full_analysis(
    video_path,
    bout_id,
    db,  # FencerDB instance
    max_parallel=1,
    worker_script: str = "worker_chunk_analyze.py",
    on_chunk_done: Optional[Callable] = None,
    on_segment_done: Optional[Callable] = None,
    chunk_evaluator: Optional[Callable] = None,  # (idx, total, chunk_data) -> eval dict
    final_evaluator: Optional[Callable] = None,   # (merged_result, segments) -> eval dict
    use_subagent_eval=False,
    keep_chunks=True,
    workdir=None,
):
    """
    Run full-length analysis on a fencing video.

    Args:
        video_path: path to source video
        bout_id:    FencerDB bout ID
        db:         FencerDB instance
        max_parallel: chunk-level parallelism (1 for single-GPU)
        worker_script: path to the chunk worker script (default: worker_chunk_analyze.py)
        on_chunk_done: optional callback (idx, total, result_dict)
        on_segment_done: optional callback (idx, total, segments_list)
        chunk_evaluator: optional callback (idx, total, chunk_data) -> eval dict
                         Called after each chunk completes. Used for subagent eval.
        final_evaluator: optional callback (merged_result, segments) -> eval dict
                         Called once after merge. Used for final sanity check.
        use_subagent_eval: if True and no evaluators set, default to subagent_eval module
        keep_chunks: if False, delete per-chunk result files after merging
        workdir: where to write temp chunks (default: <video_dir>/.chunks/)

    Returns:
        merged_result: dict with all chunk results merged
        segments: list of (type, start_s, end_s) from pause detector
        eval_results: list of chunk eval dicts + final eval dict
    """
    video_path = Path(video_path)
    workdir = Path(workdir) if workdir else video_path.parent / CHUNKS_DIR_NAME
    workdir.mkdir(parents=True, exist_ok=True)

    # Import here to avoid circular import
    from pause_detector import PauseDetector

    # === Step 1: Pause detection ===
    logger.info(f"\n[1/4] Pause detection on {video_path.name}")
    det = PauseDetector(str(video_path), verbose=True)
    det.scan_motion(mode="fast")
    segments = det.find_bout_segments()
    det.save_profile(workdir / "motion_profile.json")
    fps = det.fps

    if on_segment_done:
        on_segment_done(0, 1, segments)

    # === Step 2: Build chunk list ===
    active_segments = [(s, e) for typ, s, e in segments if typ == "active"]
    if not active_segments:
        logger.info("[scheduler] No active segments found!")
        return None, segments

    logger.info(f"\n[2/4] {len(active_segments)} active segments to analyze")

    # === Step 3: Process each chunk ===
    chunk_results = []
    t_total = time.time()

    for chunk_idx, (seg_start, seg_end) in enumerate(active_segments):
        seg_dur = seg_end - seg_start
        result_path = workdir / f"chunk_{chunk_idx:03d}.json"
        logger.info(f"\n  Chunk {chunk_idx+1}/{len(active_segments)}: "
              f"{seg_start:.1f}s - {seg_end:.1f}s ({seg_dur:.1f}s)")

        t0 = time.time()
        # Run worker_chunk_analyze.py with optional frame-range for efficiency.
        start_frame = int(seg_start * fps)
        end_frame = int(seg_end * fps)
        cmd = [
            sys.executable, worker_script,
            str(video_path), str(result_path),
            "--start-frame", str(start_frame),
            "--end-frame", str(end_frame),
            "--time-offset", str(seg_start),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800
            )
            elapsed = time.time() - t0
            if proc.returncode != 0:
                logger.warning(f"  ! Chunk failed: {proc.stderr[-500:]}")
                continue

            if not result_path.exists():
                logger.warning(f"  ! No result file written")
                continue

            with open(result_path) as f:
                chunk_data = json.load(f)
            if "error" in chunk_data:
                logger.warning(f"  ! Chunk error: {chunk_data['error']}")
                continue

            logger.info(f"  OK ({elapsed:.0f}s, {chunk_data.get('summary',{}).get('frames', 0)} frames)")
            chunk_results.append({
                "idx": chunk_idx,
                "seg_start": seg_start,
                "seg_end": seg_end,
                "result_path": str(result_path),
                "elapsed_s": elapsed,
                "data": chunk_data,
            })

            if on_chunk_done:
                on_chunk_done(chunk_idx, len(active_segments), chunk_data)

        except subprocess.TimeoutExpired:
            logger.warning(f"  ! Chunk timeout after 30min")
        except Exception as e:
            logger.warning(f"  ! Chunk exception: {e}")
            traceback.print_exc()

    total_elapsed = time.time() - t_total
    logger.info(f"\n  [scheduler] {len(chunk_results)}/{len(active_segments)} chunks "
          f"completed in {total_elapsed:.0f}s")

    # === Per-chunk evaluation ===
    eval_results = []
    if chunk_results and (chunk_evaluator or use_subagent_eval):
        evaluator = chunk_evaluator
        if evaluator is None and use_subagent_eval:
            from subagent_eval import SubagentEvaluator
            evaluator = SubagentEvaluator(db=db).eval_chunk

        logger.info(f"\n  [scheduler] Running per-chunk evaluation ({len(chunk_results)} chunks)")
        for cr in chunk_results:
            try:
                eval_dict = evaluator(cr["idx"], len(chunk_results), cr["data"])
                eval_dict["chunk_idx"] = cr["idx"]
                eval_dict["seg_start"] = cr["seg_start"]
                eval_dict["seg_end"] = cr["seg_end"]
                eval_results.append(eval_dict)
                logger.info(f"  Chunk {cr['idx']+1} eval: score={eval_dict.get('score', '?')}/5, "
                      f"issues={len(eval_dict.get('issues', []))}")
            except Exception as e:
                logger.warning(f"  ! Chunk {cr['idx']} eval failed: {e}")
                eval_results.append({"chunk_idx": cr["idx"], "error": str(e)})

    # === Step 4: Merge chunk results ===
    logger.info(f"\n[3/4] Merging chunk results")
    merged = merge_chunk_results(chunk_results)

    # === Final evaluation (cross-chunk sanity check) ===
    if final_evaluator or use_subagent_eval:
        f_eval = final_evaluator
        if f_eval is None and use_subagent_eval:
            from subagent_eval import SubagentEvaluator
            f_eval = SubagentEvaluator(db=db).eval_final
        try:
            logger.info(f"\n  [scheduler] Running final evaluation")
            final_eval = f_eval(merged, segments)
            final_eval["type"] = "final"
            eval_results.append(final_eval)
            logger.info(f"  Final eval: score={final_eval.get('score', '?')}/5, "
                  f"issues={len(final_eval.get('issues', []))}")
        except Exception as e:
            logger.warning(f"  ! Final eval failed: {e}")
            eval_results.append({"type": "final", "error": str(e)})

    # === Step 5: Persist to DB ===
    logger.info(f"\n[4/4] Persisting to database")
    _persist_to_db(db, bout_id, merged, video_path)

    # Cleanup
    if not keep_chunks:
        for cr in chunk_results:
            try:
                Path(cr["result_path"]).unlink()
            except OSError:
                pass
        try:
            (workdir / "motion_profile.json").unlink()
        except OSError:
            pass

    return merged, segments, eval_results


def merge_chunk_results(chunk_results: List[Dict]) -> Dict:
    """
    Merge per-chunk results into a single result matching worker_analyze.py schema.

    Frame data is concatenated, metric arrays extended, summary recomputed
    (averages, maxes, counts), touché candidates collected from all chunks.
    """
    if not chunk_results:
        return {"summary": {}, "frame_data": [], "chunks": []}

    merged = {
        "summary": {},
        "frame_data": [],
        # Per-frame metric arrays (extended, not appended)
        "m1_dist": [],
        "m2_m_angle": [], "m2_g_angle": [],
        "m3_m_lunge": [], "m3_g_lunge": [],
        "m4_m_path": [], "m4_g_path": [],
        "m5_m_tilt": [], "m5_g_tilt": [],
        "m6_m_acc": [], "m6_g_acc": [],
        "m7_m_steps": [], "m7_g_steps": [],
        "m8_vel_m": [], "m8_vel_g": [],
        "m9_m_hand_h": [], "m9_g_hand_h": [],
        "m10_m_ext": [], "m10_g_ext": [],
        "m11_m_stance": [], "m11_g_stance": [],
        "m12_expl": [],
        "m13_m_head": [], "m13_g_head": [],
        "m14_touches": [],
        "m15_rhythm": [],
        "m16_pressure": [],
        "chunks": [],  # metadata: list of {idx, seg_start, seg_end, frame_count}
    }

    # Per-frame scalars that need re-indexing across chunks
    raw_vel_m = []
    raw_vel_g = []
    raw_step_m = []
    raw_step_g = []

    for cr in chunk_results:
        data = cr["data"]
        merged["frame_data"].extend(data.get("frame_data", []))

        for key in ["m1_dist", "m2_m_angle", "m2_g_angle",
                    "m3_m_lunge", "m3_g_lunge",
                    "m4_m_path", "m4_g_path",
                    "m5_m_tilt", "m5_g_tilt",
                    "m6_m_acc", "m6_g_acc",
                    "m7_m_steps", "m7_g_steps",
                    "m9_m_hand_h", "m9_g_hand_h",
                    "m10_m_ext", "m10_g_ext",
                    "m11_m_stance", "m11_g_stance",
                    "m12_expl", "m13_m_head", "m13_g_head",
                    "m14_touches", "m15_rhythm",
                    "m16_pressure"]:
            if key in data:
                merged[key].extend(data[key])

        if "m8_vel_m" in data:
            raw_vel_m.extend(data["m8_vel_m"])
            raw_vel_g.extend(data["m8_vel_g"])

        merged["chunks"].append({
            "idx": cr["idx"],
            "seg_start": cr["seg_start"],
            "seg_end": cr["seg_end"],
            "elapsed_s": cr["elapsed_s"],
            "frame_count": len(data.get("frame_data", [])),
        })

    merged["m8_vel_m"] = raw_vel_m
    merged["m8_vel_g"] = raw_vel_g

    # Recompute summary
    merged["summary"] = _recompute_summary(merged)
    return merged


def _recompute_summary(merged: Dict) -> Dict:
    """Recompute summary stats from merged arrays."""
    frame_data = merged.get("frame_data", [])
    N = len(frame_data)
    fps = 30.0  # default

    # Compute fps from last t / N
    if frame_data and frame_data[-1]["t"] > 0 and N > 1:
        t_last = frame_data[-1]["t"]
        # Try to find fps from sampling: if frame_data has regular intervals
        # the rate is N/t_last
        estimated_fps = N / t_last
        if 10 <= estimated_fps <= 120:
            fps = estimated_fps
        else:
            # Fallback: use spacing between first two frames
            if len(frame_data) > 1:
                dt = frame_data[1]["t"] - frame_data[0]["t"]
                if dt > 0:
                    fps = 1.0 / dt

    sum_m1 = [d for d in merged.get("m1_dist", []) if d and d.get("cm")]
    sum_m2m = [d["deg"] for d in merged.get("m2_m_angle", []) if d.get("deg", 0) > 0]
    sum_m2g = [d["deg"] for d in merged.get("m2_g_angle", []) if d.get("deg", 0) > 0]
    sum_m6m = [d["acc"] for d in merged.get("m6_m_acc", []) if d.get("acc")]
    sum_m6g = [d["acc"] for d in merged.get("m6_g_acc", []) if d.get("acc")]
    sum_m3m = [d["px"] for d in merged.get("m3_m_lunge", []) if d.get("px")]
    sum_m3g = [d["px"] for d in merged.get("m3_g_lunge", []) if d.get("px")]
    sum_m9m = [d["px"] for d in merged.get("m9_m_hand_h", [])]
    sum_m9g = [d["px"] for d in merged.get("m9_g_hand_h", [])]
    sum_m10m = [d["px"] for d in merged.get("m10_m_ext", [])]
    sum_m10g = [d["px"] for d in merged.get("m10_g_ext", [])]
    sum_m11m = [d["px"] for d in merged.get("m11_m_stance", [])]
    sum_m11g = [d["px"] for d in merged.get("m11_g_stance", [])]
    sum_m12 = [d["cm_s"] for d in merged.get("m12_expl", []) if d.get("cm_s") is not None]
    sum_m13m = [d["px"] for d in merged.get("m13_m_head", [])]
    sum_m13g = [d["px"] for d in merged.get("m13_g_head", [])]
    sum_m15 = [r["freq_hz"] for r in merged.get("m15_rhythm", []) if r.get("freq_hz")]
    touches = merged.get("m14_touches", [])
    pressure = merged.get("m16_pressure", [])

    vel_m = merged.get("m8_vel_m", [])
    vel_g = merged.get("m8_vel_g", [])
    corr_val = 0.0
    lag_val = 0
    lag_seconds = 0.0
    if min(len(vel_m), len(vel_g)) >= 3:
        import numpy as np
        va = np.array(vel_m, dtype=float)
        vb = np.array(vel_g, dtype=float)
        if va.std() > 0 and vb.std() > 0:
            a = va - va.mean()
            b = vb - vb.mean()
            corr = np.correlate(a, b, mode="full")
            corr = corr / (va.std() * vb.std() * len(a))
            lag_val = int(np.argmax(corr) - (len(a) - 1))
            corr_val = float(np.max(corr))
            lag_seconds = round(lag_val / fps, 2) if fps > 0 else 0

    import numpy as np
    summary = {
        "duration": round(frame_data[-1]["t"], 1) if frame_data else 0,
        "frames": N,
        "fps": round(fps, 1),
        "video": {"fps": round(fps, 1),
                  "duration_s": round(frame_data[-1]["t"], 1) if frame_data else 0,
                  "w": 1920, "h": 1080},
        "dist_avg": round(np.mean([d["cm"] for d in sum_m1]), 1) if sum_m1 else 0,
        "dist_min": round(min(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "dist_max": round(max(d["cm"] for d in sum_m1), 1) if sum_m1 else 0,
        "m_angle_avg": round(np.mean(sum_m2m), 1) if sum_m2m else 0,
        "g_angle_avg": round(np.mean(sum_m2g), 1) if sum_m2g else 0,
        "m_steps": merged.get("m7_m_steps", [])[-1].get("step", 0) if merged.get("m7_m_steps") else 0,
        "g_steps": merged.get("m7_g_steps", [])[-1].get("step", 0) if merged.get("m7_g_steps") else 0,
        "m_acc_avg": round(np.mean(sum_m6m), 1) if sum_m6m else 0,
        "g_acc_avg": round(np.mean(sum_m6g), 1) if sum_m6g else 0,
        "m_acc_max": round(max(sum_m6m), 1) if sum_m6m else 0,
        "g_acc_max": round(max(sum_m6g), 1) if sum_m6g else 0,
        "m_lunge_avg": round(np.mean(sum_m3m), 1) if sum_m3m else 0,
        "g_lunge_avg": round(np.mean(sum_m3g), 1) if sum_m3g else 0,
        "m_hand_h_avg": round(np.mean(sum_m9m), 1) if sum_m9m else 0,
        "g_hand_h_avg": round(np.mean(sum_m9g), 1) if sum_m9g else 0,
        "m_ext_avg": round(np.mean(sum_m10m), 1) if sum_m10m else 0,
        "g_ext_avg": round(np.mean(sum_m10g), 1) if sum_m10g else 0,
        "m_stance_avg": round(np.mean(sum_m11m), 1) if sum_m11m else 0,
        "g_stance_avg": round(np.mean(sum_m11g), 1) if sum_m11g else 0,
        "m_head_avg": round(np.mean(sum_m13m), 1) if sum_m13m else 0,
        "g_head_avg": round(np.mean(sum_m13g), 1) if sum_m13g else 0,
        "expl_max": round(max(sum_m12), 1) if sum_m12 else 0,
        "expl_avg": round(np.mean(sum_m12), 1) if sum_m12 else 0,
        "touches": len(touches),
        "touches_high": len([t for t in touches if t.get("confidence") == "high"]),
        "rhythm_dominant": round(max(sum_m15), 2) if sum_m15 else 0,
        "correlation": round(corr_val, 3),
        "lag_frames": lag_val,
        "lag_seconds": lag_seconds,
        "m16_pressure_net": round(pressure[-1]["net_px"], 1) if pressure else 0,
        "m16_pressure_max": round(max(abs(d["net_px"]) for d in pressure), 1) if pressure else 0,
        "m16_pressure_leader": ("michael" if (pressure and pressure[-1]["net_px"] > 0)
                                else ("gegner" if (pressure and pressure[-1]["net_px"] < 0)
                                      else "neutral")),
        "chunks": len(merged.get("chunks", [])),
        "n_active_segments": len(merged.get("chunks", [])),
    }
    return summary


def _persist_to_db(db, bout_id, merged, video_path):
    """Insert metrics + annotations into FencerDB."""
    if not merged.get("frame_data"):
        return

    db.update_bout_status(bout_id, "processing")

    # Metrics
    db.insert_metrics(
        bout_id,
        merged["frame_data"],
        m1_dist=merged.get("m1_dist"),
        m2_m_angle=merged.get("m2_m_angle"),
        m2_g_angle=merged.get("m2_g_angle"),
        m3_m_lunge=merged.get("m3_m_lunge"),
        m3_g_lunge=merged.get("m3_g_lunge"),
        m5_m_tilt=merged.get("m5_m_tilt"),
        m5_g_tilt=merged.get("m5_g_tilt"),
        m6_m_acc=merged.get("m6_m_acc"),
        m6_g_acc=merged.get("m6_g_acc"),
        m8_vel_m=merged.get("m8_vel_m"),
        m8_vel_g=merged.get("m8_vel_g"),
        m9_m_hand_h=merged.get("m9_m_hand_h"),
        m9_g_hand_h=merged.get("m9_g_hand_h"),
        m10_m_ext=merged.get("m10_m_ext"),
        m10_m_ext_g=merged.get("m10_g_ext"),
        m11_m_stance=merged.get("m11_m_stance"),
        m11_g_stance=merged.get("m11_g_stance"),
        m13_m_head=merged.get("m13_m_head"),
        m13_g_head=merged.get("m13_g_head"),
        m16_pressure=merged.get("m16_pressure"),
        m4_m_path=merged.get("m4_m_path"),
        m4_g_path=merged.get("m4_g_path"),
    )

    # Annotations: touchés
    annotations = []
    for t in merged.get("m14_touches", []):
        annotations.append({
            "t": t["t"],
            "type": "touche",
            "description": f"Touché: {t.get('who', '?')} (conf={t.get('confidence', '?')})",
            "confidence": t.get("confidence", "medium"),
            "source": "auto",
        })

    # Annotations: chunk boundaries
    for c in merged.get("chunks", []):
        annotations.append({
            "t": c["seg_start"],
            "type": "marker",
            "description": f"Segment {c['idx']+1} start (frames={c['frame_count']})",
            "source": "auto",
        })

    if annotations:
        db.bulk_add_annotations(bout_id, annotations)

    # Update bout with summary stats
    s = merged.get("summary", {})
    db.update_bout_status(
        bout_id, "complete",
        completed=True,
    )
    logger.info(f"  [scheduler] Persisted {len(merged['frame_data'])} frames, "
          f"{len(annotations)} annotations")
    logger.info(f"  [scheduler] Stats: {dict((k, v) for k, v in s.items() if k in ('frames','touches','touches_high','dist_avg','m_angle_avg','g_angle_avg'))}")


# === CLI ===
if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.info("Usage: python scheduler.py <video_path> <bout_id> [db_path]")
        sys.exit(1)

    from inference_db import FencerDB

    video = sys.argv[1]
    bid = sys.argv[2]
    db_path = sys.argv[3] if len(sys.argv) > 3 else "fencing.db"

    db = FencerDB(db_path)
    merged, segments = run_full_analysis(video, bid, db)
    logger.info(f"\nFinal summary: {merged.get('summary', {})}")
