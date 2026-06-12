"""
subagent_eval.py — Quality evaluation for analyzed fencing bouts.

Three evaluator roles, each implemented as a method that calls a subagent
via `delegate_task`:

  1. eval_chunk(idx, total, chunk_data)
     - Per-chunk quality scorer
     - Checks: YOLO confidence, keypoint coverage, touché candidate plausibility,
       tracking consistency
     - Returns: {score, issues, suggestions}

  2. eval_final(merged_result, segments)
     - Cross-chunk consistency + overall sanity
     - Checks: realistic touché count for bout duration, pressure index trend,
       cross-chunk tracking continuity
     - Returns: {score, issues, suggestions}

  3. eval_quick(chunk_data)  [optional, sync heuristic-based]
     - Lightweight stats-only evaluator
     - No subagent call, runs in-process
     - Useful for fast feedback during analysis

When called outside Hermes (e.g. standalone GPU machine), falls back to
the heuristic-only evaluator.

Usage:
    from subagent_eval import SubagentEvaluator
    evaluator = SubagentEvaluator(db=db)
    eval_result = evaluator.eval_chunk(0, 4, chunk_data)
"""
import json
import time
import traceback
from typing import Dict, List, Optional, Any


# === Heuristic-only evaluator (always available) ===

class HeuristicEvaluator:
    """
    Lightweight in-process evaluator using only the chunk data.
    No subagent calls, no external dependencies.
    """

    def eval_chunk(self, idx, total, chunk_data) -> Dict[str, Any]:
        """Score a single chunk on data-quality signals."""
        issues = []
        suggestions = []
        score = 5

        summary = chunk_data.get("summary", {})
        frame_data = chunk_data.get("frame_data", [])
        m1_dist = chunk_data.get("m1_dist", [])

        # 1. Frame coverage
        n_frames = len(frame_data)
        n_with_data = sum(1 for f in frame_data if f.get("m") and f.get("g"))
        coverage = n_with_data / max(n_frames, 1)
        if coverage < 0.7:
            issues.append(f"Low frame coverage: {coverage*100:.0f}% (target: >90%)")
            score -= 2
        elif coverage < 0.9:
            issues.append(f"Moderate frame coverage: {coverage*100:.0f}%")
            score -= 1

        # 2. Distance validity
        dist_vals = [d.get("cm", 0) for d in m1_dist if d.get("cm", 0) > 0]
        if dist_vals:
            dist_median = sorted(dist_vals)[len(dist_vals)//2]
            if dist_median < 30:
                issues.append(f"Median distance only {dist_median:.0f}cm — fencers may be too close, check tracking")
                score -= 1
            elif dist_median > 400:
                issues.append(f"Median distance {dist_median:.0f}cm — fencers rarely engage, check detection")
                score -= 1
            if min(dist_vals) < 5:
                issues.append("Min distance < 5cm — possible keypoint merge or overlap detection")
                score -= 1

        # 3. Step counter sanity
        m_steps = summary.get("m_steps", 0)
        g_steps = summary.get("g_steps", 0)
        chunk_duration = summary.get("duration", 0)
        if chunk_duration > 0:
            m_step_rate = m_steps / chunk_duration
            g_step_rate = g_steps / chunk_duration
            if m_step_rate < 0.5 or m_step_rate > 8:
                issues.append(f"Michael step rate {m_step_rate:.1f}/s seems off")
                score -= 1
            if g_step_rate < 0.5 or g_step_rate > 8:
                issues.append(f"Gegner step rate {g_step_rate:.1f}/s seems off")
                score -= 1

        # 4. Touché candidate plausibility
        touches = chunk_data.get("m14_touches", [])
        n_touches = len(touches)
        if chunk_duration > 0:
            touch_rate = n_touches / (chunk_duration / 60)  # per minute
            if touch_rate > 20:
                issues.append(f"Very high touché rate: {touch_rate:.1f}/min (likely false positives)")
                suggestions.append("Consider raising M14 confidence threshold from 0.9 to 0.95")
                score -= 1
            elif touch_rate > 10 and n_touches > 0:
                suggestions.append(f"Touché rate {touch_rate:.1f}/min — verify in highlight reel")

        # 5. Pressure index drift
        pressure = chunk_data.get("m16_pressure", [])
        if pressure:
            net_values = [p.get("net_px", 0) for p in pressure]
            if max(net_values) - min(net_values) > 1000:
                suggestions.append("Large pressure swing (>1000px) — possible ID swap")

        score = max(1, min(5, score))
        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "metrics": {
                "frame_coverage": round(coverage * 100, 1),
                "dist_median_cm": round(sorted(dist_vals)[len(dist_vals)//2], 1) if dist_vals else 0,
                "touch_rate_per_min": round(n_touches / max(chunk_duration / 60, 0.01), 1),
                "n_touches": n_touches,
                "n_frames": n_frames,
            }
        }

    def eval_final(self, merged_result, segments) -> Dict[str, Any]:
        """Cross-chunk sanity check."""
        issues = []
        suggestions = []
        score = 5

        summary = merged_result.get("summary", {})
        touches = merged_result.get("m14_touches", [])
        n_touches = summary.get("touches", 0)
        n_high = summary.get("touches_high", 0)
        duration = summary.get("duration", 0)

        # 1. Realistic touché count
        if duration > 0:
            touch_per_min = n_touches / (duration / 60)
            if touch_per_min > 15:
                issues.append(f"Unrealistic touché rate: {touch_per_min:.1f}/min (typical epee: 3-8/min)")
                suggestions.append("M14 thresholds too lenient, raise min consecutive frames from 3 to 5")
                score -= 2
            elif touch_per_min < 0.5 and duration > 300:
                issues.append(f"Very low touché rate: {touch_per_min:.1f}/min — possible under-detection")
                score -= 1

            # 2. High vs medium ratio
            if n_touches > 0:
                high_ratio = n_high / n_touches
                if high_ratio < 0.3:
                    suggestions.append(f"Only {high_ratio*100:.0f}% high-confidence touchés — most uncertain")

        # 3. Cross-chunk tracking continuity
        chunks = merged_result.get("chunks", [])
        if len(chunks) > 1:
            # Check that consecutive chunks have reasonable metric continuity
            for i, c in enumerate(chunks):
                if c.get("frame_count", 0) < 50:
                    issues.append(f"Chunk {i+1} very short ({c.get('frame_count', 0)} frames) — "
                                  "may indicate tracking failure")
                    score -= 1

        # 4. Distance distribution
        m1_dist = merged_result.get("m1_dist", [])
        dist_vals = [d.get("cm", 0) for d in m1_dist if d.get("cm", 0) > 0]
        if dist_vals:
            dist_std = (sum((d - sum(dist_vals)/len(dist_vals))**2 for d in dist_vals) / len(dist_vals)) ** 0.5
            if dist_std < 20:
                issues.append(f"Distance std only {dist_std:.1f}cm — fencers barely move apart, "
                              "may indicate tracking stuck")
                score -= 1

        # 5. Pressure index trend
        pressure = merged_result.get("m16_pressure", [])
        if pressure and len(pressure) > 10:
            first_half = pressure[:len(pressure)//2]
            second_half = pressure[len(pressure)//2:]
            avg_first = sum(p.get("net_px", 0) for p in first_half) / len(first_half)
            avg_second = sum(p.get("net_px", 0) for p in second_half) / len(second_half)
            if abs(avg_second - avg_first) > 500:
                leader = "Michael" if avg_second > 0 else "Gegner"
                suggestions.append(f"{leader} dominates second half "
                                    f"(Δ {avg_second - avg_first:+.0f}px)")

        score = max(1, min(5, score))
        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "summary": {
                "n_chunks": len(chunks),
                "n_active_segments": len([s for s in segments if s[0] == "active"]),
                "n_touches": n_touches,
                "n_high": n_high,
                "duration_s": duration,
            }
        }


# === Subagent-based evaluator (calls delegate_task) ===

class SubagentEvaluator(HeuristicEvaluator):
    """
    Wrapper that uses Hermes subagents for richer evaluation.
    Falls back to HeuristicEvaluator if subagent calls fail or unavailable.
    """

    def __init__(self, db=None, use_subagent=True):
        self.db = db
        self.use_subagent = use_subagent
        # We don't store the delegate_task tool here — it's invoked by the caller
        # via the `delegate_task` parameter on the closure. Instead, we expose
        # the prompts and let the scheduler call delegate_task.
        self.heuristic = HeuristicEvaluator()

    def eval_chunk(self, idx, total, chunk_data) -> Dict[str, Any]:
        """
        Evaluate a chunk. Tries subagent first; on failure, falls back to heuristic.
        """
        if not self.use_subagent:
            return self.heuristic.eval_chunk(idx, total, chunk_data)

        # Build prompt for subagent
        try:
            prompt = self._build_chunk_prompt(idx, total, chunk_data)
            # The actual delegate_task call is in analyze_full.py — we return
            # the prompt + heuristic fallback here. This keeps the scheduler
            # framework-agnostic.
            heuristic_result = self.heuristic.eval_chunk(idx, total, chunk_data)
            heuristic_result["subagent_prompt"] = prompt
            heuristic_result["subagent_available"] = True
            return heuristic_result
        except Exception:
            return self.heuristic.eval_chunk(idx, total, chunk_data)

    def eval_final(self, merged_result, segments) -> Dict[str, Any]:
        if not self.use_subagent:
            return self.heuristic.eval_final(merged_result, segments)
        try:
            prompt = self._build_final_prompt(merged_result, segments)
            heuristic_result = self.heuristic.eval_final(merged_result, segments)
            heuristic_result["subagent_prompt"] = prompt
            heuristic_result["subagent_available"] = True
            return heuristic_result
        except Exception:
            return self.heuristic.eval_final(merged_result, segments)

    def _build_chunk_prompt(self, idx, total, chunk_data) -> str:
        """Build a prompt for a subagent to evaluate this chunk."""
        summary = chunk_data.get("summary", {})
        touches = chunk_data.get("m14_touches", [])
        chunk_dur = summary.get("duration", 0)

        return f"""Du bist ein Quality-Assurance-Agent für Fecht-Video-Analyse. Bewerte Chunk {idx+1}/{total}.

**Chunk-Daten:**
- Dauer: {chunk_dur:.1f}s
- Frames: {summary.get('frames', 0)}
- Distanz ⌀: {summary.get('dist_avg', 0):.1f}cm (min {summary.get('dist_min', 0):.0f}cm, max {summary.get('dist_max', 0):.0f}cm)
- Waffenarm-Winkel M/G: {summary.get('m_angle_avg', 0):.1f}° / {summary.get('g_angle_avg', 0):.1f}°
- Schritte M/G: {summary.get('m_steps', 0)} / {summary.get('g_steps', 0)}
- Touché-Kandidaten: {len(touches)} ({summary.get('touches_high', 0)} high-confidence)

**Deine Aufgabe:**
1. Bewerte 1-5: Wie plausibel sind diese Werte für ein Epee-Gefecht?
2. Liste 2-3 konkrete Probleme (z.B. "Distanzmedian zu klein", "Schrittrate ungewöhnlich")
3. Liste 1-2 Verbesserungsvorschläge (z.B. "M14 Threshold anheben")

**Antwortformat:**
```
SCORE: <1-5>
ISSUES: <bullet 1>, <bullet 2>, ...
SUGGESTIONS: <bullet 1>, ...
```

Kurze Antwort, kein Padding."""

    def _build_final_prompt(self, merged_result, segments) -> str:
        """Build prompt for final cross-chunk evaluation."""
        summary = merged_result.get("summary", {})
        chunks = merged_result.get("chunks", [])

        return f"""Du bist ein Sanity-Check-Agent für eine komplette Fecht-Analyse. Bewerte das Gesamtergebnis.

**Bout-Stats:**
- Dauer: {summary.get('duration', 0):.0f}s ({summary.get('duration', 0)/60:.1f} min)
- Frames: {summary.get('frames', 0)}
- Chunks: {len(chunks)}
- Touché-Kandidaten: {summary.get('touches', 0)} ({summary.get('touches_high', 0)} high)
- Touché-Rate: {summary.get('touches', 0) / max(summary.get('duration', 1)/60, 0.01):.1f}/min
- Distanz ⌀: {summary.get('dist_avg', 0):.1f}cm
- Druck-Index (winner): {summary.get('m16_pressure_leader', '?')}

**Deine Aufgabe:**
1. Bewerte 1-5: Sind diese Werte für ein Epee-Gefecht realistisch?
2. Liste 2-3 Probleme (z.B. "Touché-Rate zu hoch", "Druck-Index-Wert unplausibel")
3. Liste 1-2 Sanity-Check-Vorschläge

**Antwortformat:**
```
SCORE: <1-5>
ISSUES: <bullet 1>, <bullet 2>, ...
SUGGESTIONS: <bullet 1>, ...
```

Kurze Antwort, kein Padding."""


# === Callable adapter for scheduler ===
def make_subagent_evaluator(eval_chunk_fn=None, eval_final_fn=None):
    """
    Build an evaluator dict that can be passed to scheduler.
    Returns: {"eval_chunk": fn, "eval_final": fn}
    """
    se = SubagentEvaluator()
    return {
        "eval_chunk": eval_chunk_fn or se.eval_chunk,
        "eval_final": eval_final_fn or se.eval_final,
    }
