"""
eval_runner.py — Hermes-integrated evaluation runner.

Provides async / sync functions that use `delegate_task` (Hermes subagent
tool) to evaluate fencing analysis chunks with a real LLM in the loop.

Use this from a Hermes agent that has the `delegation` toolset enabled.
For standalone CLI usage (no Hermes), use `subagent_eval.py` directly.

Public API:
    from eval_runner import eval_chunks_with_subagent, eval_final_with_subagent
    chunk_evals = eval_chunks_with_subagent(
        chunk_results,    # list of {idx, seg_start, seg_end, data}
        prompt_template=None  # optional override
    )
    final_eval = eval_final_with_subagent(merged_result, segments)

Returns: list of eval dicts (same shape as subagent_eval.py output)
"""
import json
import re
import time
from typing import Dict, List, Any, Optional


# === Default prompt templates ===

DEFAULT_CHUNK_PROMPT = """Du bist ein Quality-Assurance-Agent für Fecht-Video-Analyse mit YOLOv8m-Pose.

**Chunk {idx}/{total}:**
- Zeit: {seg_start:.1f}s - {seg_end:.1f}s ({duration:.1f}s)
- Frames: {frames}
- Distanz ⌀: {dist_avg:.1f}cm (min {dist_min:.0f}cm, max {dist_max:.0f}cm)
- Waffenarm-Winkel M/G: {m_angle:.1f}° / {g_angle:.1f}°
- Schritte M/G: {m_steps} / {g_steps}
- Touché-Kandidaten: {n_touches} ({n_high} high-confidence)
- Touché-Rate: {touch_rate:.1f}/min

**Deine Aufgabe:**
Bewerte die Qualität 1-5 (5 = perfekt). Liste 2-3 konkrete Probleme wenn <5.
Kurze Antwort, kein Padding.

**Antwortformat (genau so):**
```
SCORE: <1-5>
ISSUES: <bullet>, <bullet>, ...
SUGGESTIONS: <bullet>, ...
```"""


DEFAULT_FINAL_PROMPT = """Du bist ein Sanity-Check-Agent für eine komplette Fecht-Analyse.

**Bout-Stats:**
- Dauer: {duration:.0f}s ({duration_min:.1f} min)
- Frames: {frames}
- Chunks: {n_chunks}
- Touché-Kandidaten: {n_touches} ({n_high} high)
- Touché-Rate: {touch_rate:.1f}/min
- Distanz ⌀: {dist_avg:.1f}cm
- Druck-Index-Sieger: {pressure_leader}

**Deine Aufgabe:**
Bewerte 1-5 (5 = perfekt). Liste 2-3 Probleme bei <5.

**Antwortformat:**
```
SCORE: <1-5>
ISSUES: <bullet>, <bullet>, ...
SUGGESTIONS: <bullet>, ...
```"""


def _format_chunk_prompt(idx, total, chunk_data, template=None) -> str:
    """Format a chunk-eval prompt from chunk data."""
    template = template or DEFAULT_CHUNK_PROMPT
    summary = chunk_data.get("summary", {})
    touches = chunk_data.get("m14_touches", [])
    chunk_dur = summary.get("duration", 1)
    n_touches = len(touches)
    touch_rate = n_touches / max(chunk_dur / 60, 0.01)

    return template.format(
        idx=idx + 1,
        total=total,
        seg_start=summary.get("duration_start_s", 0),
        seg_end=chunk_dur,
        duration=chunk_dur,
        frames=summary.get("frames", 0),
        dist_avg=summary.get("dist_avg", 0),
        dist_min=summary.get("dist_min", 0),
        dist_max=summary.get("dist_max", 0),
        m_angle=summary.get("m_angle_avg", 0),
        g_angle=summary.get("g_angle_avg", 0),
        m_steps=summary.get("m_steps", 0),
        g_steps=summary.get("g_steps", 0),
        n_touches=n_touches,
        n_high=summary.get("touches_high", 0),
        touch_rate=touch_rate,
    )


def _format_final_prompt(merged, segments, template=None) -> str:
    """Format a final-eval prompt from merged data."""
    template = template or DEFAULT_FINAL_PROMPT
    summary = merged.get("summary", {})
    duration = summary.get("duration", 0)
    n_touches = summary.get("touches", 0)
    touch_rate = n_touches / max(duration / 60, 0.01)

    return template.format(
        duration=duration,
        duration_min=duration / 60,
        frames=summary.get("frames", 0),
        n_chunks=summary.get("n_active_segments", len(merged.get("chunks", []))),
        n_touches=n_touches,
        n_high=summary.get("touches_high", 0),
        touch_rate=touch_rate,
        dist_avg=summary.get("dist_avg", 0),
        pressure_leader=summary.get("m16_pressure_leader", "?"),
    )


def _parse_eval_response(text: str) -> Dict[str, Any]:
    """Parse the LLM response into a structured eval dict.

    Expected format:
        SCORE: <1-5>
        ISSUES: <bullet>, <bullet>, ...
        SUGGESTIONS: <bullet>, ...

    Falls back to: extracting any number 1-5 as score, treating entire
    text as issues if format not found.
    """
    score = 3
    issues = []
    suggestions = []

    score_match = re.search(r"SCORE:\s*(\d)", text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
        except (ValueError, IndexError):
            pass
    else:
        # Fallback: any number 1-5
        nums = re.findall(r"\b([1-5])\b", text)
        if nums:
            score = int(nums[0])

    issues = []
    suggestions = []
    current_section = None  # "issues" or "suggestions"

    for line in text.split("\n"):
        line_stripped = line.strip()
        upper = line_stripped.upper()
        if upper.startswith("ISSUES:"):
            current_section = "issues"
            rest = line_stripped[7:].strip()
            if rest:
                issues.extend(s.strip("-*• ") for s in rest.split(",") if s.strip("-*• "))
        elif upper.startswith("SUGGESTIONS:"):
            current_section = "suggestions"
            rest = line_stripped[12:].strip()
            if rest:
                suggestions.extend(s.strip("-*• ") for s in rest.split(",") if s.strip("-*• "))
        elif current_section == "issues" and (line_stripped.startswith("-") or line_stripped.startswith("•")):
            content = line_stripped.lstrip("-*• ").strip()
            if content:
                issues.append(content)
        elif current_section == "suggestions" and (line_stripped.startswith("-") or line_stripped.startswith("•")):
            content = line_stripped.lstrip("-*• ").strip()
            if content:
                suggestions.append(content)

    # Strip empty entries
    issues = [i for i in issues if i]
    suggestions = [s for s in suggestions if s]

    return {
        "score": max(1, min(5, score)),
        "issues": issues,
        "suggestions": suggestions,
        "raw_response": text[:1000],  # cap for log readability
    }


def _call_subagent(prompt: str, context: str = "") -> str:
    """
    Call a Hermes subagent via delegate_task.

    Must be invoked from a Hermes agent context. Falls back gracefully
    (returns heuristic-style text) if delegate_task is unavailable.
    """
    try:
        from hermes_tools import delegate_task
    except ImportError:
        # Not in Hermes — return mock
        return f"SCORE: 3\nISSUES: delegate_task not available\nSUGGESTIONS: run within Hermes agent"

    full_prompt = prompt
    if context:
        full_prompt = f"{prompt}\n\n**Zusätzlicher Kontext:**\n{context}"

    try:
        result = delegate_task(
            goal=full_prompt,
            context="Fencing-Analyse Eval. Antworte in exakt dem geforderten Format.",
            toolsets=["web"],  # minimal toolset
        )
        # Result is a list (delegate_task batch shape) or string
        if isinstance(result, list) and result:
            return str(result[0])
        return str(result)
    except Exception as e:
        return f"SCORE: 3\nISSUES: subagent failed: {e}\nSUGGESTIONS: retry"


def eval_chunks_with_subagent(
    chunk_results: List[Dict],
    prompt_template: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate all chunks sequentially using Hermes subagents.

    Args:
        chunk_results: list of {idx, seg_start, seg_end, data} from scheduler
        prompt_template: optional override for the chunk prompt

    Returns:
        list of eval dicts, one per chunk
    """
    evals = []
    total = len(chunk_results)
    print(f"[eval_runner] Evaluating {total} chunks with subagent")

    for cr in chunk_results:
        prompt = _format_chunk_prompt(
            cr["idx"], total, cr["data"], prompt_template
        )
        t0 = time.time()
        response = _call_subagent(prompt)
        elapsed = time.time() - t0
        eval_dict = _parse_eval_response(response)
        eval_dict["chunk_idx"] = cr["idx"]
        eval_dict["seg_start"] = cr["seg_start"]
        eval_dict["seg_end"] = cr["seg_end"]
        eval_dict["elapsed_s"] = round(elapsed, 1)
        evals.append(eval_dict)
        print(f"  Chunk {cr['idx']+1}: {eval_dict['score']}/5 "
              f"({elapsed:.1f}s) — {len(eval_dict['issues'])} issues")

    return evals


def eval_final_with_subagent(
    merged_result: Dict,
    segments: List,
    prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate the full merged result using a Hermes subagent.
    """
    prompt = _format_final_prompt(merged_result, segments, prompt_template)
    t0 = time.time()
    response = _call_subagent(prompt)
    elapsed = time.time() - t0
    eval_dict = _parse_eval_response(response)
    eval_dict["type"] = "final"
    eval_dict["elapsed_s"] = round(elapsed, 1)
    print(f"[eval_runner] Final eval: {eval_dict['score']}/5 ({elapsed:.1f}s)")
    return eval_dict


# === Batch CLI ===

def main():
    """CLI entry: read merged JSON, evaluate, write eval JSON."""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run subagent eval on analysis results")
    parser.add_argument("merged_json", help="Path to merged analysis JSON")
    parser.add_argument("--db", help="Optional FencerDB path (for bout context)")
    parser.add_argument("--out", help="Output path for eval results (default: <input>.eval.json)")
    args = parser.parse_args()

    merged_path = Path(args.merged_json)
    if not merged_path.exists():
        print(f"ERROR: {merged_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(merged_path) as f:
        merged = json.load(f)

    chunks_meta = merged.get("chunks", [])
    if not chunks_meta:
        print("ERROR: no chunks in merged JSON (run analyze_full.py first)", file=sys.stderr)
        sys.exit(2)

    # Reconstruct chunk_results from chunks
    # Each chunk entry has seg_start, seg_end, frame_count
    # but not the full data — we use the merged arrays
    # For per-chunk eval we need actual per-chunk data, which is in
    # reports/merged_<bid>.json OR .chunks/chunk_*.json
    workdir = merged_path.parent / ".chunks"
    chunk_results = []
    for i, c in enumerate(chunks_meta):
        chunk_json = workdir / f"chunk_{i:03d}.json"
        if chunk_json.exists():
            with open(chunk_json) as f:
                chunk_data = json.load(f)
            chunk_results.append({
                "idx": i,
                "seg_start": c["seg_start"],
                "seg_end": c["seg_end"],
                "data": chunk_data,
            })
        else:
            print(f"  ! Missing chunk file: {chunk_json}", file=sys.stderr)
            chunk_results.append({
                "idx": i,
                "seg_start": c["seg_start"],
                "seg_end": c["seg_end"],
                "data": {"summary": {}, "frame_data": []},
            })

    # Run evals
    chunk_evals = eval_chunks_with_subagent(chunk_results)

    # Mock segments for final (use bout duration from summary)
    dur = merged.get("summary", {}).get("duration", 0)
    segments = [("active", 0, dur)]

    final_eval = eval_final_with_subagent(merged, segments)

    results = chunk_evals + [final_eval]

    out_path = Path(args.out) if args.out else merged_path.with_suffix(".eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
