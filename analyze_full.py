"""
analyze_full.py — One-command full-length analysis pipeline.

Wraps scheduler.py + studio_export.py for end-to-end execution:
  1. Init DB if needed
  2. Upsert fencers
  3. Create bout
  4. Run chunked analysis (pause detection + YOLO chunks + DB persist)
  5. Generate studio outputs (annotated HD, highlight reel)
  6. Generate PDF report

Usage:
    python analyze_full.py <video> \\
        --fencer-a "michael-trebis" --name-a "Michael" --last-a "Trebis" \\
        --fencer-b "richard-schmidt" --name-b "Richard" --last-b "Schmidt" \\
        --tournament "Doha 2026" --date "2026-01-15" \\
        [--score 8 15] [--no-studio] [--no-pdf]

The script is intended to run on the GPU machine. Outputs go to
  ./reports/<name>.pdf
  ./studio/<name>_annotated.mp4
  ./studio/<name>_highlights.mp4
  ./fencing.db (SQLite)
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run full-length fencing analysis end-to-end"
    )
    parser.add_argument("video", help="Path to source video")
    parser.add_argument("--db", default="fencing.db", help="FencerDB path")
    parser.add_argument("--fencer-a", required=True, help="Slug for fencer A (Michael)")
    parser.add_argument("--name-a", help="First name A")
    parser.add_argument("--last-a", help="Last name A")
    parser.add_argument("--nation-a", help="Nation A")
    parser.add_argument("--hand-a", choices=["left", "right"], help="Hand A")
    parser.add_argument("--club-a", help="Club A")

    parser.add_argument("--fencer-b", required=True, help="Slug for fencer B")
    parser.add_argument("--name-b", help="First name B")
    parser.add_argument("--last-b", help="Last name B")
    parser.add_argument("--nation-b", help="Nation B")
    parser.add_argument("--hand-b", choices=["left", "right"], help="Hand B")
    parser.add_argument("--club-b", help="Club B")

    parser.add_argument("--tournament", help="Tournament name")
    parser.add_argument("--date", help="Bout date (YYYY-MM-DD)")
    parser.add_argument("--venue", help="Venue")
    parser.add_argument("--weapon", default="epee", choices=["epee", "foil", "sabre"])
    parser.add_argument("--score", nargs=2, type=int, metavar=("A", "B"),
                        help="Final score A B")

    parser.add_argument("--no-studio", action="store_true",
                        help="Skip HD video / highlight reel output")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report")
    parser.add_argument("--no-highlights", action="store_true",
                        help="Skip highlight reel only")
    parser.add_argument("--keep-chunks", action="store_true",
                        help="Keep per-chunk result files")
    parser.add_argument("--context-s", type=float, default=5.0,
                        help="Seconds of context around each touché in highlights")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    from inference_db import FencerDB
    from scheduler import run_full_analysis
    from report_generator import generate_report
    from studio_export import export_annotated_hd, export_highlight_reel

    db = FencerDB(args.db)

    # === Step 1: Fencers ===
    print(f"[1/5] Upserting fencers...")
    fid_a = db.upsert_fencer(
        args.fencer_a, last_name=args.last_a or args.fencer_a,
        first_name=args.name_a, nation=args.nation_a,
        hand=args.hand_a, club=args.club_a
    )
    fid_b = db.upsert_fencer(
        args.fencer_b, last_name=args.last_b or args.fencer_b,
        first_name=args.name_b, nation=args.nation_b,
        hand=args.hand_b, club=args.club_b
    )
    print(f"  fencer_a: {fid_a}")
    print(f"  fencer_b: {fid_b}")

    # === Step 2: Bout ===
    print(f"\n[2/5] Creating bout record...")
    bid = db.create_bout(
        fencer_a_id=fid_a,
        fencer_b_id=fid_b,
        tournament=args.tournament,
        bout_date=args.date,
        venue=args.venue,
        weapon=args.weapon,
        fencer_a_score=args.score[0] if args.score else None,
        fencer_b_score=args.score[1] if args.score else None,
        video_path=str(video_path.resolve()),
    )
    print(f"  bout_id: {bid}")

    # === Step 3: Full analysis ===
    print(f"\n[3/5] Running full-length analysis (this may take 20-30 min on GPU)...")
    t0 = time.time()
    merged, segments = run_full_analysis(
        str(video_path), bid, db,
        keep_chunks=args.keep_chunks,
    )
    elapsed = time.time() - t0
    if merged is None:
        print("ERROR: analysis produced no results", file=sys.stderr)
        sys.exit(2)
    summary = merged.get("summary", {})
    print(f"\n  Analysis done in {elapsed:.0f}s")
    print(f"  Frames: {summary.get('frames', 0)}, "
          f"Touchés: {summary.get('touches', 0)} "
          f"({summary.get('touches_high', 0)} high)")

    # Save merged JSON
    merged_path = Path("reports") / f"merged_{bid[:8]}.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"  Merged result: {merged_path}")

    # === Step 4: PDF report ===
    if not args.no_pdf:
        print(f"\n[4/5] Generating PDF report...")
        bout_name = f"{args.tournament or 'Bout'}_{args.last_a or 'A'}_vs_{args.last_b or 'B'}"
        pdf_path, preview = generate_report(merged, bout_name)
        print(f"  PDF: {pdf_path}")
        print(preview)
    else:
        print(f"\n[4/5] Skipped (--no-pdf)")

    # === Step 5: Studio outputs ===
    if not args.no_studio:
        print(f"\n[5/5] Generating studio outputs...")
        studio_dir = Path("studio")
        studio_dir.mkdir(parents=True, exist_ok=True)
        bout_name = f"{args.tournament or 'Bout'}_{args.last_a or 'A'}_vs_{args.last_b or 'B'}"

        # Annotated HD
        try:
            hd_path = studio_dir / f"{bout_name}_annotated.mp4"
            export_annotated_hd(
                video_path,
                merged.get("frame_data", []),
                hd_path,
                cut_pauses=False,
                touches=merged.get("m14_touches", []),
            )
        except Exception as e:
            print(f"  ! HD export failed: {e}")

        # Highlight reel
        if not args.no_highlights and merged.get("m14_touches"):
            try:
                hr_path = studio_dir / f"{bout_name}_highlights.mp4"
                export_highlight_reel(
                    video_path,
                    merged["m14_touches"],
                    hr_path,
                    context_s=args.context_s,
                )
            except Exception as e:
                print(f"  ! Highlights failed: {e}")
    else:
        print(f"\n[5/5] Skipped (--no-studio)")

    print(f"\n=== DONE ===")
    print(f"Bout ID: {bid}")
    print(f"DB: {args.db}")
    print(f"PDF: reports/")
    print(f"Studio: studio/")


if __name__ == "__main__":
    main()
