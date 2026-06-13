"""
ui_full_length.py — Streamlit UI for the full-length analysis mode.

Functions:
  - full_render_form()      sidebar config
  - full_render_main(video_path)  main panel + run button
  - full_browse_bouts()     database browser
  - full_show_progress()    subprocess polling fragment

Imported by app.py.
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from inference_db import FencerDB

# Files used to communicate with the analyze_full.py subprocess
FULL_PROGRESS_FILE = "reports/.full_progress.json"
FULL_DONE_FILE = "reports/.full_done.json"
FULL_ERROR_FILE = "reports/.full_error.txt"

# === FULL-LENGTH MODE (v1.0/v1.1) ===

# Files used to communicate with the analyze_full.py subprocess
FULL_PROGRESS_FILE = "reports/.full_progress.json"
FULL_DONE_FILE = "reports/.full_done.json"
FULL_ERROR_FILE = "reports/.full_error.txt"


def full_render_form():
    """Sidebar form for Full-Length mode: fencer metadata + run button."""
    st.subheader("🎬 Full-Length Konfiguration")

    fencer_a_slug = st.text_input("Fechter A Slug (z.B. michael-trebis)",
                                   value="michael-trebis", key="fl_slug_a")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fencer_a_first = st.text_input("Vorname A", value="Michael", key="fl_first_a")
        fencer_a_nation = st.text_input("Nation A", value="GER", key="fl_nat_a", max_chars=3)
    with col_a2:
        fencer_a_last = st.text_input("Nachname A", value="Trebis", key="fl_last_a")
        fencer_a_hand = st.selectbox("Hand A", ["right", "left"], key="fl_hand_a")
    fencer_a_club = st.text_input("Club A (optional)", value="", key="fl_club_a")

    st.divider()

    fencer_b_slug = st.text_input("Fechter B Slug", value="gegner",
                                   key="fl_slug_b")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fencer_b_first = st.text_input("Vorname B", value="", key="fl_first_b")
        fencer_b_nation = st.text_input("Nation B", value="", key="fl_nat_b", max_chars=3)
    with col_b2:
        fencer_b_last = st.text_input("Nachname B", value="", key="fl_last_b")
        fencer_b_hand = st.selectbox("Hand B", ["right", "left"], key="fl_hand_b")
    fencer_b_club = st.text_input("Club B (optional)", value="", key="fl_club_b")

    st.divider()
    st.subheader("Bout-Metadaten")
    tournament = st.text_input("Turnier", value="", key="fl_tournament",
                                placeholder="z.B. Doha 2026 T16")
    bout_date = st.date_input("Datum", value=None, key="fl_date")
    venue = st.text_input("Venue (optional)", value="", key="fl_venue")
    weapon = st.selectbox("Waffe", ["epee", "foil", "sabre"],
                           index=0, key="fl_weapon")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        score_a = st.number_input("Score A", min_value=0, max_value=30,
                                   value=None, step=1, key="fl_score_a")
    with col_s2:
        score_b = st.number_input("Score B", min_value=0, max_value=30,
                                   value=None, step=1, key="fl_score_b")

    st.divider()
    st.subheader("Output-Optionen")
    do_pdf = st.checkbox("PDF-Report generieren", value=True, key="fl_pdf")
    do_studio = st.checkbox("HD-Video + Highlights", value=True, key="fl_studio")
    do_highlights = st.checkbox("Highlight-Reel", value=True, key="fl_highlights",
                                 disabled=not do_studio)
    do_eval = st.checkbox("Quality-Eval (Subagent)", value=True, key="fl_eval")
    context_s = st.slider("Touché-Kontext (s)", min_value=2, max_value=15,
                           value=5, step=1, key="fl_context")

    st.divider()
    db_path = st.text_input("DB-Pfad", value="fencing.db", key="fl_db")

    if not do_studio:
        do_highlights = False

    return {
        "fencer_a_slug": fencer_a_slug,
        "fencer_a_first": fencer_a_first,
        "fencer_a_last": fencer_a_last,
        "fencer_a_nation": fencer_a_nation,
        "fencer_a_hand": fencer_a_hand,
        "fencer_a_club": fencer_a_club,
        "fencer_b_slug": fencer_b_slug,
        "fencer_b_first": fencer_b_first,
        "fencer_b_last": fencer_b_last,
        "fencer_b_nation": fencer_b_nation,
        "fencer_b_hand": fencer_b_hand,
        "fencer_b_club": fencer_b_club,
        "tournament": tournament,
        "bout_date": bout_date.isoformat() if bout_date else None,
        "venue": venue,
        "weapon": weapon,
        "score_a": score_a,
        "score_b": score_b,
        "do_pdf": do_pdf,
        "do_studio": do_studio,
        "do_highlights": do_highlights,
        "do_eval": do_eval,
        "context_s": context_s,
        "db_path": db_path,
    }


def full_start_run(video_path, config):
    """Start the analyze_full.py subprocess with the given config."""
    for fn in [FULL_PROGRESS_FILE, FULL_DONE_FILE, FULL_ERROR_FILE]:
        p = Path(fn)
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    cmd = [
        sys.executable, "analyze_full.py", str(video_path),
        "--fencer-a", config["fencer_a_slug"],
        "--name-a", config["fencer_a_first"],
        "--last-a", config["fencer_a_last"],
        "--fencer-b", config["fencer_b_slug"],
        "--last-b", config["fencer_b_last"],
        "--db", config["db_path"],
        "--weapon", config["weapon"],
    ]
    if config["fencer_a_nation"]:
        cmd += ["--nation-a", config["fencer_a_nation"]]
    if config["fencer_a_hand"]:
        cmd += ["--hand-a", config["fencer_a_hand"]]
    if config["fencer_a_club"]:
        cmd += ["--club-a", config["fencer_a_club"]]
    if config["fencer_b_first"]:
        cmd += ["--name-b", config["fencer_b_first"]]
    if config["fencer_b_nation"]:
        cmd += ["--nation-b", config["fencer_b_nation"]]
    if config["fencer_b_hand"]:
        cmd += ["--hand-b", config["fencer_b_hand"]]
    if config["fencer_b_club"]:
        cmd += ["--club-b", config["fencer_b_club"]]
    if config["tournament"]:
        cmd += ["--tournament", config["tournament"]]
    if config["bout_date"]:
        cmd += ["--date", config["bout_date"]]
    if config["venue"]:
        cmd += ["--venue", config["venue"]]
    if config["score_a"] is not None and config["score_b"] is not None:
        cmd += ["--score", str(config["score_a"]), str(config["score_b"])]
    if not config["do_pdf"]:
        cmd += ["--no-pdf"]
    if not config["do_studio"]:
        cmd += ["--no-studio"]
    if not config["do_highlights"]:
        cmd += ["--no-highlights"]
    if not config["do_eval"]:
        cmd += ["--no-eval"]
    cmd += ["--context-s", str(config["context_s"])]

    log_path = Path("reports/.full_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
    return proc, log_path


@st.fragment(run_every=5)
def full_show_progress():
    """Polls the analyze_full.py subprocess and shows progress."""
    proc = st.session_state.get("full_proc")
    log_path = st.session_state.get("full_log_path")
    if not proc:
        st.warning("Kein Subprocess gestartet")
        return

    if proc.poll() is not None:
        st.session_state["analysis_running"] = False
        st.session_state["full_proc"] = None

        if proc.returncode == 0:
            st.success("✅ Full-Length-Analyse fertig (returncode=0)")
            db_path = st.session_state.get("fl_last_config", {}).get("db_path", "fencing.db")
            try:
                from inference_db import FencerDB
                db = FencerDB(db_path)
                bouts = db.list_bouts(limit=1)
                if bouts:
                    st.session_state["full_bout_id"] = bouts[0]["id"]
                    st.session_state["full_db_path"] = db_path
                    st.info(f"Bout-ID: {bouts[0]['id'][:8]}. "
                            f"Wechsle zu 'DB durchsuchen' zum Anzeigen.")
            except Exception as e:
                st.error(f"DB-Load fehlgeschlagen: {e}")
        else:
            st.error(f"❌ Analyse fehlgeschlagen (returncode={proc.returncode})")
            if log_path and log_path.exists():
                with st.expander("Log-Output anzeigen"):
                    st.code(log_path.read_text()[-3000:])

        st.rerun()
        return

    st.info(f"⏳ Analyse läuft (PID={proc.pid})")
    if log_path and log_path.exists():
        with st.expander("Live-Log (letzte 50 Zeilen)", expanded=True):
            log_text = log_path.read_text()
            tail = "\n".join(log_text.splitlines()[-50:])
            st.code(tail)


def full_render_main(video_path):
    """Main view: form + run button + status."""
    config = full_render_form()
    st.session_state["fl_last_config"] = config

    if not video_path:
        st.warning("⚠️ Bitte zuerst ein Video auswählen (siehe Sidebar).")
        return

    if st.button("🚀 Full-Length-Analyse starten", type="primary",
                  use_container_width=True, key="fl_start_btn"):
        st.session_state.pop("full_bout_id", None)

        proc, log_path = full_start_run(video_path, config)
        st.session_state["full_proc"] = proc
        st.session_state["full_log_path"] = log_path
        st.session_state["analysis_running"] = True
        st.rerun()


def full_browse_bouts():
    """Browse bouts in the SQLite database, show summary metrics."""
    from inference_db import FencerDB

    db_path = st.session_state.get("fl_last_config", {}).get("db_path", "fencing.db")
    db = FencerDB(db_path)

    st.subheader("📚 Fechter-Datenbank durchsuchen")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fechter", db.stats()["fencers"])
        st.metric("Bouts", db.stats()["bouts"])
    with col2:
        st.metric("Bouts complete", db.stats()["bouts_complete"])
        st.metric("Metriken-Zeilen", db.stats()["metrics_rows"])

    bouts = db.list_bouts(limit=50)
    if not bouts:
        st.info("Keine Bouts in der DB.")
        return

    rows = []
    for b in bouts:
        a = db.get_fencer(b["fencer_a_id"])
        b2 = db.get_fencer(b["fencer_b_id"])
        rows.append({
            "Datum": b.get("bout_date") or "?",
            "Turnier": b.get("tournament") or "?",
            "Fechter A": a.get("last_name") if a else "?",
            "Fechter B": b2.get("last_name") if b2 else "?",
            "Score": f"{b.get('fencer_a_score', '?')}:{b.get('fencer_b_score', '?')}",
            "Status": b.get("status", "?"),
            "Dauer (s)": round(b.get("video_duration_s") or 0, 0),
            "Bout-ID": b["id"][:8],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    selected = st.selectbox(
        "Bout auswählen für Detail-Ansicht",
        options=[b["id"][:8] for b in bouts],
        format_func=lambda bid: next(
            f"{b['bout_date'] or '?'} — {b.get('tournament') or '?'} ({bid})"
            for b in bouts if b["id"][:8] == bid
        ),
        key="fl_bout_picker",
    )
    if not selected:
        return

    bout_full = next((b for b in bouts if b["id"][:8] == selected), None)
    if not bout_full:
        return

    st.divider()
    st.subheader(f"Bout {bout_full['id'][:8]}")

    col1, col2, col3, col4 = st.columns(4)
    a = db.get_fencer(bout_full["fencer_a_id"])
    b2 = db.get_fencer(bout_full["fencer_b_id"])
    with col1:
        st.metric("Fechter A", a.get("last_name") if a else "?")
        if a:
            st.caption(f"{a.get('first_name', '')} · {a.get('nation', '')}")
    with col2:
        st.metric("Fechter B", b2.get("last_name") if b2 else "?")
        if b2:
            st.caption(f"{b2.get('first_name', '')} · {b2.get('nation', '')}")
    with col3:
        st.metric("Score", f"{bout_full.get('fencer_a_score', '?')}:"
                            f"{bout_full.get('fencer_b_score', '?')}")
    with col4:
        st.metric("Status", bout_full.get("status", "?"))

    metrics = db.get_metrics(bout_full["id"])
    if metrics:
        st.divider()
        st.subheader("Metriken aus DB")
        df = pd.DataFrame(metrics)
        keep = [c for c in ["t", "dist_cm", "arm_angle_m", "arm_angle_g",
                            "vel_m", "vel_g", "pressure_net"] if c in df.columns]
        if keep:
            st.dataframe(df[keep].head(50), use_container_width=True, hide_index=True)

        if "t" in df.columns and "dist_cm" in df.columns:
            chart_df = df[["t", "dist_cm"]].dropna()
            if not chart_df.empty:
                st.line_chart(chart_df.set_index("t"), height=250)

        annotations = db.get_annotations(bout_full["id"])
        if annotations:
            with st.expander(f"Annotationen ({len(annotations)})", expanded=False):
                for a_ in annotations[:50]:
                    st.write(f"**t={a_['t']:.1f}s** · `{a_['type']}` · {a_.get('description', '')}")

    st.divider()
    st.subheader("Generierte Outputs")
    outputs = []
    reports_dir = Path("reports")
    studio_dir = Path("studio")
    if reports_dir.exists():
        for p in reports_dir.glob("Fecht-Analyse_*.pdf"):
            outputs.append(("PDF-Report", p, "📄"))
    if studio_dir.exists():
        for p in studio_dir.glob("*_annotated.mp4"):
            outputs.append(("HD-Video", p, "🎬"))
        for p in studio_dir.glob("*_highlights.mp4"):
            outputs.append(("Highlight-Reel", p, "⭐"))

    if not outputs:
        st.caption("(noch keine — führe eine Analyse aus)")
    for label, p, icon in outputs:
        size_mb = p.stat().st_size / 1e6
        st.write(f"{icon} **{label}:** `{p}` ({size_mb:.1f} MB)")

