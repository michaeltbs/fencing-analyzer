# Fencing Analyzer v1.3

**Full-length fencing video analysis with YOLOv8m-Pose, SQLite studio pipeline, Hermes subagent quality evaluation, and Selenium-tested Streamlit UI.**

---

## What's new in v1.3

### 🤖 Subagent Evaluation Loop (Hermes-integrated)

- New `eval_runner.py` module calls Hermes `delegate_task` to evaluate each analyzed chunk and the final merged bout.
- Built-in robust parser for `SCORE:/ISSUES:/SUGGESTIONS:` responses — handles single-line, multi-line bullet lists, and mixed formats.
- CLI entry point: `python eval_runner.py reports/merged_<bout>.json --out reports/eval_<bout>.json`
- Graceful fallback when not running inside a Hermes agent.
- Updated `fencing-pose-analysis` skill documentation.

### 🧪 Selenium UI Tests

- New `tests/conftest.py` starts the Streamlit app in a subprocess + headless Chrome.
- New `tests/test_app_ui.py` with 11 automated checks:
  - App loads and title renders
  - No Streamlit API errors visible
  - Mode switch (Quick clip / Full-Length) visible
  - Video source options visible
  - GPU/CPU indicator visible
  - Navigation survives multiple page loads
  - Sidebar sections rendered
  - No severe browser console errors
  - Screenshot capture for visual review
  - Streamlit DOM structure (`stApp`, `stSidebar`)
- New `requirements-test.txt` for optional test dependencies.
- New `pytest.ini` configuration.
- `.gitignore` updated to include the `tests/` directory in the repo.

### 📦 Infrastructure

- `requirements.txt` cleaned up; test deps moved to `requirements-test.txt`.
- README (DE + EN) updated with UI test workflow.
- Changelog updated.

---

## Quick start

```bash
# Clone / update
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
pip install -r requirements.txt

# Analyze a full-length bout on a GPU machine
python analyze_full.py "bout.mp4" \
  --fencer-a "michael-trebis" --name-a "Michael" --last-a "Trebis" \
  --fencer-b "richard-schmidt" --name-b "Richard" --last-b "Schmidt" \
  --tournament "Doha 2026 T16" --date "2026-01-15" --score 8 15

# Run subagent evaluation (inside Hermes agent)
python eval_runner.py reports/merged_<bout>.json

# Launch UI
streamlit run app.py

# Run UI tests
pip install -r requirements-test.txt
pytest tests/test_app_ui.py -v
```

---

## Full module map

| File | Role |
|------|------|
| `analyze_full.py` | One-command CLI for full-length bouts |
| `scheduler.py` | Chunk orchestration, merge, DB persist |
| `worker_chunk_analyze.py` | Chunked YOLO wrapper |
| `worker_analyze.py` | Core tracking + metrics logic (v0.x legacy, unchanged) |
| `pause_detector.py` | ffmpeg scene-detect → active segments |
| `inference_db.py` | SQLite: fencers, bouts, metrics, annotations |
| `studio_export.py` | HD 1080p render + highlight reel |
| `subagent_eval.py` | v1.1 heuristic + LLM prompt builder |
| `eval_runner.py` | v1.3 Hermes subagent integration **NEW** |
| `app.py` | Streamlit UI (Quick clip + Full-Length modes) |
| `report_generator.py` | PDF report |
| `tests/conftest.py` | Selenium fixture **NEW** |
| `tests/test_app_ui.py` | 11 UI tests **NEW** |

---

## Known limitations

- `eval_runner.py` requires execution inside a Hermes agent (or the Hermes Python SDK) to call `delegate_task`; otherwise it returns a mock/fallback response.
- Full-length YOLO analysis needs a CUDA-capable GPU for practical runtimes (estimated ~22 min for a 15 min bout on mid-range GPU).
- Selenium tests require Google Chrome and a Windows/Linux environment.

---

## Asset checksums

Source tarball and zip are available from GitHub.

---

**Full commit history:** `git log --oneline 25c490b..6382b0e`

**Contributors:** Wawa / Michael Trebis
