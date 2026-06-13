# Fencing Analyzer v1.3.2

**Refactoring release: modular UI, shared video utilities, stronger eval parser, and more tests.**

---

## Highlights

### Modular UI
- `app.py` split into:
  - `ui_media_server.py` — mini HTTP server for large video streaming + player HTML
  - `ui_full_length.py` — Full-Length mode form, run button, progress fragment, DB browser
- `app.py` is now a thin router; easier to maintain and test

### Performance / Efficiency
- `video_utils.py` — shared `probe_video`, `extract_subclip`, `has_ffmpeg`, `has_ffprobe`
- `scheduler.py` now passes `--start-frame` / `--end-frame` to each chunk worker
- `worker_chunk_analyze.py` extracts only the relevant subclip via ffmpeg instead of re-reading the full source video for every chunk
- Expected speedup scales with number of chunks / pause density

### Logging
- Replaced `print()` with Python `logging` in:
  - `analyze_full.py`
  - `scheduler.py`
  - `pause_detector.py`
  - `worker_chunk_analyze.py`
- Cleaner output in UI / tests / logs

### Robust Eval Parser
- `eval_runner.py` `_parse_eval_response` now handles:
  - single-line: `SCORE: 4, ISSUES: a, b, SUGGESTIONS: c, d`
  - multiline bullet lists
  - mixed inline format
  - missing sections → fallback to raw text as suggestion
  - score clamping 1-5

### New Tests
- `tests/test_video_pipeline.py` — video utils + pause detector
- `tests/test_scheduler_integration.py` + `tests/mock_worker_chunk.py` — scheduler logic with a fake worker
- `tests/test_eval_runner.py` — 6 parametrized parser cases
- `tests/test_worker_frame_range.py` — slow benchmark for frame-range speedup (`pytest --runslow`)

### Test Results
- **19 tests passing**
  - 11 Selenium UI tests
  - 6 video-pipeline tests
  - 2 scheduler integration tests
  - 6 eval parser unit tests

---

## Assets

- Source code: `main` branch at `6eefcc2`
- Previous release: [v1.3](https://github.com/michaeltbs/fencing-analyzer/releases/tag/v1.3)

---

## How to verify

```bash
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
pip install -r requirements.txt
pip install -r requirements-test.txt
pytest tests/test_app_ui.py tests/test_video_pipeline.py tests/test_scheduler_integration.py tests/test_eval_runner.py -v
```

For the slow benchmark:
```bash
pytest tests/test_worker_frame_range.py --runslow -v
```

---

## Next steps / open items

- Real GPU end-to-end test with the 15-min Doha 2026 video (waiting for GPU machine)
- Optional: GitHub Actions CI for automated test runs on every push
- Optional: further app.py split (analysis display, player, comparison)
