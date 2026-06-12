# Fencing Analyzer ⚔️ — AI-Powered Fencing Video Analysis

YOLOv8m-Pose-based extraction of tactical metrics from fencing videos (epee).
Interactive dashboard with 15 metrics, PDF report, and live video player.

> **Target audience:** Fencers, coaches, national team coaches — **zero coding experience required.**  
> All you need: Copy & paste commands into your terminal.

---

## 📋 Features

- **15 metrics:** Distance, weapon-arm angle, lunge depth, movement path, posture, acceleration, step rhythm, reaction synchronization, heatmap, touché detection, pressure index, and more
- **2-person tracking** — automatic, frame by frame
- **Piste calibration** — automatic cm conversion
- **Tracking v2:** Side-Constraint + Velocity-Interpolation + Keypoint-Smoothing
- **Dual-Range-Slider** for precise range selection
- **Dynamic progress bar** with time estimation
- **PDF report** — 1 page with charts and statistics
- **Comparison mode** — overlay two analyses
- **Export:** JSON, CSV
- **Live video player** — toggle keypoints & distance lines on/off

---

## 🚀 Setup Guide (Absolute Beginner)

### Option A: Quick & Easy (Python Only)

> For Windows and Mac. No Docker, no Linux.  
> Duration: **~10 minutes.**

#### Step 1: Install Python

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python **3.11** (important: 3.11, not 3.13)
3. During installation, **MAKE SURE** to check:  
   ✅ **"Add Python to PATH"** (this is critical!)
4. Complete installation

**Mac:**
1. Open Terminal (Finder → Applications → Utilities → Terminal)
2. Enter this command:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Then:
```bash
brew install python@3.11 ffmpeg
```

#### Step 2: Download Fencing Analyzer

**Windows (PowerShell):**
```powershell
cd C:\Users\%USERNAME%
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
```

**Windows (without git — Download):**
1. Go to https://github.com/michaeltbs/fencing-analyzer
2. Click the green **"<> Code"** button → **"Download ZIP"**
3. Extract the ZIP
4. In the extracted folder: Right-click → **"Open in Terminal"**

**Mac:**
```bash
cd ~
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
```

#### Step 3: Install Dependencies (one-time, ~5 minutes)

**Windows:**
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Mac:**
```bash
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

> ⏳ This can take 3-5 minutes — YOLO and PyTorch are being downloaded.

#### Step 4: Start the App

**Windows:**
```powershell
streamlit run app.py --server.port 8501
```

**Mac:**
```bash
streamlit run app.py --server.port 8501
```

#### Step 5: Open in Browser

- Open Chrome, Safari, or Edge
- Type in the address bar: `http://localhost:8501`
- Done! 🎉

---

### Option B: Advanced (Docker)

> Install Docker once, then it runs identically everywhere.
> Ideal for iPad access and GPU acceleration.

#### Step 1: Install Docker Desktop

**Windows:**
1. Go to https://docs.docker.com/desktop/setup/install/windows-install/
2. Download and install Docker Desktop
3. **Restart Windows**
4. Start Docker Desktop (wait until the whale icon stops animating)

**Mac:**
1. Go to https://docs.docker.com/desktop/install/mac-install/
2. Download and install Docker Desktop
3. Start Docker Desktop

#### Step 2: Build Fencing Analyzer

**PowerShell (Windows) or Terminal (Mac):**
```bash
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
docker build -t fencing-analyzer .
```

> ⏳ First time takes ~5-10 minutes (downloads + installation).

#### Step 3: Launch

**Without GPU (default — runs on any computer):**
```bash
docker run -p 8501:8501 fencing-analyzer
```

**With NVIDIA GPU (10-20x faster):**
```bash
docker run --gpus all -p 8501:8501 fencing-analyzer:gpu
```

#### Step 4: Open Browser

`http://localhost:8501`

#### iPad / Phone Access (same WiFi required)

1. Find your **laptop's IP address:**
   - **Windows:** PowerShell → `ipconfig` → `IPv4 Address` (e.g. `192.168.1.42`)
   - **Mac:** Terminal → `ipconfig getifaddr en0` (e.g. `192.168.1.42`)
2. On your **iPad / Phone in Safari**: type `http://192.168.1.42:8501`

---

## 🖥️ GPU Acceleration (Optional)

Analysis runs on CPU at approximately **3 seconds per frame**.
With an NVIDIA GPU, this drops to **0.2 seconds per frame** — 15x faster.

### Check if you have a GPU:

**Windows:**
```powershell
nvidia-smi
```
- See a table? ✅ GPU available — build with GPU support
- "not found" error? ⚠️ No NVIDIA GPU — CPU version works fine too

### Build and run with GPU:

```bash
cd fencing-analyzer

# Build with GPU
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
  -t fencing-analyzer:gpu .

# Run with GPU
docker run --gpus all -p 8501:8501 fencing-analyzer:gpu
```

---

## 📖 How to Use (Dashboard)

1. **Upload video** — Drag & drop or select file (MP4, up to 4 GB)
2. **Select start and end time** — only the selected range is analyzed
3. **Click "Analyze"** — progress bar shows status
4. **Explore the dashboard:**
   - Live video player with overlay (skeleton, distance line)
   - 15 metrics as interactive charts
   - Toggle metrics on/off with a click
   - Download PDF report
5. **Export:** JSON or CSV

### Keyboard Shortcuts (Video Player)

| Key | Function |
|-----|----------|
| Space | Play / Pause |
| ← → | 1 second backward/forward |

---

## 📊 Metrics in Detail

| # | Metric | Description | Unit |
|---|--------|-------------|------|
| 1 | Distance | Hip-to-hip distance | cm |
| 2 | Weapon-arm angle | Shoulder-Elbow-Wrist | ° |
| 3 | Lunge depth | Vertical hip-ankle distance (front foot) | px |
| 4 | Movement path | Hip position as scatter plot | x,y |
| 5 | Body posture | Upper body lean from vertical | ° |
| 6 | Acceleration | 2nd derivative of weapon hand position | px/s² |
| 7 | Step rhythm | Individual foot tracking (half/full) | steps/s |
| 8 | Synchronization | Cross-correlation of hip velocities | r, Lag |
| 9 | Hand height | Shoulder-wrist height difference | px |
| 10 | Arm extension | Shoulder-wrist distance | px |
| 11 | Stance width | Foot-to-foot distance | px |
| 12 | Explosiveness | Distance change rate | cm/s |
| 13 | Head forward | Head protrusion past hips | px |
| 14 | Touché candidates | Arm extended + close distance | — |
| 15 | Rhythm (FFT) | Dominant bout tempo | Hz |
| 16 | Pressure index | Who's driving the bout? | ± value |

---

## 🎬 Full-Length Analysis (NEW in v1.0)

As of v1.0, the analyzer automatically processes complete bouts (15+ min).
Pauses are detected and skipped, the bout is segmented into active phases,
each segment is analyzed separately, everything is persisted to a SQLite
database — and studio-ready output (annotated HD video + highlight reel)
is generated.

### Quickstart

```bash
# One command, fully automatic
python analyze_full.py "M - T16 SCHMIDT vs TREBIS.mp4" \
    --fencer-a "michael-trebis" --name-a "Michael" --last-a "Trebis" \
              --nation-a "GER" --hand-a "right" \
    --fencer-b "richard-schmidt" --name-b "Richard" --last-b "Schmidt" \
                --nation-b "GER" \
    --tournament "Doha 2026" --date "2026-01-15" \
    --score 8 15
```

**What happens:**
1. **Pause detection** — ffmpeg-based, ~10s for 15 min video
2. **Chunked YOLO analysis** — each active segment analyzed separately
3. **SQLite persistence** — fencer data, per-frame metrics, annotations
4. **PDF report** — 1-page with all stats + charts
5. **Annotated HD video** — skeleton overlay on 1080p source
6. **Highlight reel** — 5s context around each touché

### Pipeline modules

| File | Purpose |
|------|---------|
| `pause_detector.py` | Motion-based pause detection (ffmpeg scene detect) |
| `scheduler.py` | Orchestrates chunks + DB persistence |
| `worker_chunk_analyze.py` | Wrapper for `worker_analyze.py` with time-offset |
| `inference_db.py` | SQLite schema + CRUD for fencer/bout/metrics |
| `studio_export.py` | HD render + highlight reel |
| `analyze_full.py` | One-command entry point |

### CLI options for `analyze_full.py`

```
--db PATH              SQLite file (default: fencing.db)
--no-studio            Skip HD/highlight output
--no-pdf               Skip PDF report
--no-highlights        HD video only, no highlight reel
--context-s N          Seconds of context around touché (default: 5.0)
--keep-chunks          Keep per-chunk JSON files
--no-eval              Skip subagent quality evaluation
```

### Quality Evaluation

After analysis, a subagent evaluates the quality:

- **Per-chunk:** frame coverage, distance distribution, step rate, touché plausibility
- **Final:** realistic touché rate, cross-chunk consistency, pressure index trend

Example output:
```
=== Quality Evaluation ===
Per-chunk avg score: 4.2/5 (3 chunks)
  Chunk 1: 5/5 — looks clean
  Chunk 2: 4/5 — moderate frame coverage 87%
    → Check tracking continuity
  Chunk 3: 4/5 — touch rate 6.2/min, verify in highlights

Final eval: 4/5
  → Michael dominates second half (Δ +340px)
```

Eval results are saved to `reports/eval_<bout-id>.json`.

### Querying the fencer database

```python
from inference_db import FencerDB
db = FencerDB("fencing.db")
for bout in db.list_bouts():
    print(bout["tournament"], bout["bout_date"],
          bout["fencer_a_score"], "vs", bout["fencer_b_score"])

# Fetch metrics
metrics = db.get_metrics(bout["id"])
for m in metrics[:10]:
    print(f"t={m['t']:.1f}s  dist={m['dist_cm']}cm  angle_m={m['arm_angle_m']}")

# Annotations (touchés, notes)
for a in db.get_annotations(bout["id"], type_="touche"):
    print(f"t={a['t']:.1f}s  {a['description']}")
```

---

## ❓ Frequently Asked Questions

**"The analysis is too slow!"**  
→ CPU takes ~3s per frame. A 60s clip takes about 3 minutes.  
→ With GPU (see above) it's 15x faster.  
→ Tip: Analyze short, action-packed segments (15-30s).

**"My video won't load!"**  
→ Only MP4 files. Maximum file size: 4 GB.  
→ If you have a RAW video: Convert it to MP4 first using Handbrake (free).

**"I don't see any progress!"**  
→ Analysis runs in the background. The button shows "Analyzing...".  
→ For very long clips (>60s), it can take several minutes — just wait.

**"Can I compare multiple videos?"**  
→ Yes! Analyze video 1, then video 2 — comparison mode overlays both.

**"Can I track just one fencer?"**  
→ Yes. The app defaults to tracking Michael (green) and opponent (red).  
→ For other videos: The first frame shows numbered people — pick the two relevant ones.

---

## 🔧 Troubleshooting

### "pip not found"
```powershell
# Windows: Reinstall Python — make sure to check "Add Python to PATH"!
```

### "streamlit not found"
```powershell
pip install streamlit
```

### Port 8501 is already in use
```powershell
streamlit run app.py --server.port 8502
# Then in browser: http://localhost:8502
```

### Docker: "libgl1-mesa-glx: not found"
→ Has been fixed. `git pull` and rebuild.

### Docker: "Multi-line Python error"
→ Has been fixed. `git pull` and rebuild.

---

## 📁 Project structure

```
fencing-analyzer/
├── app.py                  # Streamlit dashboard (main, short clips)
├── worker_analyze.py       # YOLO analysis (subprocess)
├── worker_chunk_analyze.py # Chunked worker with time-offset (v1.0)
├── pause_detector.py       # Motion-based pause detection (v1.0)
├── scheduler.py            # Chunk orchestration + DB persistence (v1.0)
├── inference_db.py         # SQLite fencer/bout/metrics schema (v1.0)
├── subagent_eval.py        # Quality eval with heuristic + subagent (v1.1)
├── studio_export.py        # HD render + highlight reel (v1.0)
├── analyze_full.py         # One-command full-length pipeline (v1.0)
├── preview_generator.py    # Annotated preview video
├── report_generator.py     # PDF report
├── Dockerfile              # Container build (CPU + GPU)
├── requirements.txt        # Python dependencies
├── build.sh                # Auto-build (detects GPU)
├── reports/                # Generated PDF reports + merged JSON
├── studio/                 # HD videos + highlight reels (v1.0)
├── tests/                  # Analysis scripts + motion profiles
└── README.md               # This file
```

---

## 📝 Changelog

### v1.1 (June 2026) — Quality Evaluation
- **NEW:** `subagent_eval.py` — per-chunk + final quality evaluator
- Heuristic detection of tracking errors, implausible values
- Subagent integration: prompts for LLM-based plausibility checks
- `analyze_full.py --no-eval` flag to skip
- Eval results persisted to `reports/eval_<id>.json`

### v1.0 (June 2026) — Full-Length Edition
- **NEW:** `pause_detector.py` — ffmpeg-based pause detection
- **NEW:** `scheduler.py` — chunked analysis + DB persistence
- **NEW:** `inference_db.py` — SQLite fencer/bout/metrics/annotations
- **NEW:** `studio_export.py` — HD video (1080p skeleton overlay) + highlight reel
- **NEW:** `analyze_full.py` — one-command entry point for complete bouts
- 16 metrics + touché detection now run across multiple chunks
- Chunked execution: each active segment = 1 YOLO subprocess
- SQLite output: all per-frame metrics, all annotations queryable

### v0.4 — Tracking v2 + UI
- ByteTrack + Side-Constraint + VelocityInterpolator
- 16 metrics
- Streamlit UI with live player

### v0.3 — Initial release
- YOLOv8m-Pose integration
- PDF report generator
```

---

## ⚖️ License

MIT — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8m Pose
- [Streamlit](https://streamlit.io) — Dashboard framework
- [Plotly](https://plotly.com) — Interactive charts

---

*Built with 🏃💨 for Michael Trebis — German epee fencer, military athlete, CISM 2027.*