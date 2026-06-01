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
| 9 | Heatmap | 2D histogram of piste position | density |
| 10 | Pressure index | Who is driving the bout? | ± value |
| 11-15 | More metrics | Touché, tempo, etc. | see UI |

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

## 📁 Project Structure

```
fencing-analyzer/
├── app.py                  # Streamlit dashboard (main file)
├── worker_analyze.py       # YOLO analysis (subprocess)
├── report_generator.py     # PDF report generator
├── Dockerfile              # Container build (CPU + GPU)
├── requirements.txt        # Python dependencies
├── build.sh                # Auto-build (detects GPU)
├── README.md               # German README
├── README.en.md            # English README (this file)
└── reports/                # Generated PDF reports
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