# Fencing Analyzer ⚔️ — Fecht-Video-Analyse mit KI

YOLOv8m-Pose-basierte Extraktion taktischer Metriken aus Fecht-Videos (Degen).
Interaktives Dashboard mit 15 Metriken, PDF-Report und Live-Video-Player.

> **Zielgruppe:** Fechter, Trainer, Bundestrainer — **0 Informatik-Kenntnisse nötig.**  
> Alles was du brauchst: Copy & Paste in die Kommandozeile.

---

## 📋 Features

- **15 Metriken:** Distanz, Waffenarm-Winkel, Lunge-Tiefe, Bewegungs-Pfad, Körperhaltung, Beschleunigung, Schritt-Rhythmus, Reaktions-Synchronisierung, Heatmap, Touche-Detektion, Druck-Index uvm.
- **2-Personen-Tracking** — automatisch, Frame für Frame
- **Pisten-Kalibrierung** — automatische cm-Umrechnung
- **Tracking v2:** Side-Constraint + Velocity-Interpolation + Keypoint-Smoothing
- **Dual-Range-Slider** für präzise Bereichsauswahl
- **Dynamische Progress-Bar** mit Zeit-Schätzung
- **PDF-Report** — 1 Seite, mit Diagrammen und Statistik
- **Vergleichsmodus** — zwei Analysen überlagern
- **Export:** JSON, CSV
- **Live-Video-Player** — mit ein/ausblendbaren Keypoints & Distanzlinien

---

## 🚀 Setup-Guide (für absolute Anfänger)

### Variante A: Einfach & Schnell (nur Python)

> Für Windows und Mac. Kein Docker, kein Linux.  
> Dauer: **ca. 10 Minuten.**

#### Schritt 1: Python installieren

**Windows:**
1. Gehe zu https://www.python.org/downloads/
2. Lade Python **3.11** herunter (wichtig: 3.11, nicht 3.13)
3. Beim Installieren **UNBEDINGT** den Haken setzen:  
   ✅ **"Add Python to PATH"** (das ist wichtig!)
4. Installation abschließen

**Mac:**
1. Terminal öffnen (Finder → Programme → Dienstprogramme → Terminal)
2. Diesen Befehl eingeben:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3. Danach:
```bash
brew install python@3.11 ffmpeg
```

#### Schritt 2: Fencing Analyzer herunterladen

**Windows (PowerShell):**
```powershell
cd C:\Users\%USERNAME%
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
```

**Windows (ohne git — Download):**
1. Gehe zu https://github.com/michaeltbs/fencing-analyzer
2. Klicke auf den grünen Button **"<> Code"** → **"Download ZIP"**
3. ZIP entpacken
4. Im entpackten Ordner: Rechtsklick → **"Im Terminal öffnen"**

**Mac:**
```bash
cd ~
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
```

#### Schritt 3: Abhängigkeiten installieren (einmalig, ~5 Minuten)

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

> ⏳ Das kann 3-5 Minuten dauern — YOLO und PyTorch werden heruntergeladen.

#### Schritt 4: App starten

**Windows:**
```powershell
streamlit run app.py --server.port 8501
```

**Mac:**
```bash
streamlit run app.py --server.port 8501
```

#### Schritt 5: Öffnen im Browser

- Chrome, Safari oder Edge öffnen
- In die Adresszeile eingeben: `http://localhost:8501`
- Fertig! 🎉

---

### Variante B: Für Fortgeschrittene (Docker)

> Einmalig Docker installieren, dann läuft's überall gleich.
> Ideal für iPad-Zugriff und GPU-Beschleunigung.

#### Schritt 1: Docker Desktop installieren

**Windows:**
1. Gehe zu https://docs.docker.com/desktop/setup/install/windows-install/
2. Docker Desktop herunterladen und installieren
3. **Windows neu starten**
4. Docker Desktop starten (Warten bis der Wal nicht mehr animiert ist)

**Mac:**
1. Gehe zu https://docs.docker.com/desktop/install/mac-install/
2. Docker Desktop herunterladen und installieren
3. Docker Desktop starten

#### Schritt 2: Fencing Analyzer bauen

**PowerShell (Windows) oder Terminal (Mac):**
```bash
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
docker build -t fencing-analyzer .
```

> ⏳ Beim ersten Mal ca. 5-10 Minuten (Downloads + Installation).

#### Schritt 3: Starten

**Ohne GPU (Standard — läuft auf jedem Rechner):**
```bash
docker run -p 8501:8501 fencing-analyzer
```

**Mit NVIDIA GPU (10-20x schneller):**
```bash
docker run --gpus all -p 8501:8501 fencing-analyzer:gpu
```

#### Schritt 4: Browser öffnen

`http://localhost:8501`

#### Zugriff vom iPad / Handy (gleiches WLAN nötig)

1. Auf dem **Laptop** die IP-Adresse finden:
   - **Windows:** PowerShell → `ipconfig` → `IPv4-Adresse` (z.B. `192.168.1.42`)
   - **Mac:** Terminal → `ipconfig getifaddr en0` (z.B. `192.168.1.42`)
2. Am **iPad / Handy im Safari**: `http://192.168.1.42:8501` eingeben

---

## 🖥️ GPU-Beschleunigung (optional)

Die Analyse läuft auf CPU mit ca. **3 Sekunden pro Frame**.
Mit einer NVIDIA-GPU sinkt das auf **0,2 Sekunden pro Frame** — also 15x schneller.

### Prüfen ob GPU vorhanden:

**Windows:**
```powershell
nvidia-smi
```
- Kommt eine Tabelle? ✅ GPU vorhanden — baue mit GPU-Unterstützung
- Fehler "not found"? ⚠️ Keine NVIDIA-GPU — CPU-Version läuft auch

### Mit GPU bauen und starten:

```bash
cd fencing-analyzer

# Bauen mit GPU
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 \
  -t fencing-analyzer:gpu .

# Starten mit GPU
docker run --gpus all -p 8501:8501 fencing-analyzer:gpu
```

---

## 📖 Bedienung (im Dashboard)

1. **Video hochladen** — Drag & Drop oder Datei auswählen (MP4, bis 4 GB)
2. **Start- und Endzeit wählen** — analysiert wird der ausgewählte Bereich
3. **"Analysieren" klicken** — Fortschrittsbalken zeigt den Status
4. **Dashboard erkunden:**
   - Live-Video-Player mit Overlay (Skelett, Distanzlinie)
   - 15 Metriken als interaktive Diagramme
   - Metriken per Klick ein-/ausblenden
   - PDF-Report herunterladen
5. **Export:** JSON oder CSV

### Tastenkürzel im Video-Player

| Taste | Funktion |
|-------|----------|
| Leertaste | Play / Pause |
| ← → | 1 Sekunde vor/zurück |

---

## 📊 Metriken im Detail

| Nr | Metrik | Beschreibung | Einheit |
|----|--------|-------------|---------|
| 1 | Distanz | Hüft-zu-Hüft Abstand | cm |
| 2 | Waffenarm-Winkel | Schulter-Ellbogen-Handgelenk | ° |
| 3 | Lunge-Tiefe | Vertikaler Hüft-Knöchel-Abstand | px |
| 4 | Bewegungs-Pfad | Hüft-Position als Scatter | x,y |
| 5 | Körperhaltung | Oberkörper-Neigung | ° |
| 6 | Beschleunigung | 2. Ableitung Waffenhand | px/s² |
| 7 | Schritt-Rhythmus | Einzelfuss-Tracking (halb/ganz) | Schritte/s |
| 8 | Synchronisierung | Cross-Correlation Hüft-Geschw. | r, Lag |
| 9 | Handhöhe | Schulter-Handgelenk Höhen-Differenz | px |
| 10 | Arm-Streckung | Schulter-Handgelenk Distanz | px |
| 11 | Standbreite | Fuss-zu-Fuss Abstand | px |
| 12 | Explosivität | Distanz-Änderungsrate | cm/s |
| 13 | Head-Forward | Kopf-Vorsprung vor Hüfte | px |
| 14 | Touché-Kandidaten | Arm gestreckt + nahe Distanz | — |
| 15 | Rhythmus (FFT) | Dominantes Tempo im Gefecht | Hz |
| 16 | Druck-Index | Wer treibt das Gefecht? | ± Wert |

---

## 🎬 Full-Length Analyse (NEU in v1.0)

Ab v1.0 analysiert der Analyzer komplette Gefechte (15+ min) automatisch.
Pausen werden erkannt und übersprungen, das Gefecht in aktive Segmente
zerlegt, jedes Segment einzeln analysiert, alles in einer SQLite-Datenbank
gespeichert — und Studio-Ready-Output (annotiertes HD-Video + Highlight-Reel)
generiert.

### Schnellstart

```bash
# Einzelner Befehl, alles automatisch
python analyze_full.py "M - T16 SCHMIDT vs TREBIS.mp4" \
    --fencer-a "michael-trebis" --name-a "Michael" --last-a "Trebis" \
              --nation-a "GER" --hand-a "right" \
    --fencer-b "richard-schmidt" --name-b "Richard" --last-b "Schmidt" \
                --nation-b "GER" \
    --tournament "Doha 2026" --date "2026-01-15" \
    --score 8 15
```

**Was passiert:**
1. **Pause-Detection** — ffmpeg-basiert, ~10s für 15 min Video
2. **Chunked YOLO-Analyse** — jedes aktive Segment einzeln
3. **SQLite-Persistenz** — Fechter-Stammdaten, Metriken pro Frame, Annotationen
4. **PDF-Report** — 1-Seite mit allen Stats + Charts
5. **Annotiertes HD-Video** — Skelett-Overlay auf 1080p
6. **Highlight-Reel** — 5s Kontext um jeden Touché

### Pipeline-Module

| Datei | Zweck |
|-------|-------|
| `pause_detector.py` | Motion-basierte Pausen-Erkennung (ffmpeg scene detect) |
| `scheduler.py` | Orchestriert Chunks + DB-Persistenz |
| `worker_chunk_analyze.py` | Wrapper für `worker_analyze.py` mit Time-Offset |
| `inference_db.py` | SQLite-Schema + CRUD für Fechter/Bouts/Metrics |
| `studio_export.py` | HD-Render + Highlight-Reel |
| `analyze_full.py` | One-Command-Entry-Point |

### CLI-Optionen für `analyze_full.py`

```
--db PATH              SQLite-Datei (default: fencing.db)
--no-studio            Überspringt HD/Highlight-Output
--no-pdf               Überspringt PDF-Report
--no-highlights        Nur HD-Video, kein Highlight-Reel
--context-s N          Sekunden Kontext um Touché (default: 5.0)
--keep-chunks          Behält Chunk-JSON-Dateien
--no-eval              Überspringt Subagent-Quality-Eval
```

### Quality Evaluation

Nach der Analyse bewertet ein Subagent die Qualität:

- **Per-Chunk:** Frame-Coverage, Distanzverteilung, Schrittrate, Touché-Plausibilität
- **Final:** Realistische Touché-Rate, Cross-Chunk-Konsistenz, Druck-Index-Trend

Beispiel-Output:
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

Eval-Resultate werden in `reports/eval_<bout-id>.json` gespeichert.

### Streamlit-UI mit Full-Length-Modus

Ab v1.1 hat das Streamlit-Dashboard zwei Modi (Umschalter in der Sidebar):

**Schnell-Clip (15-60s):** Der bisherige Modus — Clip auswählen, analysieren, sofort live im Player ansehen.

**Full-Length (komplettes Gefecht):** Workflow:

1. Video in der Sidebar auswählen
2. Auf "Full-Length (komplettes Gefecht)" umschalten
3. Im Hauptbereich auf "Konfiguration" bleiben
4. Fechter A + B Metadaten ausfüllen (Slug, Name, Nation, Hand, Club)
5. Bout-Daten eingeben (Turnier, Datum, Score, Waffe)
6. Output-Optionen wählen (PDF, HD-Video, Highlights, Eval)
7. "🚀 Full-Length-Analyse starten" klicken
8. Live-Log zeigt Fortschritt (alle 5s aktualisiert)
9. Nach Abschluss: "DB durchsuchen" Tab öffnen → alle gespeicherten Bouts sehen
10. Bout auswählen → Metriken + Annotationen + Output-Dateien anzeigen

Die UI startet `analyze_full.py` als Subprozess, pollt Status alle 5s, lädt nach Abschluss automatisch die Bout-Daten aus der SQLite-DB.

### Fechter-Datenbank abfragen

```python
from inference_db import FencerDB
db = FencerDB("fencing.db")
for bout in db.list_bouts():
    print(bout["tournament"], bout["bout_date"],
          bout["fencer_a_score"], "vs", bout["fencer_b_score"])

# Metriken abrufen
metrics = db.get_metrics(bout["id"])
for m in metrics[:10]:
    print(f"t={m['t']:.1f}s  dist={m['dist_cm']}cm  angle_m={m['arm_angle_m']}")

# Annotationen (Touchés, Notizen)
for a in db.get_annotations(bout["id"], type_="touche"):
    print(f"t={a['t']:.1f}s  {a['description']}")
```

---

## 🧪 UI-Tests (Selenium)

Die Streamlit-UI hat Selenium-Tests in `tests/`. Sie starten die App automatisch in einem Subprozess + Chrome headless, prüfen DOM und Screenshots.

### Setup (einmalig)

```bash
pip install -r requirements-test.txt
```

### Tests laufen lassen

```bash
# alle UI-Tests
pytest tests/test_app_ui.py -v

# einzelner Test
pytest tests/test_app_ui.py::test_app_loads -v

# mit Screenshot-Output
pytest tests/test_app_ui.py::test_capture_screenshot -v
```

### Was die Tests prüfen

- App startet ohne Fehler
- Modus-Switch (Schnell-Clip / Full-Length) sichtbar
- Video-Quellen (Upload / Pfad / YouTube) sichtbar
- Sidebar-Sektionen (Analyse-Modus, Video-Quelle)
- GPU/CPU-Indikator
- Screenshot-Capture für visuelle Reviews
- Keine Console-Errors
- Streamlit-Standard-DOM (data-testid) korrekt

Hinweis: Die Tests starten einen echten Streamlit-Server (Port 8511) und Chrome. Dauer pro Test: 5-30s.

---

## ❓ Häufige Fragen

**"Die Analyse ist zu langsam!"**  
→ CPU braucht ~3s pro Frame. Ein 60s-Clip dauert ca. 3 Minuten.  
→ Mit GPU (siehe oben) geht's 15x schneller.  
→ Tipp: Analysiere kurze, actionreiche Ausschnitte (15-30s).

**"Mein Video wird nicht geladen!"**  
→ Nur MP4-Dateien. Maximale Größe: 4 GB.  
→ Falls du ein RAW-Video hast: Vorher mit Handbrake (kostenlos) in MP4 konvertieren.

**"Ich sehe keinen Fortschritt!"**  
→ Die Analyse läuft im Hintergrund. Der Button zeigt "Analysiere...".  
→ Bei sehr langen Clips (>60s) kann es mehrere Minuten dauern — einfach warten.

**"Kann ich mehrere Videos vergleichen?"**  
→ Ja! Analysiere Video 1, dann Video 2 — der Vergleichsmodus zeigt beide überlagert.

**"Kann ich nur einen Fechter tracken?"**  
→ Ja. Die App trackt standardmäßig Michael (grün) und Gegner (rot).  
→ Für andere Videos: Der erste Frame zeigt nummerierte Personen — wähle die zwei relevanten aus.

---

## 🔧 Fehlerbehebung

### "pip wird nicht gefunden"
```powershell
# Windows: Python neu installieren — Haken "Add to PATH" setzen!
```

### "streamlit wird nicht gefunden"
```powershell
pip install streamlit
```

### Port 8501 ist schon belegt
```powershell
streamlit run app.py --server.port 8502
# Dann im Browser: http://localhost:8502
```

### Docker: "libgl1-mesa-glx: not found"
→ Wurde gefixt. `git pull` und neu bauen.

### Docker: "Multi-line Python error"
→ Wurde gefixt. `git pull` und neu bauen.

---

## 📁 Projektstruktur

```
fencing-analyzer/
├── app.py                  # Streamlit-Dashboard (Hauptdatei, kurze Clips)
├── worker_analyze.py       # YOLO-Analyse (Subprocess)
├── worker_chunk_analyze.py # Chunked Worker mit Time-Offset (v1.0)
├── pause_detector.py       # Motion-basierte Pausen-Erkennung (v1.0)
├── scheduler.py            # Chunk-Orchestrierung + DB-Persistenz (v1.0)
├── inference_db.py         # SQLite Fechter/Bout/Metrics Schema (v1.0)
├── subagent_eval.py        # Quality-Eval mit Heuristik + Subagent (v1.1)
├── studio_export.py        # HD-Render + Highlight-Reel (v1.0)
├── analyze_full.py         # One-Command Full-Length Pipeline (v1.0)
├── preview_generator.py    # Annotiertes Preview-Video
├── report_generator.py     # PDF-Report
├── Dockerfile              # Container-Build (CPU + GPU)
├── requirements.txt        # Python-Abhängigkeiten
├── build.sh                # Auto-Build (erkennt GPU)
├── reports/                # Generierte PDF-Reports + merged JSON
├── studio/                 # HD-Videos + Highlight-Reels (v1.0)
├── tests/                  # Analyse-Skripte + Motion-Profile
└── README.md               # Diese Datei
```

---

## 📝 Changelog

### v1.3 (Juni 2026) — Subagent Eval Loop + Selenium UI Tests
- **NEU:** `eval_runner.py` — Hermes-Integration mit `delegate_task`
- **NEU:** Robustes `SCORE:/ISSUES:/SUGGESTIONS:` Parser (single-line, multi-line, mixed)
- **NEU:** `eval_runner.py` CLI — `python eval_runner.py reports/merged_*.json`
- **NEU:** `tests/conftest.py` + `tests/test_app_ui.py` — 11 Selenium UI-Tests
- **NEU:** `requirements-test.txt` — separate Test-Dependencies
- **NEU:** `pytest.ini` — Pytest-Config
- **NEU:** Skill `fencing-pose-analysis` v2.2 mit eval_runner Doku
- Tests prüfen: App-Load, Mode-Switch, Video-Quellen, Sidebar, GPU/CPU, Screenshots, Console-Errors
- Test-Run: `pytest tests/test_app_ui.py -v` (~3 min, headless Chrome)

### v1.2 (Juni 2026) — Streamlit UI Full-Length Mode
- **NEU:** `app.py` Mode-Switch (Schnell-Clip vs. Full-Length) in Sidebar
- **NEU:** Full-Length-Konfigurations-Form (Fechter A/B, Bout-Metadaten, Output-Optionen)
- **NEU:** Subprozess-Wrapper ruft `analyze_full.py` mit allen Flags auf
- **NEU:** Live-Log-Fragment (5s Polling) zeigt Analyse-Fortschritt
- **NEU:** "DB durchsuchen" Tab mit Bout-Liste, Metrik-Tabelle, Distanz-Chart, Annotationen
- **NEU:** Output-Files-Liste (PDF, HD-Video, Highlight-Reel) mit Pfad + Größe

### v1.1 (Juni 2026) — Quality Evaluation
- **NEU:** `subagent_eval.py` — Per-Chunk + Final Quality Evaluator
- Heuristik-basierte Erkennung von Tracking-Fehlern, unrealistischen Werten
- Subagent-Integration: Prompts für LLM-basierte Plausibilitäts-Checks
- `analyze_full.py --no-eval` Flag zum Überspringen
- Eval-Resultate als `reports/eval_<id>.json` persistiert

### v1.0 (Juni 2026) — Full-Length Edition
- **NEU:** `pause_detector.py` — ffmpeg-basierte Pausen-Erkennung
- **NEU:** `scheduler.py` — Chunked-Analyse + DB-Persistenz
- **NEU:** `inference_db.py` — SQLite Fechter/Bout/Metrics/Annotations
- **NEU:** `studio_export.py` — HD-Video (1080p Skelett-Overlay) + Highlight-Reel
- **NEU:** `analyze_full.py` — One-Command-Entry-Point für komplette Gefechte
- 16 Metriken + Touché-Detection laufen jetzt auch über mehrere Chunks
- Chunked-Ausführung: Jedes aktive Segment = 1 YOLO-Subprozess
- SQLite-Output: alle Metriken pro Frame, alle Annotationen abrufbar

### v0.4 — Tracking v2 + UI
- ByteTrack + Side-Constraint + VelocityInterpolator
- 16 Metriken
- Streamlit UI mit Live-Player

### v0.3 — Initiales Release
- YOLOv8m-Pose Integration
- PDF-Report-Generator

---

## ⚖️ Lizenz

MIT — frei verwendbar, veränderbar, weitergebend.

---

## 🙏 Danksagung

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8m Pose
- [Streamlit](https://streamlit.io) — Dashboard-Framework
- [Plotly](https://plotly.com) — Interaktive Charts

---

*Gebaut mit 🏃💨 für Michael Trebis — Deutscher Degen-Fechter, Sportsoldat, CISM 2027.*