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
| 9 | Heatmap | 2D-Histogramm Pisten-Position | Dichte |
| 10 | Druck-Index | Wer treibt das Gefecht? | ± Wert |
| 11-15 | Weitere Metriken | Touche, Tempo, etc. | siehe UI |

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
├── app.py                  # Streamlit-Dashboard (Hauptdatei)
├── worker_analyze.py       # YOLO-Analyse (Subprocess)
├── report_generator.py     # PDF-Report
├── Dockerfile              # Container-Build (CPU + GPU)
├── requirements.txt        # Python-Abhängigkeiten
├── build.sh                # Auto-Build (erkennt GPU)
├── reports/                # Generierte PDF-Reports
└── README.md               # Diese Datei
```

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