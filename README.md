# Fencing Analyzer — Fecht-Video-Analyse mit KI

YOLOv8m-Pose-basierte Extraktion taktischer Metriken aus Fecht-Videos (Degen).  
Interaktives Streamlit-Dashboard mit 9 Metriken und Toggle-UI.

## Features

- **9 Metriken:** Distanz, Waffenarm-Winkel, Lunge-Tiefe, Bewegungs-Pfad, Körperhaltung, Beschleunigung, Schritt-Rhythmus (Halb/Ganz), Reaktions-Synchronisierung, Heatmap
- **2-Personen-Tracking:** Nearest-Center-Assignment über alle Frames
- **Pisten-Kalibrierung:** Automatische cm-Umrechnung via geschätzte Pistenbreite
- **Interaktives Dashboard:** Plotly-Charts mit Klick-Toggle pro Metrik
- **Schritt-Detail-View:** Farbmarkierte Tabelle (Halb/Ganz), kumulativer Plot
- **Export:** JSON, CSV, TXT
- **Dark Mode:** GitHub-inspiriertes Design

## Voraussetzungen

- Python 3.10+
- Ultralytics 8.x (YOLOv8m-pose)
- OpenCV 4.x
- Streamlit 1.30+
- Plotly 5.x
- ffmpeg (für Clip-Extraktion)

## Installation

```bash
git clone https://github.com/michaeltbs/fencing-analyzer.git
cd fencing-analyzer
pip install streamlit plotly ultralytics opencv-python numpy pandas Pillow scipy
```

## Nutzung

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Dann im Browser öffnen: http://localhost:8501

### Workflow
1. Video hochladen (Drag&Drop) oder lokalen Pfad eingeben
2. Clip-Startzeit und Dauer wählen (15-120s)
3. "Analyse starten" klicken (~3 Min/für 15s Clip auf CPU)
4. Dashboard erkunden — Metriken per Klick ein/ausblenden
5. Export als JSON, CSV oder TXT

## Metriken im Detail

| Nr | Metrik | Beschreibung | Einheit |
|----|--------|-------------|---------|
| 1 | Distanz | Hüft-zu-Hüft Abstand | cm |
| 2 | Waffenarm-Winkel | Schulter-Ellbogen-Handgelenk | ° |
| 3 | Lunge-Tiefe | Vertikaler Hüft-Knöchel-Abstand (vorderer Fuss) | px |
| 4 | Bewegungs-Pfad | Hüft-Position als Scatter | x,y Pixel |
| 5 | Körperhaltung | Oberkörper-Neigung zur Vertikalen | ° |
| 6 | Beschleunigung | 2. Ableitung der Waffenhand-Position | px/s² |
| 7 | Schritt-Rhythmus | Einzelfuss-Tracking (halb/ganz) | Schritte/s |
| 8 | Synchronisierung | Cross-Correlation der Hüft-Geschwindigkeiten | r, Lag |
| 9 | Heatmap | 2D-Histogramm der Pisten-Position | Dichte |

## Lizenz

MIT
