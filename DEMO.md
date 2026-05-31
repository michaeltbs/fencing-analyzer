# Fecht-Analyzer v0.3 — Enterprise Demo

## Schnellstart

```bash
cd C:\Users\micha\Desktop\fencing_analyzer
streamlit run app.py --server.port 8501
```

**→ http://127.0.0.1:8501**

## So führst du die Demo vor

### 1. Video laden
- **Lokaler Pfad** (voreingestellt): Doha 2026 Gefecht — einfach Enter drücken
- **Datei-Upload:** bis 4 GB, MP4/MOV/AVI/MKV

### 2. Bereich auswählen
- Player zeigt das komplette Video als Preview (480px, komprimiert)
- **Start/Ende** per Zahlenfeld: z.B. 60s Start, 90s Ende (30s Analyse)
- Klick: "🔍 Bereich analysieren"

### 3. Analyse läuft (3-5 Sekunden pro Sekunde Clip)
- Worker läuft im separaten Prozess — UI bleibt voll responsiv
- Fortschritt wird alle 3s gepollt
- Timeout nach 10 Minuten (bei zu langen Clips)

### 4. Ergebnisse
Nach der Analyse siehst du:

| Bereich | Was |
|---------|-----|
| **Summary** | 8 Metriken + PDF-Button + CSV-Export |
| **Auto-Kommentar** | Zusammengefasste Kampf-Analyse in einem Satz |
| **Player (separater Tab)** | Video + Skelett-Overlay (Grün=Michael, Rot=Gegner) + Toggles |
| **15 Charts** | Distanz (mit Touché-Linien), Winkel, Haltung, etc. |
| **Touché-Tabelle** | High-confidence default, Medium unter "Details" |
| **Kalibrierung** | Auto-orientation (seitlich/frontal), cm-Angaben |

### 5. Vergleichsmodus
Klick: "➕ Zweiten Bereich analysieren" → zweiten Bereich wählen → **überlagerter Distanz-Chart + Delta-Metriken**

### 6. Exporte
- **📄 PDF-Report** — 1 Seite: Stats, 3 Charts, Touché-Tabelle
- **📊 CSV Export** — eine Zeile pro Frame, alle 15 Metriken
- **▶ Player öffnen** — Skelett-Overlay in Echtzeit

## Die 15 Metriken

| # | Metrik | Einheit | Bedeutung |
|---|--------|---------|-----------|
| M1 | Distanz | cm | Abstand Hüft-Mittelpunkte |
| M2 | Waffenarm-Winkel | ° | Grad Schulter-zu-Hand |
| M3 | Lunge-Tiefe | px | Hüft-zu-Knöchel vertikal |
| M4 | Bewegungs-Pfad | px | Hüft-Trajektorie |
| M5 | Körperhaltung | ° | Vorbeuge gegen Gegner |
| M6 | Beschleunigung | px/s² | Handgelenk-Beschleunigung |
| M7 | Schritt-Rhythmus | count | Fuß-Bewegungen |
| M8 | Synchronisierung | corr | Bewegungs-Korrelation |
| M9 | Waffenhand-Höhe | px | Guard-Position |
| M10 | Arm-Streckung | px | Extension-Distanz |
| M11 | Standbreite | px | Balance/Breite |
| M12 | Distanz-Explosivität | cm/s | Geschwindigkeit des Distanzwechsels |
| M13 | Head-Forward | px | Kopf vor/über Körper |
| M14 | Touché-Kandidat | — | 90% Arm + 15% Dist. + 3-Frame-Min |
| M15 | Rhythmus-Pattern | Hz | FFT-Distanz, dominant. Frequenz |

## Tastatur-Shortcuts (Player)

- **Leertaste** — Play/Pause
- **← →** — 1s zurück/vor

## System

- **Modell:** YOLOv8m-Pose (50.8 MB, CPU ≈ 3s/frame)
- **GPU:** Automatisch erkannt (falls CUDA verfügbar)
- **Metriken:** 15 (9 Basis + 6 erweiterte)
- **Max Upload:** 4 GB
- **Export:** PDF (fpdf2), CSV (alle Rohdaten)
