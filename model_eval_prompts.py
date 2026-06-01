# === TEST-PROMPTS für Model-Evaluation ===
# Jeder Subagent bekommt exakt denselben Prompt.
# Antworten werden per Spracherkennung auf Deutsch verlangt.

PROMPT_EMAIL = """Schreibe eine professionelle Sponsoring-E-Mail auf Deutsch (max 150 Wörter). 
Absender: Bundeswehr-Fechter (Degen), Platz 19 deutsche Rangliste, trainiert am BSP Leipzig.
Empfänger: Fitness First Leipzig.
Ziel: Kostenlose Mitgliedschaft + Trainingsunterstützung im Gegenzug für Markensichtbarkeit.
Ton: warm, persönlich, aber professionell. Kein Filler.
Betreff: [selbst wählen]
Signatur: Michael Trebis"""

PROMPT_CODING = """Schreibe eine Python-Funktion `calculate_lunge_velocity(kpts_series, fps)`.
- Input: Liste von Frames, jeder Frame = [x0,y0,x1,y1,...,x16,y16] (COCO-Pose 17 Keypoints, flattened)
- Extrahiere rechten Fuss-Knöchel (Keypoint 16) und linken Fuss-Knöchel (Keypoint 15)
- Berechne Frame-zu-Frame Geschwindigkeit jedes Knöchels in px/s
- Erkenne Lunge-Bewegungen: wenn ein Knöchel sich >50px zwischen Frames bewegt UND der andere <10px
- Rückgabe: Liste von dicts mit t (Frame-Index/Fps), velocity_px_s, is_lunge (bool)
- Kurze Erklärung der Logik + vollständiger Code"""

PROMPT_ANALYSIS = """Analysiere diese Instagram-Engagement-Daten (18 Posts, 90 Tage):
Post 01-06: organisch, 90-231 Likes, kein Trend
Post 07: Sponsor-Collab Fitness First, 639 Likes
Post 08: Training-Video BSP, 412 Likes
Post 09: Turnier-Highlight Doha, 8.199 Likes (viral)
Post 10-18: nach viralem Post, 180-450 Likes
Aufgaben:
1. Identifiziere Muster und Trends (max 3 Sätze)
2. Nenne 2 konkrete Handlungsempfehlungen
3. Schätze: was machte Post 9 erfolgreich?
Antwort auf Deutsch, max 200 Wörter, kein Filler."""

PROMPT_CREATIVE = """Erstelle 3 Instagram-Post-Ideen für einen Degen-Fechter (Michael Trebis).
Ziel: Mehr Follower (aktuell 1.200), Zielgruppe 16-25, Fecht-Sportler.
Jede Idee soll enthalten:
- Hook (erste 2 Wörter/Emojis)
- Format (Reel/Carousel/Static)
- Beschreibung (1 Satz)
- Warum es funktioniert (1 Satz)
Bonuspunkt: eine Idee, die Viral-Potential hat.
Antwort auf Deutsch."""

PROMPT_VISION = """Beschreibe dieses Bild genau:
- Wie viele Personen sind zu sehen?
- Welche Sportart wird ausgeübt?
- Beschreibe die Körperhaltung der Hauptperson (Knie-Winkel, Arm-Position, Gewichtsverlagerung)
- Welche taktische Situation könnte das sein? (Angriff, Defensive, Transition?)
Antwort auf Deutsch, max 150 Wörter, präzise."""

print("TEST-PROMPTS bereit")