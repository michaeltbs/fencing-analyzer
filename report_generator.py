"""
PDF-Report Generator for Fencing Analyzer
Generates Bundestrainer-ready 1-page PDF reports from analysis results.
"""
import os, io, json, textwrap, math
from pathlib import Path
from datetime import datetime
from fpdf import FPDF

C_GREEN = "#00ff88"
C_RED = "#ff4466"
C_BLUE = "#00ccff"
C_BG = "#0d1117"
C_CARD = "#161b22"
C_TEXT = "#c9d1d9"

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def px_to_cm(px, px_per_cm):
    return round(px / px_per_cm, 1) if px_per_cm > 0 else 0

def generate_report(result, video_name="Gefecht"):
    """Main entry point — returns (pdf_path: str, preview: str)."""
    s = result["summary"]
    
    # Build filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = REPORTS_DIR / f"Fecht-Analyse_{video_name}_{ts}.pdf"
    
    pdf = FPDF(format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    
    w = pdf.w - 2 * pdf.l_margin  # usable width
    
    # === Colors (approximation for PDF) ===
    BLACK = (13, 17, 23)
    DARK = (22, 27, 34)
    BORDER = (48, 54, 61)
    TEXT_C = (201, 209, 217)
    MUTED = (139, 148, 158)
    GREEN = (0, 255, 136)
    RED = (255, 68, 102)
    BLUE = (0, 204, 255)
    ACCENT = (88, 166, 255)
    
    # === Helper functions ===
    def dark_rect(y, h, fill=DARK, border=BORDER):
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(*border)
        pdf.rect(pdf.l_margin, y, w, h, style="DF")
    
    def write_text(x, y, text, size=10, color=TEXT_C, style="", align="L"):
        # Strip unicode chars not in latin-1
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.set_text_color(*color)
        pdf.set_font("Helvetica", style, size)
        pdf.set_xy(x, y)
        if align == "R":
            pdf.cell(w - 2*pdf.l_margin, 5, safe_text, align="R")
        elif align == "C":
            pdf.cell(w, 5, safe_text, align="C")
        else:
            pdf.cell(0, 5, safe_text)
    
    def write_stat_box(y, col_x, box_w, label, value, sub=""):
        box_h = 22
        pdf.set_fill_color(*DARK)
        pdf.set_draw_color(*BORDER)
        pdf.rect(col_x, y, box_w, box_h, style="DF")
        write_text(col_x + 2, y + 1, label, 6, MUTED)
        write_text(col_x + 2, y + 7, value, 14, TEXT_C)
        if sub:
            write_text(col_x + 2, y + 16, sub, 6, MUTED)
    
    def mini_chart(pdf_obj, x, y, chart_w, chart_h, data, color=(0,200,255), label=""):
        if not data:
            return
        min_v = min(data)
        max_v = max(data)
        rng = max_v - min_v if max_v > min_v else 1
        pdf_obj.set_draw_color(*color)
        pdf_obj.set_line_width(0.6)
        
        n = len(data)
        step = chart_w / max(n-1, 1)
        
        points = []
        for i, v in enumerate(data):
            nx = x + i * step
            ny = y + chart_h - ((v - min_v) / rng) * (chart_h - 4) - 2
            points.append((nx, ny))
        
        for i in range(len(points) - 1):
            pdf_obj.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        
        if label:
            write_text(x, y - 4, label, 6, MUTED)
    
    # === TITLE ===
    title = f"Fecht-Analyse: {video_name}"
    write_text(pdf.l_margin, 8, title, 18, ACCENT, "B")
    write_text(pdf.l_margin, 17, f"{s['duration']}s @ {s['fps']}fps — {s['frames']} Frames", 8, MUTED)
    
    # === STATS ROW ===
    stat_boxes = [
        ("Distanz ⌀", f'{s.get("dist_avg", 0):.0f} cm'),
        ("Winkel M/G", f'{s.get("m_angle_avg",0):.0f}° / {s.get("g_angle_avg",0):.0f}°'),
        ("Schritte M/G", f'{s.get("m_steps",0)} / {s.get("g_steps",0)}'),
        ("Korrelation", f'{s.get("correlation",0):.2f}'),
    ]
    box_w = (w - 12) / 4
    y_stats = 24
    for i, (label, value) in enumerate(stat_boxes):
        write_stat_box(y_stats, pdf.l_margin + i * (box_w + 4), box_w, label, value)
    
    # Second row
    stat_boxes2 = [
        ("Touchés", f'{s.get("touches",0)} ({s.get("touches_high",0)} high)'),
        ("Explosivität", f'max {s.get("expl_max",0):.0f} cm/s'),
        ("Rhythmus", f'{s.get("rhythm_dominant",0):.1f} Hz dominant'),
        ("Standbreite M/G", f'{s.get("m_stance_avg",0):.0f} / {s.get("g_stance_avg",0):.0f}'),
    ]
    for i, (label, value) in enumerate(stat_boxes2):
        write_stat_box(y_stats + 26, pdf.l_margin + i * (box_w + 4), box_w, label, value)
    
    # === CHARTS ===
    y_charts = y_stats + 56
    
    # Chart 1: Distance
    dist_data = [d["cm"] for d in result["m1_dist"]] if result["m1_dist"] else [0]
    pdf.set_fill_color(*DARK)
    pdf.set_draw_color(*BORDER)
    pdf.rect(pdf.l_margin, y_charts, w, 40, style="DF")
    write_text(pdf.l_margin + 2, y_charts + 1, "Distanz (cm)", 7, MUTED)
    mini_chart(pdf, pdf.l_margin + 5, y_charts + 10, w - 10, 26, dist_data, BLUE)
    
    # Chart 2: Explosivity
    y_charts2 = y_charts + 44
    expl = [d["cm_s"] for d in result["m12_expl"]] if result["m12_expl"] else [0]
    pdf.set_fill_color(*DARK)
    pdf.set_draw_color(*BORDER)
    pdf.rect(pdf.l_margin, y_charts2, w/2 - 2, 40, style="DF")
    write_text(pdf.l_margin + 2, y_charts2 + 1, "Explosivität (cm/s)", 7, MUTED)
    mini_chart(pdf, pdf.l_margin + 3, y_charts2 + 10, w/2 - 8, 26, expl, ACCENT)
    
    # Chart 3: Sync (both velocities)
    sync_m = result.get("m8_vel_m", [0])
    sync_g = result.get("m8_vel_g", [0])
    min_l = min(len(sync_m), len(sync_g))
    pdf.set_fill_color(*DARK)
    pdf.set_draw_color(*BORDER)
    pdf.rect(pdf.l_margin + w/2 + 2, y_charts2, w/2 - 2, 40, style="DF")
    write_text(pdf.l_margin + w/2 + 4, y_charts2 + 1, "Sync Michael/Gegner", 7, MUTED)
    mini_chart(pdf, pdf.l_margin + w/2 + 5, y_charts2 + 10, w/2 - 12, 22, sync_m[:min_l], GREEN)
    mini_chart(pdf, pdf.l_margin + w/2 + 5, y_charts2 + 24, w/2 - 12, 12, sync_g[:min_l], RED)
    
    # === TOUCHE TABLE ===
    y_touche = y_charts2 + 48
    high_touches = [t for t in result.get("m14_touches", []) if t["confidence"] == "high"]
    
    if high_touches:
        pdf.set_fill_color(*DARK)
        pdf.set_draw_color(*BORDER)
        pdf.rect(pdf.l_margin, y_touche, w, 10 + 5 * len(high_touches), style="DF")
        write_text(pdf.l_margin + 2, y_touche + 2, "Touché-Kandidaten (high confidence)", 8, ACCENT)
        
        # Header row
        headers = ["Zeit", "Wer", "Distanz", "Ext M", "Ext G"]
        col_w = w / len(headers)
        pdf.set_fill_color(33, 38, 45)
        pdf.set_draw_color(*BORDER)
        for hi, hdr in enumerate(headers):
            pdf.rect(pdf.l_margin + hi * col_w, y_touche + 11, col_w, 5, style="DF")
            write_text(pdf.l_margin + hi * col_w + 1, y_touche + 11, hdr, 6, MUTED)
        
        for ti, t in enumerate(high_touches):
            row_y = y_touche + 17 + ti * 5
            vals = [f'{t["t"]:.1f}s', t["who"], f'{t["dist_cm"]:.0f}cm', f'{t["ext_m"]:.0f}', f'{t["ext_g"]:.0f}']
            for vi, v in enumerate(vals):
                write_text(pdf.l_margin + vi * col_w + 1, row_y, v, 7, TEXT_C)
    
    # === FOOTER ===
    y_footer = 275
    pdf.set_draw_color(*BORDER)
    pdf.line(pdf.l_margin, y_footer, pdf.l_margin + w, y_footer)
    write_text(pdf.l_margin, y_footer + 3, "Erstellt mit Fecht-Analyzer v0.3 — YOLOv8m-Pose", 6, MUTED)
    write_text(pdf.l_margin, y_footer + 9, f"Erstellt: {datetime.now().strftime('%d.%m.%Y %H:%M')}", 6, MUTED)
    
    pdf.output(str(pdf_path))
    
    # Generate preview text
    preview = f"""📄 **Fecht-Analyse: {video_name}**
• Dauer: {s['duration']}s ({s['frames']} Frames)
• Distanz ⌀: {s.get('dist_avg',0):.0f} cm
• Winkel M/G: {s.get('m_angle_avg',0):.0f}° / {s.get('g_angle_avg',0):.0f}°
• Schritte: M {s.get('m_steps',0)} / G {s.get('g_steps',0)}
• Korrelation: {s.get('correlation',0):.2f}
• Touchés: {s.get('touches',0)} ({s.get('touches_high',0)} high)
• Dominanter Rhythmus: {s.get('rhythm_dominant',0):.1f} Hz"""
    
    return str(pdf_path), preview


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            result = json.load(f)
        path, preview = generate_report(result, sys.argv[2] if len(sys.argv) > 2 else "Test")
        print(f"PDF saved: {path}")
        print(preview)
