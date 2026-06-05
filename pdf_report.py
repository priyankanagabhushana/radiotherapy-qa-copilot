"""
PDF report generation for NeuroQA Copilot.
Uses fpdf2 (pure Python, no system deps — works on Streamlit Cloud).
"""

import os
import datetime
import numpy as np
from fpdf import FPDF

BASE_DIR = os.path.dirname(__file__)
PREVIEW_DIR = os.path.join(BASE_DIR, "mri_previews")

RISK_COLORS = {
    "HIGH": (220, 50, 50),
    "MODERATE": (230, 160, 30),
    "LOW": (34, 180, 85),
}


class NeuroQAReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, "NeuroQA Copilot - Radiotherapy QA Decision Support", align="L")
        self.cell(0, 6, f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"NeuroQA Copilot v1.0 | Research Prototype | Not for clinical use | Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.cell(0, 9, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def risk_badge(self, risk, x=None, y=None):
        r, g, b = RISK_COLORS.get(risk, (100, 100, 100))
        if x is not None:
            self.set_xy(x, y)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 9)
        w = self.get_string_width(f"  {risk} RISK  ") + 4
        self.cell(w, 7, f"  {risk} RISK  ", fill=True, align="C")
        self.set_text_color(30, 30, 30)

    def key_value_row(self, key, value, bold_val=False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(60, 6, key, new_x="RIGHT")
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "B" if bold_val else "", 9)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def simple_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(60, 60, 60)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(250, 250, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=True, align="C")
            self.ln()


def compute_risk(row):
    if row["AI_Uncertainty_Score"] > 0.75:
        return "HIGH"
    if row["Distance_to_OAR_mm"] < 3.0:
        return "HIGH"
    if row["AI_Uncertainty_Score"] > 0.50 or row["Tumor_Volume_cc"] > 25.0:
        return "MODERATE"
    return "LOW"


def generate_patient_pdf(row, df, include_mri=True):
    """Generate a PDF report for a single patient. Returns bytes."""
    pdf = NeuroQAReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pid = row["Patient_ID"]
    risk = compute_risk(row)

    # ── Patient Header ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, f"Patient: {pid}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 7, f"Structure: {row['Structure_Name']}", new_x="RIGHT")
    pdf.risk_badge(risk)
    pdf.ln(10)

    # ── Key Metrics ──
    pdf.section_title("Key Metrics")
    vol_pct = (df["Tumor_Volume_cc"] < row["Tumor_Volume_cc"]).mean() * 100
    dist_pct = (df["Distance_to_OAR_mm"] < row["Distance_to_OAR_mm"]).mean() * 100
    unc_pct = (df["AI_Uncertainty_Score"] < row["AI_Uncertainty_Score"]).mean() * 100

    pdf.key_value_row("Tumor Volume:", f"{row['Tumor_Volume_cc']:.1f} cc  ({vol_pct:.0f}th percentile)")
    pdf.key_value_row("Distance to OAR:", f"{row['Distance_to_OAR_mm']:.1f} mm  ({dist_pct:.0f}th percentile)")
    pdf.key_value_row("AI Uncertainty Score:", f"{row['AI_Uncertainty_Score']:.3f}  ({unc_pct:.0f}th percentile)")
    pdf.key_value_row("Risk Classification:", risk, bold_val=True)
    pdf.ln(4)

    # ── MRI Preview ──
    if include_mri:
        planes_added = 0
        for plane in ["axial", "coronal", "sagittal"]:
            png_path = os.path.join(PREVIEW_DIR, pid, f"{plane}.png")
            if os.path.exists(png_path):
                if planes_added == 0:
                    pdf.section_title("MRI Previews")
                # Check if enough space for image (each ~45mm high)
                if pdf.get_y() > 210:
                    pdf.add_page()
                pdf.set_font("Helvetica", "I", 8)
                pdf.set_text_color(100, 100, 100)
                pdf.cell(0, 5, f"{plane.title()} view", new_x="LMARGIN", new_y="NEXT")
                pdf.image(png_path, w=60)
                pdf.ln(2)
                planes_added += 1
        pdf.ln(2)

    # ── Risk Triggers ──
    pdf.section_title("Risk Triggers")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)

    triggers = []
    if row["Distance_to_OAR_mm"] < 3.0:
        triggers.append(f"Proximity Alert: Distance to {row['Structure_Name']} ({row['Distance_to_OAR_mm']:.1f}mm) below 3mm threshold.")
    if row["AI_Uncertainty_Score"] > 0.75:
        triggers.append(f"Uncertainty Alert: Score ({row['AI_Uncertainty_Score']:.3f}) exceeds 0.75 critical threshold.")
    elif row["AI_Uncertainty_Score"] > 0.50:
        triggers.append(f"Moderate Uncertainty: Score ({row['AI_Uncertainty_Score']:.3f}) exceeds 0.50 moderate threshold.")
    if row["Tumor_Volume_cc"] > 25.0:
        triggers.append(f"Volume Alert: Tumor volume ({row['Tumor_Volume_cc']:.1f} cc) exceeds 25 cc.")
    if not triggers:
        triggers.append("No risk flags triggered - standard QA protocol applies.")

    for t in triggers:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(4, 5, "-", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(170, 5, t)
        pdf.ln(1)
    pdf.ln(3)

    # ── Comparison to Group ──
    pdf.section_title("Comparison to Cohort")
    grp_med_vol = df["Tumor_Volume_cc"].median()
    grp_med_dist = df["Distance_to_OAR_mm"].median()
    grp_med_unc = df["AI_Uncertainty_Score"].median()

    headers = ["Metric", "This Patient", "Group Median", "Percentile"]
    rows = [
        ["Tumor Volume", f"{row['Tumor_Volume_cc']:.1f} cc", f"{grp_med_vol:.1f} cc", f"{vol_pct:.0f}th"],
        ["Distance to OAR", f"{row['Distance_to_OAR_mm']:.1f} mm", f"{grp_med_dist:.1f} mm", f"{dist_pct:.0f}th"],
        ["AI Uncertainty", f"{row['AI_Uncertainty_Score']:.3f}", f"{grp_med_unc:.3f}", f"{unc_pct:.0f}th"],
    ]
    pdf.simple_table(headers, rows, col_widths=[50, 40, 40, 30])
    pdf.ln(6)

    # ── Recommended Actions ──
    pdf.section_title("Recommended Actions")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)

    actions = []
    if row["Distance_to_OAR_mm"] < 3.0:
        actions.append("Manual slice-by-slice review of contour near OAR boundary.")
    if row["AI_Uncertainty_Score"] > 0.75:
        actions.append("Mandatory physician verification before treatment planning.")
    elif row["AI_Uncertainty_Score"] > 0.50:
        actions.append("Spot-check high-uncertainty regions.")
    if row["Tumor_Volume_cc"] > 25.0:
        actions.append("Multi-planar contour integrity review.")
    if not actions:
        actions.append("Standard physician sign-off sufficient.")

    for i, a in enumerate(actions, 1):
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(6, 5, f"{i}.", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(168, 5, a)
        pdf.ln(1)

    # ── Disclaimer ──
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4,
        "This report is generated by a decision-support system and does not replace clinical judgment. "
        "All flagged contours require human verification before treatment delivery. "
        "Data source: BraTS 2020 (Kaggle). Guidelines: AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG Oncology.")

    return bytes(pdf.output())


def generate_group_pdf(df):
    """Generate a group/cohort PDF report. Returns bytes."""
    pdf = NeuroQAReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    high_count = int((df["Risk_Level"] == "HIGH").sum())
    mod_count = int((df["Risk_Level"] == "MODERATE").sum())
    low_count = int((df["Risk_Level"] == "LOW").sum())

    # ── Title ──
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 14, "NeuroQA Copilot - Group Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f"Cohort: {len(df)} patients | BraTS 2020 Dataset", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Risk Summary ──
    pdf.section_title("Risk Distribution")
    pdf.key_value_row("HIGH Risk:", f"{high_count} patients ({high_count/len(df)*100:.0f}%)", bold_val=True)
    pdf.key_value_row("MODERATE Risk:", f"{mod_count} patients ({mod_count/len(df)*100:.0f}%)")
    pdf.key_value_row("LOW Risk:", f"{low_count} patients ({low_count/len(df)*100:.0f}%)")
    pdf.key_value_row("Total:", f"{len(df)} patients")
    pdf.ln(4)

    # ── Descriptive Statistics ──
    pdf.section_title("Descriptive Statistics")
    numeric_cols = ["Tumor_Volume_cc", "Distance_to_OAR_mm", "AI_Uncertainty_Score"]
    headers = ["Metric", "Mean", "Median", "Std", "Min", "Max", "Skewness"]
    rows = []
    for col in numeric_cols:
        data = df[col].dropna()
        skew = float(data.skew()) if len(data) > 2 else 0
        rows.append([
            col.replace("_", " ").title(),
            f"{data.mean():.2f}",
            f"{data.median():.2f}",
            f"{data.std():.2f}",
            f"{data.min():.2f}",
            f"{data.max():.2f}",
            f"{skew:.2f}",
        ])
    pdf.simple_table(headers, rows, col_widths=[40, 22, 22, 22, 22, 22, 22])
    pdf.ln(6)

    # ── Thresholds ──
    pdf.section_title("Rule Engine Thresholds")
    headers = ["Parameter", "HIGH Threshold", "MODERATE Threshold"]
    rows = [
        ["Distance to OAR", "< 3.0 mm", "-"],
        ["AI Uncertainty", "> 0.75", "> 0.50"],
        ["Tumor Volume", "-", "> 25.0 cc"],
    ]
    pdf.simple_table(headers, rows, col_widths=[60, 50, 50])
    pdf.ln(6)

    # ── Patient List ──
    pdf.section_title("Patient List")
    sorted_df = df.sort_values(by="Risk_Level", key=lambda s: s.map({"HIGH": 0, "MODERATE": 1, "LOW": 2}))
    headers = ["Patient ID", "Structure", "Volume (cc)", "Dist OAR (mm)", "Uncertainty", "Risk"]
    rows = []
    for _, r in sorted_df.iterrows():
        rows.append([
            r["Patient_ID"],
            str(r["Structure_Name"]),
            f"{r['Tumor_Volume_cc']:.1f}",
            f"{r['Distance_to_OAR_mm']:.1f}",
            f"{r['AI_Uncertainty_Score']:.3f}",
            compute_risk(r),
        ])
    pdf.simple_table(headers, rows, col_widths=[38, 28, 25, 28, 25, 20])

    # ── Disclaimer ──
    pdf.ln(8)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4,
        "This report is generated by a decision-support system and does not replace clinical judgment. "
        "Research prototype only. Not for clinical use.")

    return bytes(pdf.output())
