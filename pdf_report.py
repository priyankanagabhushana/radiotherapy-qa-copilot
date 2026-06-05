"""
PDF report generation for NeuroQA Copilot.
Uses fpdf2 (pure Python, no system deps — works on Streamlit Cloud).
Generates rich PDFs with embedded charts, MRI previews, tables, and statistics.
"""

import os
import io
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

BASE_DIR = os.path.dirname(__file__)
PREVIEW_DIR = os.path.join(BASE_DIR, "mri_previews")

RISK_COLORS = {"HIGH": (220, 50, 50), "MODERATE": (230, 160, 30), "LOW": (34, 180, 85)}
RISK_HEX = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}
BG = "#0f172a"


def compute_risk(row):
    if row["AI_Uncertainty_Score"] > 0.75:
        return "HIGH"
    if row["Distance_to_OAR_mm"] < 3.0:
        return "HIGH"
    if row["AI_Uncertainty_Score"] > 0.50 or row["Tumor_Volume_cc"] > 25.0:
        return "MODERATE"
    return "LOW"


def _fig_to_image(fig, dpi=130):
    """Convert matplotlib figure to bytes for embedding in PDF."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


class NeuroQAPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 120, 120)
        self.cell(95, 5, "NeuroQA Copilot", align="L")
        self.cell(95, 5, datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 6)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"NeuroQA Copilot v1.0 | Research Prototype | Not for clinical use | Page {self.page_no()}/{{nb}}", align="C")

    def title_section(self, text, size=14):
        self.set_font("Helvetica", "B", size)
        self.set_text_color(30, 30, 30)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(59, 130, 246)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 80, self.get_y())
        self.set_line_width(0.2)
        self.ln(3)

    def sub_title(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def risk_badge(self, risk, x=None, y=None):
        r, g, b = RISK_COLORS.get(risk, (100, 100, 100))
        if x is not None:
            self.set_xy(x, y)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        w = self.get_string_width(f" {risk} RISK ") + 4
        self.cell(w, 6, f" {risk} RISK ", fill=True)
        self.set_text_color(30, 30, 30)

    def kv(self, key, val, bold=False):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(55, 5, key, new_x="RIGHT")
        self.set_text_color(30, 30, 30)
        self.set_font("Helvetica", "B" if bold else "", 8)
        self.cell(0, 5, str(val), new_x="LMARGIN", new_y="NEXT")

    def table(self, headers, rows, widths=None):
        if not rows:
            return
        if widths is None:
            widths = [190 / len(headers)] * len(headers)
        self.set_font("Helvetica", "B", 7)
        self.set_fill_color(235, 235, 235)
        self.set_text_color(50, 50, 50)
        for i, h in enumerate(headers):
            self.cell(widths[i], 6, str(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 7)
        self.set_text_color(30, 30, 30)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 0
            if fill:
                self.set_fill_color(248, 248, 248)
            for i, cell in enumerate(row):
                self.cell(widths[i], 5, str(cell), border=1, fill=fill, align="C")
            self.ln()

    def embed_chart(self, fig, w=170):
        """Embed a matplotlib figure as an image."""
        img_buf = _fig_to_image(fig)
        # Check if enough space (estimate 80mm height)
        if self.get_y() > 200:
            self.add_page()
        self.image(img_buf, w=w)
        self.ln(3)

    def check_space(self, needed=60):
        if self.get_y() > 300 - needed:
            self.add_page()


# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL PATIENT PDF
# ═══════════════════════════════════════════════════════════════

def generate_patient_pdf(row, df, include_mri=True):
    """Generate a full patient report PDF with MRI previews, charts, tables."""
    pdf = NeuroQAPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pid = row["Patient_ID"]
    risk = compute_risk(row)
    risk_c = RISK_COLORS[risk]
    vol_pct = (df["Tumor_Volume_cc"] < row["Tumor_Volume_cc"]).mean() * 100
    dist_pct = (df["Distance_to_OAR_mm"] < row["Distance_to_OAR_mm"]).mean() * 100
    unc_pct = (df["AI_Uncertainty_Score"] < row["AI_Uncertainty_Score"]).mean() * 100

    # ── Patient Header ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(130, 12, f"Patient: {pid}", new_x="RIGHT")
    pdf.risk_badge(risk)
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, f"Structure: {row['Structure_Name']}  |  Source: {row.get('Data_Source', 'BraTS 2020')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ── Key Metrics with percentile bars ──
    pdf.title_section("Key Metrics")
    metrics = [
        ("Tumor Volume", f"{row['Tumor_Volume_cc']:.1f} cc", vol_pct, df["Tumor_Volume_cc"]),
        ("Distance to OAR", f"{row['Distance_to_OAR_mm']:.1f} mm", dist_pct, df["Distance_to_OAR_mm"]),
        ("AI Uncertainty", f"{row['AI_Uncertainty_Score']:.3f}", unc_pct, df["AI_Uncertainty_Score"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.5), facecolor=BG)
    for ax, (name, val, pct, data) in zip(axes, metrics):
        ax.set_facecolor(BG)
        ax.barh([0], [pct], color=RISK_HEX[risk], height=0.5, alpha=0.8)
        ax.barh([0], [100], color="#1e293b", height=0.5, alpha=0.3, zorder=0)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_title(f"{name}: {val}", color="white", fontsize=9, pad=4)
        ax.text(pct, 0, f" {pct:.0f}th %ile", color="white", fontsize=7,
                va="center", ha="left" if pct < 80 else "right")
        ax.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#334155")
    fig.tight_layout()
    pdf.embed_chart(fig, w=180)

    # ── MRI Previews ──
    if include_mri:
        planes_found = []
        for plane in ["axial", "coronal", "sagittal"]:
            mask_path = os.path.join(PREVIEW_DIR, pid, f"{plane}_mask.png")
            png_path = os.path.join(PREVIEW_DIR, pid, f"{plane}.png")
            if os.path.exists(png_path) and os.path.exists(mask_path):
                planes_found.append(plane)

        if planes_found:
            pdf.title_section("MRI Previews (Axial / Coronal / Sagittal)")
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, "Segmentation overlay at opacity=0.55. Red=Tumor Core, Blue=Edema, Yellow=Enhancing.",
                     new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            # Render composite images
            from mri_viewer import render_plane_composite
            for plane in planes_found:
                pdf.check_space(65)
                composite = render_plane_composite(pid, plane, opacity=0.55)
                if composite is not None:
                    fig_mri, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
                    ax.set_facecolor(BG)
                    ax.imshow(composite)
                    ax.set_title(plane.title(), color="white", fontsize=10, pad=4)
                    ax.axis("off")
                    pdf.embed_chart(fig_mri, w=90)

    # ── Risk Triggers ──
    pdf.check_space(40)
    pdf.title_section("Risk Triggers")
    triggers = []
    if row["Distance_to_OAR_mm"] < 3.0:
        triggers.append(f"Proximity Alert: Distance to {row['Structure_Name']} ({row['Distance_to_OAR_mm']:.1f}mm) below 3mm threshold.")
    if row["AI_Uncertainty_Score"] > 0.75:
        triggers.append(f"Uncertainty Alert: Score ({row['AI_Uncertainty_Score']:.3f}) exceeds 0.75 critical threshold.")
    elif row["AI_Uncertainty_Score"] > 0.50:
        triggers.append(f"Moderate Uncertainty: Score ({row['AI_Uncertainty_Score']:.3f}) exceeds 0.50 threshold.")
    if row["Tumor_Volume_cc"] > 25.0:
        triggers.append(f"Volume Alert: Tumor volume ({row['Tumor_Volume_cc']:.1f} cc) exceeds 25 cc.")
    if "optic" in str(row["Structure_Name"]).lower() or "chiasm" in str(row["Structure_Name"]).lower():
        triggers.append(f"Sensitive Structure: '{row['Structure_Name']}' is part of the optic apparatus.")
    if "brainstem" in str(row["Structure_Name"]).lower():
        triggers.append(f"Critical Organ: '{row['Structure_Name']}' is a serial organ.")
    if not triggers:
        triggers.append("No risk flags triggered - standard QA protocol applies.")

    pdf.set_font("Helvetica", "", 8)
    for t in triggers:
        pdf.set_x(14)
        pdf.set_text_color(200, 50, 50)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(4, 5, ">", new_x="RIGHT")
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(170, 5, t)
        pdf.ln(1)
    pdf.ln(3)

    # ── Comparison to Cohort (table + chart) ──
    pdf.check_space(80)
    pdf.title_section("Comparison to Cohort")

    grp_med_vol = df["Tumor_Volume_cc"].median()
    grp_med_dist = df["Distance_to_OAR_mm"].median()
    grp_med_unc = df["AI_Uncertainty_Score"].median()

    pdf.table(
        ["Metric", "This Patient", "Group Median", "Percentile", "Interpretation"],
        [
            ["Tumor Volume", f"{row['Tumor_Volume_cc']:.1f} cc", f"{grp_med_vol:.1f} cc",
             f"{vol_pct:.0f}th", "higher" if vol_pct > 50 else "lower"],
            ["Distance to OAR", f"{row['Distance_to_OAR_mm']:.1f} mm", f"{grp_med_dist:.1f} mm",
             f"{dist_pct:.0f}th", "closer" if dist_pct < 50 else "farther"],
            ["AI Uncertainty", f"{row['AI_Uncertainty_Score']:.3f}", f"{grp_med_unc:.3f}",
             f"{unc_pct:.0f}th", "higher" if unc_pct > 50 else "lower"],
        ],
        widths=[35, 30, 30, 25, 30],
    )
    pdf.ln(4)

    # Percentile bar chart
    fig, axes = plt.subplots(1, 3, figsize=(12, 2), facecolor=BG)
    for ax, (name, pct) in zip(axes, [("Volume", vol_pct), ("Dist OAR", dist_pct), ("Uncertainty", unc_pct)]):
        ax.set_facecolor(BG)
        colors_list = ["#334155"] * 10
        bucket = min(int(pct / 10), 9)
        for i in range(10):
            if i <= bucket:
                colors_list[i] = RISK_HEX[risk]
        ax.bar(range(10), [10]*10, color=colors_list, width=1, edgecolor="#0f172a", linewidth=0.5)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name} ({pct:.0f}th %ile)", color="white", fontsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    pdf.embed_chart(fig, w=180)

    # ── Recommended Actions ──
    pdf.check_space(40)
    pdf.title_section("Recommended Actions")
    actions = []
    if row["Distance_to_OAR_mm"] < 3.0:
        actions.append("Manual slice-by-slice review of contour near OAR boundary.")
    if row["AI_Uncertainty_Score"] > 0.75:
        actions.append("Mandatory physician verification before treatment planning.")
    elif row["AI_Uncertainty_Score"] > 0.50:
        actions.append("Spot-check high-uncertainty regions. Consider MRI co-registration.")
    if row["Tumor_Volume_cc"] > 25.0:
        actions.append("Multi-planar (axial/coronal/sagittal) contour integrity review.")
    if not actions:
        actions.append("Standard physician sign-off is sufficient.")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    for i, a in enumerate(actions, 1):
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(6, 5, f"{i}.", new_x="RIGHT")
        pdf.set_font("Helvetica", "", 8)
        pdf.multi_cell(168, 5, a)
        pdf.ln(1)

    # ── Disclaimer ──
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 3,
        "This report is generated by a decision-support system and does not replace clinical judgment. "
        "All flagged contours require human verification before treatment delivery. "
        "Data: BraTS 2020 (Kaggle). Guidelines: AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG Oncology.")

    return bytes(pdf.output())


# ═══════════════════════════════════════════════════════════════
# GROUP REPORT PDF
# ═══════════════════════════════════════════════════════════════

def generate_group_pdf(df):
    """Generate a full group report PDF with charts, tables, statistics."""
    import stats_analysis as sa

    pdf = NeuroQAPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    high_n = int((df["Risk_Level"] == "HIGH").sum())
    mod_n = int((df["Risk_Level"] == "MODERATE").sum())
    low_n = int((df["Risk_Level"] == "LOW").sum())
    numeric_cols = ["Tumor_Volume_cc", "Distance_to_OAR_mm", "AI_Uncertainty_Score"]
    for c in ["Irregularity_Score", "Max_Diameter_mm"]:
        if c in df.columns:
            numeric_cols.append(c)

    # ── Title ──
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(0, 12, "NeuroQA Copilot - Group Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Cohort: {len(df)} patients | BraTS 2020 Dataset | Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # ── Risk Distribution Pie Chart ──
    pdf.title_section("Risk Distribution")
    fig, (ax_pie, ax_text) = plt.subplots(1, 2, figsize=(8, 3), facecolor=BG,
                                           gridspec_kw={"width_ratios": [1, 1.5]})
    ax_pie.set_facecolor(BG)
    sizes = [high_n, mod_n, low_n]
    wedges, _ = ax_pie.pie(sizes, labels=None, colors=["#ef4444", "#f59e0b", "#22c55e"],
                           startangle=90, wedgeprops=dict(width=0.4, edgecolor=BG))
    ax_pie.text(0, 0, str(len(df)), ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax_pie.text(0, -0.15, "patients", ha="center", va="center", fontsize=7, color="#94a3b8")
    ax_pie.legend(wedges, [f"HIGH ({high_n})", f"MOD ({mod_n})", f"LOW ({low_n})"],
                  loc="lower center", fontsize=7, facecolor="#1e293b", edgecolor="#334155",
                  labelcolor="white", ncol=3)

    ax_text.set_facecolor(BG)
    ax_text.axis("off")
    info = [
        f"HIGH Risk:    {high_n} patients ({high_n/len(df)*100:.0f}%)",
        f"MODERATE:     {mod_n} patients ({mod_n/len(df)*100:.0f}%)",
        f"LOW Risk:     {low_n} patients ({low_n/len(df)*100:.0f}%)",
        "",
        f"Thresholds:",
        f"  Distance < 3.0mm -> HIGH",
        f"  Uncertainty > 0.75 -> HIGH",
        f"  Volume > 25cc -> MODERATE",
    ]
    ax_text.text(0, 0.95, "\n".join(info), transform=ax_text.transAxes,
                 fontsize=9, color="#e2e8f0", verticalalignment="top",
                 fontfamily="monospace", linespacing=1.5)
    fig.tight_layout()
    pdf.embed_chart(fig, w=170)

    # ── Descriptive Statistics ──
    pdf.check_space(60)
    pdf.title_section("Descriptive Statistics")
    stats_df = sa.compute_descriptive_stats(df, columns=numeric_cols)
    headers = ["Metric", "N", "Mean", "Median", "Std", "Min", "Max", "Skew", "Kurt", "CV%"]
    rows = []
    for _, r in stats_df.iterrows():
        rows.append([
            str(r.get("Variable", "")),
            str(r.get("N", "")),
            f"{r.get('Mean', 0):.2f}",
            f"{r.get('Median', 0):.2f}",
            f"{r.get('Std', 0):.2f}",
            f"{r.get('Min', 0):.2f}",
            f"{r.get('Max', 0):.2f}",
            f"{r.get('Skewness', 0):.2f}",
            f"{r.get('Kurtosis', 0):.2f}",
            f"{r.get('CV%', 0):.1f}",
        ])
    pdf.table(headers, rows, widths=[30, 12, 18, 18, 18, 18, 18, 16, 16, 16])
    pdf.ln(4)

    # ── Box Plots by Risk Level ──
    pdf.check_space(70)
    pdf.sub_title("Distributions by Risk Level")
    fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(12, 3), facecolor=BG)
    if len(numeric_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, numeric_cols[:3]):
        ax.set_facecolor(BG)
        data_by_risk = [df[df["Risk_Level"] == r][col].dropna() for r in ["HIGH", "MODERATE", "LOW"]]
        bp = ax.boxplot(data_by_risk, tick_labels=["HIGH", "MOD", "LOW"], patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], ["#ef4444", "#f59e0b", "#22c55e"]):
            patch.set_facecolor(c); patch.set_alpha(0.3)
        for item in bp["whiskers"] + bp["caps"]:
            item.set_color("#cbd5e1")
        for m in bp["medians"]:
            m.set_color("white")
        ax.set_title(col.replace("_", " ").title(), color="white", fontsize=9)
        ax.tick_params(colors="#f1f5f9", labelsize=7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#475569")
        ax.spines["left"].set_color("#475569")
    fig.tight_layout()
    pdf.embed_chart(fig, w=180)

    # ── Distribution Fitting ──
    pdf.check_space(50)
    pdf.title_section("Distribution Fitting (Tumor Volume)")
    try:
        fit_df = sa.fit_distributions(df["Tumor_Volume_cc"])
        fit_headers = ["Distribution", "AIC", "KS Stat", "KS p-value", "Best"]
        fit_rows = []
        for _, r in fit_df.iterrows():
            fit_rows.append([
                str(r.get("Distribution", "")),
                f"{r.get('AIC', 0):.1f}",
                f"{r.get('KS_statistic', 0):.4f}",
                f"{r.get('KS_pvalue', 0):.4f}",
                "Yes" if r.get("Best Model") else "",
            ])
        pdf.table(fit_headers, fit_rows, widths=[40, 30, 30, 30, 20])
    except Exception:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 5, "Distribution fitting not available.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Q-Q Plot ──
    from scipy.stats import probplot
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=BG)
    ax.set_facecolor(BG)
    clean_vol = df["Tumor_Volume_cc"].dropna()
    probplot(clean_vol, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor("#3b82f6")
    ax.get_lines()[0].set_markeredgecolor("#3b82f6")
    ax.get_lines()[0].set_markersize(3)
    ax.get_lines()[1].set_color("#ef4444")
    ax.set_title("Q-Q Plot vs Normal", color="white", fontsize=10)
    ax.tick_params(colors="#f1f5f9")
    for sp in ax.spines.values():
        sp.set_color("#475569")
    pdf.embed_chart(fig, w=100)

    # ── Correlation Matrix ──
    pdf.check_space(80)
    pdf.title_section("Spearman Correlation Matrix")
    r_mat, p_mat = sa.correlation_matrix(df, columns=numeric_cols)
    sig_labels = sa.correlation_significance_labels(p_mat)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG)
    ax.set_facecolor(BG)
    im = ax.imshow(r_mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(r_mat.columns)))
    ax.set_yticks(range(len(r_mat.columns)))
    lbls = [c.replace("_", " ").title() for c in r_mat.columns]
    ax.set_xticklabels(lbls, rotation=45, ha="right", color="#f1f5f9", fontsize=8)
    ax.set_yticklabels(lbls, color="#f1f5f9", fontsize=8)
    for i in range(len(r_mat)):
        for j in range(len(r_mat)):
            val = r_mat.iloc[i, j]
            sig = sig_labels.iloc[i, j]
            tc = "white" if abs(val) > 0.5 else "#1e293b" if abs(val) > 0.2 else "#475569"
            ax.text(j, i, f"{val:.2f}\n{sig}", ha="center", va="center",
                    fontsize=8, color=tc, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho", color="#f1f5f9", fontsize=9)
    cbar.ax.tick_params(colors="#f1f5f9")
    ax.set_title("Correlation (*** p<.001, ** p<.01, * p<.05, n.s.)", color="white", fontsize=10)
    pdf.embed_chart(fig, w=130)

    # ── Risk Factor Analysis ──
    pdf.check_space(60)
    pdf.title_section("Risk Factor Analysis (HIGH vs Others)")
    try:
        risk_assoc = sa.univariate_risk_association(df)
        rf_headers = ["Variable", "Mann-Whitney U", "p-value", "Cliff's d", "Effect", "AUC"]
        rf_rows = []
        for _, r in risk_assoc.iterrows():
            rf_rows.append([
                str(r.get("Variable", "")).replace("_", " "),
                f"{r.get('U_statistic', 0):.1f}",
                f"{r.get('p_value', 0):.4f}",
                f"{r.get('Cliffs_Delta', 0):.3f}",
                str(r.get("Effect_Size", "")),
                f"{r.get('AUC', 0):.3f}",
            ])
        pdf.table(rf_headers, rf_rows, widths=[35, 28, 22, 22, 22, 22])
        pdf.ln(3)

        # Forest plot
        fig, ax = plt.subplots(figsize=(8, 2.5), facecolor=BG)
        ax.set_facecolor(BG)
        deltas = risk_assoc["Cliffs_Delta"].values
        vnames = [v.replace("_", " ").title() for v in risk_assoc["Variable"]]
        colors_list = ["#ef4444" if abs(d) > 0.33 else "#f59e0b" if abs(d) > 0.147 else "#cbd5e1" for d in deltas]
        ax.barh(range(len(deltas)), deltas, color=colors_list, height=0.5)
        ax.axvline(0, color="white", linewidth=1)
        for lv in [-0.33, -0.147, 0.147, 0.33]:
            ax.axvline(lv, color="#475569", linestyle="--", linewidth=0.7)
        ax.set_yticks(range(len(deltas)))
        ax.set_yticklabels(vnames, color="#f1f5f9", fontsize=8)
        ax.set_xlabel("Cliff's delta (HIGH vs Non-HIGH)", color="#cbd5e1", fontsize=8)
        ax.set_title("Which variables predict HIGH risk?", color="white", fontsize=10)
        ax.tick_params(colors="#f1f5f9")
        for sp in ax.spines.values():
            sp.set_color("#475569")
        pdf.embed_chart(fig, w=170)
    except Exception:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Risk factor analysis not available.", new_x="LMARGIN", new_y="NEXT")

    # ── Group Comparisons (Kruskal-Wallis) ──
    pdf.check_space(50)
    pdf.title_section("Group Comparisons (Kruskal-Wallis)")
    for col in numeric_cols[:3]:
        result = sa.compare_groups_kruskal(df, col)
        if "error" in result:
            continue
        pdf.sub_title(col.replace("_", " ").title())
        pdf.kv("H statistic:", f"{result['H_statistic']:.3f}")
        pdf.kv("p-value:", f"{result['p_value']:.6f}")
        pdf.kv("eta-squared:", f"{result['eta_squared']:.4f}")
        pdf.kv("Verdict:", "Significant" if result["significant"] else "Not Significant")
        pdf.ln(2)

    # ── Bootstrap CIs ──
    pdf.check_space(40)
    pdf.title_section("Bootstrap 95% Confidence Intervals (10K resamples)")
    boot_headers = ["Variable", "Observed", "95% CI Lower", "95% CI Upper"]
    boot_rows = []
    for col in numeric_cols[:3]:
        ci = sa.bootstrap_confidence_interval(df[col].dropna().values, n_bootstrap=10000)
        boot_rows.append([
            col.replace("_", " ").title(),
            f"{ci['observed']:.3f}",
            f"{ci['ci_lower']:.3f}",
            f"{ci['ci_upper']:.3f}",
        ])
    pdf.table(boot_headers, boot_rows, widths=[50, 35, 35, 35])
    pdf.ln(3)

    # ── Outlier Detection ──
    pdf.check_space(40)
    pdf.title_section("Outlier Detection (IQR Method)")
    try:
        outlier_df = sa.outlier_summary(df, columns=numeric_cols)
        ol_headers = list(outlier_df.columns)
        ol_rows = []
        for _, r in outlier_df.iterrows():
            ol_rows.append([str(r.get(c, "")) for c in ol_headers])
        pdf.table(ol_headers, ol_rows, widths=[190 / len(ol_headers)] * len(ol_headers))
    except Exception:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "Outlier detection not available.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # ── Patient List ──
    pdf.add_page()
    pdf.title_section("Complete Patient List")
    sorted_df = df.sort_values(by="Risk_Level", key=lambda s: s.map({"HIGH": 0, "MODERATE": 1, "LOW": 2}))
    pl_headers = ["Patient ID", "Structure", "Volume (cc)", "Dist OAR (mm)", "Uncertainty", "Risk"]
    pl_rows = []
    for _, r in sorted_df.iterrows():
        pl_rows.append([
            r["Patient_ID"],
            str(r["Structure_Name"]),
            f"{r['Tumor_Volume_cc']:.1f}",
            f"{r['Distance_to_OAR_mm']:.1f}",
            f"{r['AI_Uncertainty_Score']:.3f}",
            compute_risk(r),
        ])
    pdf.table(pl_headers, pl_rows, widths=[38, 28, 25, 28, 25, 20])

    # ── Rule Engine Thresholds ──
    pdf.ln(6)
    pdf.title_section("Rule Engine Thresholds")
    pdf.table(
        ["Parameter", "HIGH Threshold", "MODERATE Threshold"],
        [
            ["Distance to OAR", "< 3.0 mm", "-"],
            ["AI Uncertainty", "> 0.75", "> 0.50"],
            ["Tumor Volume", "-", "> 25.0 cc"],
        ],
        widths=[60, 50, 50],
    )

    # ── Disclaimer ──
    pdf.ln(6)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 6)
    pdf.set_text_color(140, 140, 140)
    pdf.multi_cell(0, 3,
        "This report is generated by a decision-support system and does not replace clinical judgment. "
        "Research prototype only. Not for clinical use. "
        "Data: BraTS 2020 (Kaggle). Guidelines: AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG Oncology.")

    return bytes(pdf.output())
