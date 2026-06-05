import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import probplot

from knowledge_base import KNOWLEDGE_BASE, retrieve_guidelines, map_triggers
from mri_viewer import (
    render_combined_view, has_nifti_data, has_plane_data,
    render_plane_composite, create_mri_plotly,
)
from pdf_report import generate_patient_pdf, generate_group_pdf
import stats_analysis as sa

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NeuroQA Copilot – Radiotherapy QA Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Theme with proper contrast
# ═══════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Background + margins ── */
    .stApp, section[data-testid="stAppViewContainer"] {
        background: #0a0e1a !important;
    }
    .main .block-container {
        background: #0a0e1a !important;
        font-family: 'Inter', sans-serif;
        padding-top: 0rem !important;
        max-width: 72rem !important;
        padding-left: 15% !important;
        padding-right: 15% !important;
    }
    header[data-testid="stHeader"] { display: none; }

    /* ── Global text: bright but NOT targeting dropdown internals ── */
    p, li, td, th, em, strong, small, code, pre,
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
        color: #e2e8f0 !important;
    }
    h1, h2, h3, h4, h5, h6 { color: #f1f5f9 !important; }
    a { color: #60a5fa !important; }

    /* ── Metric cards ── */
    .stMetric {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; border-radius: 10px; padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 600; font-size: 1.2rem !important; }
    [data-testid="stMetricDelta"] * { color: #e2e8f0 !important; }

    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: #94a3b8 !important; }

    /* ── Selectbox / Dropdown — dark bg with bright text ── */
    div[data-baseweb="select"] { background-color: #1e293b !important; }
    div[data-baseweb="select"] > div { background-color: #1e293b !important; border-color: #475569 !important; }
    div[data-baseweb="select"] input { color: #f1f5f9 !important; -webkit-text-fill-color: #f1f5f9 !important; }
    div[data-baseweb="select"] svg { fill: #94a3b8 !important; }
    div[data-baseweb="select"] [data-baseweb="tag"] { background-color: #334155 !important; }
    div[data-baseweb="select"] span { color: #f1f5f9 !important; }
    /* Dropdown popover/menu — must target deeply for portal-rendered elements */
    div[data-baseweb="popover"] { background-color: #1e293b !important; }
    div[data-baseweb="popover"] > div { background-color: #1e293b !important; }
    div[data-baseweb="menu"] { background-color: #1e293b !important; border: 1px solid #475569 !important; }
    ul[role="listbox"] { background-color: #1e293b !important; }
    li[role="option"] { background-color: #1e293b !important; color: #f1f5f9 !important; }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #334155 !important; color: #ffffff !important; }
    div[data-baseweb="option"] { background-color: #1e293b !important; color: #f1f5f9 !important; }
    div[data-baseweb="option"]:hover { background-color: #334155 !important; }
    div[data-baseweb="option"] span { color: #f1f5f9 !important; }

    /* ── Slider ── */
    .stSlider label { color: #e2e8f0 !important; }

    /* ── Alerts ── */
    .stAlert, [data-testid="stAlert"] { background: #1e293b !important; border: 1px solid #334155 !important; }
    .stAlert *, [data-testid="stAlert"] * { color: #e2e8f0 !important; }
    .stSuccess, .stSuccess * { color: #86efac !important; }
    .stWarning, .stWarning * { color: #fcd34d !important; }
    .stError, .stError * { color: #fca5a5 !important; }
    .stInfo, .stInfo * { color: #93c5fd !important; }

    /* ── Expanders ── */
    details summary, details summary * { color: #e2e8f0 !important; }
    details[open] > div { color: #e2e8f0 !important; }

    /* ── Tables in markdown ── */
    .stMarkdown table { color: #e2e8f0 !important; }
    .stMarkdown th { color: #94a3b8 !important; background: #1e293b !important; }
    .stMarkdown td { color: #e2e8f0 !important; border-color: #334155 !important; }

    /* ── DataFrame — dark bg for all tables ── */
    div[data-testid="stDataFrame"] { border: 1px solid #334155; border-radius: 8px; }
    div[data-testid="stDataFrame"] table { background: #0f172a !important; }
    div[data-testid="stDataFrame"] th { background: #1e293b !important; color: #94a3b8 !important; }
    div[data-testid="stDataFrame"] td { background: #0f172a !important; color: #e2e8f0 !important; }
    div[data-testid="stDataFrame"] tr:hover td { background: #1e293b !important; }
    div[data-testid="stDataFrame"] [role="gridcell"] { background: #0f172a !important; color: #e2e8f0 !important; }
    div[data-testid="stDataFrame"] [role="columnheader"] { background: #1e293b !important; color: #94a3b8 !important; }
    /* Arrow/dataframe internals */
    .stDataFrame [data-testid="stDataFrameContainer"] { background: #0f172a !important; }
    .stDataFrame div[class*="cell"] { background: #0f172a !important; color: #e2e8f0 !important; }
    .stDataFrame div[class*="header"] { background: #1e293b !important; color: #94a3b8 !important; }

    /* ── Risk Badges ── */
    .risk-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; }
    .risk-high { background: #7f1d1d; color: #fca5a5 !important; border: 1px solid #991b1b; }
    .risk-moderate { background: #78350f; color: #fcd34d !important; border: 1px solid #92400e; }
    .risk-low { background: #14532d; color: #86efac !important; border: 1px solid #166534; }

    /* ── Hide sidebar ── */
    section[data-testid="stSidebar"] { display: none; }

    /* ── Download button ── */
    .stDownloadButton button { background-color: #1e293b !important; color: #e2e8f0 !important; border: 1px solid #334155 !important; }
    .stDownloadButton button:hover { background-color: #334155 !important; border-color: #3b82f6 !important; }

    /* ── Plotly ── */
    .stPlotlyChart { background: #0f172a; border-radius: 10px; border: 1px solid #334155; padding: 4px; }

    /* ── Nav buttons ── */
    .stButton > button { background-color: #1e293b; color: #e2e8f0 !important; border: 1px solid #334155; border-radius: 8px; }
    .stButton > button:hover { background-color: #334155; border-color: #3b82f6; }
    .stButton > button[data-testid="stBaseButton-primary"] {
        background-color: #3b82f6 !important; color: white !important; border-color: #3b82f6 !important;
    }
    .stButton > button[data-testid="stBaseButton-secondary"] {
        background-color: #1e293b !important; color: #94a3b8 !important; border: 1px solid #334155 !important;
    }

    /* ── Dividers ── */
    hr { border-color: #1e293b !important; }

    /* ── Code blocks ── */
    .stCodeBlock, .stCodeBlock * { color: #e2e8f0 !important; }

    /* ── Tooltip ── */
    .stTooltipIcon { color: #94a3b8 !important; }

    /* ── Widget labels (selectbox, radio, checkbox, etc.) ── */
    label { color: #e2e8f0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "real_clinical_queue.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    st.error("No BraTS data found. Run `extract_real_cases.py` first.")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════
DISTANCE_THRESHOLD_MM = 3.0
UNCERTAINTY_HIGH = 0.75
UNCERTAINTY_MODERATE = 0.50
TUMOR_VOLUME_HIGH = 25.0


def compute_risk_level(row: pd.Series) -> str:
    if row["AI_Uncertainty_Score"] > UNCERTAINTY_HIGH:
        return "HIGH"
    if row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM:
        return "HIGH"
    if (
        row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE
        or row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH
    ):
        return "MODERATE"
    return "LOW"


def get_triggers(row: pd.Series) -> list[str]:
    flags: list[str] = []
    dist = row["Distance_to_OAR_mm"]
    unc = row["AI_Uncertainty_Score"]
    vol = row["Tumor_Volume_cc"]
    struct = row["Structure_Name"]

    if dist < DISTANCE_THRESHOLD_MM:
        flags.append(
            f"Proximity Alert: Contour distance to {struct} ({dist:.1f}mm) "
            f"falls below the {DISTANCE_THRESHOLD_MM:.0f}mm planning margin threshold."
        )
    if unc > UNCERTAINTY_HIGH:
        flags.append(
            f"Uncertainty Alert: AI voxel uncertainty score ({unc:.3f}) "
            f"exceeds the {UNCERTAINTY_HIGH} institutional tolerance for critical structures."
        )
    elif unc > UNCERTAINTY_MODERATE:
        flags.append(
            f"Moderate Uncertainty: AI score ({unc:.3f}) exceeds moderate "
            f"threshold of {UNCERTAINTY_MODERATE}."
        )
    if vol > TUMOR_VOLUME_HIGH:
        flags.append(
            f"Volume Alert: Gross tumor volume ({vol:.1f} cc) exceeds "
            f"{TUMOR_VOLUME_HIGH:.0f} cc — multi-planar verification recommended."
        )
    if "optic" in struct.lower() or "chiasm" in struct.lower():
        flags.append(
            f"Sensitive Structure: '{struct}' is part of the optic apparatus — "
            f"high sensitivity to contour errors."
        )
    if "brainstem" in struct.lower():
        flags.append(
            f"Critical Organ: '{struct}' is a serial organ — verify contour boundaries."
        )
    if not flags:
        flags.append("No risk flags triggered — standard QA protocol applies.")
    return flags


# ═══════════════════════════════════════════════════════════════
# COPILOT REPORT (Simulated LLM RAG)
# ═══════════════════════════════════════════════════════════════
def generate_copilot_report(row: pd.Series) -> str:
    triggers = map_triggers(
        structure_name=row["Structure_Name"],
        distance_mm=row["Distance_to_OAR_mm"],
        uncertainty=row["AI_Uncertainty_Score"],
        tumor_volume=row["Tumor_Volume_cc"],
    )
    guidelines = retrieve_guidelines(triggers)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""## NeuroQA Copilot — Clinical Review Report

**Generated:** {now} | **Patient:** {row["Patient_ID"]} | **System:** NeuroQA Copilot v1.0

---

### 1. Anatomical Risk Summary

| Parameter | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Structure | {row["Structure_Name"]} | — | — |
| Tumor Volume | {row["Tumor_Volume_cc"]:.2f} cc | > {TUMOR_VOLUME_HIGH} cc | {"Elevated" if row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH else "Normal"} |
| Distance to OAR | {row["Distance_to_OAR_mm"]:.1f} mm | < {DISTANCE_THRESHOLD_MM} mm | {"Critical" if row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM else "Adequate"} |
| AI Uncertainty | {row["AI_Uncertainty_Score"]:.3f} | > {UNCERTAINTY_HIGH} | {"High" if row["AI_Uncertainty_Score"] > UNCERTAINTY_HIGH else "Moderate" if row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE else "Low"} |

"""
    risk = compute_risk_level(row)
    risk_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk]
    report += f"**Overall Risk Classification:** {risk_emoji} **{risk}**\n\n---\n\n"

    report += "### 2. Retrieved Clinical Guidelines (RAG)\n\n"
    if guidelines:
        for key, entry in guidelines.items():
            report += f"#### {key.replace('_', ' ').title()}\n\n"
            report += f"> **Source:** {entry['source']}  \n"
            report += f"> **Risk Category:** {entry['risk_category']}  \n\n"
            report += f"{entry['excerpt']}\n\n"
    else:
        report += (
            "> No specific clinical guideline triggered for this case. "
            "Standard institutional QA protocols apply.\n\n"
        )

    report += "---\n\n"
    report += "### 3. Recommended Human-in-the-Loop Action\n\n"

    actions: list[str] = []
    if row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM:
        struct = row["Structure_Name"]
        actions.append(
            f"**Priority 1:** Recommend manual slice-by-slice review of "
            f"the auto-segmented `{struct}` contour, focusing on the "
            f"boundary closest to the PTV "
            f"(distance: {row['Distance_to_OAR_mm']:.1f} mm)."
        )
    if row["AI_Uncertainty_Score"] > UNCERTAINTY_HIGH:
        actions.append(
            "**Priority 1:** Flag for mandatory physician verification. "
            "Do NOT use the auto-segmented contour in treatment planning "
            "until manually approved."
        )
    elif row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE:
        actions.append(
            "**Priority 2:** Spot-check the high-uncertainty regions. "
            "Consider MRI co-registration if diagnostic images are available."
        )
    if row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH:
        actions.append(
            "**Priority 3:** Perform multi-planar review "
            "(axial, coronal, sagittal) to verify contour integrity "
            "around the full tumor extent."
        )
    if not actions:
        actions.append(
            "No elevated-risk actions required. Standard physician sign-off is sufficient."
        )

    for i, action in enumerate(actions, 1):
        report += f"{i}. {action}\n\n"

    report += "---\n\n"
    report += (
        "*This report is generated by a decision-support system "
        "and does not replace clinical judgment. All flagged contours "
        "require human verification before treatment delivery.*\n"
    )
    return report


# ═══════════════════════════════════════════════════════════════
# PDF / HTML EXPORT
# ═══════════════════════════════════════════════════════════════
def generate_patient_pdf_html(row: pd.Series, df: pd.DataFrame) -> str:
    triggers = get_triggers(row)
    risk = compute_risk_level(row)
    risk_color = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}[risk]

    triggers_html = "".join(
        f'<div style="background:#1e293b;border-left:3px solid #ef4444;'
        f'border-radius:0 6px 6px 0;padding:8px 12px;margin:4px 0;font-size:0.82rem;'
        f'color:#e2e8f0;">{t}</div>'
        for t in triggers
    )

    vol_pct = (df["Tumor_Volume_cc"] < row["Tumor_Volume_cc"]).mean() * 100
    dist_pct = (df["Distance_to_OAR_mm"] < row["Distance_to_OAR_mm"]).mean() * 100
    unc_pct = (df["AI_Uncertainty_Score"] < row["AI_Uncertainty_Score"]).mean() * 100

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>NeuroQA Report — {row["Patient_ID"]}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0;
         padding: 30px; max-width: 800px; margin: 0 auto; }}
  h1 {{ color: #f1f5f9; font-size: 1.5rem; }}
  h2 {{ color: #f1f5f9; font-size: 1.1rem; margin-top: 24px; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 16px; margin: 12px 0; border: 1px solid #334155; }}
  .metric-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
  .metric {{ flex: 1; min-width: 100px; text-align: center; background: #0f172a;
             border-radius: 8px; padding: 10px; }}
  .metric-val {{ font-size: 1.2rem; font-weight: 700; color: #f1f5f9; }}
  .metric-lbl {{ font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #1e293b; color: #94a3b8; padding: 6px 8px; text-align: left; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #1e293b; color: #e2e8f0; }}
  .risk-badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px;
                 font-weight: 600; font-size: 0.75rem; }}
  .footer {{ margin-top: 30px; padding-top: 12px; border-top: 1px solid #1e293b;
             color: #475569; font-size: 0.7rem; text-align: center; }}
  @media print {{ body {{ background: #fff; color: #111; }} .card {{ background: #f8f8f8; border:1px solid #ddd; }} }}
</style></head><body>
<h1>NeuroQA Copilot — Patient Report</h1>
<div class="card">
  <span style="font-size:1.2rem;font-weight:700;color:#f1f5f9;">{row["Patient_ID"]}</span>
  <span style="color:#94a3b8;margin-left:12px;">{row["Structure_Name"]}</span>
  <span class="risk-badge" style="background:{risk_color}22;color:{risk_color};margin-left:12px;">{risk} RISK</span>
</div>
<div class="metric-row">
  <div class="metric"><div class="metric-val">{row["Tumor_Volume_cc"]:.1f} cc</div><div class="metric-lbl">Tumor Volume ({vol_pct:.0f}th %ile)</div></div>
  <div class="metric"><div class="metric-val">{row["Distance_to_OAR_mm"]:.1f} mm</div><div class="metric-lbl">Dist to OAR ({dist_pct:.0f}th %ile)</div></div>
  <div class="metric"><div class="metric-val">{row["AI_Uncertainty_Score"]:.3f}</div><div class="metric-lbl">AI Uncertainty ({unc_pct:.0f}th %ile)</div></div>
</div>
<h2>Risk Triggers</h2>
{triggers_html}
<div class="footer">NeuroQA Copilot v1.0 — Research Prototype — Not for clinical use</div>
</body></html>"""
    return html


# ═══════════════════════════════════════════════════════════════
# PATIENT COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════
def generate_patient_comparison(row: pd.Series, df: pd.DataFrame) -> str:
    risk = compute_risk_level(row)
    risk_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk]

    vol_pct = (df["Tumor_Volume_cc"] < row["Tumor_Volume_cc"]).mean() * 100
    dist_pct = (df["Distance_to_OAR_mm"] < row["Distance_to_OAR_mm"]).mean() * 100
    unc_pct = (df["AI_Uncertainty_Score"] < row["AI_Uncertainty_Score"]).mean() * 100

    grp_med_vol = df["Tumor_Volume_cc"].median()
    grp_med_dist = df["Distance_to_OAR_mm"].median()
    grp_med_unc = df["AI_Uncertainty_Score"].median()

    high_n = int((df["Risk_Level"] == "HIGH").sum())
    same_risk_n = int((df["Risk_Level"] == risk).sum())
    triggers = get_triggers(row)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    higher_lower_vol = "higher" if vol_pct > 50 else "lower"
    higher_lower_dist = "higher" if dist_pct > 50 else "lower"
    higher_lower_unc = "higher" if unc_pct > 50 else "lower"

    report = f"""## Patient Summary — {row["Patient_ID"]}

**Generated:** {now} | **Risk:** {risk_emoji} {risk}

---

### How This Patient Compares to the Group (n={len(df)})

| Metric | This Patient | Group Median | Percentile | Interpretation |
|--------|-------------|-------------|------------|----------------|
| Tumor Volume | {row["Tumor_Volume_cc"]:.1f} cc | {grp_med_vol:.1f} cc | {vol_pct:.0f}th | {higher_lower_vol} than {abs(vol_pct - 100) if vol_pct > 50 else vol_pct:.0f}% of patients |
| Distance to OAR | {row["Distance_to_OAR_mm"]:.1f} mm | {grp_med_dist:.1f} mm | {dist_pct:.0f}th | {higher_lower_dist} than {abs(dist_pct - 100) if dist_pct > 50 else dist_pct:.0f}% of patients |
| AI Uncertainty | {row["AI_Uncertainty_Score"]:.3f} | {grp_med_unc:.3f} | {unc_pct:.0f}th | {higher_lower_unc} than {abs(unc_pct - 100) if unc_pct > 50 else unc_pct:.0f}% of patients |

### Risk Triggers

"""
    for t in triggers:
        report += f"- {t}\n"

    report += f"""
### Clinical Context

- **{same_risk_n}** out of {len(df)} patients ({same_risk_n/len(df)*100:.0f}%) are also classified as **{risk} RISK**
- Structure affected: **{row["Structure_Name"]}**
- Data source: **{row.get('Data_Source', 'BraTS 2020')}**

### Recommended Actions

"""
    if row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM:
        report += (
            f"- **CRITICAL:** Manual slice-by-slice review of `{row['Structure_Name']}` "
            f"contour (distance: {row['Distance_to_OAR_mm']:.1f} mm)\n"
        )
    if row["AI_Uncertainty_Score"] > UNCERTAINTY_HIGH:
        report += (
            "- **CRITICAL:** Do NOT use auto-segmented contour until "
            "physician verification is complete\n"
        )
    elif row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE:
        report += (
            "- **WARNING:** Spot-check high-uncertainty regions before approval\n"
        )
    if row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH:
        report += (
            "- Multi-planar (axial/coronal/sagittal) contour integrity review\n"
        )
    if not any([row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM,
                row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE,
                row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH]):
        report += "- Standard physician sign-off is sufficient\n"

    report += (
        "\n---\n"
        "*Decision-support only. All flagged contours require human verification.*\n"
    )
    return report


# ═══════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════
if "review_patient_id" not in st.session_state:
    st.session_state["review_patient_id"] = None
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = 1  # Default to Patient Queue
if "chart_key" not in st.session_state:
    st.session_state["chart_key"] = 0


def navigate_to_review(patient_id: str):
    """Set patient and switch to Patient Review tab."""
    st.session_state["review_patient_id"] = patient_id
    st.session_state["active_tab"] = 2
    st.session_state["chart_key"] = st.session_state.get("chart_key", 0) + 1
    st.rerun()


def _extract_plotly_patient(event) -> str | None:
    """Safely extract patient ID from a Plotly on_select event."""
    if event is None:
        return None
    try:
        sel = getattr(event, "selection", None) or (event.get("selection") if isinstance(event, dict) else None) or {}
        pts = getattr(sel, "points", None) or (sel.get("points", []) if isinstance(sel, dict) else [])
        if not pts:
            return None
        p = pts[0]
        cd = getattr(p, "customdata", None) if not isinstance(p, dict) else p.get("customdata")
        if cd is not None:
            return str(cd[0] if isinstance(cd, (list, tuple)) else cd)
        x_val = getattr(p, "x", None) if not isinstance(p, dict) else p.get("x")
        if x_val is not None:
            return str(x_val)
    except Exception:
        pass
    return None


def _extract_df_patient(event, source_df: pd.DataFrame) -> str | None:
    """Safely extract patient ID from a dataframe on_select event."""
    if event is None:
        return None
    try:
        sel = getattr(event, "selection", None) or (event.get("selection") if isinstance(event, dict) else None) or {}
        rows = getattr(sel, "rows", None) or (sel.get("rows", []) if isinstance(sel, dict) else [])
        if rows:
            return str(source_df.iloc[rows[0]]["Patient_ID"])
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    "<div style='background:linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%); "
    "border:1px solid #334155; border-radius:14px; padding:24px 28px; margin-bottom:8px;'>"
    "<h1 style='margin:0 0 8px 0; font-size:2rem; color:#f1f5f9;'>"
    "\U0001f9e0 NeuroQA Copilot</h1>"
    "<p style='color:#94a3b8; font-size:1.15rem; margin:0; line-height:1.5;'>"
    "Human-in-the-Loop Radiotherapy QA Decision Support for Brain Tumor Auto-Segmentation</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
df = load_data()
if df.empty:
    st.stop()
df["Risk_Level"] = df.apply(compute_risk_level, axis=1)

if st.session_state["review_patient_id"] is None:
    st.session_state["review_patient_id"] = df["Patient_ID"].iloc[0]

high_count = int((df["Risk_Level"] == "HIGH").sum())
mod_count = int((df["Risk_Level"] == "MODERATE").sum())
low_count = int((df["Risk_Level"] == "LOW").sum())

numeric_cols = ["Tumor_Volume_cc", "Distance_to_OAR_mm", "AI_Uncertainty_Score"]
if "Irregularity_Score" in df.columns:
    numeric_cols.append("Irregularity_Score")
if "Max_Diameter_mm" in df.columns:
    numeric_cols.append("Max_Diameter_mm")


# ═══════════════════════════════════════════════════════════════
# NAVIGATION — Button tabs with reliable programmatic switching
# ═══════════════════════════════════════════════════════════════

TAB_LABELS = [
    "\U0001f4ca Group Statistics",
    "\U0001f4cb Patient Queue",
    "\U0001f50d Patient Review",
    "\U0001f4da Clinical Reference",
]

nav_cols = st.columns(len(TAB_LABELS))
for i, (col, label) in enumerate(zip(nav_cols, TAB_LABELS)):
    with col:
        if st.button(
            label,
            use_container_width=True,
            key=f"nav_tab_{i}",
            type="primary" if st.session_state.active_tab == i else "secondary",
        ):
            st.session_state.active_tab = i
            st.rerun()

st.markdown("---")

active = st.session_state.active_tab


# ═══════════════════════════════════════════════════════════════
# TAB 1 — GROUP STATISTICS
# ═══════════════════════════════════════════════════════════════

if active == 0:
    gk1, gk2 = st.columns([1, 1.5])
    with gk1:
        fig, ax = plt.subplots(figsize=(3.5, 3), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        sizes = [high_count, mod_count, low_count]
        wedges, _texts = ax.pie(
            sizes, labels=None, colors=["#ef4444", "#f59e0b", "#22c55e"],
            startangle=90, wedgeprops=dict(width=0.4, edgecolor="#0f172a"),
        )
        ax.text(0, 0, f"{len(df)}", ha="center", va="center",
                fontsize=22, fontweight="bold", color="white")
        ax.text(0, -0.2, "patients", ha="center", va="center",
                fontsize=8, color="#94a3b8")
        ax.legend(wedges, [f"HIGH ({high_count})", f"MOD ({mod_count})", f"LOW ({low_count})"],
                  loc="lower center", fontsize=7, facecolor="#1e293b",
                  edgecolor="#334155", labelcolor="white", ncol=3)
        st.pyplot(fig)
        plt.close(fig)

    with gk2:
        st.markdown(
            '<p style="font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-top:10px;">'
            'Group Statistics</p>'
            '<p style="color:#94a3b8; font-size:0.9rem;">'
            'Cohort-level analysis across all 50 patients from BraTS 2020. '
            'Click any bar or point in the charts below to jump to that patient\'s review.</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Thresholds: Distance < {DISTANCE_THRESHOLD_MM}mm -> HIGH | "
            f"Uncertainty > {UNCERTAINTY_HIGH} -> HIGH | Volume > {TUMOR_VOLUME_HIGH}cc -> MODERATE"
        )

        @st.cache_data
        def _get_group_report_bytes():
            try:
                return generate_group_pdf(df)
            except Exception as e:
                return generate_group_pdf(df)

        st.download_button(
            label="\U0001f4c4 Download Group Report (PDF)",
            data=_get_group_report_bytes(),
            file_name="NeuroQA_Group_Report.pdf",
            mime="application/pdf", use_container_width=True,
            key="dl_group_top",
        )

    st.markdown("---")

    # ── Cohort Charts (Plotly — Interactive with click events) ──
    st.markdown("## Cohort Overview Charts")
    st.markdown("*Click any bar or point to open that patient's review.*")

    risk_color_map = {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}

    ch1, ch2 = st.columns(2)

    # ── Tumor Volume Bar Chart (Plotly) ──
    with ch1:
        vol_df = df.sort_values("Tumor_Volume_cc", ascending=False).copy()
        vol_fig = go.Figure()
        for risk_level in ["HIGH", "MODERATE", "LOW"]:
            mask = vol_df["Risk_Level"] == risk_level
            vol_fig.add_trace(go.Bar(
                x=vol_df.loc[mask, "Patient_ID"],
                y=vol_df.loc[mask, "Tumor_Volume_cc"],
                name=risk_level,
                marker_color=risk_color_map[risk_level],
                customdata=vol_df.loc[mask, "Patient_ID"],
                hovertemplate="%{x}<br>Volume: %{y:.1f} cc<extra></extra>",
            ))
        vol_fig.update_layout(
            title=dict(text="Tumor Volume by Patient", font=dict(color="#f1f5f9", size=14)),
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=380,
            margin=dict(t=40, b=80, l=50, r=20),
            xaxis=dict(tickangle=-90, tickfont=dict(size=7, color="#94a3b8"),
                       gridcolor="#1e293b"),
            yaxis=dict(title="Volume (cc)", tickfont=dict(color="#94a3b8"),
                       gridcolor="#1e293b"),
            legend=dict(font=dict(color="#94a3b8", size=10)),
            barmode="group",
        )
        vol_event = st.plotly_chart(vol_fig, on_select="rerun", key=f"vol_chart_{st.session_state.chart_key}",
                                     use_container_width=True)

    # ── AI Uncertainty Bar Chart (Plotly) ──
    with ch2:
        unc_df = df.sort_values("AI_Uncertainty_Score", ascending=False).copy()
        unc_fig = go.Figure()
        for risk_level in ["HIGH", "MODERATE", "LOW"]:
            mask = unc_df["Risk_Level"] == risk_level
            unc_fig.add_trace(go.Bar(
                x=unc_df.loc[mask, "Patient_ID"],
                y=unc_df.loc[mask, "AI_Uncertainty_Score"],
                name=risk_level,
                marker_color=risk_color_map[risk_level],
                customdata=unc_df.loc[mask, "Patient_ID"],
                hovertemplate="%{x}<br>Uncertainty: %{y:.3f}<extra></extra>",
            ))
        unc_fig.update_layout(
            title=dict(text="AI Uncertainty Score", font=dict(color="#f1f5f9", size=14)),
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=380,
            margin=dict(t=40, b=80, l=50, r=20),
            xaxis=dict(tickangle=-90, tickfont=dict(size=7, color="#94a3b8"),
                       gridcolor="#1e293b"),
            yaxis=dict(title="Uncertainty", tickfont=dict(color="#94a3b8"),
                       gridcolor="#1e293b", range=[0, 1]),
            legend=dict(font=dict(color="#94a3b8", size=10)),
            barmode="group",
        )
        unc_event = st.plotly_chart(unc_fig, on_select="rerun", key=f"unc_chart_{st.session_state.chart_key}",
                                     use_container_width=True)

    # ── Scatter Plot (Plotly) ──
    st.markdown("**Tumor Volume vs AI Uncertainty by Risk Level**")
    scatter_fig = px.scatter(
        df, x="Tumor_Volume_cc", y="AI_Uncertainty_Score",
        color="Risk_Level",
        color_discrete_map=risk_color_map,
        hover_name="Patient_ID",
        hover_data={"Tumor_Volume_cc": ":.1f", "AI_Uncertainty_Score": ":.3f", "Risk_Level": True},
        custom_data=["Patient_ID"],
    )
    scatter_fig.update_layout(
        title=dict(text="Volume vs Uncertainty", font=dict(color="#f1f5f9", size=14)),
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        height=400,
        margin=dict(t=40, b=40, l=50, r=20),
        xaxis=dict(title="Tumor Volume (cc)", tickfont=dict(color="#94a3b8"),
                   gridcolor="#1e293b"),
        yaxis=dict(title="AI Uncertainty", tickfont=dict(color="#94a3b8"),
                   gridcolor="#1e293b"),
        legend=dict(font=dict(color="#94a3b8", size=10)),
    )
    scatter_event = st.plotly_chart(scatter_fig, on_select="rerun", key=f"scatter_chart_{st.session_state.chart_key}",
                                     use_container_width=True)

    # ── Handle Plotly Click Events ──
    clicked_patient = (
        _extract_plotly_patient(vol_event)
        or _extract_plotly_patient(unc_event)
        or _extract_plotly_patient(scatter_event)
    )

    if clicked_patient and clicked_patient in df["Patient_ID"].values:
        navigate_to_review(clicked_patient)

    st.markdown("---")

    # ── 1. Descriptive Statistics ───────────────────────────────
    st.markdown("## Descriptive Statistics")
    st.markdown(
        "**Why this matters:** Understanding the distribution of tumor volumes, "
        "distances, and uncertainty scores tells us what's 'normal' in our cohort. "
        "Skewed distributions (common in tumor data) mean we should use non-parametric "
        "tests and medians rather than means."
    )
    with st.expander("How to interpret each statistic", expanded=False):
        st.markdown("""
        | Statistic | What it tells us |
        |-----------|-----------------|
        | **Skewness** | > 0 = right-tailed (some very large tumors pull the mean up) |
        | **Kurtosis** | > 0 = heavy tails (more extreme outliers than a normal distribution) |
        | **CV (%)** | Coefficient of variation = SD/Mean x 100. > 100% = very high spread |
        | **SEM** | Standard Error of Mean — smaller = more precise estimate of the true mean |
        """)

    stats_df = sa.compute_descriptive_stats(df, columns=numeric_cols)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    st.markdown("#### By Risk Level")
    grouped_stats = sa.compute_descriptive_stats(df, columns=numeric_cols, group_by="Risk_Level")
    st.dataframe(grouped_stats, use_container_width=True, hide_index=True)

    for col in numeric_cols[:3]:
        fig, ax = plt.subplots(figsize=(7, 2.2), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        data_by_risk = [df[df["Risk_Level"] == r][col].dropna() for r in ["HIGH", "MODERATE", "LOW"]]
        bp = ax.boxplot(data_by_risk, tick_labels=["HIGH", "MODERATE", "LOW"],
                        patch_artist=True, widths=0.5)
        for patch, c in zip(bp["boxes"], ["#ef4444", "#f59e0b", "#22c55e"]):
            patch.set_facecolor(c); patch.set_alpha(0.3)
        for item in bp["whiskers"] + bp["caps"]:
            item.set_color("#cbd5e1")
        for m in bp["medians"]:
            m.set_color("white")
        ax.set_title(col.replace("_", " ").title(), color="white", fontsize=11)
        ax.tick_params(colors="#f1f5f9", labelsize=9)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#475569"); ax.spines["left"].set_color("#475569")
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── 2. Distribution Fitting ─────────────────────────────────
    st.markdown("## Distribution Fitting")
    st.markdown(
        "**Why this matters:** In oncology, tumor volumes are classically **log-normal** — "
        "most tumors are small-to-medium, with a few very large ones. Confirming this "
        "determines which statistical tests are appropriate."
    )
    fit_df = sa.fit_distributions(df["Tumor_Volume_cc"])
    st.dataframe(fit_df, use_container_width=True, hide_index=True)
    best_row = fit_df[fit_df["Best Model"] == True]
    if len(best_row) > 0:
        st.success(f"Best fitting distribution: **{best_row.iloc[0]['Distribution']}**")

    qc1, qc2 = st.columns(2)
    with qc1:
        clean_vol = df["Tumor_Volume_cc"].dropna()
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        probplot(clean_vol, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("#3b82f6")
        ax.get_lines()[0].set_markeredgecolor("#3b82f6"); ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color("#ef4444")
        ax.set_title("Q-Q Plot vs Normal", color="white", fontsize=11)
        ax.tick_params(colors="#f1f5f9")
        for sp in ax.spines.values(): sp.set_color("#475569")
        st.pyplot(fig); plt.close(fig)
    with qc2:
        n_result = sa.test_normality(df["Tumor_Volume_cc"])
        ln_result = sa.test_lognormality(df["Tumor_Volume_cc"])
        st.metric("Shapiro-Wilk (raw)", f"p = {n_result['p_value']:.4f}")
        st.caption(n_result["interpretation"])
        st.metric("Shapiro-Wilk (log)", f"p = {ln_result['p_value']:.4f}")
        st.caption(ln_result["interpretation"])

    st.markdown("---")

    # ── 3. Correlation Matrix ───────────────────────────────────
    st.markdown("## Correlations")
    st.markdown(
        "**Why this matters:** Correlations reveal relationships between variables. "
        "For example, do larger tumors have higher AI uncertainty? Spearman (rank) "
        "correlation is used because clinical data rarely follows a normal distribution."
    )
    r_mat, p_mat = sa.correlation_matrix(df, columns=numeric_cols)
    sig_labels = sa.correlation_significance_labels(p_mat)

    fig, ax = plt.subplots(figsize=(7, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    im = ax.imshow(r_mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(r_mat.columns))); ax.set_yticks(range(len(r_mat.columns)))
    lbls = [c.replace("_", " ").title() for c in r_mat.columns]
    ax.set_xticklabels(lbls, rotation=45, ha="right", color="#f1f5f9", fontsize=10)
    ax.set_yticklabels(lbls, color="#f1f5f9", fontsize=10)
    for i in range(len(r_mat)):
        for j in range(len(r_mat)):
            val = r_mat.iloc[i, j]; sig = sig_labels.iloc[i, j]
            if abs(val) > 0.5:
                tc = "white"
            elif abs(val) > 0.2:
                tc = "#1e293b"
            else:
                tc = "#475569"
            ax.text(j, i, f"{val:.2f}\n{sig}", ha="center", va="center",
                    fontsize=10, color=tc, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Spearman rho", color="#f1f5f9", fontsize=10); cbar.ax.tick_params(colors="#f1f5f9")
    ax.set_title("Spearman Correlation (*** p<.001, ** p<.01, * p<.05, n.s.=not significant)", color="white", fontsize=11, pad=12)
    st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── 4. Risk Factor Analysis ─────────────────────────────────
    st.markdown("## Risk Factor Analysis")
    st.markdown(
        "**Why this matters:** Identifies which variables best discriminate "
        "HIGH-risk patients from others. Uses **Mann-Whitney U** (non-parametric), "
        "**Cliff's delta** (robust effect size), and **AUC** (discriminative power)."
    )
    risk_assoc = sa.univariate_risk_association(df)
    st.dataframe(risk_assoc.rename(columns={
        "Variable": "Variable", "Cliffs_Delta": "Cliff's delta",
        "Effect_Size": "Effect", "AUC": "AUC",
    }), use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 2.8), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    deltas = risk_assoc["Cliffs_Delta"].values
    vnames = [v.replace("_", " ").title() for v in risk_assoc["Variable"]]
    colors = ["#ef4444" if abs(d) > 0.33 else "#f59e0b" if abs(d) > 0.147 else "#cbd5e1" for d in deltas]
    ax.barh(range(len(deltas)), deltas, color=colors, height=0.5)
    ax.axvline(0, color="white", linewidth=1)
    for lv in [-0.33, -0.147, 0.147, 0.33]:
        ax.axvline(lv, color="#475569", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(deltas)))
    ax.set_yticklabels(vnames, color="#f1f5f9", fontsize=10)
    ax.set_xlabel("Cliff's delta (HIGH vs Non-HIGH)", color="#cbd5e1")
    ax.set_title("Which variables best predict HIGH risk?", color="white", fontsize=11)
    ax.tick_params(colors="#f1f5f9")
    for sp in ax.spines.values(): sp.set_color("#475569")
    st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # ── 5. Group Comparisons ────────────────────────────────────
    st.markdown("## Group Comparisons")
    st.markdown(
        "**Why this matters:** Kruskal-Wallis tests whether HIGH, MODERATE, and LOW "
        "risk groups truly differ on each metric. **eta-squared** tells us the proportion "
        "of variance explained by risk level."
    )
    for col in numeric_cols[:3]:
        result = sa.compare_groups_kruskal(df, col)
        if "error" in result: continue
        sig_icon = "Significant" if result["significant"] else "Not Significant"
        st.markdown(f"**{col.replace('_', ' ').title()}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("H", f"{result['H_statistic']:.3f}")
        c2.metric("p", f"{result['p_value']:.6f}")
        c3.metric("eta-sq", f"{result['eta_squared']:.4f}")
        c4.metric("Verdict", sig_icon)
        with st.expander("Pairwise post-hoc (Mann-Whitney U + Cliff's delta)"):
            st.dataframe(pd.DataFrame(result["pairwise_comparisons"]), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 6. Bootstrap CIs ────────────────────────────────────────
    st.markdown("## Bootstrap Confidence Intervals")
    st.markdown(
        "**Why this matters:** Bootstrap resampling (10,000 iterations) estimates "
        "how precise our measurements are without assuming any distribution."
    )
    boot_col = st.selectbox("Variable", numeric_cols, key="group_boot_col")
    ci_result = sa.bootstrap_confidence_interval(df[boot_col].dropna().values, n_bootstrap=10_000)
    bc1, bc2, bc3 = st.columns(3)
    bc1.metric("Observed Mean", f"{ci_result['observed']:.3f}")
    bc2.metric("95% CI Lower", f"{ci_result['ci_lower']:.3f}")
    bc3.metric("95% CI Upper", f"{ci_result['ci_upper']:.3f}")
    st.caption(ci_result["interpretation"])

    st.markdown("---")

    # ── 7. Outliers & Power ─────────────────────────────────────
    st.markdown("## Outliers & Statistical Power")
    st.markdown(
        "**Why this matters:** Outliers can be clinical anomalies or data errors. "
        "Power analysis tells us what effect sizes we can reliably detect with 50 patients."
    )
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**Outlier Detection (IQR)**")
        outlier_summary = sa.outlier_summary(df, columns=numeric_cols)
        st.dataframe(outlier_summary, use_container_width=True, hide_index=True)
    with oc2:
        st.markdown("**Minimum Detectable Effect (80% power)**")
        for col in numeric_cols[:3]:
            mde = sa.minimum_detectable_effect(df[col])
            if "error" not in mde:
                st.metric(col.replace("_", " ").title(), f"d >= {mde['min_detectable_d']:.3f}")
        st.info("With n=50, we can detect medium+ effects (d >= ~0.4). Smaller effects may be missed.")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PATIENT QUEUE
# ═══════════════════════════════════════════════════════════════

elif active == 1:
    # Pie chart + queue header
    pq1, pq2 = st.columns([1, 2.5])
    with pq1:
        fig, ax = plt.subplots(figsize=(3, 2.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        sizes = [high_count, mod_count, low_count]
        wedges, _ = ax.pie(
            sizes, labels=None, colors=["#ef4444", "#f59e0b", "#22c55e"],
            startangle=90, wedgeprops=dict(width=0.4, edgecolor="#0f172a"),
        )
        ax.text(0, 0, f"{len(df)}", ha="center", va="center",
                fontsize=18, fontweight="bold", color="white")
        ax.text(0, -0.17, "patients", ha="center", va="center",
                fontsize=7, color="#94a3b8")
        ax.legend(wedges, [f"HIGH ({high_count})", f"MOD ({mod_count})", f"LOW ({low_count})"],
                  loc="lower center", fontsize=6, facecolor="#1e293b",
                  edgecolor="#334155", labelcolor="white", ncol=3)
        st.pyplot(fig)
        plt.close(fig)

    with pq2:
        st.markdown(
            '<p style="font-size:1.4rem; font-weight:700; color:#f1f5f9;">Patient Triage Queue</p>'
            '<p style="color:#94a3b8; font-size:0.85rem; margin-bottom:8px;">'
            'Patients are sorted by risk level (HIGH first). '
            'Select a patient below to open their full review.</p>',
            unsafe_allow_html=True,
        )

    sorted_df = df.sort_values(
        by="Risk_Level",
        key=lambda s: s.map({"HIGH": 0, "MODERATE": 1, "LOW": 2}),
    ).reset_index(drop=True)

    # Patient selector (primary mechanism — like HTML report's selectPatient)
    sel1, sel2, sel3 = st.columns([5, 2, 1])
    with sel1:
        queue_patient = st.selectbox(
            "Select patient",
            sorted_df["Patient_ID"].tolist(),
            key="queue_select",
            label_visibility="collapsed",
            placeholder="Choose a patient to review...",
        )
    with sel2:
        if queue_patient:
            prow = sorted_df[sorted_df["Patient_ID"] == queue_patient].iloc[0]
            prisk = compute_risk_level(prow)
            risk_emoji = {"HIGH": "\U0001f534", "MODERATE": "\U0001f7e1", "LOW": "\U0001f7e2"}[prisk]
            risk_class = {"HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}[prisk]
            st.markdown(
                f'<span class="risk-badge {risk_class}" style="font-size:0.9rem; '
                f'padding:6px 16px; display:inline-block; margin-top:4px;">'
                f'{risk_emoji} {prisk} RISK</span>',
                unsafe_allow_html=True,
            )
    with sel3:
        if st.button("Review \u2192", key="queue_go", use_container_width=True, type="primary"):
            navigate_to_review(queue_patient)

    st.markdown("")

    # Also allow clicking rows in the dataframe below
    display_cols = ["Patient_ID", "Structure_Name", "Tumor_Volume_cc",
                    "Distance_to_OAR_mm", "AI_Uncertainty_Score", "Risk_Level"]

    queue_event = st.dataframe(
        sorted_df[display_cols],
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        height=480,
        key="queue_df",
        column_config={
            "Patient_ID": st.column_config.TextColumn("Patient ID", width="medium"),
            "Structure_Name": st.column_config.TextColumn("Structure", width="medium"),
            "Tumor_Volume_cc": st.column_config.NumberColumn("Volume (cc)", format="%.1f"),
            "Distance_to_OAR_mm": st.column_config.NumberColumn("Dist OAR (mm)", format="%.1f"),
            "AI_Uncertainty_Score": st.column_config.NumberColumn("Uncertainty", format="%.3f"),
            "Risk_Level": st.column_config.TextColumn("Risk", width="small"),
        },
    )

    queue_pid = _extract_df_patient(queue_event, sorted_df)
    if queue_pid:
        navigate_to_review(queue_pid)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — PATIENT REVIEW
# ═══════════════════════════════════════════════════════════════

elif active == 2:
    patient_ids = df["Patient_ID"].tolist()
    default_patient = st.session_state.get("review_patient_id", patient_ids[0])
    default_idx = patient_ids.index(default_patient) if default_patient in patient_ids else 0

    pc1, pc2, pc3, pc4 = st.columns([2, 5, 1.5, 1.5])
    with pc1:
        st.markdown(
            '<p style="font-size:1.3rem; font-weight:700; color:#f1f5f9; '
            'margin:6px 0 0 0; white-space:nowrap;">Patient ID:</p>',
            unsafe_allow_html=True,
        )
    with pc2:
        selected_id = st.selectbox(
            "Patient", patient_ids, index=default_idx,
            label_visibility="collapsed", key="patient_selector",
        )

    selected_row = df[df["Patient_ID"] == selected_id].iloc[0]
    risk = compute_risk_level(selected_row)
    risk_emoji = {"HIGH": "\U0001f534", "MODERATE": "\U0001f7e1", "LOW": "\U0001f7e2"}[risk]
    risk_class = {"HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}[risk]

    with pc3:
        st.markdown(
            f'<span class="risk-badge {risk_class}" style="font-size:1.1rem; '
            f'padding:8px 20px; display:inline-block; margin-top:8px;">'
            f'{risk_emoji} {risk} RISK</span>',
            unsafe_allow_html=True,
        )

    with pc4:
        patient_pdf = generate_patient_pdf(selected_row, df)
        st.download_button(
            label="\U0001f4c4 Download Report (PDF)",
            data=patient_pdf,
            file_name=f"NeuroQA_{selected_row['Patient_ID']}.pdf",
            mime="application/pdf", use_container_width=True,
            key="dl_patient_top",
        )

    st.markdown("")

    # MRI Viewer — PNG-based with opacity slider + Plotly zoom
    any_plane = any(has_plane_data(selected_id, p) for p in ["axial", "coronal", "sagittal"])

    if any_plane:
        opacity = st.slider(
            "Segmentation Overlay  (0 = MRI only, 1 = full color fill)",
            min_value=0.0, max_value=1.0, value=0.55, step=0.05, key="opacity",
        )
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.8rem;">'
            'Scroll to zoom into each image. Drag to pan.</p>',
            unsafe_allow_html=True,
        )

        plane_cols = st.columns(3)
        for col, plane in zip(plane_cols, ["axial", "coronal", "sagittal"]):
            with col:
                if has_plane_data(selected_id, plane):
                    composite = render_plane_composite(selected_id, plane, opacity)
                    if composite is not None:
                        fig = create_mri_plotly(composite, title=plane.title(), height=380)
                        st.plotly_chart(
                            fig, use_container_width=True,
                            config={"scrollZoom": True, "displayModeBar": False},
                            key=f"mri_{plane}_{selected_id}",
                        )
                else:
                    st.info(f"No {plane} data")

    elif has_nifti_data(selected_id):
        opacity = st.slider(
            "Segmentation Overlay  (0 = MRI only, 1 = full color fill)",
            min_value=0.0, max_value=1.0, value=0.55, step=0.05, key="opacity",
        )
        combined_buf = render_combined_view(selected_id, overlay_opacity=opacity)
        if combined_buf:
            st.image(combined_buf, use_container_width=True)

    else:
        st.info(
            "**MRI scans appear here** when NIfTI files "
            "are in `data/` or `brats_real/`. "
            "[Download from Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)"
        )

    st.markdown("")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Tumor Volume", f"{selected_row['Tumor_Volume_cc']:.1f} cc")
    mc2.metric("Dist. to OAR", f"{selected_row['Distance_to_OAR_mm']:.1f} mm")
    mc3.metric("AI Uncertainty", f"{selected_row['AI_Uncertainty_Score']:.3f}")
    mc4.metric("Structure", str(selected_row["Structure_Name"]))

    st.markdown("---")
    st.markdown("#### How This Patient Compares to the Group")
    comparison_report = generate_patient_comparison(selected_row, df)
    st.markdown(comparison_report)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — CLINICAL REFERENCE
# ═══════════════════════════════════════════════════════════════

elif active == 3:
    st.markdown(
        '<p style="font-size:1.4rem; font-weight:700; color:#f1f5f9;">'
        'Clinical Knowledge Base</p>'
        '<p style="color:#94a3b8; font-size:0.9rem;">'
        f'{len(KNOWLEDGE_BASE)} indexed guideline entries from AAPM TG-132, '
        'ESTRO, QUANTEC, ASTRO, and NRG Oncology.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    for key, entry in KNOWLEDGE_BASE.items():
        badge_cls = {"Critical": "risk-high", "High": "risk-high",
                     "Moderate": "risk-moderate", "Low": "risk-low"}.get(
            entry["risk_category"], "risk-low")
        with st.expander(
            f"{key.replace('_', ' ').title()} — "
            f"[{entry['risk_category']}]",
        ):
            st.markdown(f"**Source:** {entry['source']}")
            st.markdown(entry["excerpt"])
            st.markdown(f"**Action:** {entry['recommended_action']}")
    st.markdown("---")
    st.markdown("#### Rule Engine Thresholds")
    st.markdown(
        f"""
        | Parameter | HIGH Threshold | MODERATE Threshold |
        |-----------|---------------|-------------------|
        | Distance to OAR | < {DISTANCE_THRESHOLD_MM}.0 mm | — |
        | AI Uncertainty | > {UNCERTAINTY_HIGH} | > {UNCERTAINTY_MODERATE} |
        | Tumor Volume | — | > {TUMOR_VOLUME_HIGH}.0 cc |
        """
    )

# ── Footer ──────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "NeuroQA Copilot v1.0 — Research Prototype — Not for clinical use. "
    "Data: BraTS 2020 (Kaggle). Guidelines: AAPM TG-132, ESTRO, QUANTEC, ASTRO, NRG Oncology."
)
