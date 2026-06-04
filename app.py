import streamlit as st
import pandas as pd
import datetime
import os
import numpy as np

from knowledge_base import KNOWLEDGE_BASE, retrieve_guidelines, map_triggers
from mri_viewer import render_mri_with_overlay, render_all_planes, has_nifti_data

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroQA Copilot – Radiotherapy QA Decision Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        max-width: 1200px;
    }

    .stMetric {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    .stMetric label { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 600; }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 8px;
        overflow: hidden;
    }

    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-high { background: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }
    .risk-moderate { background: #78350f; color: #fcd34d; border: 1px solid #92400e; }
    .risk-low { background: #14532d; color: #86efac; border: 1px solid #166534; }

    .trigger-card {
        background: #1e293b;
        border-left: 3px solid #ef4444;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .trigger-card.moderate { border-left-color: #f59e0b; }
    .trigger-card.low { border-left-color: #22c55e; }

    .report-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 16px;
    }

    .copilot-section {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        margin: 12px 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1a1a2e 100%);
        border-right: 1px solid #1e293b;
    }

    h1, h2, h3 { color: #f1f5f9 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data Loading ────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "real_clinical_queue.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    st.error(
        "No BraTS data found. Run `extract_real_cases.py` first."
    )
    return pd.DataFrame()


# ── Rule Engine ─────────────────────────────────────────────────────
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
            f"🚩 Distance to OAR ({dist:.1f} mm) is below the "
            f"{DISTANCE_THRESHOLD_MM:.0f} mm safety threshold"
        )
    if unc > UNCERTAINTY_HIGH:
        flags.append(
            f"🚩 AI Uncertainty Score ({unc:.3f}) exceeds critical "
            f"threshold of {UNCERTAINTY_HIGH}"
        )
    elif unc > UNCERTAINTY_MODERATE:
        flags.append(
            f"⚠️ AI Uncertainty Score ({unc:.3f}) exceeds moderate "
            f"threshold of {UNCERTAINTY_MODERATE}"
        )
    if vol > TUMOR_VOLUME_HIGH:
        flags.append(
            f"⚠️ Tumor volume ({vol:.1f} cc) exceeds "
            f"{TUMOR_VOLUME_HIGH:.0f} cc — increased segmentation risk"
        )
    if "optic" in struct.lower() or "chiasm" in struct.lower():
        flags.append(
            f"⚠️ Structure '{struct}' is part of the optic apparatus "
            f"— high sensitivity to contour errors"
        )
    if "brainstem" in struct.lower():
        flags.append(
            f"⚠️ Structure '{struct}' is a critical serial organ "
            f"— verify contour boundaries"
        )
    if not flags:
        flags.append(
            "✅ No risk flags triggered — standard QA protocol applies"
        )
    return flags


# ── Simulated LLM RAG Report ───────────────────────────────────────
def generate_copilot_report(row: pd.Series) -> str:
    triggers = map_triggers(
        structure_name=row["Structure_Name"],
        distance_mm=row["Distance_to_OAR_mm"],
        uncertainty=row["AI_Uncertainty_Score"],
        tumor_volume=row["Tumor_Volume_cc"],
    )
    guidelines = retrieve_guidelines(triggers)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    report = f"""## 🧠 NeuroQA Copilot — Clinical Review Report

**Generated:** {now} &nbsp;|&nbsp; **Patient:** {row["Patient_ID"]} &nbsp;|&nbsp; **System:** NeuroQA Copilot v1.0

---

### 1. Anatomical Risk Summary

| Parameter | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Structure | {row["Structure_Name"]} | — | — |
| Tumor Volume | {row["Tumor_Volume_cc"]:.2f} cc | > {TUMOR_VOLUME_HIGH} cc | {"⚠️ Elevated" if row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH else "✅ Normal"} |
| Distance to OAR | {row["Distance_to_OAR_mm"]:.1f} mm | < {DISTANCE_THRESHOLD_MM} mm | {"🚩 Critical" if row["Distance_to_OAR_mm"] < DISTANCE_THRESHOLD_MM else "✅ Adequate"} |
| AI Uncertainty | {row["AI_Uncertainty_Score"]:.3f} | > {UNCERTAINTY_HIGH} | {"🚩 High" if row["AI_Uncertainty_Score"] > UNCERTAINTY_HIGH else "⚠️ Moderate" if row["AI_Uncertainty_Score"] > UNCERTAINTY_MODERATE else "✅ Low"} |

"""
    risk = compute_risk_level(row)
    risk_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk]
    report += (
        f"**Overall Risk Classification:** {risk_emoji} **{risk}**\n\n---\n\n"
    )

    report += "### 2. Retrieved Clinical Guidelines (RAG)\n\n"
    if guidelines:
        for key, entry in guidelines.items():
            report += f"#### 📖 {key.replace('_', ' ').title()}\n\n"
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
            "Consider MRI co-registration if diagnostic images are "
            "available."
        )
    if row["Tumor_Volume_cc"] > TUMOR_VOLUME_HIGH:
        actions.append(
            "**Priority 3:** Perform multi-planar review "
            "(axial, coronal, sagittal) to verify contour integrity "
            "around the full tumor extent. Check for over-segmentation "
            "into edema and under-segmentation at necrotic regions."
        )
    if not actions:
        actions.append(
            "No elevated-risk actions required. Standard physician "
            "sign-off is sufficient."
        )

    for i, action in enumerate(actions, 1):
        report += f"{i}. {action}\n\n"

    report += "---\n\n"
    report += (
        "*⚠️ This report is generated by a decision-support system "
        "and does not replace clinical judgment. All flagged contours "
        "require human verification before treatment delivery.*\n"
    )
    return report


# ── PDF Export ──────────────────────────────────────────────────────
def build_pdf_content(row: pd.Series, report_md: str) -> bytes:
    risk = compute_risk_level(row)
    triggers = get_triggers(row)
    lines = [
        "=" * 70,
        "NEUROQA COPILOT - CLINICAL REVIEW REPORT",
        "=" * 70,
        "",
        f"Report Date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Patient ID:        {row['Patient_ID']}",
        f"Structure:         {row['Structure_Name']}",
        f"Risk Level:        {risk}",
        "",
        "-" * 70,
        "RISK PARAMETERS",
        "-" * 70,
        f"  Tumor Volume:        {row['Tumor_Volume_cc']:.2f} cc",
        f"  Distance to OAR:     {row['Distance_to_OAR_mm']:.1f} mm",
        f"  AI Uncertainty:      {row['AI_Uncertainty_Score']:.3f}",
        "",
        "-" * 70,
        "TRIGGER FLAGS",
        "-" * 70,
    ]
    for t in triggers:
        lines.append(f"  {t}")
    lines += [
        "",
        "-" * 70,
        "COPILOT REVIEW",
        "-" * 70,
    ]
    plain_report = report_md.replace("## ", "").replace("### ", "")
    plain_report = plain_report.replace("#### ", "")
    plain_report = plain_report.replace("**", "").replace("*", "")
    plain_report = plain_report.replace("> ", "  ")
    for line in plain_report.split("\n"):
        lines.append(f"  {line}")
    lines += [
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
        "",
        "This report is generated by a decision-support system and does not",
        "replace clinical judgment. All flagged contours require human",
        "verification before treatment delivery.",
    ]
    return "\n".join(lines).encode("utf-8")


# ── Chart Helpers ───────────────────────────────────────────────────
def _risk_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Risk_Level"].value_counts().reindex(
        ["HIGH", "MODERATE", "LOW"], fill_value=0
    )
    return pd.DataFrame({
        "Risk Level": counts.index,
        "Count": counts.values,
    })


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NeuroQA Copilot")
    st.markdown(
        '<span style="color:#64748b; font-size:0.8rem;">'
        "Human-in-the-Loop Radiotherapy QA Decision Support</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### ⚙️ Rule Engine Thresholds")
    st.markdown(
        f"""
        | Parameter | Threshold |
        |-----------|-----------|
        | Distance to OAR | < {DISTANCE_THRESHOLD_MM} mm → HIGH |
        | AI Uncertainty | > {UNCERTAINTY_HIGH} → HIGH |
        | AI Uncertainty | > {UNCERTAINTY_MODERATE} → MODERATE |
        | Tumor Volume | > {TUMOR_VOLUME_HIGH} cc → MODERATE |
        """
    )
    st.divider()
    st.markdown("#### 📊 Knowledge Base")
    st.markdown(f"**{len(KNOWLEDGE_BASE)}** clinical guideline entries loaded")
    st.markdown(
        "Sources: AAPM TG-132, ESTRO Auto-Contouring Guidelines, "
        "QUANTEC, ASTRO AI Best Practices, NRG Oncology"
    )
    st.divider()

    st.markdown("#### 📤 Share Report")
    st.markdown(
        '<span style="color:#64748b; font-size:0.75rem;">'
        "Download a self-contained HTML report with all charts "
        "and MRI images. Opens in any browser — no server needed. "
        "Attach to email to share.</span>",
        unsafe_allow_html=True,
    )
    if st.button("📄 Generate HTML Report", use_container_width=True):
        with st.spinner("Generating report with MRI images..."):
            from generate_report import generate_html_report
            report_path = generate_html_report(
                include_mri=True, max_mri_patients=10
            )
            with open(report_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Report (HTML)",
                    data=f.read(),
                    file_name="NeuroQA_Report.html",
                    mime="text/html",
                    use_container_width=True,
                )

    st.divider()
    st.markdown(
        '<span style="color:#475569; font-size:0.7rem;">'
        "v1.0 — Research Prototype<br>Not for clinical use</span>",
        unsafe_allow_html=True,
    )


# ── Main Layout ─────────────────────────────────────────────────────
st.markdown(
    "# 🧠 NeuroQA Copilot\n"
    "<span style='color:#64748b; font-size:1rem;'>"
    "Human-in-the-Loop Radiotherapy QA Decision Support System "
    "for Brain Tumor Auto-Segmentation</span>",
    unsafe_allow_html=True,
)
st.markdown("")

# ── Intro Explanation ───────────────────────────────────────────────
with st.expander("ℹ️ What does this app do? (click to expand)", expanded=True):
    st.markdown(
        """
        **Problem:** When treating brain tumors with radiation, doctors use AI
        to automatically outline the tumor on MRI scans. But AI isn't perfect —
        mistakes near sensitive areas (optic nerve, brainstem) can cause serious
        side effects.

        **Solution:** This dashboard acts as a **safety review system** for
        **50 real brain tumor patients** from the BraTS 2020 challenge dataset
        (Kaggle). For each patient it checks three things:

        | Check | What it means |
        |-------|---------------|
        | **Tumor Size** | Bigger tumors are harder for AI to outline correctly |
        | **Distance to critical structures** | Closer = more dangerous if the AI makes a mistake |
        | **AI Confidence** | Low confidence = the AI is unsure, needs human double-check |

        Based on these checks, each patient gets a **risk level**:

        - 🔴 **HIGH** — Mandatory human review before treatment
        - 🟡 **MODERATE** — Spot-check recommended
        - 🟢 **LOW** — Standard sign-off is enough

        **Use the tabs below** to browse the patient queue, review individual
        cases with real MRI images, and view data charts.
        """
    )

st.markdown("")

df = load_data()
df["Risk_Level"] = df.apply(compute_risk_level, axis=1)

# ── KPI Metrics ─────────────────────────────────────────────────────
high_count = (df["Risk_Level"] == "HIGH").sum()
mod_count = (df["Risk_Level"] == "MODERATE").sum()
low_count = (df["Risk_Level"] == "LOW").sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Cases", len(df))
m2.metric("🔴 HIGH Risk", high_count)
m3.metric("🟡 MODERATE Risk", mod_count)
m4.metric("🟢 LOW Risk", low_count)

st.markdown("")

# ── Tabs ────────────────────────────────────────────────────────────
tab_queue, tab_review, tab_viz, tab_kb = st.tabs(
    [
        "📋 Triage Queue",
        "🔍 Case Review & Copilot",
        "📊 Data Visualisation",
        "📚 Knowledge Base",
    ]
)

# ── Tab 1: Triage Queue ────────────────────────────────────────────
with tab_queue:
    st.markdown("### Patient Triage Queue")
    st.markdown(
        "Cases are automatically classified by the **rule engine** "
        "based on AI uncertainty, OAR proximity, and tumor volume. "
        "Select a case below to begin the review workflow."
    )

    filter_risk = st.multiselect(
        "Filter by Risk Level",
        options=["HIGH", "MODERATE", "LOW"],
        default=["HIGH", "MODERATE", "LOW"],
    )

    filtered = df[df["Risk_Level"].isin(filter_risk)].copy()
    filtered = filtered.sort_values(
        by="Risk_Level",
        key=lambda s: s.map({"HIGH": 0, "MODERATE": 1, "LOW": 2}),
    )

    def color_risk(val: str) -> str:
        if val == "HIGH":
            return "background-color: #7f1d1d; color: #fca5a5"
        if val == "MODERATE":
            return "background-color: #78350f; color: #fcd34d"
        return "background-color: #14532d; color: #86efac"

    cols_to_show = [
        "Patient_ID", "Structure_Name", "Tumor_Volume_cc",
        "Distance_to_OAR_mm", "AI_Uncertainty_Score", "Risk_Level",
    ]
    display_df = filtered[cols_to_show].copy()
    display_df.columns = [
        "Patient ID", "Structure", "Volume (cc)",
        "Dist to OAR (mm)", "Uncertainty", "Risk",
    ]

    st.dataframe(
        display_df.style.map(
            color_risk, subset=["Risk"]
        ).format(
            {
                "Volume (cc)": "{:.2f}",
                "Dist to OAR (mm)": "{:.1f}",
                "Uncertainty": "{:.3f}",
            }
        ),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

# ── Tab 2: Case Review ─────────────────────────────────────────────
with tab_review:
    st.markdown("### 🔍 Case Review")

    patient_ids = df["Patient_ID"].tolist()
    selected_id = st.selectbox("Select Patient", patient_ids, index=0)
    selected_row = df[df["Patient_ID"] == selected_id].iloc[0]

    risk = compute_risk_level(selected_row)
    risk_color = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}[risk]

    col_info, col_params = st.columns([1, 1])
    with col_info:
        st.markdown(
            f"""
            <div class="report-header">
                <div style="font-size:0.75rem;color:#64748b;
                text-transform:uppercase;letter-spacing:0.1em;">
                Patient</div>
                <div style="font-size:1.8rem;font-weight:700;
                color:#f1f5f9;margin:4px 0;">
                {selected_row['Patient_ID']}</div>
                <div style="font-size:1rem;color:#94a3b8;">
                Structure: <strong style="color:#e2e8f0;">
                {selected_row['Structure_Name']}</strong></div>
                <div style="margin-top:12px;font-size:1.1rem;">
                Risk Level: {risk_color} <strong>{risk}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_params:
        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Tumor Volume",
            f"{selected_row['Tumor_Volume_cc']:.2f} cc",
        )
        p2.metric(
            "Distance to OAR",
            f"{selected_row['Distance_to_OAR_mm']:.1f} mm",
        )
        p3.metric(
            "AI Uncertainty",
            f"{selected_row['AI_Uncertainty_Score']:.3f}",
        )

    st.markdown("")

    # ── MRI Visualisation ───────────────────────────────────────────
    if has_nifti_data(selected_id):
        st.markdown("#### 🧬 MRI Scan with Tumor Segmentation Overlay")
        st.markdown(
            "Real MRI slices showing the tumor region. "
            "Colors: <span style='color:#ff4444;'>■ Tumor Core</span> "
            "<span style='color:#4444ff;'>■ Edema</span> "
            "<span style='color:#ffff00;'>■ Enhancing Tumor</span>",
            unsafe_allow_html=True,
        )

        plane_tabs = st.tabs(["Axial", "Coronal", "Sagittal"])
        planes = ["axial", "coronal", "sagittal"]
        for tab, plane in zip(plane_tabs, planes):
            with tab:
                buf = render_mri_with_overlay(selected_id, plane)
                if buf:
                    st.image(buf, use_column_width=True)
                else:
                    st.info(
                        f"No NIfTI data available for {plane} view."
                    )

        st.markdown("")

    # ── Explainability Panel ─────────────────────────────────────────
    with st.expander(
        "🔎 **Explainability — Rule Engine Triggers**", expanded=True
    ):
        triggers = get_triggers(selected_row)
        for t in triggers:
            if "🚩" in t:
                css_class = "trigger-card"
            elif "⚠️" in t:
                css_class = "trigger-card moderate"
            else:
                css_class = "trigger-card low"
            st.markdown(
                f'<div class="{css_class}">{t}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    st.markdown("#### 🤖 LLM RAG Copilot")
    st.markdown(
        "Generate a structured clinical review by retrieving relevant "
        "guidelines from the knowledge base based on the patient's "
        "specific risk triggers."
    )

    if st.button(
        "🚀 Generate Copilot Review",
        type="primary",
        use_container_width=False,
    ):
        with st.spinner(
            "Retrieving clinical guidelines and generating review..."
        ):
            import time
            time.sleep(1.2)
            report = generate_copilot_report(selected_row)

        st.session_state["copilot_report"] = report
        st.session_state["copilot_patient"] = selected_id

    if (
        "copilot_report" in st.session_state
        and st.session_state.get("copilot_patient") == selected_id
    ):
        st.markdown("")
        st.markdown(
            '<div class="copilot-section">',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state["copilot_report"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        pdf_bytes = build_pdf_content(
            selected_row, st.session_state["copilot_report"]
        )
        st.download_button(
            label="📄 Export Review as PDF-Ready Report",
            data=pdf_bytes,
            file_name=(
                f"NeuroQA_Review_{selected_row['Patient_ID']}_"
                f"{datetime.datetime.now().strftime('%Y%m%d')}.txt"
            ),
            mime="text/plain",
            use_container_width=False,
        )

# ── Tab 3: Data Visualisation ──────────────────────────────────────
with tab_viz:
    st.markdown("### 📊 Data Visualisation")
    st.markdown(
        f"Interactive charts for the **BraTS 2020** dataset "
        f"(n={len(df)} patients)."
    )

    # ── Tumor Volume Distribution ────────────────────────────────────
    st.markdown("#### 🧬 Tumor Volume Distribution")
    vol_data = df[["Patient_ID", "Tumor_Volume_cc"]].set_index("Patient_ID")
    st.bar_chart(vol_data, use_container_width=True, color="#8b5cf6")

    vc1, vc2 = st.columns(2)
    with vc1:
        st.markdown("#### 📏 Distance to OAR")
        dist_data = df[["Patient_ID", "Distance_to_OAR_mm"]].set_index(
            "Patient_ID"
        )
        st.bar_chart(dist_data, use_container_width=True, color="#06b6d4")
    with vc2:
        st.markdown("#### 🤖 AI Uncertainty Score")
        unc_data = df[["Patient_ID", "AI_Uncertainty_Score"]].set_index(
            "Patient_ID"
        )
        st.bar_chart(unc_data, use_container_width=True, color="#f59e0b")

    st.markdown("---")

    # ── Risk Pie Chart ───────────────────────────────────────────────
    st.markdown("#### 🎯 Risk Level Breakdown")
    risk_counts = df["Risk_Level"].value_counts()
    risk_df = pd.DataFrame({
        "Risk Level": risk_counts.index,
        "Count": risk_counts.values,
    })
    st.bar_chart(
        risk_df.set_index("Risk Level"),
        use_container_width=True,
        color="#ef4444",
    )

    st.markdown("---")

    # ── Structure Distribution ───────────────────────────────────────
    st.markdown("#### 🏥 Structure Distribution")
    struct_counts = df["Structure_Name"].value_counts()
    struct_df = pd.DataFrame({
        "Structure": struct_counts.index,
        "Count": struct_counts.values,
    })
    st.bar_chart(
        struct_df.set_index("Structure"),
        use_container_width=True,
        color="#22c55e",
    )

    st.markdown("---")

    # ── Scatter: Volume vs Uncertainty ───────────────────────────────
    st.markdown("#### 🔬 Tumor Volume vs AI Uncertainty")
    scatter_df = df[["Tumor_Volume_cc", "AI_Uncertainty_Score",
                     "Risk_Level"]].copy()
    scatter_df.columns = ["Tumor Volume (cc)", "AI Uncertainty", "Risk"]
    st.scatter_chart(
        scatter_df,
        x="Tumor Volume (cc)",
        y="AI Uncertainty",
        color="Risk",
        use_container_width=True,
    )

    # ── Irregularity (if available) ──────────────────────────────────
    if "Irregularity_Score" in df.columns:
        st.markdown("---")
        st.markdown("#### 🔀 Tumor Irregularity Score")
        irr_data = df[["Patient_ID", "Irregularity_Score"]].set_index(
            "Patient_ID"
        )
        st.bar_chart(irr_data, use_container_width=True, color="#ec4899")

    # ── Max Diameter (if available) ──────────────────────────────────
    if "Max_Diameter_mm" in df.columns:
        st.markdown("#### 📐 Max Tumor Diameter (mm)")
        dia_data = df[["Patient_ID", "Max_Diameter_mm"]].set_index(
            "Patient_ID"
        )
        st.bar_chart(dia_data, use_container_width=True, color="#14b8a6")

# ── Tab 5: Knowledge Base ──────────────────────────────────────────
with tab_kb:
    st.markdown("### 📚 Clinical Knowledge Base")
    st.markdown(
        "The following clinical guideline excerpts are indexed in the "
        "RAG retrieval system. When the Copilot generates a review, "
        "it retrieves the most relevant entries based on the patient's "
        "specific risk triggers."
    )

    for key, entry in KNOWLEDGE_BASE.items():
        risk_badge_class = {
            "Critical": "risk-high",
            "High": "risk-high",
            "Moderate": "risk-moderate",
            "Low": "risk-low",
        }.get(entry["risk_category"], "risk-low")

        with st.expander(
            f"📖 {key.replace('_', ' ').title()} "
            f"— {entry['risk_category']}"
        ):
            st.markdown(f"**Source:** {entry['source']}")
            st.markdown(
                f'<span class="risk-badge {risk_badge_class}">'
                f'{entry["risk_category"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown(entry["excerpt"])
            st.markdown(
                f"**Recommended Action:** {entry['recommended_action']}"
            )
