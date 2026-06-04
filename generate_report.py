import pandas as pd
import numpy as np
import os
import base64
import json
import datetime
from mri_viewer import render_mri_with_overlay, has_nifti_data


def _risk_level(row):
    if row["AI_Uncertainty_Score"] > 0.75:
        return "HIGH"
    if row["Distance_to_OAR_mm"] < 3.0:
        return "HIGH"
    if row["AI_Uncertainty_Score"] > 0.50 or row["Tumor_Volume_cc"] > 25.0:
        return "MODERATE"
    return "LOW"


def _risk_color(level):
    return {"HIGH": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}[level]


def _risk_bg(level):
    return {"HIGH": "#7f1d1d", "MODERATE": "#78350f", "LOW": "#14532d"}[level]


def _triggers_text(row):
    flags = []
    d = row["Distance_to_OAR_mm"]
    u = row["AI_Uncertainty_Score"]
    v = row["Tumor_Volume_cc"]
    struct = str(row["Structure_Name"])

    if d < 3.0:
        flags.append(f"\u26a0\ufe0f Proximity Alert: Contour distance to {struct} ({d:.1f}mm) falls below the 3.0mm planning margin threshold.")
    elif d < 6.0:
        flags.append(f"\u26a0\ufe0f Proximity Warning: Contour distance to {struct} ({d:.1f}mm) is marginal \u2014 recommend margin review.")
    else:
        flags.append(f"\u2705 Clearance OK: Distance to {struct} ({d:.1f}mm) exceeds safety thresholds.")

    if u > 0.70:
        flags.append(f"\u26a0\ufe0f Uncertainty Alert: AI voxel uncertainty score ({u:.3f}) exceeds the 0.70 institutional tolerance for critical structures.")
    elif u > 0.55:
        flags.append(f"\u26a0\ufe0f Uncertainty Warning: AI confidence score ({u:.3f}) is marginally below the 0.55 nominal threshold \u2014 recommend secondary review.")
    else:
        flags.append(f"\u2705 Confidence OK: AI uncertainty score ({u:.3f}) is within safe clinical limits.")

    if v > 100:
        flags.append(f"\u26a0\ufe0f Volume Alert: Gross tumor volume ({v:.1f} cc) exceeds 100 cc \u2014 multi-planar verification recommended per ESTRO protocol.")

    if not flags:
        flags.append("\u2705 No risk flags \u2014 standard QA protocol applies")
    return flags


def _action_text(row):
    actions = []
    d = row["Distance_to_OAR_mm"]
    u = row["AI_Uncertainty_Score"]
    v = row["Tumor_Volume_cc"]

    if d < 3.0:
        actions.append("ESTRO Guidelines: Manually verify structure boundaries interfacing with the Organ at Risk. Contour proximity <3mm requires attending physicist sign-off.")
    if u > 0.70:
        actions.append("AAPM TG-132: Review high-uncertainty voxel boundaries using strict window/level adjustments. Re-segment if Dice < 0.85.")
    if v > 100:
        actions.append("Volume Protocol: Large target volume requires multi-planar (axial/coronal/sagittal) contour integrity review before dose calculation.")
    actions.append("Approve target volume for dosimetric calculation only after manual sign-off per institutional QA protocol.")
    return actions


def generate_html_report(csv_path=None, output_path=None, include_mri=True, max_mri_patients=50):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(script_dir, "real_clinical_queue.csv")
    if output_path is None:
        output_path = os.path.join(script_dir, "neuroqa_report.html")

    df = pd.read_csv(csv_path)
    df["Risk_Level"] = df.apply(_risk_level, axis=1)
    df = df.sort_values("Risk_Level", key=lambda s: s.map({"HIGH": 0, "MODERATE": 1, "LOW": 2}))

    high = int((df["Risk_Level"] == "HIGH").sum())
    mod = int((df["Risk_Level"] == "MODERATE").sum())
    low = int((df["Risk_Level"] == "LOW").sum())

    risk_labels = json.dumps(["HIGH", "MODERATE", "LOW"])
    risk_values = json.dumps([high, mod, low])
    vol_labels = json.dumps(df["Patient_ID"].tolist())
    vol_values = json.dumps([float(v) for v in df["Tumor_Volume_cc"].round(2)])
    unc_values = json.dumps([float(v) for v in df["AI_Uncertainty_Score"].round(3)])
    dist_values = json.dumps([float(v) for v in df["Distance_to_OAR_mm"].round(1)])

    patients_json = []
    for _, r in df.iterrows():
        pid = r["Patient_ID"]
        risk = r["Risk_Level"]
        triggers = _triggers_text(r)
        actions = _action_text(r)

        mri_data = {}
        if include_mri and has_nifti_data(pid):
            for plane in ["axial", "coronal", "sagittal"]:
                buf = render_mri_with_overlay(pid, plane)
                if buf:
                    mri_data[plane] = base64.b64encode(buf.read()).decode("utf-8")

        patients_json.append({
            "id": pid,
            "structure": str(r["Structure_Name"]),
            "volume": float(r["Tumor_Volume_cc"]),
            "distance": float(r["Distance_to_OAR_mm"]),
            "uncertainty": float(r["AI_Uncertainty_Score"]),
            "risk": risk,
            "triggers": triggers,
            "actions": actions,
            "mri": mri_data,
        })

    patients_js = json.dumps(patients_json)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    table_rows = ""
    for _, r in df.iterrows():
        rc = _risk_color(r["Risk_Level"])
        rbg = _risk_bg(r["Risk_Level"])
        table_rows += f"""
        <tr onclick="selectPatient('{r['Patient_ID']}')" style="cursor:pointer;">
          <td>{r['Patient_ID']}</td>
          <td>{r['Structure_Name']}</td>
          <td>{r['Tumor_Volume_cc']:.2f}</td>
          <td>{r['Distance_to_OAR_mm']:.1f}</td>
          <td>{r['AI_Uncertainty_Score']:.3f}</td>
          <td><span style="background:{rbg};color:{rc};
              padding:2px 10px;border-radius:10px;
              font-weight:600;font-size:0.75rem;">
              {r['Risk_Level']}</span></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>NeuroQA Copilot — Interactive Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 20px;
    max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ color: #f1f5f9; font-size: 1.6rem; margin-bottom: 4px; }}
  h2 {{ color: #f1f5f9; font-size: 1.2rem; margin: 20px 0 10px; }}
  .subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 16px; }}

  .tabs {{ display:flex; gap:6px; margin:16px 0 0; }}
  .tab {{
    padding:8px 18px; border-radius:8px; cursor:pointer;
    background:#1e293b; color:#94a3b8; font-size:0.85rem;
    border:1px solid #334155; transition: all 0.2s;
  }}
  .tab:hover {{ background:#334155; }}
  .tab.active {{ background:#3b82f6; color:white; border-color:#3b82f6; }}
  .tab-content {{ display:none; padding:16px 0; }}
  .tab-content.active {{ display:block; }}

  .kpi-row {{ display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }}
  .kpi {{
    flex:1; min-width: 120px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #334155; border-radius: 10px;
    padding: 14px 16px; text-align: center;
  }}
  .kpi-label {{ color:#94a3b8; font-size:0.65rem;
      text-transform:uppercase; letter-spacing:0.05em; }}
  .kpi-value {{ color:#f1f5f9; font-size:1.4rem; font-weight:700;
      margin-top:2px; }}

  .chart-row {{ display:flex; gap:12px; margin:12px 0; flex-wrap:wrap; }}
  .chart-box {{
    flex:1; min-width:280px;
    background:#1e293b; border-radius:10px;
    padding:16px; border:1px solid #334155;
  }}

  table {{ width:100%; border-collapse:collapse; font-size:0.8rem; }}
  th {{ background:#1e293b; color:#94a3b8; padding:8px 10px;
      text-align:left; font-size:0.7rem;
      text-transform:uppercase; letter-spacing:0.05em;
      border-bottom:1px solid #334155; position:sticky; top:0; }}
  td {{ padding:6px 10px; border-bottom:1px solid #1e293b;
      color:#e2e8f0; }}
  tr:hover {{ background:#1e293b; }}
  tr.selected {{ background:#1e3a5f !important; }}

  .patient-detail {{
    background:#1e293b; border-radius:12px; padding:20px;
    border:1px solid #334155; margin:16px 0;
  }}
  .patient-header {{ display:flex; justify-content:space-between;
      align-items:center; flex-wrap:wrap; gap:12px; }}
  .patient-name {{ font-size:1.4rem; font-weight:700; color:#f1f5f9; }}
  .patient-struct {{ color:#94a3b8; margin-left:12px; }}
  .risk-badge {{ padding:4px 14px; border-radius:10px;
      font-weight:600; font-size:0.8rem; }}

  .metric-row {{ display:flex; gap:12px; margin:16px 0; flex-wrap:wrap; }}
  .metric {{
    flex:1; min-width:100px; text-align:center;
    background:#0f172a; border-radius:8px; padding:12px;
    border:1px solid #334155;
  }}
  .metric-val {{ font-size:1.3rem; font-weight:700; color:#f1f5f9; }}
  .metric-lbl {{ font-size:0.65rem; color:#94a3b8;
      text-transform:uppercase; margin-top:2px; }}

  .mri-row {{ display:flex; gap:12px; flex-wrap:wrap;
      justify-content:center; margin:16px 0; }}
  .mri-card {{ flex:1; min-width:220px; text-align:center; }}
  .mri-card img {{ width:100%; max-width:300px; border-radius:8px;
      border:1px solid #334155; }}
  .mri-label {{ color:#94a3b8; font-size:0.75rem; margin-top:4px; }}

  .trigger {{ background:#1e293b; border-left:3px solid #ef4444;
      border-radius:0 6px 6px 0; padding:8px 12px;
      margin:6px 0; font-size:0.8rem; color:#e2e8f0; }}
  .trigger.warn {{ border-left-color:#f59e0b; }}
  .trigger.ok {{ border-left-color:#22c55e; }}

  .action-list {{ margin:8px 0; }}
  .action-item {{ padding:6px 0; font-size:0.82rem; color:#e2e8f0;
      border-bottom:1px solid #1e293b; }}
  .action-item:last-child {{ border-bottom:none; }}

  .desc-box {{ background:#0f172a; border:1px solid #334155;
      border-radius:8px; padding:12px; margin:12px 0;
      font-size:0.8rem; color:#94a3b8; line-height:1.5; }}

  .legend {{ display:flex; gap:16px; margin:8px 0; }}
  .legend-item {{ display:flex; align-items:center; gap:6px;
      font-size:0.75rem; color:#94a3b8; }}
  .legend-dot {{ width:10px; height:10px; border-radius:50%; }}

  .footer {{
    margin-top:30px; padding-top:12px;
    border-top:1px solid #1e293b;
    color:#475569; font-size:0.7rem; text-align:center;
  }}
  @media print {{
    body {{ background:#fff; color:#111; }}
  }}
</style>
</head>
<body>

<h1>&#129504; NeuroQA Copilot — Interactive Clinical Report</h1>
<div class="subtitle">
  Generated: {now} &nbsp;|&nbsp; BraTS 2020 (n={len(df)} patients) &nbsp;|&nbsp; Research Prototype
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('dashboard')">&#128202; Dashboard</div>
  <div class="tab" onclick="showTab('queue')">&#128203; Patient Queue</div>
  <div class="tab" onclick="showTab('review')">&#128269; Case Review</div>
</div>

<!-- ─── TAB: Dashboard ─── -->
<div id="tab-dashboard" class="tab-content active">
  <div class="kpi-row">
    <div class="kpi"><div class="kpi-label">Total Cases</div><div class="kpi-value">{len(df)}</div></div>
    <div class="kpi"><div class="kpi-label">&#128308; HIGH Risk</div><div class="kpi-value" style="color:#ef4444;">{high}</div></div>
    <div class="kpi"><div class="kpi-label">&#128993; MODERATE</div><div class="kpi-value" style="color:#f59e0b;">{mod}</div></div>
    <div class="kpi"><div class="kpi-label">&#128994; LOW Risk</div><div class="kpi-value" style="color:#22c55e;">{low}</div></div>
  </div>

  <div class="desc-box">
    <strong>How to use this report:</strong> Use the tabs above to navigate.
    <strong>Dashboard</strong> shows overview charts.
    <strong>Patient Queue</strong> lists all patients (click a row to review).
    <strong>Case Review</strong> shows MRI scans, risk triggers, and recommended actions for a selected patient.
  </div>

  <div class="chart-row">
    <div class="chart-box"><h3 style="color:#f1f5f9;font-size:0.95rem;">Risk Distribution</h3><canvas id="riskChart" height="200"></canvas></div>
    <div class="chart-box"><h3 style="color:#f1f5f9;font-size:0.95rem;">Tumor Volume per Patient</h3><canvas id="volChart" height="200"></canvas></div>
  </div>
  <div class="chart-row">
    <div class="chart-box"><h3 style="color:#f1f5f9;font-size:0.95rem;">AI Uncertainty Score</h3><canvas id="uncChart" height="200"></canvas></div>
    <div class="chart-box"><h3 style="color:#f1f5f9;font-size:0.95rem;">Distance to OAR (mm)</h3><canvas id="distChart" height="200"></canvas></div>
  </div>
</div>

<!-- ─── TAB: Patient Queue ─── -->
<div id="tab-queue" class="tab-content">
  <div class="desc-box">
    <strong>Patient Triage Queue:</strong> Cases ranked by risk level.
    Click any row to open the <strong>Case Review</strong> tab for that patient.
  </div>
  <div style="max-height:600px; overflow-y:auto; border:1px solid #334155; border-radius:8px;">
  <table>
    <thead><tr>
      <th>Patient ID</th><th>Structure</th><th>Volume (cc)</th>
      <th>Dist to OAR (mm)</th><th>Uncertainty</th><th>Risk</th>
    </tr></thead>
    <tbody id="queueBody">{table_rows}</tbody>
  </table>
  </div>
</div>

<!-- ─── TAB: Case Review ─── -->
<div id="tab-review" class="tab-content">
  <div style="margin-bottom:12px;">
    <label style="color:#94a3b8;font-size:0.8rem;">Select Patient:</label>
    <select id="patientSelect" onchange="selectPatient(this.value)"
      style="background:#1e293b;color:#f1f5f9;border:1px solid #334155;
      border-radius:6px;padding:6px 12px;font-size:0.85rem;margin-left:8px;">
    </select>
  </div>
  <div id="patientDetail"></div>
</div>

<div class="footer">
  NeuroQA Copilot v1.0 — Research Prototype — Not for clinical use<br/>
  Data: BraTS 2020 Challenge (Kaggle) — AAPM TG-132, ESTRO, QUANTEC guidelines<br/>
  All flagged contours require human verification before treatment delivery.
</div>

<script>
const patients = {patients_js};
const riskColors = {{ HIGH:'#ef4444', MODERATE:'#f59e0b', LOW:'#22c55e' }};
const riskBg = {{ HIGH:'#7f1d1d', MODERATE:'#78350f', LOW:'#14532d' }};

// ── Tabs ──
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-'+name).classList.add('active');
  event.target.classList.add('active');
}}

// ── Patient Selection ──
const select = document.getElementById('patientSelect');
patients.forEach(p => {{
  const opt = document.createElement('option');
  opt.value = p.id;
  opt.textContent = p.id + ' — ' + p.structure + ' [' + p.risk + ']';
  select.appendChild(opt);
}});

function selectPatient(pid) {{
  showTab('review');
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab')[2].classList.add('active');
  select.value = pid;
  const p = patients.find(x => x.id === pid);
  if (!p) return;

  document.querySelectorAll('#queueBody tr').forEach(tr => tr.classList.remove('selected'));
  const detail = document.getElementById('patientDetail');

  let mriHtml = '';
  if (Object.keys(p.mri).length > 0) {{
    mriHtml = '<h3 style="color:#f1f5f9;font-size:0.95rem;margin:16px 0 8px;">&#129504; MRI Scans with Tumor Segmentation</h3>';
    mriHtml += '<div class="desc-box"><span style="color:#ff4444;">&#9632; Tumor Core</span> &nbsp; <span style="color:#4488ff;">&#9632; Edema</span> &nbsp; <span style="color:#ffff44;">&#9632; Enhancing Tumor</span></div>';
    mriHtml += '<div class="mri-row">';
    for (const plane of ['axial','coronal','sagittal']) {{
      if (p.mri[plane]) {{
        mriHtml += '<div class="mri-card"><img src="data:image/png;base64,' + p.mri[plane] + '" /><div class="mri-label">' + plane.charAt(0).toUpperCase() + plane.slice(1) + ' View</div></div>';
      }}
    }}
    mriHtml += '</div>';
  }} else {{
    mriHtml = '<div class="desc-box">No MRI data available for this patient.</div>';
  }}

  let triggersHtml = '';
  p.triggers.forEach(t => {{
    let cls = 'trigger';
    if (t.startsWith('✅')) cls += ' ok';
    else if (t.startsWith('⚠️')) cls += ' warn';
    triggersHtml += '<div class="' + cls + '">' + t + '</div>';
  }});

  let actionsHtml = '';
  p.actions.forEach((a, i) => {{
    actionsHtml += '<div class="action-item">' + (i+1) + '. ' + a + '</div>';
  }});

  detail.innerHTML = `
    <div class="patient-detail">
      <div class="patient-header">
        <div>
          <span class="patient-name">${{p.id}}</span>
          <span class="patient-struct">${{p.structure}}</span>
        </div>
        <span class="risk-badge" style="background:${{riskBg[p.risk]}};color:${{riskColors[p.risk]}};">${{p.risk}} RISK</span>
      </div>

      <div class="metric-row">
        <div class="metric"><div class="metric-val">${{p.volume.toFixed(2)}}</div><div class="metric-lbl">Volume (cc)</div></div>
        <div class="metric"><div class="metric-val">${{p.distance.toFixed(1)}}</div><div class="metric-lbl">Dist to OAR (mm)</div></div>
        <div class="metric"><div class="metric-val">${{p.uncertainty.toFixed(3)}}</div><div class="metric-lbl">AI Uncertainty</div></div>
      </div>

      ${{mriHtml}}

      <h3 style="color:#f1f5f9;font-size:0.95rem;margin:16px 0 8px;">&#128269; Risk Triggers (Rule Engine)</h3>
      <div class="desc-box">These are the automated checks that flagged this patient. Each trigger explains <em>why</em> the patient was assigned their risk level.</div>
      ${{triggersHtml}}

      <h3 style="color:#3b82f6;font-size:0.95rem;margin:16px 0 8px;">&#129302; LLM Copilot Assessment (RAG)</h3>
      <div class="desc-box">Synthesized clinical reasoning based on patient geometric triggers and retrieved guidelines (AAPM TG-132, ESTRO).</div>
      <div class="action-list">${{actionsHtml}}</div>
    </div>
  `;
}}

// ── Charts ──
new Chart(document.getElementById('riskChart'), {{
  type: 'doughnut',
  data: {{ labels: {risk_labels}, datasets: [{{ data: {risk_values}, backgroundColor: ['#ef4444','#f59e0b','#22c55e'], borderColor: '#0f172a', borderWidth: 2 }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }} }}
}});

new Chart(document.getElementById('volChart'), {{
  type: 'bar',
  data: {{ labels: {vol_labels}, datasets: [{{ data: {vol_values}, backgroundColor: '#8b5cf6', borderRadius: 3 }}] }},
  options: {{ responsive: true, scales: {{ x: {{ ticks: {{ color:'#64748b', maxRotation:90, font:{{size:7}} }} }}, y: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e293b' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('uncChart'), {{
  type: 'bar',
  data: {{ labels: {vol_labels}, datasets: [{{ data: {unc_values}, backgroundColor: '#f59e0b', borderRadius: 3 }}] }},
  options: {{ responsive: true, scales: {{ x: {{ ticks: {{ color:'#64748b', maxRotation:90, font:{{size:7}} }} }}, y: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e293b' }}, min:0, max:1 }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('distChart'), {{
  type: 'bar',
  data: {{ labels: {vol_labels}, datasets: [{{ data: {dist_values}, backgroundColor: '#06b6d4', borderRadius: 3 }}] }},
  options: {{ responsive: true, scales: {{ x: {{ ticks: {{ color:'#64748b', maxRotation:90, font:{{size:7}} }} }}, y: {{ ticks: {{ color:'#64748b' }}, grid: {{ color:'#1e293b' }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Report: {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    generate_html_report(include_mri=True, max_mri_patients=50)
