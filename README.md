# NeuroQA Copilot

**Human-in-the-Loop Radiotherapy Quality Assurance Decision Support System for Brain Tumor Auto-Segmentation**

> Research Prototype — Not for clinical use

---

## Overview

NeuroQA Copilot is a clinical decision-support tool that helps medical physicists review AI-generated brain tumor segmentations before radiation therapy treatment planning. It combines:

- **Rule-based risk engine** — automatically triages 50 patients by tumor volume, proximity to critical structures, and AI uncertainty scores
- **RAG-simulated LLM copilot** — retrieves relevant clinical guidelines (AAPM TG-132, ESTRO, QUANTEC) matched to each patient's specific risk triggers
- **MRI visualisation** — renders real BraTS 2020 NIfTI scans with tumor segmentation overlays (Core, Edema, Enhancing)

## Features

| Feature | Description |
|---------|-------------|
| **Dashboard** | KPI cards, risk distribution doughnut chart, per-patient volume/uncertainty/distance bar charts |
| **Patient Queue** | Triaged table of all 50 patients ranked by risk level |
| **Case Review** | Click any patient to see MRI scans (axial/coronal/sagittal), risk triggers, and LLM copilot actions |
| **Knowledge Base** | Indexed clinical guidelines from AAPM, ESTRO, QUANTEC, ASTRO, NRG Oncology |
| **HTML Report** | Self-contained, shareable HTML report with embedded MRI images |
| **Streamlit App** | Interactive web dashboard with PDF export |

## Project Structure

```
radiotherapy-qa-copilot/
├── app.py                    # Streamlit web application
├── generate_report.py        # Generates self-contained HTML report with embedded MRI
├── mri_viewer.py             # NIfTI renderer: MRI + segmentation overlay
├── knowledge_base.py         # Clinical guideline RAG knowledge base
├── extract_real_cases.py     # Extracts patient data from BraTS dataset
├── generate_cases.py         # Generates synthetic clinical queue data
├── generate_brats_nifti.py   # Creates synthetic NIfTI volumes
├── real_clinical_queue.csv   # Patient data (50 patients from BraTS 2020)
├── clinical_queue.csv        # Synthetic clinical queue data
├── neuroqa_report.html       # Generated HTML report (create with generate_report.py)
├── PROJECT_GUIDE.md          # Plain-language project explanation
├── requirements.txt          # Python dependencies
└── .gitignore
```

> **Note:** Large MRI data directories (`brats_real/`, `brats_data/`, ~4.3 GB) are excluded from the repository via `.gitignore`. See [Data Setup](#data-setup) below.

## Quick Start

### Prerequisites

- Python 3.10+
- BraTS 2020 dataset (for MRI visualisation) — [Download from Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/radiotherapy-qa-copilot.git
cd radiotherapy-qa-copilot
pip install -r requirements.txt
```

### Run the Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Generate the HTML Report

```bash
python generate_report.py
```

Creates `neuroqa_report.html` (opens in any browser, shareable via email).

## Data Setup

The repository ships with `real_clinical_queue.csv` (patient parameters for 50 BraTS 2020 cases). To enable **MRI visualisation**:

1. Download the BraTS 2020 Training Data from [Kaggle](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
2. Extract patient folders into `brats_real/`:
   ```
   brats_real/
   ├── BraTS20_Training_001/
   │   ├── BraTS20_Training_001_flair.nii
   │   ├── BraTS20_Training_001_t1ce.nii
   │   ├── BraTS20_Training_001_seg.nii
   │   └── ...
   └── ...
   ```
3. The MRI viewer (`mri_viewer.py`) automatically detects available NIfTI files

## Risk Engine Logic

Each patient is classified by three parameters:

| Parameter | HIGH Threshold | MODERATE Threshold |
|-----------|---------------|-------------------|
| AI Uncertainty Score | > 0.70 | > 0.55 |
| Distance to OAR (mm) | < 3.0 | < 6.0 |
| Tumor Volume (cc) | — | > 100 (flag only) |

A patient is classified as **HIGH** if *any* HIGH threshold is breached, **MODERATE** if any MODERATE threshold is breached, otherwise **LOW**.

## Clinical Guidelines Referenced

- **AAPM TG-132** — Use of Image Registration and Fusion Algorithms in Radiation Oncology (2017)
- **ESTRO** — Guideline on Automated Contouring in Radiotherapy (2020)
- **QUANTEC** — Spinal Cord / Brainstem Dose Constraints (2010)
- **ASTRO** — AI in Radiation Oncology Best Practices (2023)
- **NRG Oncology CC001** — Hippocampal Avoidance Guidelines (2018)

## Tech Stack

- **Frontend:** Streamlit, Chart.js (HTML report)
- **Backend:** Python, Pandas, NumPy
- **MRI Processing:** NiBabel, Matplotlib, SciPy
- **Data:** BraTS 2020 Challenge (Kaggle)

## License

This is a research prototype. Not intended for clinical use. All flagged contours require human verification before treatment delivery.

## Acknowledgments

- [BraTS 2020 Challenge](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) for the brain tumor MRI dataset
- AAPM, ESTRO, QUANTEC, ASTRO, and NRG Oncology for clinical guideline references
