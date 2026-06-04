# Project Guide — NeuroQA Copilot

**In plain language: what this project does, why it exists, and how every piece fits together.**

---

## The Problem This Solves

When a patient has a brain tumor, doctors can use radiation to shrink it. Before treatment, someone needs to carefully outline the tumor on MRI scans so the radiation beam targets *only* the tumor and avoids healthy brain tissue.

Today, hospitals use **AI programs** to automatically draw these outlines. The AI is fast, but it's not perfect. If the AI makes a mistake near a sensitive area — like the optic nerve (which controls vision) or the brainstem (which controls breathing) — the consequences can be serious.

**This project is a safety net.** It reviews the AI's work *before* a human doctor signs off, flagging cases that need extra attention.

---

## How It Works (Step by Step)

### Step 1: Patient Data
We have MRI scan data from **50 real brain tumor patients** (from the BraTS 2020 medical dataset). For each patient, we know:

- **How big the tumor is** (volume in cubic centimeters)
- **How close the tumor is to critical structures** (distance in millimeters)
- **How confident the AI was** when it drew the tumor outline (a score from 0 to 1 — lower means the AI was unsure)

### Step 2: The Rule Engine
A simple set of rules automatically assigns a risk level to each patient:

| Risk Level | What triggers it |
|------------|-----------------|
| 🔴 **HIGH** | The AI was very unsure (score > 0.70) OR the tumor is dangerously close (< 3mm) to a critical structure |
| 🟡 **MODERATE** | The AI was somewhat unsure (score > 0.55) OR the tumor is nearby (< 6mm) |
| 🟢 **LOW** | Everything looks safe |

### Step 3: The Copilot (RAG System)
For each flagged patient, the system searches a **knowledge base** of real clinical guidelines from:

- **AAPM TG-132** — Rules for using AI in radiation planning
- **ESTRO** — European guidelines for auto-contouring
- **QUANTEC** — Dose limits for sensitive organs
- **ASTRO** — Best practices for AI in cancer treatment
- **NRG Oncology** — Rules for protecting memory centers (hippocampus)

It pulls the most relevant guideline excerpts and generates a **recommended action list** — telling the physicist exactly what to double-check.

### Step 4: MRI Visualisation
For patients where we have the actual MRI scan files, the system renders the brain scan with a **color overlay** showing where the AI thinks the tumor is:

- 🔴 **Red** = Tumor Core (the dangerous part)
- 🔵 **Blue** = Edema (swelling around the tumor)
- 🟡 **Yellow** = Enhancing Tumor (actively growing)

This lets the reviewer visually confirm whether the AI's outline looks reasonable.

### Step 5: Report Generation
The system can produce two types of output:

1. **Interactive Web App** (Streamlit) — A dashboard you can open in your browser, click through patients, view MRI scans, and generate copilot reviews
2. **HTML Report** — A single self-contained file (with MRI images embedded) that can be emailed to colleagues or opened on any computer without installing software

---

## The Files Explained

Here's what each file in the project does:

| File | What it does | Who needs it |
|------|-------------|--------------|
| `app.py` | The main web dashboard (Streamlit). This is what you launch with `streamlit run app.py`. | Everyone — this is the main interface |
| `generate_report.py` | Creates a standalone HTML report with MRI images baked in. Run with `python generate_report.py`. | Anyone who needs a shareable report file |
| `mri_viewer.py` | Reads MRI brain scans (NIfTI format) and draws the tumor segmentation overlay on top. | Used internally by the other scripts |
| `knowledge_base.py` | Contains all the clinical guideline text (from AAPM, ESTRO, etc.) that the copilot searches through. | Used internally by `app.py` |
| `extract_real_cases.py` | One-time script that extracts patient data from the BraTS dataset into the CSV file. | Only if you need to rebuild the CSV |
| `generate_cases.py` | Creates synthetic (fake) patient data for testing. | Only for development/testing |
| `generate_brats_nifti.py` | Creates synthetic MRI scan files for testing. | Only for development/testing |
| `real_clinical_queue.csv` | The actual patient data (50 rows). Each row has: patient ID, structure name, tumor volume, distance to OAR, AI uncertainty score. | Core data file — needed by everything |
| `clinical_queue.csv` | Synthetic patient data (for testing). | Only for development |
| `neuroqa_report.html` | The generated HTML report. Open in any browser. | Anyone who wants the standalone report |
| `requirements.txt` | List of Python packages needed to run the project. | Everyone — `pip install -r requirements.txt` |

---

## The Data Flow

```
BraTS 2020 MRI Scans (NIfTI files)
         │
         ▼
   extract_real_cases.py
         │
         ▼
   real_clinical_queue.csv ──────────┐
         │                          │
         ▼                          ▼
     app.py                   generate_report.py
   (Streamlit)                    │
         │                        ▼
         ▼                 neuroqa_report.html
   http://localhost:8501    (open in browser)
```

---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. In our project:

1. **Retrieval** — When a patient is flagged, the system searches the knowledge base (`knowledge_base.py`) for the most relevant clinical guideline. For example, if a tumor is 1.5mm from the optic chiasm, it retrieves the AAPM TG-132 guideline about optic chiasm proximity.

2. **Augmented** — The retrieved guideline text is combined with the patient's specific data (tumor size, distance, uncertainty score).

3. **Generation** — The system produces a structured review report that includes the guideline recommendation, the patient's specific risk factors, and a prioritized action list.

In a production system, this would use a real Large Language Model (like GPT-4). In this research prototype, the "generation" step uses template-based text to demonstrate the workflow without requiring an LLM API.

---

## What is NIfTI?

NIfTI (Neuroimaging Informatics Technology Initiative) is the standard file format for brain scans. A NIfTI file (`.nii` or `.nii.gz`) contains a 3D volume of the brain — like stacking hundreds of thin MRI slices into a cube. Our `mri_viewer.py` reads these 3D volumes and displays a 2D slice with the tumor overlay.

---

## Key Thresholds Explained

| Threshold | Value | Why this number? |
|-----------|-------|-----------------|
| Distance to OAR < 3mm | HIGH risk | 3mm is the typical planning margin. If the tumor is closer than this, even a tiny contour error could irradiate the critical structure. |
| AI Uncertainty > 0.70 | HIGH risk | At this level, the AI model has significant doubt about its own contour. Published research shows error rates increase sharply above 0.70. |
| AI Uncertainty > 0.55 | MODERATE risk | The AI is somewhat unsure. A spot-check review is recommended but not mandatory. |
| Tumor Volume > 100cc | Trigger only | Large tumors have irregular shapes that AI struggles with. This doesn't automatically assign HIGH risk but triggers a volume alert. |

---

## Running the Project

### First time setup
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/radiotherapy-qa-copilot.git
cd radiotherapy-qa-copilot

# Install Python dependencies
pip install -r requirements.txt
```

### Launch the interactive dashboard
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

### Generate the HTML report
```bash
python generate_report.py
```
This creates `neuroqa_report.html`. Double-click it to open in your browser.

### Enable MRI images (optional)
Download the BraTS 2020 dataset from Kaggle and place the patient folders in `brats_real/`. The MRI viewer will automatically find and display them.

---

## Who Is This For?

- **Medical Physicists** — Primary users who review AI auto-segmentations before treatment
- **Radiation Oncologists** — Physicians who make the final treatment decision
- **Researchers** — Anyone studying AI-assisted radiotherapy workflows
- **Students** — Learning about clinical AI, RAG systems, and radiation therapy QA

---

## Limitations

- This is a **research prototype**, not a medical device
- The LLM copilot uses template-based text, not a real language model
- Only 10 of the 50 patients have actual MRI scan files; the rest show "No MRI data available"
- All risk thresholds are configurable and should be validated by each institution
- **Never use auto-segmented contours without human verification**
