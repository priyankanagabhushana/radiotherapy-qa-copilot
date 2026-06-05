"""
Render individual MRI plane PNGs + segmentation masks + YOLO-Seg contour files.

For each patient, generates:
  mri_previews/{patient_id}/
    axial.png, coronal.png, sagittal.png           — Grayscale MRI slices
    axial_mask.png, coronal_mask.png, sagittal_mask.png — Segmentation masks (0/1/2/4)
    axial.txt, coronal.txt, sagittal.txt           — YOLO-Seg contour polygons

YOLO-Seg format per line:
  class_id x1 y1 x2 y2 ... xn yn
  (coordinates normalized 0-1, origin top-left)

Class IDs (remapped from BraTS):
  0 = Tumor Core (BraTS label 1)
  1 = Edema (BraTS label 2)
  2 = Enhancing Tumor (BraTS label 4)
"""

import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


BRATS_TO_YOLO = {1: 0, 2: 1, 4: 2}
SEG_COLORS = {0: (1.0, 0.2, 0.2), 1: (0.2, 0.5, 1.0), 2: (1.0, 1.0, 0.2)}
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "mri_previews")
DATA_DIRS = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "brats_real"),
    os.path.join(BASE_DIR, "brats_data"),
]


def find_nifti(patient_id):
    for base in DATA_DIRS:
        if not os.path.exists(base):
            continue
        for d in os.listdir(base):
            if patient_id.lower() in d.lower() or d.lower() in patient_id.lower():
                d_path = os.path.join(base, d)
                if not os.path.isdir(d_path):
                    continue
                flair = t1ce = seg = None
                for f in os.listdir(d_path):
                    fp = os.path.join(d_path, f)
                    if "seg" in f and f.endswith((".nii", ".nii.gz")):
                        seg = fp
                    elif "flair" in f and f.endswith((".nii", ".nii.gz")):
                        flair = fp
                    elif "t1ce" in f and f.endswith((".nii", ".nii.gz")):
                        t1ce = fp
                mri = t1ce if t1ce else flair
                if mri and seg:
                    return mri, seg
    return None, None


def normalize_slice(mri_slice):
    brain = mri_slice > 0
    if brain.any():
        pl, ph = np.percentile(mri_slice[brain], 1), np.percentile(mri_slice[brain], 99)
    else:
        pl, ph = 0, 1
    mri_slice = np.clip(mri_slice, pl, ph)
    if ph > pl:
        mri_slice = (mri_slice - pl) / (ph - pl)
    return mri_slice


def extract_contours(seg_slice):
    """Extract contour polygons from segmentation. Returns list of (yolo_class, Nx2 normalized coords)."""
    h, w = seg_slice.shape
    results = []
    for brats_label, yolo_class in sorted(BRATS_TO_YOLO.items()):
        mask = seg_slice == brats_label
        if not mask.any():
            continue
        try:
            fig, ax = plt.subplots(figsize=(1, 1))
            cs = ax.contour(mask.astype(float), levels=[0.5], origin="upper")
            for seg_list in cs.allsegs:
                for seg in seg_list:
                    if len(seg) >= 3:
                        norm = seg.copy()
                        norm[:, 0] = np.clip(norm[:, 0] / w, 0, 1)
                        norm[:, 1] = np.clip(norm[:, 1] / h, 0, 1)
                        results.append((yolo_class, norm))
            plt.close(fig)
        except Exception:
            plt.close("all")
    return results


def save_yolo(contours, filepath):
    with open(filepath, "w") as f:
        for yolo_class, polygon in contours:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            f.write(f"{yolo_class} {coords}\n")


def render_patient(patient_id):
    mri_path, seg_path = find_nifti(patient_id)
    if not mri_path:
        return False

    mri_data = nib.load(mri_path).get_fdata()
    seg_data = nib.load(seg_path).get_fdata()

    center = np.argwhere(seg_data > 0)
    if len(center) == 0:
        center = np.array(seg_data.shape) // 2
    else:
        center = center.mean(axis=0).astype(int)

    planes = {
        "axial":    (mri_data[:, :, min(center[2], mri_data.shape[2]-1)],
                     seg_data[:, :, min(center[2], seg_data.shape[2]-1)]),
        "coronal":  (mri_data[:, min(center[1], mri_data.shape[1]-1), :],
                     seg_data[:, min(center[1], seg_data.shape[1]-1), :]),
        "sagittal": (mri_data[min(center[0], mri_data.shape[0]-1), :, :],
                     seg_data[min(center[0], seg_data.shape[0]-1), :, :]),
    }

    patient_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(patient_dir, exist_ok=True)

    for plane_name, (mri_slice, seg_slice) in planes.items():
        # Save grayscale MRI PNG
        norm = normalize_slice(mri_slice)
        img_uint8 = (norm * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(
            os.path.join(patient_dir, f"{plane_name}.png")
        )

        # Save segmentation mask PNG (raw label values: 0/1/2/4)
        Image.fromarray(seg_slice.astype(np.uint8), mode="L").save(
            os.path.join(patient_dir, f"{plane_name}_mask.png")
        )

        # Save YOLO-Seg contours (remapped class IDs: 0/1/2)
        contours = extract_contours(seg_slice)
        save_yolo(contours, os.path.join(patient_dir, f"{plane_name}.txt"))

    return True


def main():
    # Clean old data
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(BASE_DIR, "real_clinical_queue.csv"))

    success = 0
    for i, row in df.iterrows():
        pid = row["Patient_ID"]
        print(f"  [{i+1}/{len(df)}] {pid} — rendering...", end=" ", flush=True)
        try:
            if render_patient(pid):
                out_dir = os.path.join(OUTPUT_DIR, pid)
                size = sum(
                    os.path.getsize(os.path.join(out_dir, f))
                    for f in os.listdir(out_dir)
                )
                print(f"OK ({size/1024:.0f} KB)")
                success += 1
            else:
                print("FAILED (no NIfTI)")
        except Exception as e:
            print(f"ERROR: {e}")

    total_bytes = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            total_bytes += os.path.getsize(os.path.join(root, f))

    print(f"\nDone: {success}/{len(df)} patients")
    print(f"Total size: {total_bytes / (1024*1024):.1f} MB")
    print(f"Files per patient: 3 PNGs + 3 mask PNGs + 3 TXT contours = 9 files")
    print(f"Location: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
