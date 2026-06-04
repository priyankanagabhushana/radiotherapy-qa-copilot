import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import io
from scipy.ndimage import binary_erosion


SEG_COLORS = {
    1: (1.0, 0.2, 0.2),
    2: (0.2, 0.5, 1.0),
    4: (1.0, 1.0, 0.2),
}
SEG_LABELS = {1: "Tumor Core (Necrosis)", 2: "Edema (Swelling)", 4: "Enhancing Tumor (Active)"}


def _find_nifti_files(patient_id, base_dirs=None):
    if base_dirs is None:
        base_dirs = [
            os.path.join(os.path.dirname(__file__), "brats_real"),
            os.path.join(os.path.dirname(__file__), "brats_data"),
        ]

    for base in base_dirs:
        if not os.path.exists(base):
            continue
        for d in os.listdir(base):
            d_path = os.path.join(base, d)
            if not os.path.isdir(d_path):
                continue
            if patient_id in d:
                flair = None
                t1ce = None
                seg = None
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


def _get_tumor_center_slice(seg_data):
    coords = np.argwhere(seg_data > 0)
    if len(coords) == 0:
        return np.array(seg_data.shape) // 2
    return coords.mean(axis=0).astype(int)


def _make_overlay(seg_slice):
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        if mask.any():
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
            rgba[mask, 3] = 0.55
    return rgba


def _draw_contours(ax, seg_slice):
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        if mask.any():
            eroded = binary_erosion(mask)
            boundary = mask & ~eroded
            ys, xs = np.where(boundary)
            if len(xs) > 0:
                ax.scatter(xs, ys, c=[color], s=0.3, alpha=0.9,
                           marker='.', linewidths=0)


def render_mri_with_overlay(patient_id, plane="axial"):
    mri_path, seg_path = _find_nifti_files(patient_id)
    if mri_path is None:
        return None

    mri_img = nib.load(mri_path)
    seg_img = nib.load(seg_path)

    mri_data = mri_img.get_fdata()
    seg_data = seg_img.get_fdata()

    center = _get_tumor_center_slice(seg_data)

    if plane == "axial":
        slice_idx = min(center[2], mri_data.shape[2] - 1)
        mri_slice = mri_data[:, :, slice_idx]
        seg_slice = seg_data[:, :, slice_idx]
    elif plane == "coronal":
        slice_idx = min(center[1], mri_data.shape[1] - 1)
        mri_slice = mri_data[:, slice_idx, :]
        seg_slice = seg_data[:, slice_idx, :]
    else:
        slice_idx = min(center[0], mri_data.shape[0] - 1)
        mri_slice = mri_data[slice_idx, :, :]
        seg_slice = seg_data[slice_idx, :, :]

    brain_mask = mri_slice > 0
    if brain_mask.any():
        p_low = np.percentile(mri_slice[brain_mask], 1)
        p_high = np.percentile(mri_slice[brain_mask], 99)
    else:
        p_low, p_high = 0, 1
    mri_slice = np.clip(mri_slice, p_low, p_high)
    if p_high > p_low:
        mri_slice = (mri_slice - p_low) / (p_high - p_low)

    overlay_rgba = _make_overlay(seg_slice)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.imshow(mri_slice, cmap="gray", origin="lower", interpolation="bilinear")
    ax.imshow(overlay_rgba, origin="lower", interpolation="nearest")
    _draw_contours(ax, seg_slice)

    tumor_types = []
    if np.any(seg_slice == 1):
        tumor_types.append("Core")
    if np.any(seg_slice == 2):
        tumor_types.append("Edema")
    if np.any(seg_slice == 4):
        tumor_types.append("Enhancing")
    tumor_str = ", ".join(tumor_types) if tumor_types else "None in slice"

    ax.set_title(
        f"{plane.title()} view — slice {slice_idx}  |  Tumor: {tumor_str}",
        color="white", fontsize=10, pad=10,
    )
    ax.axis("off")

    legend_patches = []
    for label, name in SEG_LABELS.items():
        color = SEG_COLORS[label]
        legend_patches.append(
            mpatches.Patch(facecolor=color, edgecolor="white",
                           linewidth=0.5, alpha=0.7, label=name)
        )
    ax.legend(
        handles=legend_patches, loc="lower right",
        fontsize=7, facecolor="#1e293b", edgecolor="#334155",
        labelcolor="white", framealpha=0.9,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                pad_inches=0.1, facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return buf


def render_all_planes(patient_id):
    results = {}
    for plane in ["axial", "coronal", "sagittal"]:
        buf = render_mri_with_overlay(patient_id, plane)
        if buf:
            results[plane] = buf
    return results


def has_nifti_data(patient_id):
    mri_path, seg_path = _find_nifti_files(patient_id)
    return mri_path is not None and seg_path is not None
