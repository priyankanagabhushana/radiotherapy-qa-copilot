import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import io
from PIL import Image
from scipy.ndimage import binary_erosion
import plotly.graph_objects as go


SEG_COLORS = {
    1: (1.0, 0.2, 0.2),
    2: (0.2, 0.5, 1.0),
    4: (1.0, 1.0, 0.2),
}
SEG_LABELS = {1: "Tumor Core (Necrosis)", 2: "Edema (Swelling)", 4: "Enhancing Tumor (Active)"}
PREVIEW_DIR = os.path.join(os.path.dirname(__file__), "mri_previews")


def _find_nifti_files(patient_id, base_dirs=None):
    if base_dirs is None:
        base_dirs = [
            os.path.join(os.path.dirname(__file__), "data"),
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
            if patient_id.lower() in d.lower() or d.lower() in patient_id.lower():
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


def _make_overlay(seg_slice, alpha=0.55):
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        if mask.any():
            rgba[mask, 0] = color[0]
            rgba[mask, 1] = color[1]
            rgba[mask, 2] = color[2]
            rgba[mask, 3] = alpha
    return rgba


def _draw_contours(ax, seg_slice):
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        if mask.any():
            padded = np.pad(mask, 1, mode='constant', constant_values=False)
            eroded = binary_erosion(padded)
            eroded = eroded[1:-1, 1:-1]
            boundary = mask & ~eroded
            boundary_float = boundary.astype(float)
            try:
                ax.contour(boundary_float, levels=[0.5], colors=[color],
                           linewidths=1.2, origin='lower')
            except Exception:
                ys, xs = np.where(boundary)
                if len(xs) > 0:
                    ax.scatter(xs, ys, c=[color], s=0.5, alpha=0.9,
                               marker='.', linewidths=0)


def render_mri_with_overlay(patient_id, plane="axial", overlay_opacity=0.55):
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

    overlay_rgba = _make_overlay(seg_slice, alpha=overlay_opacity)

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


def render_all_planes(patient_id, overlay_opacity=0.55):
    results = {}
    for plane in ["axial", "coronal", "sagittal"]:
        buf = render_mri_with_overlay(patient_id, plane, overlay_opacity=overlay_opacity)
        if buf:
            results[plane] = buf
    return results


def render_combined_view(patient_id, overlay_opacity=0.55):
    """Render all 3 planes (axial, coronal, sagittal) in a single figure."""
    mri_path, seg_path = _find_nifti_files(patient_id)
    if mri_path is None:
        return None

    mri_img = nib.load(mri_path)
    seg_img = nib.load(seg_path)
    mri_data = mri_img.get_fdata()
    seg_data = seg_img.get_fdata()
    center = _get_tumor_center_slice(seg_data)

    planes = {
        "Axial": (2, lambda s: (mri_data[:, :, s], seg_data[:, :, s]),
                  min(center[2], mri_data.shape[2] - 1)),
        "Coronal": (1, lambda s: (mri_data[:, s, :], seg_data[:, s, :]),
                    min(center[1], mri_data.shape[1] - 1)),
        "Sagittal": (0, lambda s: (mri_data[s, :, :], seg_data[s, :, :]),
                     min(center[0], mri_data.shape[0] - 1)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f172a")

    for ax, (name, (axis_idx, slicer, slc)) in zip(axes, planes.items()):
        ax.set_facecolor("#0f172a")
        mri_slice, seg_slice = slicer(slc)

        # Normalize MRI
        brain_mask = mri_slice > 0
        if brain_mask.any():
            pl, ph = np.percentile(mri_slice[brain_mask], 1), np.percentile(mri_slice[brain_mask], 99)
        else:
            pl, ph = 0, 1
        mri_slice = np.clip(mri_slice, pl, ph)
        if ph > pl:
            mri_slice = (mri_slice - pl) / (ph - pl)

        overlay_rgba = _make_overlay(seg_slice, alpha=overlay_opacity)
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
        tumor_str = ", ".join(tumor_types) if tumor_types else "None"

        ax.set_title(f"{name}  |  Tumor: {tumor_str}", color="white", fontsize=12, pad=8)
        ax.axis("off")

    # Single shared legend
    legend_patches = [
        mpatches.Patch(facecolor=SEG_COLORS[l], edgecolor="white",
                       linewidth=0.5, alpha=0.7, label=SEG_LABELS[l])
        for l in SEG_LABELS
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, facecolor="#1e293b", edgecolor="#334155",
               labelcolor="white", framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                pad_inches=0.1, facecolor="#0f172a")
    plt.close(fig)
    buf.seek(0)
    return buf


def has_nifti_data(patient_id):
    mri_path, seg_path = _find_nifti_files(patient_id)
    return mri_path is not None and seg_path is not None


# ═══════════════════════════════════════════════════════════════
# PNG + Mask-based MRI rendering (lightweight, no NIfTI needed)
# ═══════════════════════════════════════════════════════════════

def has_plane_data(patient_id, plane):
    """Check if PNG + mask PNG exists for a specific plane."""
    d = os.path.join(PREVIEW_DIR, patient_id)
    return (os.path.exists(os.path.join(d, f"{plane}.png")) and
            os.path.exists(os.path.join(d, f"{plane}_mask.png")))


def load_plane_image(patient_id, plane):
    """Load grayscale PNG as float32 array [0, 1]. Returns (H, W) or None."""
    path = os.path.join(PREVIEW_DIR, patient_id, f"{plane}.png")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path)).astype(np.float32) / 255.0


def load_plane_mask(patient_id, plane):
    """Load mask PNG. Returns (H, W) uint8 with BraTS labels 0/1/2/4."""
    path = os.path.join(PREVIEW_DIR, patient_id, f"{plane}_mask.png")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def render_plane_composite(patient_id, plane, opacity=0.55):
    """Create RGB composite: grayscale MRI + mask overlay at given opacity.
    Returns (H, W, 3) uint8 array or None."""
    base = load_plane_image(patient_id, plane)
    if base is None:
        return None
    h, w = base.shape

    # Start with grayscale RGB
    rgb = np.stack([base, base, base], axis=-1)

    # Apply mask overlay with opacity
    mask = load_plane_mask(patient_id, plane)
    if mask is not None and opacity > 0:
        for brats_label, color in SEG_COLORS.items():
            m = mask == brats_label
            if m.any():
                rgb[m, 0] = rgb[m, 0] * (1 - opacity) + color[0] * opacity
                rgb[m, 1] = rgb[m, 1] * (1 - opacity) + color[1] * opacity
                rgb[m, 2] = rgb[m, 2] * (1 - opacity) + color[2] * opacity

    return (rgb * 255).clip(0, 255).astype(np.uint8)


def create_mri_plotly(composite_rgb, title="", height=450):
    """Create Plotly figure with scroll-to-zoom for MRI display.
    composite_rgb: (H, W, 3) uint8 array."""
    fig = go.Figure()
    fig.add_trace(go.Image(z=composite_rgb, name=title))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9", size=13)),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f172a",
        height=height,
        margin=dict(l=0, r=0, t=35, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", autorange="reversed"),
    )
    return fig
