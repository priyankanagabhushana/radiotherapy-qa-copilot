import nibabel as nib
import numpy as np
import os

np.random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "brats_data")
VOLUME_SHAPE = (240, 240, 155)
VOXEL_SPACING = (1.0, 1.0, 1.0)


def generate_brain_mask(shape):
    center = np.array(shape) / 2.0
    zz, yy, xx = np.mgrid[
        0:shape[0], 0:shape[1], 0:shape[2]
    ]
    dist = np.sqrt(
        ((zz - center[0]) / (shape[0] * 0.35)) ** 2
        + ((yy - center[1]) / (shape[1] * 0.4)) ** 2
        + ((xx - center[2]) / (shape[2] * 0.45)) ** 2
    )
    return (dist < 1.0).astype(np.float32)


def generate_tumor_mask(shape, tumor_center, tumor_radius, irregularity=0.3):
    zz, yy, xx = np.mgrid[
        0:shape[0], 0:shape[1], 0:shape[2]
    ]
    dist = np.sqrt(
        (zz - tumor_center[0]) ** 2
        + (yy - tumor_center[1]) ** 2
        + (xx - tumor_center[2]) ** 2
    )
    noise = np.random.randn(*shape) * irregularity * tumor_radius
    tumor_core = (dist < tumor_radius * 0.5 + noise * 0.3).astype(np.uint8)
    enhancing = ((dist >= tumor_radius * 0.3) & (dist < tumor_radius * 0.7 + noise * 0.2)).astype(np.uint8)
    edema = ((dist >= tumor_radius * 0.5) & (dist < tumor_radius + noise * 0.5)).astype(np.uint8)
    seg = np.zeros(shape, dtype=np.uint16)
    seg[edema == 1] = 2
    seg[enhancing == 1] = 4
    seg[tumor_core == 1] = 1
    return seg


def generate_modality(brain_mask, seg_mask, noise_level=0.1, contrast_scale=1.0):
    base = brain_mask * 0.5
    base += seg_mask.astype(np.float32) * 0.15 * contrast_scale
    noise = np.random.randn(*brain_mask.shape).astype(np.float32) * noise_level
    return (base + noise) * brain_mask


PATIENT_CONFIGS = [
    {"id": "BraTS_Real_001", "center": (120, 110, 75),  "radius": 18, "irregularity": 0.25},
    {"id": "BraTS_Real_002", "center": (130, 125, 80),  "radius": 30, "irregularity": 0.45},
    {"id": "BraTS_Real_003", "center": (115, 100, 70),  "radius": 12, "irregularity": 0.15},
    {"id": "BraTS_Real_004", "center": (125, 115, 85),  "radius": 40, "irregularity": 0.55},
    {"id": "BraTS_Real_005", "center": (110, 130, 65),  "radius": 22, "irregularity": 0.35},
]


def generate_patient_data(patient_config):
    pid = patient_config["id"]
    patient_dir = os.path.join(OUTPUT_DIR, pid)
    os.makedirs(patient_dir, exist_ok=True)

    brain_mask = generate_brain_mask(VOLUME_SHAPE)
    seg_mask = generate_tumor_mask(
        VOLUME_SHAPE,
        patient_config["center"],
        patient_config["radius"],
        patient_config["irregularity"],
    )

    affine = np.diag([VOXEL_SPACING[0], VOXEL_SPACING[1], VOXEL_SPACING[2], 1.0]).astype(np.float64)

    for modality, contrast, noise in [
        ("flair", 1.2, 0.08),
        ("t1", 0.9, 0.06),
        ("t1ce", 1.5, 0.07),
        ("t2", 1.0, 0.10),
    ]:
        data = generate_modality(brain_mask, seg_mask, noise_level=noise, contrast_scale=contrast)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, os.path.join(patient_dir, f"{pid}_{modality}.nii.gz"))

    seg_img = nib.Nifti1Image(seg_mask.astype(np.uint16), affine)
    nib.save(seg_img, os.path.join(patient_dir, f"{pid}_seg.nii.gz"))

    return patient_dir


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating BraTS-style NIfTI data in: {OUTPUT_DIR}")
    for cfg in PATIENT_CONFIGS:
        path = generate_patient_data(cfg)
        print(f"  Generated: {path}")
    print(f"\nDone. {len(PATIENT_CONFIGS)} patients generated.")
    print("Each patient has: flair, t1, t1ce, t2 (.nii.gz) + seg mask (.nii.gz)")
