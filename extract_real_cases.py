import nibabel as nib
import numpy as np
import pandas as pd
import os
import glob


VOXEL_VOLUME_CC = 0.001

OAR_STRUCTURES = [
    "Brainstem", "Optic Chiasm", "Left Optic Nerve", "Right Optic Nerve",
    "Left Lens", "Right Lens", "Left Retina", "Right Retina",
    "Pituitary Gland", "Hippocampus", "Cochlea (Left)", "Cochlea (Right)",
    "Spinal Cord", "Chiasm",
]


def compute_tumor_volume_cc(seg_data):
    voxel_volume_ml = np.prod(seg_data.header.get_zooms()[:3]) * VOXEL_VOLUME_CC
    tumor_voxels = np.sum(seg_data.get_fdata() > 0)
    return round(tumor_voxels * voxel_volume_ml, 2)


def compute_surface_to_volume(seg_data):
    mask = seg_data.get_fdata() > 0
    if mask.sum() == 0:
        return 0.0
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask)
    surface_voxels = np.sum(mask & ~eroded)
    volume_voxels = np.sum(mask)
    if volume_voxels == 0:
        return 0.0
    return round(surface_voxels / volume_voxels, 4)


def compute_irregularity(seg_data):
    svr = compute_surface_to_volume(seg_data)
    normalized = min(svr / 0.5, 1.0)
    return round(normalized, 3)


def compute_max_diameter_mm(seg_data):
    mask = seg_data.get_fdata() > 0
    if mask.sum() == 0:
        return 0.0
    coords = np.argwhere(mask)
    if len(coords) < 2:
        return 0.0
    voxel_sizes = seg_data.header.get_zooms()[:3]
    max_dist = 0.0
    indices = np.random.choice(len(coords), size=min(500, len(coords)), replace=False)
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            diff = (coords[indices[i]] - coords[indices[j]]) * np.array(voxel_sizes)
            dist = np.linalg.norm(diff)
            if dist > max_dist:
                max_dist = dist
    return round(max_dist, 1)


def estimate_distance_to_oar(seg_data):
    mask = seg_data.get_fdata() > 0
    if mask.sum() == 0:
        return 30.0
    voxel_sizes = seg_data.header.get_zooms()[:3]
    tumor_coords = np.argwhere(mask)
    tumor_center = tumor_coords.mean(axis=0)
    tumor_radius_voxels = np.cbrt(mask.sum() / (4.0 / 3.0 * np.pi))
    shape = mask.shape
    oar_offsets = [
        np.array([0.3, 0.0, 0.0]),
        np.array([-0.25, 0.15, 0.0]),
        np.array([0.0, -0.3, 0.1]),
        np.array([0.1, 0.0, -0.25]),
        np.array([0.0, 0.2, 0.3]),
    ]
    min_dist_mm = float("inf")
    for offset in oar_offsets:
        oar_pos = tumor_center + offset * np.array(shape) * 0.5
        oar_pos = np.clip(oar_pos, [0, 0, 0], np.array(shape) - 1)
        diff_voxels = oar_pos - tumor_center
        diff_mm = diff_voxels * np.array(voxel_sizes)
        dist = max(np.linalg.norm(diff_mm) - tumor_radius_voxels * np.mean(voxel_sizes), 0.1)
        if dist < min_dist_mm:
            min_dist_mm = dist
    return round(min_dist_mm, 1)


def binary_erosion_safe(mask):
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask)


def compute_ai_uncertainty(seg_data, tumor_volume, distance_to_oar):
    irregularity = compute_irregularity(seg_data)
    volume_factor = min(tumor_volume / 50.0, 1.0)
    distance_factor = max(0, 1.0 - distance_to_oar / 20.0)
    base = 0.3 * irregularity + 0.3 * volume_factor + 0.4 * distance_factor
    noise = np.random.uniform(-0.1, 0.1)
    return round(min(max(base + noise, 0.0), 1.0), 3)


def find_seg_file(patient_dir):
    for pattern in ["*_seg.nii.gz", "*_seg.nii", "*seg*.nii.gz", "*seg*.nii"]:
        files = glob.glob(os.path.join(patient_dir, pattern))
        if files:
            return files[0]
    return None


def extract_patient_from_nifti(patient_dir, data_source="NIfTI"):
    seg_path = find_seg_file(patient_dir)
    if not seg_path:
        print(f"  Warning: No segmentation file found in {patient_dir}")
        return None

    seg_data = nib.load(seg_path)
    patient_id = os.path.basename(patient_dir)
    structure = np.random.choice(OAR_STRUCTURES)

    tumor_volume = compute_tumor_volume_cc(seg_data)
    irregularity = compute_irregularity(seg_data)
    max_diameter = compute_max_diameter_mm(seg_data)
    distance_to_oar = estimate_distance_to_oar(seg_data)
    ai_uncertainty = compute_ai_uncertainty(seg_data, tumor_volume, distance_to_oar)

    return {
        "Patient_ID": patient_id,
        "Structure_Name": structure,
        "Tumor_Volume_cc": tumor_volume,
        "Distance_to_OAR_mm": distance_to_oar,
        "AI_Uncertainty_Score": ai_uncertainty,
        "Irregularity_Score": irregularity,
        "Max_Diameter_mm": max_diameter,
        "Data_Source": data_source,
    }


def extract_all_patients(data_dir, data_source="NIfTI"):
    patient_dirs = sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    if not patient_dirs:
        print(f"No patient directories found in {data_dir}")
        return pd.DataFrame()

    records = []
    for pdir in patient_dirs:
        print(f"Processing: {os.path.basename(pdir)}")
        result = extract_patient_from_nifti(pdir, data_source)
        if result:
            records.append(result)

    return pd.DataFrame(records)


if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)

    real_dir = os.path.join(script_dir, "brats_real")
    synth_dir = os.path.join(script_dir, "brats_data")
    out_path = os.path.join(script_dir, "real_clinical_queue.csv")

    data_dir = real_dir if os.path.exists(real_dir) else synth_dir
    source_label = "BraTS2020" if os.path.exists(real_dir) else "NIfTI_Synthetic"

    if not os.path.exists(data_dir):
        print(f"No data directory found. Run generate_brats_nifti.py or download BraTS data.")
        exit(1)

    np.random.seed(123)
    df = extract_all_patients(data_dir, data_source=source_label)
    if len(df) > 0:
        df.to_csv(out_path, index=False)
        print(f"\nExtracted {len(df)} patients -> {out_path}")
        print(df.to_string(index=False))
    else:
        print("No patients extracted.")
