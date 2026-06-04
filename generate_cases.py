import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_CASES = 50

OAR_STRUCTURES = [
    "Brainstem",
    "Optic Chiasm",
    "Left Optic Nerve",
    "Right Optic Nerve",
    "Left Lens",
    "Right Lens",
    "Left Retina",
    "Right Retina",
    "Pituitary Gland",
    "Hippocampus",
    "Cochlea (Left)",
    "Cochlea (Right)",
    "Spinal Cord",
    "Chiasm",
]

TUMOR_VOLUMES = {
    "small": (0.5, 5.0),
    "medium": (5.0, 25.0),
    "large": (25.0, 80.0),
}


def generate_patient_cases(n: int = NUM_CASES) -> pd.DataFrame:
    records = []
    for i in range(1, n + 1):
        patient_id = f"PT-{i:04d}"
        structure = np.random.choice(OAR_STRUCTURES)

        vol_bucket = np.random.choice(["small", "medium", "large"], p=[0.35, 0.45, 0.20])
        vol_lo, vol_hi = TUMOR_VOLUMES[vol_bucket]
        tumor_volume = round(np.random.uniform(vol_lo, vol_hi), 2)

        if structure in ("Optic Chiasm", "Chiasm", "Left Optic Nerve", "Right Optic Nerve"):
            distance = round(np.random.exponential(scale=4.0), 1)
        elif structure in ("Brainstem", "Spinal Cord"):
            distance = round(np.random.exponential(scale=8.0), 1)
        else:
            distance = round(np.random.exponential(scale=10.0), 1)
        distance = max(0.1, min(distance, 40.0))

        if distance < 3.0:
            unc_base = np.random.uniform(0.6, 1.0)
        elif tumor_volume > 25.0:
            unc_base = np.random.uniform(0.4, 0.85)
        else:
            unc_base = np.random.uniform(0.05, 0.6)
        uncertainty = round(min(unc_base + np.random.normal(0, 0.05), 1.0), 3)
        uncertainty = max(0.0, uncertainty)

        records.append({
            "Patient_ID": patient_id,
            "Structure_Name": structure,
            "Tumor_Volume_cc": tumor_volume,
            "Distance_to_OAR_mm": distance,
            "AI_Uncertainty_Score": uncertainty,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_patient_cases()
    out_path = os.path.join(os.path.dirname(__file__), "clinical_queue.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} cases -> {out_path}")
