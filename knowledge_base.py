KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "optic_chiasm_proximity": {
        "source": "AAPM TG-132: Use of Image Registration and Fusion Algorithms in Radiation Oncology (2017)",
        "excerpt": (
            "When the planning target volume (PTV) is within 3 mm of the optic chiasm, "
            "image registration uncertainty must be carefully evaluated. Misregistration of even 1–2 mm "
            "can result in clinically significant dose deviation to the optic apparatus. "
            "Recommendation: Perform manual review of auto-segmented contours for all structures "
            "within 3 mm of the optic pathways. Visual inspection of the superior and inferior "
            "contour boundaries is mandatory."
        ),
        "risk_category": "Critical",
        "recommended_action": "Mandatory physician contour review with slice-by-slice verification.",
    },
    "optic_nerve_proximity": {
        "source": "ESTRO Guideline on Automated Contouring in Radiotherapy (2020)",
        "excerpt": (
            "The optic nerve is a serial organ with a tolerance dose of approximately 55 Gy. "
            "Auto-segmented contours of the optic nerve must be verified when the AI uncertainty "
            "score exceeds 0.5 or when the structure boundary is within 5 mm of the PTV margin. "
            "Particular attention should be paid to the posterior segment near the optic canal "
            "where CT contrast is limited."
        ),
        "risk_category": "High",
        "recommended_action": "Physician review of posterior nerve boundary; consider MRI co-registration.",
    },
    "brainstem_proximity": {
        "source": "AAPM TG-132 / QUANTEC Brainstem Constraints (2010)",
        "excerpt": (
            "The brainstem maximum point dose should not exceed 54 Gy (conventional fractionation). "
            "When auto-segmented brainstem contours are within 5 mm of the target volume, "
            "registration accuracy must be verified. Auto-contouring algorithms may over-segment "
            "the brainstem at the pontomedullary junction. Recommend manual verification of the "
            "inferior contour extent."
        ),
        "risk_category": "High",
        "recommended_action": "Review inferior brainstem boundary; verify dose-volume histogram constraints.",
    },
    "high_uncertainty": {
        "source": "AI in Radiation Oncology: Best Practice Guidelines (ASTRO, 2023)",
        "excerpt": (
            "An AI uncertainty score above 0.75 indicates regions where the auto-segmentation model "
            "has low confidence. These areas commonly occur at tissue interfaces with poor CT contrast "
            "(e.g., soft tissue–bone boundaries) or in the presence of imaging artifacts. "
            "All contours with uncertainty > 0.75 should undergo mandatory human review before "
            "being used in treatment planning."
        ),
        "risk_category": "Critical",
        "recommended_action": "Flag for mandatory manual review. Do not use auto-contour without verification.",
    },
    "moderate_uncertainty": {
        "source": "AI in Radiation Oncology: Best Practice Guidelines (ASTRO, 2023)",
        "excerpt": (
            "AI uncertainty scores between 0.5 and 0.75 represent moderate confidence regions. "
            "While auto-segmented contours in this range may be acceptable for clinical use, "
            "a spot-check review is recommended, focusing on anatomical boundary regions. "
            "Institutions should maintain audit logs of all moderate-uncertainty contours "
            "accepted without modification."
        ),
        "risk_category": "Moderate",
        "recommended_action": "Spot-check review recommended. Log acceptance decision in QA system.",
    },
    "large_tumor_volume": {
        "source": "ESTRO Guideline on Auto-Contouring Quality Assurance (2020)",
        "excerpt": (
            "Large tumor volumes (> 25 cc) in the brain present increased risk of auto-segmentation "
            "error due to irregular shapes, heterogeneous enhancement patterns, and potential "
            "edema infiltration. The auto-contour should be evaluated for over-segmentation "
            "into adjacent edema regions and under-segmentation at necrotic cores. "
            "Multi-slice review in axial, coronal, and sagittal planes is recommended."
        ),
        "risk_category": "Moderate",
        "recommended_action": "Multi-planar contour review; compare with diagnostic MRI if available.",
    },
    "spinal_cord_proximity": {
        "source": "QUANTEC Review of Spinal Cord Tolerance (2010)",
        "excerpt": (
            "The spinal cord is a serial organ with a strict tolerance dose. Auto-segmented "
            "spinal cord contours near the PTV must be verified for lateral boundary accuracy. "
            "CT-based auto-segmentation may fail to distinguish the cord from the thecal sac. "
            "When distance to PTV < 5 mm, recommend MRI co-registration and manual contour "
            "verification."
        ),
        "risk_category": "High",
        "recommended_action": "MRI co-registration recommended. Verify lateral cord boundaries.",
    },
    "hippocampus_proximity": {
        "source": "NRG Oncology CC001 / Hippocampal Avoidance Guidelines (2018)",
        "excerpt": (
            "Hippocampal avoidance during whole-brain radiotherapy is associated with preservation "
            "of cognitive function. Auto-segmented hippocampal contours have shown systematic "
            "variability of 1.5–2.0 mm at the hippocampal tail. When the hippocampus is a "
            "dose-limiting structure, manual review of the tail and head boundaries is essential. "
            "Consider a 5 mm PRV expansion for planning purposes."
        ),
        "risk_category": "Moderate",
        "recommended_action": "Review hippocampal tail contour; apply institutional PRV margin.",
    },
    "default_low_risk": {
        "source": "Institutional Standard Operating Procedure – Routine QA",
        "excerpt": (
            "For auto-segmented structures with AI uncertainty < 0.5 and adequate distance (> 5 mm) "
            "from the PTV, standard institutional QA protocols apply. A single-physician review "
            "of the contour is sufficient. No additional co-registration or multi-planar review "
            "is required unless clinically indicated."
        ),
        "risk_category": "Low",
        "recommended_action": "Standard physician sign-off. No additional steps required.",
    },
}


def retrieve_guidelines(triggers: list[str]) -> dict[str, dict[str, str]]:
    """Given a list of trigger keys, return matching guideline entries."""
    results = {}
    for trigger in triggers:
        if trigger in KNOWLEDGE_BASE:
            results[trigger] = KNOWLEDGE_BASE[trigger]
    return results


TRIGGER_MAPPING: dict[str, list[str]] = {
    "distance_optic_chiasm": ["optic_chiasm_proximity"],
    "distance_optic_nerve": ["optic_nerve_proximity"],
    "distance_brainstem": ["brainstem_proximity"],
    "distance_spinal_cord": ["spinal_cord_proximity"],
    "distance_hippocampus": ["hippocampus_proximity"],
    "high_uncertainty": ["high_uncertainty"],
    "moderate_uncertainty": ["moderate_uncertainty"],
    "large_tumor": ["large_tumor_volume"],
}


def map_triggers(
    structure_name: str,
    distance_mm: float,
    uncertainty: float,
    tumor_volume: float,
) -> list[str]:
    """Determine which knowledge-base keys apply based on patient parameters."""
    triggers: list[str] = []
    structure_lower = structure_name.lower()

    if distance_mm < 3.0:
        if "chiasm" in structure_lower:
            triggers.append("distance_optic_chiasm")
        elif "optic" in structure_lower and "nerve" in structure_lower:
            triggers.append("distance_optic_nerve")
        elif "brainstem" in structure_lower:
            triggers.append("distance_brainstem")
        elif "spinal" in structure_lower or "cord" in structure_lower:
            triggers.append("distance_spinal_cord")
        elif "hippocampus" in structure_lower:
            triggers.append("distance_hippocampus")

    if uncertainty > 0.75:
        triggers.append("high_uncertainty")
    elif uncertainty > 0.5:
        triggers.append("moderate_uncertainty")

    if tumor_volume > 25.0:
        triggers.append("large_tumor")

    if not triggers:
        triggers.append("default_low_risk")

    return triggers
