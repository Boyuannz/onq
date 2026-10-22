# Source lineage: project-owned QAT orchestration, contracts, or tests.
# Upstream integrations are recorded in OnQ/UPSTREAMS.md.
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationRecipe:
    name: str
    backend_dataset: str
    requires_calibration_path: bool
    reasoning_ratio: float
    general_source: str | None


def build_calibration_recipe(name: str) -> CalibrationRecipe:
    if name == "paper_tokens":
        return CalibrationRecipe(name, name, True, 0.0, None)
    if name == "official":
        return CalibrationRecipe(name, "sweep_0.8", False, 0.8, "fineweb_edu")
    if name == "filtered_reasoning":
        return CalibrationRecipe(name, name, True, 0.8, "fineweb_edu")
    if name == "math_train":
        return CalibrationRecipe(name, name, True, 1.0, None)
    if name == "math_ot_mix":
        return CalibrationRecipe(name, name, True, 0.8, "openthoughts")
    raise ValueError(f"unknown Calibration recipe {name!r}")
