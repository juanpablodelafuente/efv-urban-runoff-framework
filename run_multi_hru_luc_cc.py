"""
Multi-HRU land-use and climate-change scenario framework for surface-subsurface
runoff response analysis.

This script is designed as Script 2, building on the stable reference engine in:

    run_infiltration_comparison.py

Scientific purpose
------------------
Script 1 compares infiltration/loss methods under a single soil and a controlled
storm-routing framework. This Script 2 adds a catchment-composition layer so that
land-use change (LUC) and climate-change/storm-stress (CC) scenarios can be tested
without changing the core infiltration and routing logic.

The central modelling idea is:

    climate-stress storm
        -> HRU-specific rainfall losses / infiltration / runoff excess
        -> area-weighted catchment-average excess rainfall
        -> common unit-hydrograph routing
        -> hydrograph and scenario metrics

The HRU layer allows the catchment to be represented as a mixture of:
    - connected impervious surface
    - compacted pervious urban soil
    - loam pervious soil
    - green-infrastructure / engineered soil
    - clay pervious soil

Land-use change is represented by changing the HRU area fractions.
Climate change is represented by changing storm depth, duration, temporal pattern,
and antecedent moisture condition.

Important modelling convention
------------------------------
For pervious HRUs, this script uses the same infiltration methods available in
Script 1: Green-Ampt, Horton, SCS-CN, Philip, and EFV. For impervious HRUs, it uses
a simple impervious runoff abstraction with optional depression storage and runoff
coefficient.

For HRUs with depression/storage capacity, storage is applied to generated excess
rainfall before routing. Stored water is treated as event-scale retained water, not
routed surface runoff. This is a deliberately simple representation of local surface
storage / GI retention and should be described as a conceptual scenario assumption.

Outputs
-------
The script saves four CSV files in a folder called multi_hru_outputs:
    1) multi_hru_summary.csv
    2) multi_hru_excess_series.csv
    3) multi_hru_hydrographs.csv
    4) multi_hru_hru_contributions.csv

It also saves two groups of PNG figures:
    MAIN figures:
        - one selected method across climate scenarios, for a fixed land-use composition
        - one selected method across land-use compositions, for a fixed climate scenario
        - peak-flow change vs baseline, for the selected method
        - runoff-volume change vs baseline, for the selected method

    OPTIONAL diagnostic figures:
        - method comparison for one fixed land-use + climate scenario
        - EFV-relative method-bias plots

The default main method is EFV, but this can be changed by editing main_plot_method_key.

How to run
----------
Place this file in the SAME folder as:

    run_infiltration_comparison.py

Then run this file in Spyder. The import block below automatically imports the
stable functions/classes from Script 1 without executing Script 1's main block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Optional, Tuple
import sys
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================================
# 0. IMPORT THE STABLE SCRIPT-1 ENGINE
# ======================================================================================

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # Fallback for interactive notebooks/cells where __file__ may not exist.
    SCRIPT_DIR = Path.cwd()

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from run_infiltration_comparison import (
        SOILS,
        SoilScenario,
        CatchmentScenario,
        ExperimentConfig,
        build_storm_from_pattern,
        hyetograph_from_coefficients,
        run_method_scs,
        run_method_horton,
        run_method_philip,
        run_method_green_ampt,
        run_method_efv,
        triangular_unit_hydrograph,
        convolve_excess_with_uh,
        hydrograph_metrics,
    )
except Exception as exc:
    raise ImportError(
        "Could not import the required functions/classes from "
        "run_infiltration_comparison.py.\n\n"
        "Make sure this Script 2 file is saved in the SAME folder as "
        "run_infiltration_comparison.py, and that Script 1 runs correctly by itself.\n\n"
        f"Original import error:\n{exc}"
    ) from exc


# ======================================================================================
# 1. SMALL NUMERICAL HELPERS
# ======================================================================================

def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """
    Version-safe trapezoidal integration.

    Some Python environments have np.trapezoid, while older NumPy versions only have
    np.trapz. This helper avoids the environment-dependent error you encountered before.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


def safe_relative_value(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """Return numerator / denominator, or NaN if the denominator is too close to zero."""
    if abs(float(denominator)) <= eps:
        return float("nan")
    return float(numerator / denominator)


def method_display_name(method_name: str) -> str:
    """Convert internal method key into a clean label used in outputs."""
    method_name = method_name.lower()
    if method_name in ["ga", "green-ampt", "green_ampt"]:
        return "Green-Ampt"
    if method_name == "horton":
        return "Horton"
    if method_name == "scs":
        return "SCS-CN"
    if method_name == "philip":
        return "Philip"
    if method_name == "efv":
        return "EFV"
    raise ValueError("Unknown method. Use one of: ga, horton, scs, philip, efv")


def validate_time_series_compatible(df_a: pd.DataFrame, df_b: pd.DataFrame) -> None:
    """Check that two pulse time series have the same time grid."""
    if len(df_a) != len(df_b):
        raise ValueError("Time series have different lengths.")

    a = df_a["time_min"].to_numpy(dtype=float)
    b = df_b["time_min"].to_numpy(dtype=float)

    if not np.allclose(a, b):
        raise ValueError("Time series have different time_min values.")


# ======================================================================================
# 2. HRU, LAND-USE, AND CLIMATE DATA STRUCTURES
# ======================================================================================

@dataclass
class HRUDefinition:
    """
    Hydrological Response Unit (HRU) definition.

    Parameters
    ----------
    key : str
        Stable identifier used in CSV outputs.

    name : str
        Human-readable HRU name.

    area_fraction : float
        Fraction of the total catchment area assigned to this HRU. Fractions in a
        catchment composition must sum to 1.0.

    hru_type : str
        Either "impervious" or "pervious".

    soil_key : Optional[str]
        Required for pervious HRUs. Must exist in SOILS from Script 1.

    depression_storage_mm : float
        Event-scale local storage depth [mm]. For impervious HRUs, this represents
        small surface depression storage. For GI/pervious HRUs, this can represent
        local surface storage or retention capacity before runoff is routed.

    runoff_coefficient : float
        Used only for impervious HRUs. A value of 1.0 means all rainfall beyond
        depression storage becomes runoff. Values below 1.0 represent additional
        local losses/disconnection.
    """
    key: str
    name: str
    area_fraction: float
    hru_type: str
    soil_key: Optional[str] = None
    depression_storage_mm: float = 0.0
    runoff_coefficient: float = 1.0

    def validate(self) -> None:
        """Validate the HRU definition."""
        if self.area_fraction < 0.0:
            raise ValueError(f"HRU '{self.key}' has a negative area fraction.")

        if self.hru_type not in ["impervious", "pervious"]:
            raise ValueError(f"HRU '{self.key}' hru_type must be 'impervious' or 'pervious'.")

        if self.depression_storage_mm < 0.0:
            raise ValueError(f"HRU '{self.key}' has negative depression storage.")

        if not (0.0 <= self.runoff_coefficient <= 1.0):
            raise ValueError(f"HRU '{self.key}' runoff_coefficient must be between 0 and 1.")

        if self.hru_type == "pervious":
            if self.soil_key is None:
                raise ValueError(f"Pervious HRU '{self.key}' must define soil_key.")
            if self.soil_key not in SOILS:
                raise ValueError(f"Pervious HRU '{self.key}' uses unknown soil_key '{self.soil_key}'.")


@dataclass
class CatchmentComposition:
    """
    A land-use / land-cover catchment composition expressed as HRU area fractions.
    """
    key: str
    name: str
    description: str
    hrus: List[HRUDefinition]

    def validate(self, tol: float = 1e-8) -> None:
        """Validate HRUs and check that area fractions sum to 1.0."""
        if len(self.hrus) == 0:
            raise ValueError(f"Composition '{self.key}' has no HRUs.")

        for hru in self.hrus:
            hru.validate()

        total = sum(hru.area_fraction for hru in self.hrus)
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"Composition '{self.key}' area fractions must sum to 1.0. "
                f"Current sum = {total:.12f}"
            )

    def descriptors(self) -> Dict[str, float]:
        """Return useful area-fraction descriptors for summary tables."""
        self.validate()

        impervious_fraction = sum(
            h.area_fraction for h in self.hrus if h.hru_type == "impervious"
        )
        pervious_fraction = 1.0 - impervious_fraction
        gi_fraction = sum(
            h.area_fraction for h in self.hrus if h.soil_key == "gi"
        )
        compacted_fraction = sum(
            h.area_fraction for h in self.hrus if h.soil_key == "compacted"
        )
        loam_fraction = sum(
            h.area_fraction for h in self.hrus if h.soil_key == "loam"
        )
        clay_fraction = sum(
            h.area_fraction for h in self.hrus if h.soil_key == "clay"
        )

        weighted_storage_mm = sum(
            h.area_fraction * h.depression_storage_mm for h in self.hrus
        )

        return {
            "impervious_fraction": float(impervious_fraction),
            "pervious_fraction": float(pervious_fraction),
            "gi_fraction": float(gi_fraction),
            "compacted_fraction": float(compacted_fraction),
            "loam_fraction": float(loam_fraction),
            "clay_fraction": float(clay_fraction),
            "area_weighted_storage_mm": float(weighted_storage_mm),
        }


@dataclass
class ClimateScenario:
    """
    Climate/storm-stress scenario.

    This framework uses scenario-level changes in:
        - storm total depth,
        - storm duration,
        - storm temporal pattern,
        - antecedent moisture condition.

    These are not formal UKCP projections by themselves. They are controlled climate
    stress tests that let you ask how sensitive the surface response is to plausible
    rainfall intensification and wetness changes.
    """
    key: str
    name: str
    description: str
    storm_duration_min: int
    total_depth_mm: float
    pattern: str
    antecedent: str
    custom_depths_mm: Optional[List[float]] = None

    def validate(self, dt_min: int) -> None:
        """Validate the climate scenario."""
        if self.custom_depths_mm is None:
            if self.storm_duration_min <= 0:
                raise ValueError(f"Climate scenario '{self.key}' has non-positive duration.")
            if self.storm_duration_min % dt_min != 0:
                raise ValueError(
                    f"Climate scenario '{self.key}' duration must be a multiple of dt_min."
                )
            if self.total_depth_mm <= 0.0:
                raise ValueError(f"Climate scenario '{self.key}' has non-positive depth.")
        else:
            if len(self.custom_depths_mm) == 0:
                raise ValueError(f"Climate scenario '{self.key}' custom_depths_mm is empty.")
            if any(p < 0.0 for p in self.custom_depths_mm):
                raise ValueError(f"Climate scenario '{self.key}' has negative custom rainfall.")

        if self.antecedent not in ["dry", "avg", "wet"]:
            raise ValueError(f"Climate scenario '{self.key}' antecedent must be dry, avg, or wet.")


# ======================================================================================
# 3. LAND-USE / LAND-COVER SCENARIO LIBRARY
# ======================================================================================

COMPOSITIONS: Dict[str, CatchmentComposition] = {
    "baseline_urban": CatchmentComposition(
        key="baseline_urban",
        name="Baseline urban catchment",
        description=(
            "Mixed urban catchment with connected impervious area, compacted pervious "
            "soil, ordinary loam pervious area, and a modest GI fraction."
        ),
        hrus=[
            HRUDefinition(
                key="impervious_connected",
                name="Connected impervious surface",
                area_fraction=0.50,
                hru_type="impervious",
                depression_storage_mm=1.5,
                runoff_coefficient=0.98,
            ),
            HRUDefinition(
                key="compacted_pervious",
                name="Compacted pervious urban soil",
                area_fraction=0.20,
                hru_type="pervious",
                soil_key="compacted",
                depression_storage_mm=2.0,
            ),
            HRUDefinition(
                key="loam_pervious",
                name="Loam pervious soil",
                area_fraction=0.20,
                hru_type="pervious",
                soil_key="loam",
                depression_storage_mm=3.0,
            ),
            HRUDefinition(
                key="gi_soil",
                name="Green infrastructure / engineered soil",
                area_fraction=0.10,
                hru_type="pervious",
                soil_key="gi",
                depression_storage_mm=15.0,
            ),
        ],
    ),

    "densified_urban": CatchmentComposition(
        key="densified_urban",
        name="Densified urban catchment",
        description=(
            "Land-use change scenario with larger connected impervious fraction and "
            "reduced GI/pervious area."
        ),
        hrus=[
            HRUDefinition(
                key="impervious_connected",
                name="Connected impervious surface",
                area_fraction=0.70,
                hru_type="impervious",
                depression_storage_mm=1.2,
                runoff_coefficient=0.99,
            ),
            HRUDefinition(
                key="compacted_pervious",
                name="Compacted pervious urban soil",
                area_fraction=0.20,
                hru_type="pervious",
                soil_key="compacted",
                depression_storage_mm=1.5,
            ),
            HRUDefinition(
                key="loam_pervious",
                name="Loam pervious soil",
                area_fraction=0.08,
                hru_type="pervious",
                soil_key="loam",
                depression_storage_mm=2.0,
            ),
            HRUDefinition(
                key="gi_soil",
                name="Green infrastructure / engineered soil",
                area_fraction=0.02,
                hru_type="pervious",
                soil_key="gi",
                depression_storage_mm=8.0,
            ),
        ],
    ),

    "gi_retrofit": CatchmentComposition(
        key="gi_retrofit",
        name="GI retrofit catchment",
        description=(
            "Retrofit scenario with reduced connected imperviousness and increased "
            "engineered/GI soil fraction."
        ),
        hrus=[
            HRUDefinition(
                key="impervious_connected",
                name="Connected impervious surface",
                area_fraction=0.40,
                hru_type="impervious",
                depression_storage_mm=1.5,
                runoff_coefficient=0.95,
            ),
            HRUDefinition(
                key="compacted_pervious",
                name="Compacted pervious urban soil",
                area_fraction=0.15,
                hru_type="pervious",
                soil_key="compacted",
                depression_storage_mm=2.0,
            ),
            HRUDefinition(
                key="loam_pervious",
                name="Loam pervious soil",
                area_fraction=0.25,
                hru_type="pervious",
                soil_key="loam",
                depression_storage_mm=3.0,
            ),
            HRUDefinition(
                key="gi_soil",
                name="Green infrastructure / engineered soil",
                area_fraction=0.20,
                hru_type="pervious",
                soil_key="gi",
                depression_storage_mm=25.0,
            ),
        ],
    ),

    "degraded_gi": CatchmentComposition(
        key="degraded_gi",
        name="Degraded GI / compacted urban catchment",
        description=(
            "Scenario where GI/pervious areas exist but are partly degraded or compacted, "
            "reducing effective infiltration/storage."
        ),
        hrus=[
            HRUDefinition(
                key="impervious_connected",
                name="Connected impervious surface",
                area_fraction=0.55,
                hru_type="impervious",
                depression_storage_mm=1.2,
                runoff_coefficient=0.98,
            ),
            HRUDefinition(
                key="compacted_pervious",
                name="Compacted pervious urban soil",
                area_fraction=0.30,
                hru_type="pervious",
                soil_key="compacted",
                depression_storage_mm=1.5,
            ),
            HRUDefinition(
                key="loam_pervious",
                name="Loam pervious soil",
                area_fraction=0.10,
                hru_type="pervious",
                soil_key="loam",
                depression_storage_mm=2.0,
            ),
            HRUDefinition(
                key="gi_soil",
                name="Green infrastructure / engineered soil",
                area_fraction=0.05,
                hru_type="pervious",
                soil_key="gi",
                depression_storage_mm=8.0,
            ),
        ],
    ),

    "clay_dominated": CatchmentComposition(
        key="clay_dominated",
        name="Clay-dominated neighbourhood",
        description=(
            "Urbanised neighbourhood with a large low-permeability clay pervious fraction."
        ),
        hrus=[
            HRUDefinition(
                key="impervious_connected",
                name="Connected impervious surface",
                area_fraction=0.45,
                hru_type="impervious",
                depression_storage_mm=1.5,
                runoff_coefficient=0.98,
            ),
            HRUDefinition(
                key="clay_pervious",
                name="Clay pervious soil",
                area_fraction=0.35,
                hru_type="pervious",
                soil_key="clay",
                depression_storage_mm=2.0,
            ),
            HRUDefinition(
                key="loam_pervious",
                name="Loam pervious soil",
                area_fraction=0.15,
                hru_type="pervious",
                soil_key="loam",
                depression_storage_mm=3.0,
            ),
            HRUDefinition(
                key="gi_soil",
                name="Green infrastructure / engineered soil",
                area_fraction=0.05,
                hru_type="pervious",
                soil_key="gi",
                depression_storage_mm=15.0,
            ),
        ],
    ),
}


# ======================================================================================
# 4. CLIMATE / STORM-STRESS SCENARIO LIBRARY
# ======================================================================================

CLIMATE_SCENARIOS: Dict[str, ClimateScenario] = {
    "present": ClimateScenario(
        key="present",
        name="Present reference storm",
        description="Reference present-style design storm with average antecedent moisture.",
        storm_duration_min=120,
        total_depth_mm=80.0,
        pattern="center_peaked",
        antecedent="avg",
    ),

    "intensified_depth": ClimateScenario(
        key="intensified_depth",
        name="Climate stress: increased storm depth",
        description="Same duration and temporal structure as present, but larger total depth.",
        storm_duration_min=120,
        total_depth_mm=125.0,
        pattern="center_peaked",
        antecedent="avg",
    ),

    "intensified_peakiness": ClimateScenario(
        key="intensified_peakiness",
        name="Climate stress: peakier storm",
        description="Same duration and total depth as present, but more rainfall concentrated near the centre.",
        storm_duration_min=120,
        total_depth_mm=80.0,
        pattern="sharp_center_peaked",
        antecedent="avg",
    ),

    "shorter_more_intense": ClimateScenario(
        key="shorter_more_intense",
        name="Climate stress: shorter, more intense storm",
        description="Same total depth as present compressed into a shorter duration.",
        storm_duration_min=60,
        total_depth_mm=80.0,
        pattern="center_peaked",
        antecedent="avg",
    ),

    "wetter_antecedent": ClimateScenario(
        key="wetter_antecedent",
        name="Climate stress: wetter antecedent condition",
        description="Same storm as present but with wet antecedent soil moisture.",
        storm_duration_min=120,
        total_depth_mm=80.0,
        pattern="center_peaked",
        antecedent="wet",
    ),
}


# ======================================================================================
# 5. STORM GENERATION FOR EXTENDED CLIMATE PATTERNS
# ======================================================================================

STANDARD_PATTERNS = {"constant", "center_peaked", "front_loaded", "back_loaded"}


def build_extended_dimensionless_pattern(n_pulses: int, pattern: str) -> np.ndarray:
    """
    Build dimensionless storm pattern for Script 2.

    For standard patterns, Script 1 is used through build_storm_from_pattern(). This
    function only handles additional patterns required by the LUC/CC extension.
    """
    if n_pulses <= 0:
        raise ValueError("n_pulses must be positive.")

    x = np.arange(n_pulses, dtype=float)

    if pattern == "sharp_center_peaked":
        # Gaussian-like central peak. The small background value avoids exact zero tails.
        center = 0.5 * (n_pulses - 1)
        sigma = max(n_pulses / 8.0, 0.75)
        w = np.exp(-0.5 * ((x - center) / sigma) ** 2) + 0.02
    else:
        raise ValueError(
            "Unknown extended pattern. Use one of the standard patterns or 'sharp_center_peaked'."
        )

    return w / w.sum()


def build_storm_for_climate(climate: ClimateScenario, dt_min: int) -> pd.DataFrame:
    """Build a storm DataFrame for a climate/storm-stress scenario."""
    climate.validate(dt_min)

    if climate.custom_depths_mm is not None:
        depths = np.asarray(climate.custom_depths_mm, dtype=float)
        time_min = np.arange(1, len(depths) + 1) * dt_min
        storm_df = pd.DataFrame({"time_min": time_min, "P_mm": depths})
        storm_df["i_mm_h"] = storm_df["P_mm"] * 60.0 / dt_min
        return storm_df

    if climate.pattern in STANDARD_PATTERNS:
        return build_storm_from_pattern(
            storm_duration_min=climate.storm_duration_min,
            total_depth_mm=climate.total_depth_mm,
            dt_min=dt_min,
            pattern=climate.pattern,
        )

    n_pulses = int(climate.storm_duration_min / dt_min)
    coeff = build_extended_dimensionless_pattern(n_pulses, climate.pattern)

    return hyetograph_from_coefficients(
        coefficients=coeff.tolist(),
        total_depth_mm=climate.total_depth_mm,
        dt_min=dt_min,
        normalise=True,
    )


def storm_summary(storm_df: pd.DataFrame) -> Dict[str, float]:
    """Return storm descriptors independent of any soil/HRU."""
    total_depth_mm = float(storm_df["P_mm"].sum())
    duration_min = float(storm_df["time_min"].iloc[-1])
    duration_h = duration_min / 60.0
    mean_i = total_depth_mm / duration_h if duration_h > 0.0 else float("nan")
    peak_i = float(storm_df["i_mm_h"].max())

    return {
        "storm_total_depth_mm_check": total_depth_mm,
        "storm_duration_min_check": duration_min,
        "mean_intensity_mm_h": mean_i,
        "peak_intensity_mm_h": peak_i,
    }


# ======================================================================================
# 6. HRU-LEVEL RUNOFF GENERATION
# ======================================================================================

def apply_event_storage_to_excess(method_df: pd.DataFrame, storage_mm: float) -> pd.DataFrame:
    """
    Apply finite event-scale storage to generated excess rainfall.

    The input method_df contains rainfall, infiltration/loss, and excess rainfall. This
    function removes excess rainfall until the HRU storage capacity is filled. It then
    recomputes cumulative excess and cumulative infiltration/loss.

    This represents local surface storage / GI retention before runoff is routed.
    """
    out = method_df.copy().reset_index(drop=True)

    if storage_mm <= 0.0:
        out["storage_retained_mm"] = 0.0
        out["storage_remaining_mm"] = 0.0
        return out

    remaining_storage = float(storage_mm)
    new_excess = []
    retained = []
    remaining_values = []

    for excess in out["excess_mm"].to_numpy(dtype=float):
        retained_now = min(max(excess, 0.0), remaining_storage)
        routed_excess = max(excess - retained_now, 0.0)
        remaining_storage -= retained_now

        retained.append(retained_now)
        new_excess.append(routed_excess)
        remaining_values.append(remaining_storage)

    out["storage_retained_mm"] = np.asarray(retained, dtype=float)
    out["storage_remaining_mm"] = np.asarray(remaining_values, dtype=float)
    out["excess_mm_before_storage"] = out["excess_mm"].to_numpy(dtype=float)
    out["excess_mm"] = np.asarray(new_excess, dtype=float)

    # In the event-scale accounting, anything not routed as excess is treated as a
    # rainfall loss/retention term. This keeps P = infiltration/loss + routed excess.
    out["infil_mm"] = out["P_mm"].to_numpy(dtype=float) - out["excess_mm"].to_numpy(dtype=float)
    out["cum_infil_mm"] = out["infil_mm"].cumsum()
    out["cum_excess_mm"] = out["excess_mm"].cumsum()

    return out


def run_impervious_hru(storm_df: pd.DataFrame, hru: HRUDefinition) -> pd.DataFrame:
    """
    Compute runoff excess from an impervious HRU.

    Rainfall first fills depression storage. Rainfall exceeding remaining storage becomes
    runoff according to the HRU runoff coefficient.
    """
    hru.validate()
    if hru.hru_type != "impervious":
        raise ValueError("run_impervious_hru requires an impervious HRU.")

    out = storm_df.copy().reset_index(drop=True)

    remaining_storage = float(hru.depression_storage_mm)
    excess_values = []
    loss_values = []
    retained_values = []
    remaining_values = []

    for P in out["P_mm"].to_numpy(dtype=float):
        retained_now = min(max(P, 0.0), remaining_storage)
        available_after_storage = max(P - retained_now, 0.0)
        excess_now = available_after_storage * hru.runoff_coefficient
        extra_loss_now = available_after_storage * (1.0 - hru.runoff_coefficient)
        loss_now = retained_now + extra_loss_now

        remaining_storage -= retained_now

        retained_values.append(retained_now)
        remaining_values.append(remaining_storage)
        excess_values.append(excess_now)
        loss_values.append(loss_now)

    out["method"] = "Impervious abstraction"
    out["infil_mm"] = np.asarray(loss_values, dtype=float)
    out["excess_mm"] = np.asarray(excess_values, dtype=float)
    out["storage_retained_mm"] = np.asarray(retained_values, dtype=float)
    out["storage_remaining_mm"] = np.asarray(remaining_values, dtype=float)
    out["cum_infil_mm"] = out["infil_mm"].cumsum()
    out["cum_excess_mm"] = out["excess_mm"].cumsum()

    return out


def run_pervious_hru(
    method_name: str,
    storm_df: pd.DataFrame,
    hru: HRUDefinition,
    antecedent: str,
    dt_min: int,
    experiment: ExperimentConfig,
) -> pd.DataFrame:
    """Run the chosen infiltration/loss method for a pervious HRU."""
    hru.validate()
    if hru.hru_type != "pervious":
        raise ValueError("run_pervious_hru requires a pervious HRU.")

    soil_key = hru.soil_key
    assert soil_key is not None
    soil = SOILS[soil_key]

    method = method_name.lower()

    if method == "scs":
        out = run_method_scs(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_scs_params=None,
        )
    elif method == "horton":
        out = run_method_horton(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_horton_params=None,
        )
    elif method == "philip":
        out = run_method_philip(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_philip_params=None,
        )
    elif method in ["ga", "green-ampt", "green_ampt"]:
        out = run_method_green_ampt(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
        )
    elif method == "efv":
        out = run_method_efv(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            experiment=experiment,
        )
    else:
        raise ValueError("Unknown method. Use one of: scs, horton, philip, ga, efv")

    # Apply HRU-specific event-scale storage/retention after runoff generation.
    out = apply_event_storage_to_excess(out, storage_mm=hru.depression_storage_mm)

    return out


def run_hru_response(
    method_name: str,
    storm_df: pd.DataFrame,
    hru: HRUDefinition,
    climate: ClimateScenario,
    dt_min: int,
    experiment: ExperimentConfig,
) -> pd.DataFrame:
    """Run either impervious or pervious HRU response and add HRU metadata."""
    if hru.hru_type == "impervious":
        out = run_impervious_hru(storm_df, hru)
    elif hru.hru_type == "pervious":
        out = run_pervious_hru(
            method_name=method_name,
            storm_df=storm_df,
            hru=hru,
            antecedent=climate.antecedent,
            dt_min=dt_min,
            experiment=experiment,
        )
    else:
        raise ValueError(f"Unknown HRU type: {hru.hru_type}")

    out = out.copy().reset_index(drop=True)
    out["hru_key"] = hru.key
    out["hru_name"] = hru.name
    out["hru_type"] = hru.hru_type
    out["hru_area_fraction"] = hru.area_fraction
    out["hru_soil_key"] = hru.soil_key if hru.soil_key is not None else "none"
    out["hru_storage_mm"] = hru.depression_storage_mm
    out["hru_runoff_coefficient"] = hru.runoff_coefficient

    out["weighted_excess_mm"] = out["excess_mm"].to_numpy(dtype=float) * hru.area_fraction
    out["weighted_infil_mm"] = out["infil_mm"].to_numpy(dtype=float) * hru.area_fraction

    return out


# ======================================================================================
# 7. CATCHMENT-LEVEL HRU COMBINATION AND ROUTING
# ======================================================================================

def combine_hru_outputs(
    storm_df: pd.DataFrame,
    hru_outputs: List[pd.DataFrame],
) -> pd.DataFrame:
    """Combine HRU outputs into a catchment-average excess rainfall time series."""
    if len(hru_outputs) == 0:
        raise ValueError("No HRU outputs were supplied for combination.")

    for hru_df in hru_outputs:
        validate_time_series_compatible(storm_df, hru_df)

    total_weighted_excess = np.zeros(len(storm_df), dtype=float)
    total_weighted_infil = np.zeros(len(storm_df), dtype=float)

    for hru_df in hru_outputs:
        total_weighted_excess += hru_df["weighted_excess_mm"].to_numpy(dtype=float)
        total_weighted_infil += hru_df["weighted_infil_mm"].to_numpy(dtype=float)

    out = storm_df.copy().reset_index(drop=True)
    out["excess_mm"] = total_weighted_excess
    out["infil_mm"] = total_weighted_infil
    out["cum_excess_mm"] = out["excess_mm"].cumsum()
    out["cum_infil_mm"] = out["infil_mm"].cumsum()

    # This should be close to rainfall depth if HRU area fractions sum to 1.0 and all
    # rainfall is either excess or retained/lost.
    out["water_accounting_residual_mm"] = (
        out["P_mm"].to_numpy(dtype=float)
        - out["excess_mm"].to_numpy(dtype=float)
        - out["infil_mm"].to_numpy(dtype=float)
    )

    return out


def route_catchment_excess(
    catchment_excess_df: pd.DataFrame,
    catchment: CatchmentScenario,
    dt_min: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Route catchment-average excess rainfall through the common unit hydrograph."""
    uh_df = triangular_unit_hydrograph(catchment, dt_min)

    Q = convolve_excess_with_uh(
        excess_mm=catchment_excess_df["excess_mm"].to_numpy(dtype=float),
        uh_m3s_per_mm=uh_df["UH_m3s_per_mm"].to_numpy(dtype=float),
        baseflow_m3_s=catchment.baseflow_m3_s,
    )

    hydro_time_min = np.arange(len(Q), dtype=float) * dt_min

    hydro_df = pd.DataFrame({
        "time_min": hydro_time_min,
        "Q_m3_s": Q,
    })

    metrics = hydrograph_metrics(
        time_min=hydro_time_min,
        Q_m3_s=Q,
    )

    return hydro_df, metrics


def run_composition_method_climate(
    composition: CatchmentComposition,
    climate: ClimateScenario,
    method_name: str,
    storm_df: pd.DataFrame,
    catchment: CatchmentScenario,
    dt_min: int,
    experiment: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[pd.DataFrame]]:
    """
    Run one method for one land-use composition and one climate scenario.

    Returns
    -------
    catchment_excess_df : pd.DataFrame
        Catchment-average excess rainfall before routing.

    hydro_df : pd.DataFrame
        Routed hydrograph.

    metrics : dict
        Hydrograph metrics plus runtime.

    hru_outputs : list[pd.DataFrame]
        Detailed HRU-level pulse outputs.
    """
    composition.validate()

    t0 = perf_counter()

    hru_outputs = []
    for hru in composition.hrus:
        hru_df = run_hru_response(
            method_name=method_name,
            storm_df=storm_df,
            hru=hru,
            climate=climate,
            dt_min=dt_min,
            experiment=experiment,
        )
        hru_outputs.append(hru_df)

    catchment_excess_df = combine_hru_outputs(
        storm_df=storm_df,
        hru_outputs=hru_outputs,
    )

    hydro_df, metrics = route_catchment_excess(
        catchment_excess_df=catchment_excess_df,
        catchment=catchment,
        dt_min=dt_min,
    )

    metrics["method_runtime_s"] = perf_counter() - t0

    return catchment_excess_df, hydro_df, metrics, hru_outputs


# ======================================================================================
# 8. EFV-RELATIVE AND BASELINE-RELATIVE METRICS
# ======================================================================================

def compare_catchment_to_efv(
    ref_excess_df: pd.DataFrame,
    test_excess_df: pd.DataFrame,
    ref_hydro_df: pd.DataFrame,
    test_hydro_df: pd.DataFrame,
) -> Dict[str, float]:
    """Compare a catchment-level method response against EFV for the same scenario."""
    validate_time_series_compatible(ref_excess_df, test_excess_df)

    if len(ref_hydro_df) != len(test_hydro_df):
        raise ValueError("Hydrograph DataFrames must have the same length for EFV comparison.")

    ref_excess = ref_excess_df["excess_mm"].to_numpy(dtype=float)
    test_excess = test_excess_df["excess_mm"].to_numpy(dtype=float)

    ref_cum_excess = float(ref_excess_df["cum_excess_mm"].iloc[-1])
    test_cum_excess = float(test_excess_df["cum_excess_mm"].iloc[-1])

    ref_cum_infil = float(ref_excess_df["cum_infil_mm"].iloc[-1])
    test_cum_infil = float(test_excess_df["cum_infil_mm"].iloc[-1])

    ref_peak = float(ref_hydro_df["Q_m3_s"].max())
    test_peak = float(test_hydro_df["Q_m3_s"].max())

    ref_time_s = ref_hydro_df["time_min"].to_numpy(dtype=float) * 60.0
    test_time_s = test_hydro_df["time_min"].to_numpy(dtype=float) * 60.0

    ref_volume = trapezoid_integral(ref_hydro_df["Q_m3_s"].to_numpy(dtype=float), ref_time_s)
    test_volume = trapezoid_integral(test_hydro_df["Q_m3_s"].to_numpy(dtype=float), test_time_s)

    rmse_excess = float(np.sqrt(np.mean((test_excess - ref_excess) ** 2)))

    return {
        "bias_cum_infil_mm_vs_efv": test_cum_infil - ref_cum_infil,
        "bias_cum_excess_mm_vs_efv": test_cum_excess - ref_cum_excess,
        "rmse_excess_mm_vs_efv": rmse_excess,
        "nrmse_excess_vs_efv": safe_relative_value(rmse_excess, ref_cum_excess),
        "peak_bias_m3_s_vs_efv": test_peak - ref_peak,
        "peak_rel_bias_vs_efv": safe_relative_value(test_peak - ref_peak, ref_peak),
        "volume_bias_m3_vs_efv": test_volume - ref_volume,
        "volume_rel_bias_vs_efv": safe_relative_value(test_volume - ref_volume, ref_volume),
    }


def add_baseline_relative_metrics(
    summary_df: pd.DataFrame,
    baseline_composition_key: str,
    baseline_climate_key: str,
) -> pd.DataFrame:
    """
    Add scenario-to-baseline changes for each method.

    The baseline is normally:
        composition = baseline_urban
        climate     = present

    Changes are computed within the same method so that, for example, EFV GI-retrofit is
    compared against EFV baseline, Green-Ampt GI-retrofit against Green-Ampt baseline, etc.
    """
    df = summary_df.copy()

    baseline_cols = [
        "method",
        "peak_flow_m3_s",
        "time_to_peak_min",
        "runoff_volume_m3",
        "cum_excess_mm",
        "runoff_coefficient_event",
    ]

    baseline_df = df[
        (df["composition_key"] == baseline_composition_key)
        & (df["climate_key"] == baseline_climate_key)
    ][baseline_cols].copy()

    if baseline_df.empty:
        raise ValueError(
            "Could not find the requested baseline scenario in summary_df: "
            f"composition={baseline_composition_key}, climate={baseline_climate_key}"
        )

    baseline_df = baseline_df.rename(columns={
        "peak_flow_m3_s": "baseline_peak_flow_m3_s",
        "time_to_peak_min": "baseline_time_to_peak_min",
        "runoff_volume_m3": "baseline_runoff_volume_m3",
        "cum_excess_mm": "baseline_cum_excess_mm",
        "runoff_coefficient_event": "baseline_runoff_coefficient_event",
    })

    df = df.merge(baseline_df, on="method", how="left")

    df["delta_peak_flow_m3_s_vs_baseline"] = (
        df["peak_flow_m3_s"] - df["baseline_peak_flow_m3_s"]
    )
    df["pct_peak_flow_change_vs_baseline"] = 100.0 * df.apply(
        lambda r: safe_relative_value(
            r["delta_peak_flow_m3_s_vs_baseline"],
            r["baseline_peak_flow_m3_s"],
        ),
        axis=1,
    )

    df["delta_runoff_volume_m3_vs_baseline"] = (
        df["runoff_volume_m3"] - df["baseline_runoff_volume_m3"]
    )
    df["pct_volume_change_vs_baseline"] = 100.0 * df.apply(
        lambda r: safe_relative_value(
            r["delta_runoff_volume_m3_vs_baseline"],
            r["baseline_runoff_volume_m3"],
        ),
        axis=1,
    )

    df["delta_cum_excess_mm_vs_baseline"] = (
        df["cum_excess_mm"] - df["baseline_cum_excess_mm"]
    )
    df["pct_cum_excess_change_vs_baseline"] = 100.0 * df.apply(
        lambda r: safe_relative_value(
            r["delta_cum_excess_mm_vs_baseline"],
            r["baseline_cum_excess_mm"],
        ),
        axis=1,
    )

    df["delta_time_to_peak_min_vs_baseline"] = (
        df["time_to_peak_min"] - df["baseline_time_to_peak_min"]
    )

    df["delta_runoff_coefficient_vs_baseline"] = (
        df["runoff_coefficient_event"] - df["baseline_runoff_coefficient_event"]
    )

    return df


# ======================================================================================
# 9. OUTPUT AND PLOTTING UTILITIES
# ======================================================================================

def save_csv_safely(df: pd.DataFrame, path: Path) -> Path:
    """
    Save a CSV file. If the target is locked/open, save a timestamped fallback file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback, index=False)
        return fallback


def _clean_filename_label(value: str) -> str:
    """Return a safe, compact string for filenames."""
    return str(value).lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _method_label_from_key_or_label(method_key_or_label: str) -> str:
    """
    Accept either an internal method key, e.g. 'efv', or an existing display label,
    e.g. 'EFV', and return the display label used in output DataFrames.
    """
    value = str(method_key_or_label).strip()

    known_labels = {"EFV", "Green-Ampt", "Horton", "Philip", "SCS-CN"}
    if value in known_labels:
        return value

    return method_display_name(value)


def _plot_empty_message(figure_name: str, subset: pd.DataFrame) -> bool:
    """Return True and print a message when the subset is empty."""
    if subset.empty:
        print(f"No data available for figure: {figure_name}")
        return True
    return False


# --------------------------------------------------------------------------------------
# MAIN PLOTS: SCENARIO COMPARISON FOR ONE SELECTED METHOD
# --------------------------------------------------------------------------------------

def plot_method_hydrographs_across_climates(
    hydrographs_df: pd.DataFrame,
    method_key_or_label: str,
    composition_key: str,
    climate_keys: List[str],
    output_dir: Path,
    show: bool = True,
) -> Optional[Path]:
    """
    MAIN FIGURE 1.

    Plot hydrographs for one selected infiltration method across several climate-stress
    scenarios, while keeping land-use composition fixed.

    This is the clearest figure for the climate-change part of the chapter because it
    answers:

        For the selected method, normally EFV, how does runoff response change when
        rainfall depth, rainfall temporal concentration, duration, or antecedent wetness
        changes?
    """
    method_label = _method_label_from_key_or_label(method_key_or_label)

    subset = hydrographs_df[
        (hydrographs_df["method"] == method_label)
        & (hydrographs_df["composition_key"] == composition_key)
        & (hydrographs_df["climate_key"].isin(climate_keys))
    ].copy()

    if _plot_empty_message("main hydrographs across climates", subset):
        return None

    plt.figure(figsize=(10, 5))

    for climate_key in climate_keys:
        group = subset[subset["climate_key"] == climate_key].copy()
        if group.empty:
            continue
        group = group.sort_values("time_min")
        climate_name = group["climate_name"].iloc[0]
        plt.plot(group["time_min"], group["Q_m3_s"], label=climate_name)

    plt.xlabel("Time (min)")
    plt.ylabel("Discharge (m³/s)")
    plt.title(
        f"Climate change: {method_label} hydrographs across climate scenarios\n"
        f"Fixed land-use composition: {composition_key}"
    )
    plt.legend()
    plt.tight_layout()

    filename = (
        f"MAIN_{_clean_filename_label(method_label)}_hydrographs_across_climates_"
        f"fixed_{_clean_filename_label(composition_key)}.png"
    )
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_method_hydrographs_across_compositions(
    hydrographs_df: pd.DataFrame,
    method_key_or_label: str,
    climate_key: str,
    composition_keys: List[str],
    output_dir: Path,
    show: bool = True,
) -> Optional[Path]:
    """
    MAIN FIGURE 2.

    Plot hydrographs for one selected infiltration method across several land-use
    compositions, while keeping climate scenario fixed.

    This is the clearest figure for the land-use-change part of the chapter because it
    answers:

        For the selected method, normally EFV, how does runoff response change when the
        catchment is densified, retrofitted with GI, degraded, or clay-dominated?
    """
    method_label = _method_label_from_key_or_label(method_key_or_label)

    subset = hydrographs_df[
        (hydrographs_df["method"] == method_label)
        & (hydrographs_df["climate_key"] == climate_key)
        & (hydrographs_df["composition_key"].isin(composition_keys))
    ].copy()

    if _plot_empty_message("main hydrographs across compositions", subset):
        return None

    plt.figure(figsize=(10, 5))

    for composition_key in composition_keys:
        group = subset[subset["composition_key"] == composition_key].copy()
        if group.empty:
            continue
        group = group.sort_values("time_min")
        composition_name = group["composition_name"].iloc[0]
        plt.plot(group["time_min"], group["Q_m3_s"], label=composition_name)

    plt.xlabel("Time (min)")
    plt.ylabel("Discharge (m³/s)")
    plt.title(
        f"Land-use change: {method_label} hydrographs across land-use compositions\n"
        f"Fixed climate scenario: {climate_key}"
    )
    plt.legend()
    plt.tight_layout()

    filename = (
        f"MAIN_{_clean_filename_label(method_label)}_hydrographs_across_land_use_"
        f"fixed_{_clean_filename_label(climate_key)}.png"
    )
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_method_peak_change_vs_baseline(
    summary_df: pd.DataFrame,
    method_key_or_label: str,
    output_dir: Path,
    baseline_composition_key: str,
    baseline_climate_key: str,
    show: bool = True,
) -> Optional[Path]:
    """
    MAIN FIGURE 3.

    Plot percentage peak-flow change relative to the selected baseline for one method.

    This figure turns the hydrograph comparison into a compact flood-impact result:
    positive values mean larger flood peaks than the baseline; negative values mean lower
    flood peaks than the baseline.
    """
    method_label = _method_label_from_key_or_label(method_key_or_label)
    subset = summary_df[summary_df["method"] == method_label].copy()

    if _plot_empty_message("main peak-flow change vs baseline", subset):
        return None

    subset["scenario_label"] = subset["composition_key"] + " | " + subset["climate_key"]
    baseline_label = f"{baseline_composition_key} | {baseline_climate_key}"

    subset["_baseline_first"] = np.where(subset["scenario_label"] == baseline_label, 0, 1)
    subset = subset.sort_values(["_baseline_first", "composition_key", "climate_key"])

    plt.figure(figsize=(13, 5))
    plt.bar(subset["scenario_label"], subset["pct_peak_flow_change_vs_baseline"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Land-use composition | climate scenario")
    plt.ylabel("Peak-flow change vs baseline (%)")
    plt.title(
        f"{method_label} peak-flow change relative to baseline\n"
        f"Baseline: {baseline_label}"
    )
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    filename = f"MAIN_{_clean_filename_label(method_label)}_peak_change_vs_baseline.png"
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_method_volume_change_vs_baseline(
    summary_df: pd.DataFrame,
    method_key_or_label: str,
    output_dir: Path,
    baseline_composition_key: str,
    baseline_climate_key: str,
    show: bool = True,
) -> Optional[Path]:
    """
    MAIN FIGURE 4.

    Plot percentage runoff-volume change relative to the selected baseline for one method.

    This figure summarises the event-scale water-balance consequence of land-use and
    climate-stress scenarios.
    """
    method_label = _method_label_from_key_or_label(method_key_or_label)
    subset = summary_df[summary_df["method"] == method_label].copy()

    if _plot_empty_message("main runoff-volume change vs baseline", subset):
        return None

    subset["scenario_label"] = subset["composition_key"] + " | " + subset["climate_key"]
    baseline_label = f"{baseline_composition_key} | {baseline_climate_key}"

    subset["_baseline_first"] = np.where(subset["scenario_label"] == baseline_label, 0, 1)
    subset = subset.sort_values(["_baseline_first", "composition_key", "climate_key"])

    plt.figure(figsize=(13, 5))
    plt.bar(subset["scenario_label"], subset["pct_volume_change_vs_baseline"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Land-use composition | climate scenario")
    plt.ylabel("Runoff-volume change vs baseline (%)")
    plt.title(
        f"{method_label} runoff-volume change relative to baseline\n"
        f"Baseline: {baseline_label}"
    )
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()

    filename = f"MAIN_{_clean_filename_label(method_label)}_volume_change_vs_baseline.png"
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


# --------------------------------------------------------------------------------------
# OPTIONAL PLOTS: METHOD COMPARISON FOR ONE FIXED SCENARIO
# --------------------------------------------------------------------------------------

def plot_optional_method_comparison_hydrographs(
    hydrographs_df: pd.DataFrame,
    composition_key: str,
    climate_key: str,
    output_dir: Path,
    show: bool = True,
) -> Optional[Path]:
    """
    OPTIONAL DIAGNOSTIC FIGURE.

    Plot all infiltration methods for one fixed land-use composition and one fixed
    climate scenario. This is not the main LUC/CC plot. It supports the methodological
    question: how much do simpler methods differ from EFV under the same multi-HRU setup?
    """
    subset = hydrographs_df[
        (hydrographs_df["composition_key"] == composition_key)
        & (hydrographs_df["climate_key"] == climate_key)
    ].copy()

    if _plot_empty_message("optional method-comparison hydrographs", subset):
        return None

    method_order = ["EFV", "Green-Ampt", "Horton", "Philip", "SCS-CN"]

    plt.figure(figsize=(10, 5))
    for method in method_order:
        group = subset[subset["method"] == method].copy()
        if group.empty:
            continue
        group = group.sort_values("time_min")
        plt.plot(group["time_min"], group["Q_m3_s"], label=method)

    plt.xlabel("Time (min)")
    plt.ylabel("Discharge (m³/s)")
    plt.title(
        f"OPTIONAL: method comparison for one fixed scenario\n"
        f"Composition: {composition_key}; climate: {climate_key}"
    )
    plt.legend()
    plt.tight_layout()

    filename = (
        f"OPTIONAL_method_comparison_hydrographs_"
        f"{_clean_filename_label(composition_key)}_{_clean_filename_label(climate_key)}.png"
    )
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_optional_method_comparison_excess(
    excess_df: pd.DataFrame,
    composition_key: str,
    climate_key: str,
    output_dir: Path,
    show: bool = True,
) -> Optional[Path]:
    """
    OPTIONAL DIAGNOSTIC FIGURE.

    Plot catchment-average excess rainfall from all methods for one fixed land-use and
    climate scenario. This helps diagnose why the hydrographs differ.
    """
    subset = excess_df[
        (excess_df["composition_key"] == composition_key)
        & (excess_df["climate_key"] == climate_key)
    ].copy()

    if _plot_empty_message("optional method-comparison excess rainfall", subset):
        return None

    method_order = ["EFV", "Green-Ampt", "Horton", "Philip", "SCS-CN"]

    plt.figure(figsize=(10, 5))
    for method in method_order:
        group = subset[subset["method"] == method].copy()
        if group.empty:
            continue
        group = group.sort_values("time_min")
        plt.plot(group["time_min"], group["excess_mm"], marker="o", label=method)

    plt.xlabel("Time (min)")
    plt.ylabel("Catchment-average excess rainfall (mm per pulse)")
    plt.title(
        f"OPTIONAL: method comparison of excess rainfall\n"
        f"Composition: {composition_key}; climate: {climate_key}"
    )
    plt.legend()
    plt.tight_layout()

    filename = (
        f"OPTIONAL_method_comparison_excess_"
        f"{_clean_filename_label(composition_key)}_{_clean_filename_label(climate_key)}.png"
    )
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_optional_efv_relative_peak_bias(
    summary_df: pd.DataFrame,
    composition_key: str,
    climate_key: str,
    output_dir: Path,
    show: bool = True,
) -> Optional[Path]:
    """
    OPTIONAL DIAGNOSTIC FIGURE.

    Plot EFV-relative peak-flow bias by method for one selected scenario. EFV itself is
    zero by definition. This figure is useful only as supporting evidence that simpler
    methods can distort the EFV-based runoff response.
    """
    subset = summary_df[
        (summary_df["composition_key"] == composition_key)
        & (summary_df["climate_key"] == climate_key)
    ].copy()

    if _plot_empty_message("optional EFV-relative peak bias", subset):
        return None

    method_order = ["Green-Ampt", "Horton", "Philip", "SCS-CN"]
    subset = subset[subset["method"].isin(method_order)].copy()
    subset["method"] = pd.Categorical(subset["method"], categories=method_order, ordered=True)
    subset = subset.sort_values("method")

    plt.figure(figsize=(8, 4))
    plt.bar(subset["method"].astype(str), 100.0 * subset["peak_rel_bias_vs_efv"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Method")
    plt.ylabel("Peak-flow bias vs EFV (%)")
    plt.title(
        f"OPTIONAL: EFV-relative method bias\n"
        f"Composition: {composition_key}; climate: {climate_key}"
    )
    plt.tight_layout()

    filename = (
        f"OPTIONAL_efv_relative_peak_bias_"
        f"{_clean_filename_label(composition_key)}_{_clean_filename_label(climate_key)}.png"
    )
    path = output_dir / filename
    plt.savefig(path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()
    return path


def export_main_method_summary_table(
    summary_df: pd.DataFrame,
    method_key_or_label: str,
    output_dir: Path,
) -> Path:
    """
    Export a compact table for the selected main method.

    This table is intended for thesis/PPT interpretation. It contains only the most
    relevant scenario-comparison variables, not the full detailed CSV.
    """
    method_label = _method_label_from_key_or_label(method_key_or_label)
    subset = summary_df[summary_df["method"] == method_label].copy()

    cols = [
        "method",
        "composition_key",
        "climate_key",
        "antecedent",
        "impervious_fraction",
        "gi_fraction",
        "area_weighted_storage_mm",
        "storm_total_depth_mm_check",
        "storm_duration_min_check",
        "mean_intensity_mm_h",
        "peak_intensity_mm_h",
        "cum_excess_mm",
        "runoff_coefficient_event",
        "peak_flow_m3_s",
        "time_to_peak_min",
        "runoff_volume_m3",
        "pct_cum_excess_change_vs_baseline",
        "pct_peak_flow_change_vs_baseline",
        "pct_volume_change_vs_baseline",
    ]

    existing_cols = [c for c in cols if c in subset.columns]
    out = subset[existing_cols].sort_values(["composition_key", "climate_key"])

    filename = f"MAIN_{_clean_filename_label(method_label)}_scenario_summary_table.csv"
    path = output_dir / filename
    return save_csv_safely(out, path)


# ======================================================================================
# 10. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # 1) USER-CONTROLLED SETTINGS
    # ------------------------------------------------------------------------------

    dt_min = 10

    experiment = ExperimentConfig(
        comparison_mode="structural",
        efv_bottom_bc="free_drainage",
        reference_method="efv",
    )

    catchment = CatchmentScenario(
        name="Conceptual urban catchment",
        area_km2=27.0,
        flow_length_m=2500.0,
        hmax_m=100.0,
        hmin_m=60.0,
        baseflow_m3_s=0.0,
    )

    # All methods are still run so that EFV-relative method bias remains available.
    # However, the MAIN plots below focus on one selected method, normally EFV.
    selected_methods = ["ga", "horton", "scs", "philip", "efv"]

    # MAIN plotting method. Keep "efv" for the thesis application story.
    # You may change this to "ga", "horton", "scs", or "philip" if you want
    # scenario-comparison plots for another method.
    main_plot_method_key = "efv"

    selected_compositions = [
        "baseline_urban",
        "densified_urban",
        "gi_retrofit",
        "degraded_gi",
        "clay_dominated",
    ]

    selected_climates = [
        "present",
        "intensified_depth",
        "intensified_peakiness",
        "shorter_more_intense",
        "wetter_antecedent",
    ]

    baseline_composition_key = "baseline_urban"
    baseline_climate_key = "present"

    # MAIN plot controls.
    # 1) Climate comparison: keep land-use fixed and vary climate scenarios.
    main_fixed_composition_for_climate_plot = "baseline_urban"

    # 2) Land-use comparison: keep climate fixed and vary land-use compositions.
    main_fixed_climate_for_land_use_plot = "present"

    # OPTIONAL method-comparison controls.
    # These plots are not the main LUC/CC argument. They diagnose how simpler methods
    # differ from EFV for one selected fixed scenario.
    make_optional_method_comparison_plots = True
    optional_comparison_composition_key = "baseline_urban"
    optional_comparison_climate_key = "present"

    # Output folder.
    output_dir = SCRIPT_DIR / "multi_hru_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------------
    # 2) VALIDATE SCENARIO LIBRARIES
    # ------------------------------------------------------------------------------

    for comp_key in selected_compositions:
        if comp_key not in COMPOSITIONS:
            raise ValueError(f"Unknown composition key: {comp_key}")
        COMPOSITIONS[comp_key].validate()

    for climate_key in selected_climates:
        if climate_key not in CLIMATE_SCENARIOS:
            raise ValueError(f"Unknown climate scenario key: {climate_key}")
        CLIMATE_SCENARIOS[climate_key].validate(dt_min)

    if "efv" not in [m.lower() for m in selected_methods]:
        raise ValueError("selected_methods must include 'efv' so EFV-relative biases can be computed.")

    # ------------------------------------------------------------------------------
    # 3) RUN ALL COMPOSITION x CLIMATE x METHOD SCENARIOS
    # ------------------------------------------------------------------------------

    print("Running Script 2: multi-HRU LUC/CC framework")
    print("Methods run:", selected_methods)
    print("Main plotting method:", _method_label_from_key_or_label(main_plot_method_key))
    print("Compositions:", selected_compositions)
    print("Climate scenarios:", selected_climates)
    print("Output directory:", output_dir)

    summary_rows: List[Dict[str, float]] = []
    excess_records: List[pd.DataFrame] = []
    hydro_records: List[pd.DataFrame] = []
    hru_records: List[pd.DataFrame] = []

    for composition_key in selected_compositions:
        composition = COMPOSITIONS[composition_key]
        composition_desc = composition.descriptors()

        for climate_key in selected_climates:
            climate = CLIMATE_SCENARIOS[climate_key]
            storm_df = build_storm_for_climate(climate, dt_min=dt_min)
            storm_desc = storm_summary(storm_df)

            scenario_excess_by_method: Dict[str, pd.DataFrame] = {}
            scenario_hydro_by_method: Dict[str, pd.DataFrame] = {}
            scenario_metrics_by_method: Dict[str, Dict[str, float]] = {}

            for method_key in selected_methods:
                method_label = method_display_name(method_key)

                catchment_excess_df, hydro_df, metrics, hru_outputs = run_composition_method_climate(
                    composition=composition,
                    climate=climate,
                    method_name=method_key,
                    storm_df=storm_df,
                    catchment=catchment,
                    dt_min=dt_min,
                    experiment=experiment,
                )

                # Add metadata to catchment-average excess series.
                catchment_excess_out = catchment_excess_df.copy()
                catchment_excess_out["method"] = method_label
                catchment_excess_out["method_key"] = method_key
                catchment_excess_out["composition_key"] = composition_key
                catchment_excess_out["composition_name"] = composition.name
                catchment_excess_out["climate_key"] = climate_key
                catchment_excess_out["climate_name"] = climate.name
                catchment_excess_out["antecedent"] = climate.antecedent
                excess_records.append(catchment_excess_out)

                # Add metadata to hydrograph series.
                hydro_out = hydro_df.copy()
                hydro_out["method"] = method_label
                hydro_out["method_key"] = method_key
                hydro_out["composition_key"] = composition_key
                hydro_out["composition_name"] = composition.name
                hydro_out["climate_key"] = climate_key
                hydro_out["climate_name"] = climate.name
                hydro_out["antecedent"] = climate.antecedent
                hydro_records.append(hydro_out)

                # Add metadata to HRU-level outputs.
                for hru_df in hru_outputs:
                    hru_out = hru_df.copy()
                    hru_out["method"] = method_label
                    hru_out["method_key"] = method_key
                    hru_out["composition_key"] = composition_key
                    hru_out["composition_name"] = composition.name
                    hru_out["climate_key"] = climate_key
                    hru_out["climate_name"] = climate.name
                    hru_out["antecedent"] = climate.antecedent
                    hru_records.append(hru_out)

                scenario_excess_by_method[method_label] = catchment_excess_df
                scenario_hydro_by_method[method_label] = hydro_df
                scenario_metrics_by_method[method_label] = metrics

            # ------------------------------------------------------------------
            # EFV-relative comparison for this composition and climate scenario.
            # ------------------------------------------------------------------
            efv_excess_df = scenario_excess_by_method["EFV"]
            efv_hydro_df = scenario_hydro_by_method["EFV"]

            for method_label, catchment_excess_df in scenario_excess_by_method.items():
                hydro_df = scenario_hydro_by_method[method_label]
                metrics = scenario_metrics_by_method[method_label]

                efv_bias_metrics = compare_catchment_to_efv(
                    ref_excess_df=efv_excess_df,
                    test_excess_df=catchment_excess_df,
                    ref_hydro_df=efv_hydro_df,
                    test_hydro_df=hydro_df,
                )

                cum_excess_mm = float(catchment_excess_df["cum_excess_mm"].iloc[-1])
                cum_infil_mm = float(catchment_excess_df["cum_infil_mm"].iloc[-1])
                total_depth_mm = float(storm_df["P_mm"].sum())

                runoff_coefficient_event = safe_relative_value(cum_excess_mm, total_depth_mm)
                retained_or_infil_fraction = safe_relative_value(cum_infil_mm, total_depth_mm)

                row = {
                    "method": method_label,
                    "composition_key": composition_key,
                    "composition_name": composition.name,
                    "composition_description": composition.description,
                    "climate_key": climate_key,
                    "climate_name": climate.name,
                    "climate_description": climate.description,
                    "antecedent": climate.antecedent,
                    "storm_duration_min": climate.storm_duration_min,
                    "storm_total_depth_mm": climate.total_depth_mm,
                    "storm_pattern": climate.pattern,
                    "comparison_mode": experiment.comparison_mode,
                    "efv_bottom_bc": experiment.efv_bottom_bc,
                    "catchment_area_km2": catchment.area_km2,
                    "catchment_tc_min": catchment.tc_min,
                    "cum_excess_mm": cum_excess_mm,
                    "cum_infil_or_retained_mm": cum_infil_mm,
                    "runoff_coefficient_event": runoff_coefficient_event,
                    "retained_or_infil_fraction_event": retained_or_infil_fraction,
                    "peak_flow_m3_s": metrics["peak_flow_m3_s"],
                    "time_to_peak_min": metrics["time_to_peak_min"],
                    "runoff_volume_m3": metrics["runoff_volume_m3"],
                    "method_runtime_s": metrics["method_runtime_s"],
                    "max_abs_water_accounting_residual_mm": float(
                        np.abs(catchment_excess_df["water_accounting_residual_mm"]).max()
                    ),
                    **composition_desc,
                    **storm_desc,
                    **efv_bias_metrics,
                }

                summary_rows.append(row)

    # ------------------------------------------------------------------------------
    # 4) BUILD OUTPUT DATAFRAMES AND ADD BASELINE-RELATIVE METRICS
    # ------------------------------------------------------------------------------

    summary_df = pd.DataFrame(summary_rows)
    summary_df = add_baseline_relative_metrics(
        summary_df=summary_df,
        baseline_composition_key=baseline_composition_key,
        baseline_climate_key=baseline_climate_key,
    )

    excess_df = pd.concat(excess_records, ignore_index=True)
    hydrographs_df = pd.concat(hydro_records, ignore_index=True)
    hru_contrib_df = pd.concat(hru_records, ignore_index=True)

    # ------------------------------------------------------------------------------
    # 5) SAVE OUTPUTS
    # ------------------------------------------------------------------------------

    summary_path = save_csv_safely(summary_df, output_dir / "multi_hru_summary.csv")
    excess_path = save_csv_safely(excess_df, output_dir / "multi_hru_excess_series.csv")
    hydro_path = save_csv_safely(hydrographs_df, output_dir / "multi_hru_hydrographs.csv")
    hru_path = save_csv_safely(hru_contrib_df, output_dir / "multi_hru_hru_contributions.csv")

    print("\nSaved outputs:")
    print("Summary:", summary_path)
    print("Excess series:", excess_path)
    print("Hydrographs:", hydro_path)
    print("HRU contributions:", hru_path)

    # ------------------------------------------------------------------------------
    # 6) DIAGNOSTIC PRINTS
    # ------------------------------------------------------------------------------

    print("\n=== Selected baseline rows ===")
    print(
        summary_df[
            (summary_df["composition_key"] == baseline_composition_key)
            & (summary_df["climate_key"] == baseline_climate_key)
        ][[
            "method",
            "composition_key",
            "climate_key",
            "cum_excess_mm",
            "runoff_coefficient_event",
            "peak_flow_m3_s",
            "runoff_volume_m3",
            "nrmse_excess_vs_efv",
            "peak_rel_bias_vs_efv",
            "volume_rel_bias_vs_efv",
        ]]
    )

    print("\n=== EFV scenario changes relative to baseline ===")
    print(
        summary_df[summary_df["method"] == "EFV"][[
            "composition_key",
            "climate_key",
            "cum_excess_mm",
            "peak_flow_m3_s",
            "runoff_volume_m3",
            "pct_cum_excess_change_vs_baseline",
            "pct_peak_flow_change_vs_baseline",
            "pct_volume_change_vs_baseline",
        ]].sort_values(["composition_key", "climate_key"])
    )

    # ------------------------------------------------------------------------------
    # 7) MAIN FIGURES: SCENARIO COMPARISON FOR ONE SELECTED METHOD
    # ------------------------------------------------------------------------------

    print("\nCreating MAIN figures for scenario comparison...")

    main_method_summary_path = export_main_method_summary_table(
        summary_df=summary_df,
        method_key_or_label=main_plot_method_key,
        output_dir=output_dir,
    )
    print("Main-method scenario table:", main_method_summary_path)

    path = plot_method_hydrographs_across_climates(
        hydrographs_df=hydrographs_df,
        method_key_or_label=main_plot_method_key,
        composition_key=main_fixed_composition_for_climate_plot,
        climate_keys=selected_climates,
        output_dir=output_dir,
    )
    if path is not None:
        print("Main climate-comparison hydrograph figure:", path)

    path = plot_method_hydrographs_across_compositions(
        hydrographs_df=hydrographs_df,
        method_key_or_label=main_plot_method_key,
        climate_key=main_fixed_climate_for_land_use_plot,
        composition_keys=selected_compositions,
        output_dir=output_dir,
    )
    if path is not None:
        print("Main land-use-comparison hydrograph figure:", path)

    path = plot_method_peak_change_vs_baseline(
        summary_df=summary_df,
        method_key_or_label=main_plot_method_key,
        output_dir=output_dir,
        baseline_composition_key=baseline_composition_key,
        baseline_climate_key=baseline_climate_key,
    )
    if path is not None:
        print("Main peak-change figure:", path)

    path = plot_method_volume_change_vs_baseline(
        summary_df=summary_df,
        method_key_or_label=main_plot_method_key,
        output_dir=output_dir,
        baseline_composition_key=baseline_composition_key,
        baseline_climate_key=baseline_climate_key,
    )
    if path is not None:
        print("Main volume-change figure:", path)

    # ------------------------------------------------------------------------------
    # 8) OPTIONAL FIGURES: METHOD COMPARISON FOR ONE FIXED SCENARIO
    # ------------------------------------------------------------------------------

    if make_optional_method_comparison_plots:
        print("\nCreating OPTIONAL method-comparison diagnostic figures...")

        path = plot_optional_method_comparison_excess(
            excess_df=excess_df,
            composition_key=optional_comparison_composition_key,
            climate_key=optional_comparison_climate_key,
            output_dir=output_dir,
        )
        if path is not None:
            print("Optional method-comparison excess figure:", path)

        path = plot_optional_method_comparison_hydrographs(
            hydrographs_df=hydrographs_df,
            composition_key=optional_comparison_composition_key,
            climate_key=optional_comparison_climate_key,
            output_dir=output_dir,
        )
        if path is not None:
            print("Optional method-comparison hydrograph figure:", path)

        path = plot_optional_efv_relative_peak_bias(
            summary_df=summary_df,
            composition_key=optional_comparison_composition_key,
            climate_key=optional_comparison_climate_key,
            output_dir=output_dir,
        )
        if path is not None:
            print("Optional EFV-relative peak-bias figure:", path)

    print("\nScript 2 completed successfully.")
