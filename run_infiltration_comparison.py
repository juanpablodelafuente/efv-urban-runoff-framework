"""
Unified, fully commented Python framework for comparing infiltration / loss methods
under the SAME design storm and the SAME unit-hydrograph routing.

Why this script exists
----------------------
Your uploaded Colab notebooks already follow the correct overall philosophy:

    design storm -> excess rainfall -> unit hydrograph -> runoff hydrograph

This script reorganises that idea into a cleaner and more defensible framework for a
PhD chapter / paper. The key scientific rule is:

    ONLY the infiltration / loss method changes.

Everything else must stay fixed:
    - same design storm
    - same time step
    - same catchment area and routing parameters
    - same unit hydrograph
    - same baseflow (if used)

This way, differences in the final hydrograph come mainly from the infiltration model,
not from unrelated changes in routing or catchment setup.

Included methods
----------------
1) Green-Ampt (physically based lumped infiltration)
2) Philip (semi-analytical infiltration approximation)
3) Horton (empirical infiltration-decay model)
4) SCS Curve Number (event-based runoff abstraction)
5) EFV (Richards-based explicit FV infiltration model embedded in the framework)

Important note about EFV
------------------------
This script does not attempt to replace the full standalone EFV research code.
However, it does include an embedded EFV event-simulation driver so that EFV can be
run consistently inside the same comparative framework as the simpler infiltration methods.

Recommended workflow
--------------------
A) Prepare a design storm hyetograph at a fixed time step (e.g. 10 min).
B) Choose one master soil description.
C) Derive the parameters for all simpler methods from that same soil.
D) Compute excess rainfall with each method.
E) Route all excess rainfall series through the same unit hydrograph.
F) Compare hydrographs and metrics.

Why the parameters are defensible
---------------------------------
This script uses the idea of a MASTER SOIL DESCRIPTION.
For each soil we first define a common set of physically meaningful properties:

    - Ks       : saturated hydraulic conductivity
    - theta_s  : saturated volumetric water content
    - theta_r  : residual volumetric water content
    - theta_i  : initial volumetric water content (depends on antecedent condition)
    - psi_f    : representative Green-Ampt wetting-front suction head
    - CN_II    : SCS Curve Number under average antecedent condition

Then, each method is parameterised from that SAME soil:

    EFV      -> uses the full hydraulic description
    GA       -> uses Ks, theta_s, theta_i, psi_f
    Philip   -> uses the SAME Ks and a sorptivity S linked to the same soil state
    Horton   -> uses fc = Ks and f0, k chosen to mimic the same soil tendency
    SCS-CN   -> uses CN values consistent with the same soil permeability class

This is much more defensible than choosing unrelated parameter sets for each method.

Units used throughout
---------------------
Rainfall depths per time step: mm
Time step dt: minutes (converted internally to hours or seconds when needed)
Hydraulic conductivity for infiltration functions: mm/h in this script
Catchment area: km^2
Discharge: m^3/s

Author's note
-------------
This is a RESEARCH FRAMEWORK script, not a black-box design tool.
It is intentionally verbose and heavily commented so that every step is easy to audit,
modify, and explain in a thesis chapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path
from datetime import datetime

# ======================================================================================
# 1. DATA CLASSES
# ======================================================================================

@dataclass
class SoilScenario:
    """
    Master soil description.

    These parameters should be interpreted as the COMMON underlying description of the
    soil, from which the simpler methods are derived.

    Attributes
    ----------
    name : str
        A short descriptive name for the soil scenario.

    Ks_mm_h : float
        Saturated hydraulic conductivity [mm/h].
        This is one of the most important quantities in the comparison because it can
        be used consistently in EFV, Green-Ampt, Philip, and Horton.

    theta_s : float
        Saturated volumetric water content [-].

    theta_r : float
        Residual volumetric water content [-].

    theta_i_dry, theta_i_avg, theta_i_wet : float
        Initial volumetric water content for three antecedent moisture states.
        These are the states that should be used to explore the sensitivity to wetness.

    psi_f_mm : float
        Representative Green-Ampt wetting-front suction head [mm].
        This is not exactly the same quantity as pressure head in Richards,
        but it is physically related to capillary suction and should be chosen to be
        consistent with the same soil class.

    CN_II : float
        SCS Curve Number for average antecedent moisture condition (AMC II).
        Lower CN means more infiltration potential.
        Higher CN means more runoff potential.

    philip_S_dry, philip_S_avg, philip_S_wet : float
        Sorptivity [mm / h^0.5] for Philip's model under each antecedent condition.
        In a more rigorous workflow, these should be fitted from early-time EFV results.
        Here we provide plausible starting values to make the framework usable now.

    horton_f0_dry, horton_f0_avg, horton_f0_wet : float
        Initial infiltration capacity [mm/h] for Horton.
        This should be larger than Ks and normally decreases with wetter antecedent state.

    horton_k_h_inv : float
        Horton decay coefficient [1/h].
        Larger values mean the infiltration capacity drops more rapidly from f0 to fc.
    """
    name: str
    Ks_mm_h: float
    theta_s: float
    theta_r: float
    theta_i_dry: float
    theta_i_avg: float
    theta_i_wet: float
    psi_f_mm: float
    CN_II: float
    philip_S_dry: float
    philip_S_avg: float
    philip_S_wet: float
    horton_f0_dry: float
    horton_f0_avg: float
    horton_f0_wet: float
    horton_k_h_inv: float
    efv_alpha_cm_inv: Optional[float] = None
    efv_n: Optional[float] = None
    efv_nz: int = 25
    efv_dz_cm: float = 5.0
    efv_dt_sub_s: float = 5.0
    efv_avg_kind: int = 0
    efv_surface_theta: Optional[float] = None

    def theta_i(self, antecedent: str) -> float:
        """Return the initial water content corresponding to the chosen antecedent state."""
        antecedent = antecedent.lower()
        if antecedent == "dry":
            return self.theta_i_dry
        if antecedent in ["avg", "average", "normal"]:
            return self.theta_i_avg
        if antecedent == "wet":
            return self.theta_i_wet
        raise ValueError("antecedent must be 'dry', 'avg', or 'wet'")

    def philip_S(self, antecedent: str) -> float:
        """Return Philip sorptivity for the chosen antecedent state."""
        antecedent = antecedent.lower()
        if antecedent == "dry":
            return self.philip_S_dry
        if antecedent in ["avg", "average", "normal"]:
            return self.philip_S_avg
        if antecedent == "wet":
            return self.philip_S_wet
        raise ValueError("antecedent must be 'dry', 'avg', or 'wet'")

    def horton_f0(self, antecedent: str) -> float:
        """Return Horton initial infiltration capacity for the chosen antecedent state."""
        antecedent = antecedent.lower()
        if antecedent == "dry":
            return self.horton_f0_dry
        if antecedent in ["avg", "average", "normal"]:
            return self.horton_f0_avg
        if antecedent == "wet":
            return self.horton_f0_wet
        raise ValueError("antecedent must be 'dry', 'avg', or 'wet'")

    def efv_surface_theta_value(self) -> float:
        """Return the surface moisture content used for EFV surface-capacity evaluation."""
        return self.theta_s if self.efv_surface_theta is None else self.efv_surface_theta

    def efv_m_value(self) -> float:
        """Return van Genuchten m computed from n using m = 1 - 1/n."""
        if self.efv_n is None:
            raise ValueError("EFV parameter efv_n is missing.")
        return 1.0 - 1.0 / self.efv_n


@dataclass
class CatchmentScenario:
    """
    Conceptual catchment and routing description.

    The point of this class is to keep the routing model IDENTICAL across methods.

    Attributes
    ----------
    name : str
        Descriptive name of the conceptual catchment.

    area_km2 : float
        Catchment area [km^2].
        This is needed to convert effective rainfall into discharge in the unit hydrograph.

    flow_length_m : float
        Main hydraulic / flow path length [m].
        Used in the Kirpich equation below as one practical way to estimate concentration time.

    hmax_m, hmin_m : float
        Maximum and minimum elevation [m]. These are only used to estimate slope.

    baseflow_m3_s : float
        Constant baseflow added to the hydrograph. Keep this fixed for all methods.
        If you do not want baseflow, simply set it to 0.0.
    """
    name: str
    area_km2: float
    flow_length_m: float
    hmax_m: float
    hmin_m: float
    baseflow_m3_s: float = 0.0

    @property
    def slope_m_m(self) -> float:
        """Catchment slope [m/m] based on the very simple relief/length estimate."""
        return (self.hmax_m - self.hmin_m) / self.flow_length_m

    @property
    def tc_h(self) -> float:
        """
        Time of concentration [hours] using the Kirpich formula.

        This is a simple practical estimate, not a universal truth.
        It is acceptable for a comparative framework as long as it is held fixed across methods.
        """
        slp = self.slope_m_m
        return (0.0195 * (self.flow_length_m / math.sqrt(slp)) ** 0.77) / 60.0

    @property
    def tc_min(self) -> float:
        """Time of concentration [minutes]."""
        return self.tc_h * 60.0

@dataclass
class ExperimentConfig:
    comparison_mode: str = "structural"   # "structural" or "surrogate"
    efv_bottom_bc: str = "free_drainage"  # "free_drainage" or "no_flow"
    reference_method: str = "efv"         # used only in surrogate mode


# ======================================================================================
# 2. SOIL LIBRARY
# ======================================================================================

# These values are intended as PHYSICALLY REASONABLE STARTING POINTS for a controlled
# comparative experiment. They are not meant to claim site calibration.
#
# The main idea is that permeability and storage potential increase from:
#
#   compacted urban soil  -> urban loam -> GI / engineered soil
#
# and that the antecedent wetness states are internally consistent.

SOILS: Dict[str, SoilScenario] = {
    "compacted": SoilScenario(
        name="Compacted urban soil",
        # Compacted-loam surrogate (not a standard USDA class)
        Ks_mm_h=5.0,
        theta_s=0.40,
        theta_r=0.09,
        theta_i_dry=0.18,
        theta_i_avg=0.26,
        theta_i_wet=0.34,
        psi_f_mm=180.0,
        CN_II=90.0,
        philip_S_dry=26.0,
        philip_S_avg=18.0,
        philip_S_wet=10.0,
        horton_f0_dry=45.0,
        horton_f0_avg=25.0,
        horton_f0_wet=12.0,
        horton_k_h_inv=2.0,
        efv_alpha_cm_inv=0.012,
        efv_n=1.25,
        efv_nz=25,
        efv_dz_cm=5.0,
        efv_dt_sub_s=5.0,
        efv_avg_kind=0,
        efv_surface_theta=None,
    ),

    "loam": SoilScenario(
        name="Urban loam",
        # Standard loam (Carsel-Parrish / HYDRUS-style)
        Ks_mm_h=10.4,       # 24.96 cm/day
        theta_s=0.43,
        theta_r=0.078,
        theta_i_dry=0.16,
        theta_i_avg=0.24,
        theta_i_wet=0.32,
        psi_f_mm=88.9,
        CN_II=80.0,
        philip_S_dry=45.0,
        philip_S_avg=32.0,
        philip_S_wet=20.0,
        horton_f0_dry=70.0,
        horton_f0_avg=40.0,
        horton_f0_wet=20.0,
        horton_k_h_inv=2.2,
        efv_alpha_cm_inv=0.036,
        efv_n=1.56,
        efv_nz=25,
        efv_dz_cm=5.0,
        efv_dt_sub_s=5.0,
        efv_avg_kind=0,
        efv_surface_theta=None,
    ),

    "gi": SoilScenario(
        name="GI / engineered soil",
        # Loamy-sand surrogate for engineered sandy bioretention-type media
        Ks_mm_h=60.8,       # 145.9 cm/day
        theta_s=0.41,
        theta_r=0.057,
        theta_i_dry=0.10,
        theta_i_avg=0.18,
        theta_i_wet=0.26,
        psi_f_mm=61.3,
        CN_II=70.0,
        philip_S_dry=70.0,
        philip_S_avg=52.0,
        philip_S_wet=34.0,
        horton_f0_dry=120.0,
        horton_f0_avg=80.0,
        horton_f0_wet=45.0,
        horton_k_h_inv=2.5,
        efv_alpha_cm_inv=0.124,
        efv_n=2.28,
        efv_nz=25,
        efv_dz_cm=5.0,
        efv_dt_sub_s=5.0,
        efv_avg_kind=0,
        efv_surface_theta=None,
    ),

    "clay": SoilScenario(
        name="Urban clay",
        # Standard clay
        Ks_mm_h=2.0,        # 4.8 cm/day
        theta_s=0.38,
        theta_r=0.068,
        theta_i_dry=0.22,
        theta_i_avg=0.30,
        theta_i_wet=0.36,
        psi_f_mm=316.3,
        CN_II=88.0,
        philip_S_dry=20.0,
        philip_S_avg=14.0,
        philip_S_wet=8.0,
        horton_f0_dry=35.0,
        horton_f0_avg=20.0,
        horton_f0_wet=10.0,
        horton_k_h_inv=2.2,
        efv_alpha_cm_inv=0.008,
        efv_n=1.09,
        efv_nz=25,
        efv_dz_cm=5.0,
        efv_dt_sub_s=5.0,
        efv_avg_kind=0,
        efv_surface_theta=None,
    ),
}


# ======================================================================================
# 3. SIMPLE DESIGN-STORM UTILITIES
# ======================================================================================

# Your previous Colab notebooks created design storms through L-moments + IDF + alternating
# blocks. That is fine when building a rainfall-frequency workflow.
#
# For the comparative chapter, however, it is perfectly acceptable to START from a design
# hyetograph already prepared. That simplifies the experiment dramatically.
#
# We therefore provide two options here:
#   (a) directly input a list/array of rainfall depths per time step
#   (b) generate an alternating-block storm from an IDF relationship if desired


def alternating_block_hyetograph(
    coef_K: float,
    coef_m: float,
    coef_n: float,
    return_period_years: float,
    storm_duration_min: int,
    dt_min: int,
) -> pd.DataFrame:
    """
    Create a design hyetograph using the alternating block method.

    Parameters
    ----------
    coef_K, coef_m, coef_n : float
        Coefficients of an IDF equation of the form:

            i = (K * T^m) / D^n

        where:
            i = intensity [mm/h]
            T = return period [years]
            D = duration [minutes]

    return_period_years : float
        Chosen return period of the design storm.

    storm_duration_min : int
        Total storm duration [minutes]. Must be a multiple of dt_min.

    dt_min : int
        Time step of the design storm [minutes].

    Returns
    -------
    DataFrame with columns:
        time_min : cumulative time [min] at the end of each pulse
        P_mm     : rainfall depth in each pulse [mm]
        i_mm_h   : equivalent pulse intensity [mm/h]

    Notes
    -----
    The alternating-block method rearranges incremental rainfall depths so that the
    largest pulse is near the middle of the storm, with the remaining pulses alternating
    before and after it. This is commonly used for synthetic design storms.
    """
    durations = np.arange(dt_min, storm_duration_min + dt_min, dt_min, dtype=float)

    # IDF intensities for each cumulative duration.
    intensities_mm_h = (coef_K * (return_period_years ** coef_m)) / (durations ** coef_n)

    # Convert cumulative depth to incremental depth.
    cumulative_depth_mm = intensities_mm_h * durations / 60.0
    delta_depth_mm = np.diff(np.append(0.0, cumulative_depth_mm))

    # Alternating-block reordering.
    order = np.append(np.arange(1, len(durations) + 1, 2)[::-1], np.arange(2, len(durations) + 1, 2))
    reordered_depth_mm = np.asarray([delta_depth_mm[idx - 1] for idx in order], dtype=float)

    df = pd.DataFrame({
        "time_min": durations,
        "P_mm": reordered_depth_mm,
    })
    df["i_mm_h"] = df["P_mm"] * 60.0 / dt_min
    return df


def hyetograph_from_depths(depths_mm: List[float], dt_min: int) -> pd.DataFrame:
    """
    Build a hyetograph DataFrame from a simple list of rainfall depths per time step.

    This is often the easiest option for the comparative experiment.

    Example
    -------
    depths_mm = [2, 4, 7, 12, 9, 5, 3, 1]

    means that each time step contains that rainfall depth in mm.
    """
    depths_mm = np.asarray(depths_mm, dtype=float)
    time_min = np.arange(1, len(depths_mm) + 1) * dt_min
    df = pd.DataFrame({
        "time_min": time_min,
        "P_mm": depths_mm,
    })
    df["i_mm_h"] = df["P_mm"] * 60.0 / dt_min
    return df


def hyetograph_from_coefficients(
    coefficients: List[float],
    total_depth_mm: float,
    dt_min: int,
    normalise: bool = True,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a hyetograph from dimensionless storm coefficients and a total rainfall depth.

    Parameters
    ----------
    coefficients : list of float
        Dimensionless coefficients for each pulse.
        These should ideally sum to 1.0. If they do not, and normalise=True,
        they are automatically rescaled so that their sum becomes 1.0.

    total_depth_mm : float
        Total storm depth [mm].

    dt_min : int
        Time step [minutes].

    normalise : bool, default=True
        If True, coefficients are divided by their sum before creating the storm.

    tol : float, default=1e-6
        Tolerance used only for information messages if desired later.

    Returns
    -------
    pd.DataFrame
        DataFrame with:
            time_min : cumulative time at the end of each pulse [min]
            P_mm     : rainfall depth in each pulse [mm]
            i_mm_h   : pulse intensity [mm/h]
    """
    coeff = np.asarray(coefficients, dtype=float)

    if np.any(coeff < 0.0):
        raise ValueError("Storm coefficients must be non-negative.")

    coeff_sum = float(coeff.sum())

    if coeff_sum <= 0.0:
        raise ValueError("Storm coefficients must have a positive sum.")

    if normalise:
        coeff = coeff / coeff_sum

    pulse_depths_mm = total_depth_mm * coeff

    return hyetograph_from_depths(
        depths_mm=pulse_depths_mm.tolist(),
        dt_min=dt_min,
    )


def build_dimensionless_pattern(n_pulses: int, pattern: str) -> np.ndarray:
    """
    Build a dimensionless storm pattern that sums to 1.0.

    Parameters
    ----------
    n_pulses : int
        Number of rainfall pulses.

    pattern : str
        One of:
            - "constant"
            - "center_peaked"
            - "front_loaded"
            - "back_loaded"

    Returns
    -------
    np.ndarray
        Dimensionless pulse weights summing to 1.0
    """
    if n_pulses <= 0:
        raise ValueError("n_pulses must be positive.")

    x = np.arange(n_pulses, dtype=float)

    if pattern == "constant":
        w = np.ones(n_pulses, dtype=float)

    elif pattern == "center_peaked":
        # triangular pattern with maximum at the center
        center = 0.5 * (n_pulses - 1)
        w = 1.0 - np.abs(x - center) / max(center + 1.0, 1.0)
        w = np.clip(w, 0.05, None)   # avoid exact zeros at the tails

    elif pattern == "front_loaded":
        # larger rain at the beginning, decreasing later
        w = np.arange(n_pulses, 0, -1, dtype=float)

    elif pattern == "back_loaded":
        # smaller rain at the beginning, increasing later
        w = np.arange(1, n_pulses + 1, dtype=float)

    else:
        raise ValueError(
            "pattern must be one of: 'constant', 'center_peaked', "
            "'front_loaded', 'back_loaded'"
        )

    return w / w.sum()


def build_storm_from_pattern(
    storm_duration_min: int,
    total_depth_mm: float,
    dt_min: int,
    pattern: str,
) -> pd.DataFrame:
    """
    Build a storm from:
        - total duration
        - total depth
        - pulse time step
        - temporal pattern kind
    """
    if storm_duration_min % dt_min != 0:
        raise ValueError("storm_duration_min must be a multiple of dt_min.")

    n_pulses = int(storm_duration_min / dt_min)

    coeff = build_dimensionless_pattern(
        n_pulses=n_pulses,
        pattern=pattern,
    )

    return hyetograph_from_coefficients(
        coefficients=coeff.tolist(),
        total_depth_mm=total_depth_mm,
        dt_min=dt_min,
        normalise=True,
    )


# ======================================================================================
# 4. ANTECEDENT-CONDITION HELPERS
# ======================================================================================


def scs_adjust_cn_for_antecedent(CN_II: float, antecedent: str) -> float:
    """
    Convert average-condition CN (AMC II) to dry or wet values using common formulas.

    Formulas used
    -------------
    CN_I   = CN_II / (2.281 - 0.01281 * CN_II)
    CN_III = CN_II / (0.427 + 0.00573 * CN_II)

    Interpretation
    --------------
    - Dry antecedent condition  -> lower CN -> more infiltration, less runoff
    - Average antecedent        -> CN_II
    - Wet antecedent condition  -> higher CN -> less infiltration, more runoff
    """
    antecedent = antecedent.lower()
    if antecedent == "dry":
        return CN_II / (2.281 - 0.01281 * CN_II)
    if antecedent in ["avg", "average", "normal"]:
        return CN_II
    if antecedent == "wet":
        return CN_II / (0.427 + 0.00573 * CN_II)
    raise ValueError("antecedent must be 'dry', 'avg', or 'wet'")


# ======================================================================================
# 5. INFILTRATION / LOSS METHODS
# ======================================================================================

# All functions below return a DataFrame with, at minimum:
#   - P_mm           : rainfall depth during each pulse [mm]
#   - infil_mm       : infiltrated depth during each pulse [mm]
#   - excess_mm      : effective rainfall / runoff excess [mm]
#   - cum_infil_mm   : cumulative infiltration [mm]
#   - cum_excess_mm  : cumulative excess rainfall [mm]
#
# This makes it easy to compare methods directly before routing.


def scs_cn_excess(storm_df: pd.DataFrame, CN: float) -> pd.DataFrame:
    """
    Compute effective rainfall by the SCS Curve Number method.

    Important conceptual note
    -------------------------
    SCS-CN is fundamentally event-based, not fully dynamic in the same sense as
    Richards, Green-Ampt, or Philip. The standard runoff equation is written in terms
    of CUMULATIVE rainfall depth P.

    Standard equation
    -----------------
    S  = (25400 / CN) - 254     [mm]
    Ia = 0.2 * S                [mm]

    If P <= Ia:
        Q = 0
    else:
        Q = (P - Ia)^2 / (P + 0.8 S)

    where:
        Q = cumulative runoff excess [mm]

    Here we compute cumulative Q from cumulative P and then difference it to obtain
    incremental excess rainfall per pulse.
    """
    df = storm_df.copy().reset_index(drop=True)

    # Retention parameter S [mm]. Higher CN -> lower S -> more runoff.
    S_mm = (25400.0 / CN) - 254.0

    # Initial abstraction Ia [mm]. This lumps interception, depression storage, etc.
    Ia_mm = 0.2 * S_mm

    # Cumulative rainfall depth [mm].
    df["cumP_mm"] = df["P_mm"].cumsum()

    # Compute cumulative runoff excess Q [mm] for each cumulative rainfall depth.
    Qcum = []
    for Pcum in df["cumP_mm"].to_numpy():
        if Pcum <= Ia_mm:
            Qcum.append(0.0)
        else:
            Qcum.append(((Pcum - Ia_mm) ** 2) / (Pcum + 0.8 * S_mm))

    df["cum_excess_mm"] = np.asarray(Qcum, dtype=float)

    # Convert cumulative excess to incremental excess per pulse.
    df["excess_mm"] = df["cum_excess_mm"].diff().fillna(df["cum_excess_mm"])

    # In this event-based abstraction, infiltration/losses are simply the remainder.
    df["infil_mm"] = df["P_mm"] - df["excess_mm"]
    df["cum_infil_mm"] = df["infil_mm"].cumsum()

    # Store useful method parameters for transparency.
    df["CN"] = CN
    df["S_mm"] = S_mm
    df["Ia_mm"] = Ia_mm

    return df



def horton_excess(
    storm_df: pd.DataFrame,
    f0_mm_h: float,
    fc_mm_h: float,
    k_h_inv: float,
    dt_min: int,
) -> pd.DataFrame:
    """
    Compute excess rainfall using the Horton infiltration-capacity equation.

    Horton equation
    ---------------
    f(t) = fc + (f0 - fc) * exp(-k t)

    where:
        f0 = initial infiltration capacity [mm/h]
        fc = final infiltration capacity [mm/h]
        k  = decay coefficient [1/h]
        t  = elapsed time since rainfall started [h]

    Interpretation
    --------------
    Horton is empirical. It assumes infiltration capacity decays exponentially from an
    initially high value toward a final value. It does not explicitly track moisture storage
    or ponding the way Green-Ampt / Richards do.

    Practical implementation used here
    ----------------------------------
    For each pulse:
        infiltration depth = min(rainfall depth, infiltration capacity * dt)
        excess depth       = rainfall depth - infiltration depth
    """
    df = storm_df.copy().reset_index(drop=True)

    dt_h = dt_min / 60.0
    rain = df["P_mm"].to_numpy(dtype=float)

    elapsed_h = np.zeros(len(df), dtype=float)
    wet_time = 0.0
    for i in range(len(df)):
        elapsed_h[i] = wet_time
        if rain[i] > 0.0:
            wet_time += dt_h

    # Infiltration capacity at the start of each pulse [mm/h].
    fcap_mm_h = fc_mm_h + (f0_mm_h - fc_mm_h) * np.exp(-k_h_inv * elapsed_h)

    # Maximum infiltration depth allowed during each pulse [mm].
    max_infil_mm = fcap_mm_h * dt_h

    # Actual infiltration cannot exceed available rainfall depth.
    infil_mm = np.minimum(df["P_mm"].to_numpy(), max_infil_mm)
    excess_mm = df["P_mm"].to_numpy() - infil_mm

    df["fcap_mm_h"] = fcap_mm_h
    df["infil_mm"] = infil_mm
    df["excess_mm"] = excess_mm
    df["cum_infil_mm"] = df["infil_mm"].cumsum()
    df["cum_excess_mm"] = df["excess_mm"].cumsum()

    return df



def philip_excess(
    storm_df: pd.DataFrame,
    S_mm_h05: float,
    Ks_mm_h: float,
    dt_min: int,
) -> pd.DataFrame:
    """
    Compute excess rainfall using Philip's infiltration model.

    Philip equations
    ----------------
    Cumulative infiltration:
        F(t) = S * t^(1/2) + Ks * t

    Infiltration rate:
        f(t) = 0.5 * S / sqrt(t) + Ks

    where:
        S  = sorptivity [mm / h^0.5]
        Ks = saturated hydraulic conductivity [mm/h]
        t  = time [h]

    Important practical note
    ------------------------
    The formal Philip infiltration rate tends to infinity as t -> 0.
    Numerically that is inconvenient. A robust way to use the model with time steps is:

        - compute cumulative infiltration F(t_end)
        - subtract F(t_start)
        - that gives the infiltration capacity over the pulse
        - actual infiltration = min(rainfall depth, capacity over pulse)

    This is what we do here.
    """
    df = storm_df.copy().reset_index(drop=True)

    dt_h = dt_min / 60.0
    rain = df["P_mm"].to_numpy(dtype=float)

    t_start_h = np.zeros(len(df), dtype=float)
    t_end_h = np.zeros(len(df), dtype=float)

    wet_time = 0.0
    for i in range(len(df)):
        t_start_h[i] = wet_time
        if rain[i] > 0.0:
            wet_time += dt_h
        t_end_h[i] = wet_time

    def F_philip(t_h: np.ndarray) -> np.ndarray:
        """Cumulative Philip infiltration [mm] at time t [h]."""
        return S_mm_h05 * np.sqrt(t_h) + Ks_mm_h * t_h

    # Cumulative infiltration at the beginning and end of each pulse.
    F_end = F_philip(t_end_h)
    F_start = F_philip(t_start_h)

    # Potential infiltration during each pulse [mm].
    potential_infil_mm = F_end - F_start

    # Actual infiltration cannot exceed rainfall depth in the pulse.
    infil_mm = np.minimum(df["P_mm"].to_numpy(), potential_infil_mm)
    excess_mm = df["P_mm"].to_numpy() - infil_mm

    df["potential_infil_mm"] = potential_infil_mm
    df["infil_mm"] = infil_mm
    df["excess_mm"] = excess_mm
    df["cum_infil_mm"] = df["infil_mm"].cumsum()
    df["cum_excess_mm"] = df["excess_mm"].cumsum()

    return df


# -------------------------------
# Green-Ampt support: exact step
# -------------------------------

def _green_ampt_deltaF_ponded(
    F0_mm: float,
    dt_h: float,
    Ks_mm_h: float,
    A_mm: float,
    itmax: int = 60,
    tol: float = 1e-10,
) -> float:
    """
    Exact Green-Ampt cumulative-infiltration increment over a fully ponded interval.

    Green-Ampt ponded equation (integrated form)
    --------------------------------------------
    For ponded infiltration:

        dF - A ln((F0 + dF + A)/(F0 + A)) = Ks * dt

    where:
        A = psi_f * (theta_s - theta_i)

    This equation is nonlinear in dF, so we solve it with Newton iterations.

    Why this helper is important
    ----------------------------
    A crude explicit approximation can become inaccurate around the onset of ponding.
    This exact integrated step is much more robust and is in the same spirit as the more
    careful implementation you already used in your GA notebook.
    """
    dF = max(Ks_mm_h * dt_h, 1e-12)  # robust initial guess

    for _ in range(itmax):
        denom = F0_mm + dF + A_mm
        denom = max(denom, 1e-16)

        g = dF - A_mm * np.log(denom / (F0_mm + A_mm)) - Ks_mm_h * dt_h
        dg = 1.0 - A_mm / denom
        dg = dg if abs(dg) > 1e-16 else 1e-16

        dF_new = dF - g / dg

        if abs(dF_new - dF) <= tol * max(1.0, abs(dF)):
            dF = dF_new
            break
        dF = dF_new

    return max(dF, 0.0)



def green_ampt_excess(
    storm_df: pd.DataFrame,
    Ks_mm_h: float,
    psi_f_mm: float,
    theta_s: float,
    theta_i: float,
    dt_min: int,
) -> pd.DataFrame:
    """
    Compute excess rainfall using a Green-Ampt infiltration model with ponding detection.

    Green-Ampt physical idea
    ------------------------
    The soil above the wetting front is assumed to become saturated, while the soil below
    the front remains at its initial water content.

    The main capacity expression can be written as:

        f = Ks * (1 + A / F)

    where:
        A = psi_f * (theta_s - theta_i)
        F = cumulative infiltration

    Interpretation
    --------------
    - early in the event, F is small, so infiltration capacity can be very high
    - as infiltration accumulates, F increases and capacity decreases toward Ks

    Why this method is a good comparator to Richards / EFV
    ------------------------------------------------------
    Because it uses some of the SAME core physical quantities:
        - Ks
        - theta_s
        - theta_i
        - capillary suction scale (psi_f)

    That makes it much more physically related to your EFV solver than SCS or Horton.
    """
    df = storm_df.copy().reset_index(drop=True)

    dt_h = dt_min / 60.0
    rainfall_depth_mm = df["P_mm"].to_numpy(dtype=float)
    rainfall_intensity_mm_h = rainfall_depth_mm / dt_h

    delta_theta = theta_s - theta_i
    if delta_theta <= 0:
        raise ValueError("theta_s must be greater than theta_i for Green-Ampt.")

    A_mm = psi_f_mm * delta_theta
    F_mm = 1e-9

    infil_mm_list = []
    excess_mm_list = []
    fcap_mm_h_list = []
    ponded_list = []

    for Ppulse_mm, r_mm_h in zip(rainfall_depth_mm, rainfall_intensity_mm_h):
        fcap_start_mm_h = Ks_mm_h * (1.0 + A_mm / max(F_mm, 1e-9))

        if Ppulse_mm <= 0.0:
            infil_mm = 0.0
            excess_mm = 0.0
            ponded = False

        elif r_mm_h <= Ks_mm_h:
            # Rainfall can never exceed long-time capacity in this pulse
            infil_mm = Ppulse_mm
            excess_mm = 0.0
            ponded = False
            F_mm += infil_mm

        elif r_mm_h >= fcap_start_mm_h:
            # Already ponded at pulse start
            dF_mm = _green_ampt_deltaF_ponded(
                F0_mm=F_mm,
                dt_h=dt_h,
                Ks_mm_h=Ks_mm_h,
                A_mm=A_mm,
            )
            infil_mm = min(dF_mm, Ppulse_mm)
            excess_mm = Ppulse_mm - infil_mm
            ponded = True
            F_mm += infil_mm

        else:
            # Rainfall below start capacity but above Ks, so ponding may begin mid-step
            Fp_mm = A_mm / (r_mm_h / Ks_mm_h - 1.0)

            if F_mm >= Fp_mm:
                dF_mm = _green_ampt_deltaF_ponded(
                    F0_mm=F_mm,
                    dt_h=dt_h,
                    Ks_mm_h=Ks_mm_h,
                    A_mm=A_mm,
                )
                infil_mm = min(dF_mm, Ppulse_mm)
                excess_mm = Ppulse_mm - infil_mm
                ponded = True
                F_mm += infil_mm
            else:
                t_pond_h = (Fp_mm - F_mm) / r_mm_h

                if t_pond_h >= dt_h:
                    infil_mm = Ppulse_mm
                    excess_mm = 0.0
                    ponded = False
                    F_mm += infil_mm
                else:
                    infil_before_mm = r_mm_h * t_pond_h
                    dF_after_mm = _green_ampt_deltaF_ponded(
                        F0_mm=Fp_mm,
                        dt_h=dt_h - t_pond_h,
                        Ks_mm_h=Ks_mm_h,
                        A_mm=A_mm,
                    )
                    infil_mm = min(infil_before_mm + dF_after_mm, Ppulse_mm)
                    excess_mm = Ppulse_mm - infil_mm
                    ponded = True
                    F_mm += infil_mm

        infil_mm_list.append(infil_mm)
        excess_mm_list.append(excess_mm)
        fcap_mm_h_list.append(fcap_start_mm_h)
        ponded_list.append(ponded)

    df["fcap_start_mm_h"] = fcap_mm_h_list
    df["ponded"] = ponded_list
    df["infil_mm"] = infil_mm_list
    df["excess_mm"] = excess_mm_list
    df["cum_infil_mm"] = df["infil_mm"].cumsum()
    df["cum_excess_mm"] = df["excess_mm"].cumsum()

    return df


# -------------------------------
# EFV hook / placeholder wrapper
# -------------------------------

def _efv_condf(theta: float, thr: float, ths: float, alpha_cm_inv: float, m_vg: float, Ks_cm_s: float) -> float:
    """
    van Genuchten-Mualem hydraulic conductivity function used by the EFV notebook.

    Parameters
    ----------
    theta : float
        Volumetric water content [-].
    thr, ths : float
        Residual and saturated water content [-].
    alpha_cm_inv : float
        van Genuchten alpha [1/cm]. Included for consistency with the head function.
    m_vg : float
        van Genuchten m parameter [-].
    Ks_cm_s : float
        Saturated hydraulic conductivity [cm/s].

    Returns
    -------
    float
        Unsaturated hydraulic conductivity [cm/s].
    """
    if theta >= ths:
        return Ks_cm_s

    se = (theta - thr) / (ths - thr)
    se = min(max(se, 1e-12), 1.0)

    return Ks_cm_s * (se ** 0.5) * (1.0 - (1.0 - se ** (1.0 / m_vg)) ** m_vg) ** 2


def _efv_headf(theta: float, thr: float, ths: float, alpha_cm_inv: float, m_vg: float) -> float:
    """
    Pressure head function [cm] used in the uploaded EFV notebook.
    """
    if theta >= ths:
        return 0.0

    n_vg = 1.0 / (1.0 - m_vg)

    se = (theta - thr) / (ths - thr)
    se = min(max(se, 1e-12), 1.0)

    return -((se ** (-1.0 / m_vg) - 1.0) ** (1.0 / n_vg)) / alpha_cm_inv


def _efv_surface_capacity_flux(
    theta0: float,
    thr: float,
    ths: float,
    alpha_cm_inv: float,
    m_vg: float,
    Ks_cm_s: float,
    wet_theta: float,
    dz_cm: float,
    avg_kind: int = 0,
) -> float:
    """
    Surface infiltration capacity flux [cm/s] assuming ponding at the upper boundary.

    Positive flux means downward infiltration.
    """
    h_old = _efv_headf(wet_theta, thr, ths, alpha_cm_inv, m_vg)
    K_old = _efv_condf(wet_theta, thr, ths, alpha_cm_inv, m_vg, Ks_cm_s)

    h0 = _efv_headf(theta0, thr, ths, alpha_cm_inv, m_vg)
    K0 = _efv_condf(theta0, thr, ths, alpha_cm_inv, m_vg, Ks_cm_s)

    if avg_kind == 0:
        Kbar = 0.5 * (K_old + K0)
    else:
        Kbar = np.sqrt(K_old * K0) if K_old > 0 and K0 > 0 else 0.0

    return -Kbar * ((h0 - h_old) / dz_cm - 1.0)


def _efv_interior_fluxes(
    head_arr: np.ndarray,
    K_arr: np.ndarray,
    dz_cm: float,
    avg_kind: int = 0,
) -> np.ndarray:
    """
    Compute interior fluxes q_{i+1/2} [cm/s] between adjacent nodes.
    Positive flux means downward.
    """
    nz = len(head_arr)
    qmid = np.zeros(nz - 1)

    for i in range(nz - 1):
        if avg_kind == 0:
            Kbar = 0.5 * (K_arr[i] + K_arr[i + 1])
        else:
            Ka, Kb = K_arr[i], K_arr[i + 1]
            Kbar = np.sqrt(Ka * Kb) if Ka > 0 and Kb > 0 else 0.0

        qmid[i] = -Kbar * ((head_arr[i + 1] - head_arr[i]) / dz_cm - 1.0)

    return qmid


def _efv_step_explicit_fluxBC(
    theta: np.ndarray,
    dt_s: float,
    dz_cm: float,
    q_surface_cm_s: float,
    thr: float,
    ths: float,
    alpha_cm_inv: float,
    m_vg: float,
    Ks_cm_s: float,
    avg_kind: int = 0,
    bottom_bc: str = "free_drainage",
) -> np.ndarray:
    """
    One explicit EFV theta-step with:
      - top boundary: prescribed flux q_surface_cm_s [cm/s]
      - bottom boundary: "free_drainage" or "no_flow"
    """
    theta = theta.copy()

    h = np.array([_efv_headf(th, thr, ths, alpha_cm_inv, m_vg) for th in theta])
    K = np.array([_efv_condf(th, thr, ths, alpha_cm_inv, m_vg, Ks_cm_s) for th in theta])
    qmid = _efv_interior_fluxes(h, K, dz_cm, avg_kind)

    # Top node
    theta[0] += -(dt_s / dz_cm) * (qmid[0] - q_surface_cm_s)

    # Interior nodes
    for i in range(1, len(theta) - 1):
        theta[i] += -(dt_s / dz_cm) * (qmid[i] - qmid[i - 1])

    # Bottom node
    if bottom_bc == "no_flow":
        q_bottom = 0.0
    elif bottom_bc == "free_drainage":
        q_bottom = K[-1]   # unit-gradient drainage, positive downward
    else:
        raise ValueError("bottom_bc must be 'no_flow' or 'free_drainage'")

    theta[-1] += -(dt_s / dz_cm) * (q_bottom - qmid[-1])

    np.clip(theta, thr, ths, out=theta)
    return theta


def efv_excess(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    bottom_bc: str = "free_drainage",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Run the EFV Richards-based infiltration model extracted from the uploaded notebook.

    Inputs
    ------
    storm_df must contain:
        - time_min : cumulative end-of-pulse time [min]
        - P_mm     : rainfall depth during each pulse [mm]

    Soil requirements
    -----------------
    The soil entry must provide:
        - theta_r
        - theta_s
        - theta_i(antecedent)
        - Ks_mm_h
        - efv_alpha_cm_inv
        - efv_n

    Returns
    -------
    out_df : pd.DataFrame
        Pulse-by-pulse infiltration and excess rainfall results.
    theta_final : np.ndarray
        Final soil moisture profile.
    """
    if soil.efv_alpha_cm_inv is None or soil.efv_n is None:
        raise ValueError(
            f"EFV parameters are missing for soil '{soil.name}'. "
            "Please define efv_alpha_cm_inv and efv_n in the soil library."
        )

    # Convert the current framework storm format into the format expected by the notebook logic
    blocks_df = pd.DataFrame({
        "Dur_min": storm_df["time_min"].astype(float).values,
        "P_mm": storm_df["P_mm"].astype(float).values,
    })

    dur = blocks_df["Dur_min"].values
    dt_min_arr = np.empty_like(dur)
    dt_min_arr[0] = dur[0]
    dt_min_arr[1:] = np.diff(dur)

    if np.any(dt_min_arr <= 0):
        raise ValueError("Storm cumulative times must be strictly increasing.")

    P_mm_arr = blocks_df["P_mm"].values

    # Soil / EFV parameters
    nz = soil.efv_nz
    dz_cm = soil.efv_dz_cm
    dt_sub_s = soil.efv_dt_sub_s
    avg_kind = soil.efv_avg_kind

    thr = soil.theta_r
    ths = soil.theta_s
    theta_ini = soil.theta_i(antecedent)
    wet_theta = soil.efv_surface_theta_value()

    alpha_cm_inv = soil.efv_alpha_cm_inv
    m_vg = soil.efv_m_value()
    Ks_cm_s = soil.Ks_mm_h / 36000.0  # mm/h -> cm/s

    # Initial profile: uniform theta
    theta = np.full(nz, theta_ini, dtype=float)

    infil_per_pulse = []
    runoff_per_pulse = []
    qcap_start_mmph = []

    for k in range(len(blocks_df)):
        block_dt_s = dt_min_arr[k] * 60.0
        rainfall_cm_s = (P_mm_arr[k] / 10.0) / block_dt_s

        qcap0_cm_s = _efv_surface_capacity_flux(
            theta0=theta[0],
            thr=thr,
            ths=ths,
            alpha_cm_inv=alpha_cm_inv,
            m_vg=m_vg,
            Ks_cm_s=Ks_cm_s,
            wet_theta=wet_theta,
            dz_cm=dz_cm,
            avg_kind=avg_kind,
        )
        qcap_start_mmph.append(qcap0_cm_s * 3600.0 * 10.0)

        infil_cm = 0.0
        nsub = int(np.ceil(block_dt_s / dt_sub_s))
        dt_eff_s = block_dt_s / nsub

        for _ in range(nsub):
            qcap_cm_s = _efv_surface_capacity_flux(
                theta0=theta[0],
                thr=thr,
                ths=ths,
                alpha_cm_inv=alpha_cm_inv,
                m_vg=m_vg,
                Ks_cm_s=Ks_cm_s,
                wet_theta=wet_theta,
                dz_cm=dz_cm,
                avg_kind=avg_kind,
            )

            q_surface_cm_s = min(max(rainfall_cm_s, 0.0), max(qcap_cm_s, 0.0))

            theta = _efv_step_explicit_fluxBC(
                theta=theta,
                dt_s=dt_eff_s,
                dz_cm=dz_cm,
                q_surface_cm_s=q_surface_cm_s,
                thr=thr,
                ths=ths,
                alpha_cm_inv=alpha_cm_inv,
                m_vg=m_vg,
                Ks_cm_s=Ks_cm_s,
                avg_kind=avg_kind,
                bottom_bc=bottom_bc,
            )

            infil_cm += q_surface_cm_s * dt_eff_s

        infil_mm = infil_cm * 10.0
        runoff_mm = max(P_mm_arr[k] - infil_mm, 0.0)

        infil_per_pulse.append(infil_mm)
        runoff_per_pulse.append(runoff_mm)

    out_df = storm_df.copy().reset_index(drop=True)
    out_df["qcap_start_mmph"] = qcap_start_mmph
    out_df["infil_mm"] = infil_per_pulse
    out_df["excess_mm"] = runoff_per_pulse
    out_df["cum_infil_mm"] = out_df["infil_mm"].cumsum()
    out_df["cum_excess_mm"] = out_df["excess_mm"].cumsum()

    return out_df, theta


# ======================================================================================
# 6. COMMON UNIT-HYDROGRAPH ROUTING
# ======================================================================================

# To make the hydrograph comparison fair, the SAME routing is used for all methods.
#
# Below we keep a simple triangular unit hydrograph similar in spirit to your previous
# notebooks. You could later replace it with an FEH-style UH if you wish, but the key
# scientific requirement remains the same: one routing method for all infiltration methods.


def triangular_unit_hydrograph(catchment: CatchmentScenario, dt_min: int) -> pd.DataFrame:
    """
    Build a simple triangular unit hydrograph (UH).

    Formulas used (same spirit as your notebooks)
    ---------------------------------------------
    tlag = 0.6 * tc
    de   = 2 * sqrt(tc)
    tp   = tlag + 0.5 * de
    tr   = 1.67 * tp
    tb   = 2.67 * tp
    qp   = 0.208 * A / tp

    where:
        tc = concentration time [h]
        A  = catchment area [km^2]
        tp = peak time [h]
        tb = base time [h]
        qp = unit peak flow [m^3/s per mm of excess rainfall]

    Why this is acceptable here
    ---------------------------
    The purpose of the chapter is NOT to innovate the routing method.
    The purpose is to isolate how infiltration representation changes runoff response.
    Therefore, a simple and fixed UH is appropriate.
    """
    tc_h = catchment.tc_h
    A_km2 = catchment.area_km2

    tlag_h = 0.6 * tc_h
    de_h = 2.0 * math.sqrt(tc_h)
    tp_h = tlag_h + 0.5 * de_h
    tr_h = 1.67 * tp_h
    tb_h = 2.67 * tp_h
    qp = 0.208 * A_km2 / tp_h

    # Number of UH pulses at the chosen dt.
    tb_min = tb_h * 60.0
    n_pulses = int(np.ceil(tb_min / dt_min)) + 1

    time_min = np.arange(n_pulses) * dt_min
    time_h = time_min / 60.0

    uh = []
    for t_h in time_h:
        if t_h < tp_h:
            q = (t_h / tp_h) * qp
        else:
            q = qp - ((t_h - tp_h) / (tb_h - tp_h)) * qp
        uh.append(max(q, 0.0))

    return pd.DataFrame({
        "time_min": time_min,
        "UH_m3s_per_mm": uh,
        "tp_h": tp_h,
        "tb_h": tb_h,
        "qp_m3s_per_mm": qp,
        "tc_h": tc_h,
        "tr_h": tr_h,
        "de_h": de_h,
        "tlag_h": tlag_h,
    })



def convolve_excess_with_uh(
    excess_mm: np.ndarray,
    uh_m3s_per_mm: np.ndarray,
    baseflow_m3_s: float = 0.0,
) -> np.ndarray:
    """
    Route incremental excess rainfall through the unit hydrograph by discrete convolution.

    Interpretation
    --------------
    Each pulse of excess rainfall acts like a small impulse that produces a shifted UH
    response. The total hydrograph is the sum of all shifted responses.

    Numerically, this is exactly the discrete convolution of:
        excess rainfall pulses [mm]
    with:
        unit hydrograph ordinates [m^3/s per mm]

    We then optionally add a constant baseflow.
    """
    direct_runoff = np.convolve(excess_mm, uh_m3s_per_mm)
    return direct_runoff + baseflow_m3_s


# ======================================================================================
# 7. METRICS AND SUMMARY FUNCTIONS
# ======================================================================================

def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """
    Version-safe trapezoidal integration.

    Uses np.trapezoid when available, and falls back to np.trapz for older
    NumPy versions or environments where np.trapezoid is missing.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))

    return float(np.trapz(y, x))


def safe_relative_value(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    """
    Safe relative ratio numerator / denominator.

    Returns NaN if the denominator is too close to zero.
    """
    if abs(denominator) <= eps:
        return float("nan")
    return float(numerator / denominator)


def hydrograph_metrics(time_min: np.ndarray, Q_m3_s: np.ndarray) -> Dict[str, float]:
    """
    Compute a small set of hydrograph metrics.

    Metrics returned
    ----------------
    peak_flow_m3_s : hydrograph peak
    time_to_peak_min : time of peak discharge
    runoff_volume_m3 : integral of discharge over time

    Note
    ----
    Runoff volume here is computed from the routed hydrograph as the time integral of Q.
    If a constant baseflow is included, then the total volume includes baseflow too.
    If you want direct-runoff volume only, set baseflow to zero or subtract it beforehand.
    """
    peak_idx = int(np.argmax(Q_m3_s))
    peak_flow = float(Q_m3_s[peak_idx])
    t_peak = float(time_min[peak_idx])

    # Time integration using the trapezoidal rule.
    Q_m3_s = np.asarray(Q_m3_s, dtype=float)
    time_s = np.asarray(time_min, dtype=float) * 60.0
    volume_m3 = trapezoid_integral(Q_m3_s, time_s)

    return {
        "peak_flow_m3_s": peak_flow,
        "time_to_peak_min": t_peak,
        "runoff_volume_m3": volume_m3,
    }


def compare_to_reference(
    ref_method_df: pd.DataFrame,
    test_method_df: pd.DataFrame,
    ref_hydro_df: pd.DataFrame,
    test_hydro_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compare a tested method against a reference method (normally EFV).

    Returns
    -------
    dict
        Absolute and relative errors / biases relative to the reference.
    """
    if len(ref_method_df) != len(test_method_df):
        raise ValueError("Method DataFrames must have the same length for comparison.")

    if len(ref_hydro_df) != len(test_hydro_df):
        raise ValueError("Hydrograph DataFrames must have the same length for comparison.")

    ref_excess = ref_method_df["excess_mm"].to_numpy(dtype=float)
    test_excess = test_method_df["excess_mm"].to_numpy(dtype=float)

    ref_infil_final = float(ref_method_df["cum_infil_mm"].iloc[-1])
    test_infil_final = float(test_method_df["cum_infil_mm"].iloc[-1])

    ref_excess_final = float(ref_method_df["cum_excess_mm"].iloc[-1])
    test_excess_final = float(test_method_df["cum_excess_mm"].iloc[-1])

    ref_peak = float(ref_hydro_df["Q_m3_s"].max())
    test_peak = float(test_hydro_df["Q_m3_s"].max())

    ref_time_s = ref_hydro_df["time_min"].to_numpy(dtype=float) * 60.0
    test_time_s = test_hydro_df["time_min"].to_numpy(dtype=float) * 60.0

    ref_volume = trapezoid_integral(
        ref_hydro_df["Q_m3_s"].to_numpy(dtype=float),
        ref_time_s,
    )
    test_volume = trapezoid_integral(
        test_hydro_df["Q_m3_s"].to_numpy(dtype=float),
        test_time_s,
    )

    rmse_excess = float(np.sqrt(np.mean((test_excess - ref_excess) ** 2)))

    # Normalise RMSE of excess by final cumulative EFV excess depth.
    # This makes the metric more comparable across storms of different magnitudes.
    nrmse_excess = safe_relative_value(rmse_excess, ref_excess_final)

    # Relative hydrograph biases
    peak_rel_bias = safe_relative_value(test_peak - ref_peak, ref_peak)
    volume_rel_bias = safe_relative_value(test_volume - ref_volume, ref_volume)

    return {
        "bias_cum_infil_mm_vs_ref": test_infil_final - ref_infil_final,
        "bias_cum_excess_mm_vs_ref": test_excess_final - ref_excess_final,
        "rmse_excess_mm_vs_ref": rmse_excess,
        "nrmse_excess_vs_ref": nrmse_excess,
        "peak_bias_m3_s_vs_ref": test_peak - ref_peak,
        "peak_rel_bias_vs_ref": peak_rel_bias,
        "volume_bias_m3_vs_ref": test_volume - ref_volume,
        "volume_rel_bias_vs_ref": volume_rel_bias,
    }


def storm_descriptors(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
) -> Dict[str, float]:
    """
    Compute storm and soil-state descriptors useful for synthesis plots.
    """
    total_depth_mm = float(storm_df["P_mm"].sum())
    duration_min = float(storm_df["time_min"].iloc[-1])
    duration_h = duration_min / 60.0

    mean_intensity_mm_h = total_depth_mm / duration_h if duration_h > 0 else np.nan
    peak_intensity_mm_h = float(storm_df["i_mm_h"].max())

    Ks_mm_h = float(soil.Ks_mm_h)

    theta_i = float(soil.theta_i(antecedent))
    theta_deficit = float(soil.theta_s - theta_i)

    # Optional storage-deficit estimate over the EFV profile depth
    profile_depth_cm = soil.efv_nz * soil.efv_dz_cm
    profile_storage_deficit_mm = profile_depth_cm * theta_deficit * 10.0

    return {
        "storm_total_depth_mm_check": total_depth_mm,
        "storm_duration_min_check": duration_min,
        "mean_intensity_mm_h": mean_intensity_mm_h,
        "peak_intensity_mm_h": peak_intensity_mm_h,
        "Ks_mm_h": Ks_mm_h,
        "mean_i_over_Ks": mean_intensity_mm_h / Ks_mm_h if Ks_mm_h > 0 else np.nan,
        "peak_i_over_Ks": peak_intensity_mm_h / Ks_mm_h if Ks_mm_h > 0 else np.nan,
        "theta_i": theta_i,
        "theta_deficit": theta_deficit,
        "profile_storage_deficit_mm": profile_storage_deficit_mm,
    }


# ======================================================================================
# 8. METHOD WRAPPERS USING THE COMMON SOIL DESCRIPTION
# ======================================================================================

# These functions link the simplified methods to the same underlying soil description.
#
# In structural mode:
#   - Green-Ampt uses direct soil parameters
#   - EFV uses direct hydraulic parameters
#   - Horton uses direct scenario parameters (f0, fc, k)
#   - Philip uses direct scenario sorptivity S
#   - SCS-CN uses direct CN_II adjusted for antecedent condition
#
# In surrogate mode:
#   - Horton, Philip and SCS-CN may instead use calibrated parameters
#     fitted to a reference method over separate calibration storms.


def calibrate_horton_from_green_ampt(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
    k_search_min: float = 0.05,
    k_search_max: float = 3.0,
    k_search_steps: int = 400,
) -> Dict[str, float]:
    """
    Calibrate Horton parameters from Green-Ampt for one soil and antecedent condition.

    Why this calibration is needed
    ------------------------------
    Horton uses:
        - f0 : initial infiltration capacity
        - fc : final infiltration capacity
        - k  : exponential decay rate

    Of these:
        - fc can be linked directly to Ks
        - f0 and k are empirical shape parameters

    Therefore Horton cannot be assigned directly from the soil in the same way as
    Green-Ampt. Instead, it is calibrated so that its cumulative infiltration response
    resembles the Green-Ampt response for the same soil and antecedent condition.

    Calibration strategy
    --------------------
    1. Run Green-Ampt for the chosen reference storm.
    2. Fix Horton fc equal to the same soil Ks.
    3. Estimate Horton f0 from the mean early Green-Ampt actual infiltration rate.
    4. Search for the Horton k value that best matches Green-Ampt cumulative infiltration.

    Returns
    -------
    dict with:
        f0_mm_h
        fc_mm_h
        k_h_inv
        rmse_cum_infil_mm
    """

    ga_df = green_ampt_excess(
        storm_df=storm_df,
        Ks_mm_h=soil.Ks_mm_h,
        psi_f_mm=soil.psi_f_mm,
        theta_s=soil.theta_s,
        theta_i=soil.theta_i(antecedent),
        dt_min=dt_min,
    )

    # Final Horton capacity is tied directly to the same saturated conductivity
    fc_mm_h = soil.Ks_mm_h
    dt_h = dt_min / 60.0

    # Estimate an early effective infiltration rate from the first two pulses
    # This avoids using the theoretical Green-Ampt initial capacity directly,
    # which can be unrealistically large when cumulative infiltration is near zero.
    n_early = min(2, len(ga_df))
    early_infil_mm = ga_df["infil_mm"].iloc[:n_early].to_numpy()
    f0_mm_h = float(np.mean(early_infil_mm / dt_h))

    # Keep f0 physically sensible
    f0_mm_h = max(f0_mm_h, fc_mm_h + 1e-6)
    f0_mm_h = min(f0_mm_h, 150.0)

    ga_cum_infil = ga_df["cum_infil_mm"].to_numpy()

    best_k = None
    best_rmse = np.inf

    k_values = np.linspace(k_search_min, k_search_max, k_search_steps)

    for k in k_values:
        h_df = horton_excess(
            storm_df=storm_df,
            f0_mm_h=f0_mm_h,
            fc_mm_h=fc_mm_h,
            k_h_inv=k,
            dt_min=dt_min,
        )

        h_cum_infil = h_df["cum_infil_mm"].to_numpy()
        rmse = float(np.sqrt(np.mean((h_cum_infil - ga_cum_infil) ** 2)))

        if rmse < best_rmse:
            best_rmse = rmse
            best_k = k

    return {
        "f0_mm_h": float(f0_mm_h),
        "fc_mm_h": float(fc_mm_h),
        "k_h_inv": float(best_k),
        "rmse_cum_infil_mm": float(best_rmse),
    }


def calibrate_cn_from_green_ampt(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
    cn_min: float = 45.0,
    cn_max: float = 95.0,
    cn_steps: int = 500,
) -> Dict[str, float]:
    """
    Calibrate SCS Curve Number from Green-Ampt for one soil and antecedent condition.

    Why this calibration is needed
    ------------------------------
    SCS-CN does not use direct soil hydraulic parameters such as Ks or theta.
    Instead, it represents runoff generation through a lumped event-scale parameter:
        CN

    Therefore CN cannot be assigned as directly as Green-Ampt parameters.
    It must be chosen so that SCS-CN reproduces the same overall runoff tendency
    as the reference method for the same soil condition.

    Why the calibration storm must be peaked
    ----------------------------------------
    In this script, SCS-CN is applied through cumulative rainfall in time.
    Therefore storm temporal structure still matters.

    For that reason, SCS-CN should be calibrated using a realistic peaked reference storm,
    not a constant storm. A constant storm is suitable for Horton calibration, but not
    for SCS-CN calibration.

    Calibration strategy
    --------------------
    1. Run Green-Ampt under the chosen peaked reference storm.
    2. Extract the final cumulative runoff excess depth.
    3. Search for the CN value that makes SCS-CN reproduce that same final runoff depth.

    Returns
    -------
    dict with:
        CN_match
        Q_ga_final_mm
        Q_scs_final_mm
        abs_error_mm
    """

    ga_df = green_ampt_excess(
        storm_df=storm_df,
        Ks_mm_h=soil.Ks_mm_h,
        psi_f_mm=soil.psi_f_mm,
        theta_s=soil.theta_s,
        theta_i=soil.theta_i(antecedent),
        dt_min=dt_min,
    )

    Q_ga_final_mm = float(ga_df["cum_excess_mm"].iloc[-1])

    best_cn = None
    best_err = np.inf
    best_q_scs = None

    cn_values = np.linspace(cn_min, cn_max, cn_steps)

    for cn in cn_values:
        scs_df = scs_cn_excess(storm_df, CN=cn)
        Q_scs_final_mm = float(scs_df["cum_excess_mm"].iloc[-1])

        err = abs(Q_scs_final_mm - Q_ga_final_mm)

        if err < best_err:
            best_err = err
            best_cn = cn
            best_q_scs = Q_scs_final_mm

    return {
        "CN_match": float(best_cn),
        "Q_ga_final_mm": float(Q_ga_final_mm),
        "Q_scs_final_mm": float(best_q_scs),
        "abs_error_mm": float(best_err),
    }


def build_calibrated_horton_parameters(
    reference_storm_df: pd.DataFrame,
    soils: Dict[str, SoilScenario],
    dt_min: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build Horton parameter dictionaries for all soils and antecedent states.

    Why this uses a constant storm
    ------------------------------
    Horton is an infiltration-capacity decay model. A capacity-controlled constant storm
    is a reasonable way to expose the key behaviour Horton is meant to represent:
        - initial infiltration capacity
        - long-term infiltration capacity
        - decay through time
    """
    horton_params = {}

    for soil_key, soil in soils.items():
        horton_params[soil_key] = {}

        for antecedent in ["dry", "avg", "wet"]:
            horton_params[soil_key][antecedent] = calibrate_horton_from_green_ampt(
                storm_df=reference_storm_df,
                soil=soil,
                antecedent=antecedent,
                dt_min=dt_min,
            )

    return horton_params


def build_calibrated_scs_parameters(
    reference_storm_df: pd.DataFrame,
    soils: Dict[str, SoilScenario],
    dt_min: int,
) -> Dict[str, Dict[str, float]]:
    """
    Build SCS-CN parameter dictionaries for all soils.

    Why this uses a peaked storm
    ----------------------------
    SCS-CN is an event-based runoff method and, in this script, it is applied using
    cumulative rainfall through time. Therefore its calibration should reflect a realistic
    storm temporal pattern similar to the family of storms used in evaluation.

    The average antecedent condition is calibrated first as CN_II.
    Then CN_I and CN_III are derived using standard antecedent-moisture formulas.
    """
    scs_params = {}

    for soil_key, soil in soils.items():
        cn_fit = calibrate_cn_from_green_ampt(
            storm_df=reference_storm_df,
            soil=soil,
            antecedent="avg",
            dt_min=dt_min,
        )

        CN_II = cn_fit["CN_match"]
        CN_I = CN_II / (2.281 - 0.01281 * CN_II)
        CN_III = CN_II / (0.427 + 0.00573 * CN_II)

        scs_params[soil_key] = {
            "CN_I": float(CN_I),
            "CN_II": float(CN_II),
            "CN_III": float(CN_III),
            "fit_abs_error_mm": float(cn_fit["abs_error_mm"]),
            "Q_ga_ref_mm": float(cn_fit["Q_ga_final_mm"]),
            "Q_scs_ref_mm": float(cn_fit["Q_scs_final_mm"]),
        }

    return scs_params

def estimate_philip_sorptivity_from_efv(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
    n_fit_pulses: int = 3,
) -> Dict[str, float]:
    """
    Estimate Philip sorptivity S [mm / h^0.5] from early-time EFV cumulative infiltration.

    Why this is done
    ----------------
    Philip's equation is:

        F(t) = S * sqrt(t) + Ks * t

    where:
        F(t) = cumulative infiltration [mm]
        S    = sorptivity [mm / h^0.5]
        Ks   = saturated hydraulic conductivity [mm/h]
        t    = time [h]

    In this framework, EFV is the most physical Richards-based reference model.
    Therefore, the most consistent way to parameterise Philip is to estimate S
    from early EFV infiltration, while keeping Ks equal to the same soil Ks.

    Strategy
    --------
    1. Run EFV under a chosen reference storm.
    2. Take the first few pulses, where sorptivity effects are most important.
    3. Use the Philip equation rearranged as:

           S = (F - Ks * t) / sqrt(t)

    4. Compute S at each early time and average the valid values.

    Parameters
    ----------
    storm_df : pd.DataFrame
        Reference storm used to estimate Philip sorptivity.

    soil : SoilScenario
        Soil for which S will be estimated.

    antecedent : str
        Antecedent moisture condition.

    dt_min : int
        Storm time step [minutes].

    n_fit_pulses : int
        Number of early pulses used for estimating S.

    Returns
    -------
    dict with:
        S_mm_h05
        n_points_used
    """
    efv_df, _ = efv_excess(
        storm_df=storm_df,
        soil=soil,
        antecedent=antecedent,
    )

    Ks_mm_h = soil.Ks_mm_h

    # Use cumulative time in hours at the end of each pulse
    t_h = efv_df["time_min"].to_numpy(dtype=float) / 60.0
    F_mm = efv_df["cum_infil_mm"].to_numpy(dtype=float)

    s_values = []

    # Scan through the event and keep the first n_fit_pulses VALID positive estimates.
    # This is more robust than forcing the first few pulses, which may still be rainfall-limited.
    for i in range(len(efv_df)):
        ti = t_h[i]
        Fi = F_mm[i]

        if ti <= 0.0:
            continue

        numerator = Fi - Ks_mm_h * ti
        denominator = np.sqrt(ti)

        S_i = numerator / denominator

        # Only keep physically meaningful positive values.
        if np.isfinite(S_i) and S_i > 0.0:
            s_values.append(S_i)

        if len(s_values) >= n_fit_pulses:
            break

    if len(s_values) == 0:
        raise ValueError(
            f"Could not estimate Philip sorptivity from EFV for soil '{soil.name}' "
            f"and antecedent '{antecedent}'."
        )

    return {
        "S_mm_h05": float(np.mean(s_values)),
        "n_points_used": int(len(s_values)),
    }


def build_calibrated_philip_parameters(
    reference_storm_df: pd.DataFrame,
    soils: Dict[str, SoilScenario],
    dt_min: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Build Philip sorptivity parameters for all soils and antecedent conditions
    using EFV as the reference model.

    Returns
    -------
    dict
        Nested dictionary:
            philip_params[soil_key][antecedent] = {
                "S_mm_h05": ...,
                "n_points_used": ...
            }
    """
    philip_params = {}

    for soil_key, soil in soils.items():
        philip_params[soil_key] = {}

        for antecedent in ["dry", "avg", "wet"]:
            philip_params[soil_key][antecedent] = estimate_philip_sorptivity_from_efv(
                storm_df=reference_storm_df,
                soil=soil,
                antecedent=antecedent,
                dt_min=dt_min,
            )

    return philip_params


def run_method_scs(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    comparison_mode: str,
    soil_key: str,
    calibrated_scs_params: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:

    antecedent = antecedent.lower()

    if comparison_mode == "structural":
        CN = scs_adjust_cn_for_antecedent(soil.CN_II, antecedent)

    elif comparison_mode == "surrogate":
        if calibrated_scs_params is None:
            raise ValueError("calibrated_scs_params required in surrogate mode.")

        if antecedent == "dry":
            CN = calibrated_scs_params[soil_key]["CN_I"]
        elif antecedent in ["avg", "average", "normal"]:
            CN = calibrated_scs_params[soil_key]["CN_II"]
        elif antecedent == "wet":
            CN = calibrated_scs_params[soil_key]["CN_III"]
        else:
            raise ValueError("antecedent must be 'dry', 'avg', or 'wet'")

    else:
        raise ValueError("comparison_mode must be 'structural' or 'surrogate'")

    out = scs_cn_excess(storm_df, CN=CN)
    out["method"] = "SCS-CN"
    out["antecedent"] = antecedent
    out["soil"] = soil.name
    out["CN_used"] = CN
    out["comparison_mode"] = comparison_mode
    return out


def run_method_horton(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
    comparison_mode: str,
    soil_key: str,
    calibrated_horton_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> pd.DataFrame:

    if comparison_mode == "structural":
        f0 = soil.horton_f0(antecedent)
        fc = soil.Ks_mm_h
        k = soil.horton_k_h_inv

    elif comparison_mode == "surrogate":
        if calibrated_horton_params is None:
            raise ValueError("calibrated_horton_params required in surrogate mode.")
        params = calibrated_horton_params[soil_key][antecedent]
        f0 = params["f0_mm_h"]
        fc = params["fc_mm_h"]
        k = params["k_h_inv"]

    else:
        raise ValueError("comparison_mode must be 'structural' or 'surrogate'")

    out = horton_excess(
        storm_df=storm_df,
        f0_mm_h=f0,
        fc_mm_h=fc,
        k_h_inv=k,
        dt_min=dt_min,
    )

    out["method"] = "Horton"
    out["antecedent"] = antecedent
    out["soil"] = soil.name
    out["f0_mm_h"] = f0
    out["fc_mm_h"] = fc
    out["k_h_inv"] = k
    out["comparison_mode"] = comparison_mode
    return out


def run_method_philip(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
    comparison_mode: str,
    soil_key: str,
    calibrated_philip_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> pd.DataFrame:

    if comparison_mode == "structural":
        S = soil.philip_S(antecedent)

    elif comparison_mode == "surrogate":
        if calibrated_philip_params is None:
            raise ValueError("calibrated_philip_params required in surrogate mode.")
        S = calibrated_philip_params[soil_key][antecedent]["S_mm_h05"]

    else:
        raise ValueError("comparison_mode must be 'structural' or 'surrogate'")

    Ks = soil.Ks_mm_h

    out = philip_excess(
        storm_df=storm_df,
        S_mm_h05=S,
        Ks_mm_h=Ks,
        dt_min=dt_min,
    )

    out["method"] = "Philip"
    out["antecedent"] = antecedent
    out["soil"] = soil.name
    out["philip_S_mm_h05"] = S
    out["comparison_mode"] = comparison_mode
    return out


def run_method_green_ampt(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    dt_min: int,
) -> pd.DataFrame:
    """
    Run Green-Ampt using parameters taken directly from the master soil description.

    Green-Ampt does not require calibration here because it already uses physically
    interpretable soil quantities:
        - Ks
        - theta_s
        - theta_i
        - psi_f
    """
    out = green_ampt_excess(
        storm_df=storm_df,
        Ks_mm_h=soil.Ks_mm_h,
        psi_f_mm=soil.psi_f_mm,
        theta_s=soil.theta_s,
        theta_i=soil.theta_i(antecedent),
        dt_min=dt_min,
    )

    out["method"] = "Green-Ampt"
    out["antecedent"] = antecedent
    out["soil"] = soil.name
    return out


def run_method_efv(
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    antecedent: str,
    experiment: ExperimentConfig,
) -> pd.DataFrame:

    out, theta_final = efv_excess(
        storm_df=storm_df,
        soil=soil,
        antecedent=antecedent,
        bottom_bc=experiment.efv_bottom_bc,
    )

    out["method"] = "EFV"
    out["antecedent"] = antecedent
    out["soil"] = soil.name
    out["efv_bottom_bc"] = experiment.efv_bottom_bc
    out.attrs["theta_final"] = theta_final
    return out

# ======================================================================================
# 9. MAIN EXPERIMENT DRIVER
# ======================================================================================

def run_one_method_and_route(
    method_name: str,
    storm_df: pd.DataFrame,
    soil: SoilScenario,
    soil_key: str,
    antecedent: str,
    dt_min: int,
    catchment: CatchmentScenario,
    experiment: ExperimentConfig,
    calibrated_horton_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    calibrated_scs_params: Optional[Dict[str, Dict[str, float]]] = None,
    calibrated_philip_params: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run one infiltration/loss method and then route its excess rainfall through the
    common unit hydrograph.
    """
    method_name = method_name.lower()

    # ==========================================================
    # NEW: START TIMER
    # ==========================================================
    t0 = perf_counter()

    # ==========================================================
    # RUN THE SELECTED INFILTRATION METHOD
    # ==========================================================
    if method_name == "scs":
        if experiment.comparison_mode == "surrogate" and calibrated_scs_params is None:
            raise ValueError(
                "calibrated_scs_params must be provided when running the SCS-CN method in surrogate mode."
            )

        method_df = run_method_scs(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_scs_params=calibrated_scs_params,
        )

    elif method_name == "horton":
        if experiment.comparison_mode == "surrogate" and calibrated_horton_params is None:
            raise ValueError(
                "calibrated_horton_params must be provided when running the Horton method in surrogate mode."
            )

        method_df = run_method_horton(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_horton_params=calibrated_horton_params,
        )

    elif method_name == "philip":
        if experiment.comparison_mode == "surrogate" and calibrated_philip_params is None:
            raise ValueError(
                "calibrated_philip_params must be provided when running the Philip method in surrogate mode."
            )

        method_df = run_method_philip(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
            comparison_mode=experiment.comparison_mode,
            soil_key=soil_key,
            calibrated_philip_params=calibrated_philip_params,
        )

    elif method_name in ["ga", "green-ampt", "green_ampt"]:
        method_df = run_method_green_ampt(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            dt_min=dt_min,
        )

    elif method_name == "efv":
        method_df = run_method_efv(
            storm_df=storm_df,
            soil=soil,
            antecedent=antecedent,
            experiment=experiment,
        )

    else:
        raise ValueError("Unknown method. Use one of: scs, horton, philip, ga, efv")

    # ==========================================================
    # NEW: STOP TIMER
    # This measures ONLY the infiltration/loss method runtime
    # ==========================================================
    method_runtime_s = perf_counter() - t0

    # ==========================================================
    # BUILD THE COMMON UNIT HYDROGRAPH
    # ==========================================================
    uh_df = triangular_unit_hydrograph(catchment, dt_min)

    # ==========================================================
    # ROUTE EXCESS RAINFALL
    # ==========================================================
    Q = convolve_excess_with_uh(
        excess_mm=method_df["excess_mm"].to_numpy(),
        uh_m3s_per_mm=uh_df["UH_m3s_per_mm"].to_numpy(),
        baseflow_m3_s=catchment.baseflow_m3_s,
    )

    hydro_time_min = np.arange(len(Q)) * dt_min

    metrics = hydrograph_metrics(hydro_time_min, Q)

    # ==========================================================
    # NEW: ADD RUNTIME TO METRICS DICTIONARY
    # ==========================================================
    metrics["method_runtime_s"] = method_runtime_s

    hydro_df = pd.DataFrame({
        "time_min": hydro_time_min,
        "Q_m3_s": Q,
    })

    return method_df, hydro_df, metrics


# ======================================================================================
# 10. PLOTTING FUNCTIONS
# ======================================================================================

def plot_storm(storm_df: pd.DataFrame, title: str = "Design storm") -> None:
    """Plot a rainfall hyetograph."""
    plt.figure(figsize=(10, 4))
    plt.bar(
        storm_df["time_min"],
        storm_df["P_mm"],
        width=0.8 * (storm_df["time_min"].diff().median()),
        align="center",
    )
    plt.xlabel("Time (min)")
    plt.ylabel("Rainfall depth per pulse (mm)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_excess_comparison(
    results: Dict[str, pd.DataFrame],
    title: str = "Excess rainfall comparison",
) -> None:
    """Plot incremental excess rainfall for multiple methods."""
    plt.figure(figsize=(10, 5))
    for method_name, df in results.items():
        plt.plot(df["time_min"], df["excess_mm"], marker="o", label=method_name)
    plt.xlabel("Time (min)")
    plt.ylabel("Excess rainfall per pulse (mm)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_infiltration(
    results: Dict[str, pd.DataFrame],
    title: str = "Cumulative infiltration",
) -> None:
    """Plot cumulative infiltration for multiple methods."""
    plt.figure(figsize=(10, 5))
    for method_name, df in results.items():
        plt.plot(df["time_min"], df["cum_infil_mm"], marker="o", label=method_name)
    plt.xlabel("Time (min)")
    plt.ylabel("Cumulative infiltration (mm)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_hydrographs(
    hydrographs: Dict[str, pd.DataFrame],
    title: str = "Hydrograph comparison",
) -> None:
    """Plot routed hydrographs for multiple methods."""
    plt.figure(figsize=(10, 5))
    for method_name, df in hydrographs.items():
        plt.plot(df["time_min"], df["Q_m3_s"], label=method_name)
    plt.xlabel("Time (min)")
    plt.ylabel("Discharge (m³/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ======================================================================================
# 11. MAIN EXECUTION BLOCK
# ======================================================================================

if __name__ == "__main__":
    # ==============================================================================
    # 1. USER INPUTS
    # ==============================================================================
    #
    # This section defines:
    #   - the time step
    #   - the calibration storm for Horton
    #   - the storm-pattern family used for synthetic symmetric storms
    #   - the SCS-CN calibration storm
    #   - the Philip calibration storm
    #   - the evaluation storm
    #   - the catchment
    #   - which methods, soils, and antecedent states to run

    dt_min = 10

    experiment = ExperimentConfig(
        comparison_mode="structural",
        efv_bottom_bc="free_drainage",
        reference_method="efv",
    )

    # ==============================================================================
    # 2) STORM FACTORS FOR THE FACTORIAL EXPERIMENT
    # ==============================================================================
    #
    # These define the scenario matrix.
    # The actual storm_df will be built INSIDE the experiment loops later.

    storm_durations_min = [60, 120, 240]
    storm_total_depths_mm = [80.0, 125.0, 150.0]
    storm_pattern_kinds = ["constant", "center_peaked", "front_loaded", "back_loaded"]

    # ==============================================================================
    # 3) REFERENCE / CALIBRATION STORMS
    # ==============================================================================
    #
    # These are only needed in surrogate mode.
    # In structural mode, they are ignored.

    calibration_storm_df = None
    reference_peaked_storm_df = None
    philip_calibration_storm_df = None

    if experiment.comparison_mode == "surrogate":
        # Horton surrogate calibration: constant storm
        calibration_storm_df = build_storm_from_pattern(
            storm_duration_min=120,
            total_depth_mm=50.0,
            dt_min=dt_min,
            pattern="constant",
        )

        # SCS-CN surrogate calibration: peaked storm
        reference_peaked_storm_df = build_storm_from_pattern(
            storm_duration_min=120,
            total_depth_mm=50.0,
            dt_min=dt_min,
            pattern="center_peaked",
        )

        # Philip surrogate calibration: strong constant storm
        philip_calibration_storm_df = build_storm_from_pattern(
            storm_duration_min=120,
            total_depth_mm=120.0,
            dt_min=dt_min,
            pattern="constant",
        )

    # ==============================================================================
    # 4) CATCHMENT
    # ==============================================================================
    #
    # The catchment and routing setup must remain fixed for all methods.
    # Otherwise, hydrograph differences would reflect both:
    #   - infiltration/runoff-generation differences
    #   - routing differences
    #
    # which would make the comparison unfair.
    catchment = CatchmentScenario(
        name="Conceptual urban catchment",
        area_km2=27.0,
        flow_length_m=2500.0,
        hmax_m=100.0,
        hmin_m=60.0,
        baseflow_m3_s=0.0,
    )

    # ==============================================================================
    # 5) METHOD SELECTION
    # ==============================================================================
    #
    # Valid method names:
    #   "ga"      -> Green-Ampt
    #   "philip"  -> Philip
    #   "horton"  -> Horton
    #   "scs"     -> SCS Curve Number
    #   "efv"     -> EFV
    selected_methods = ["ga", "horton", "scs", "philip", "efv"]

    # ==============================================================================
    # 6) SOILS AND ANTECEDENT CONDITIONS
    # ==============================================================================
    selected_soils = ["compacted", "loam", "gi", "clay"]
    selected_antecedents = ["dry", "avg", "wet"]

    # Reduced soil dictionary to calibrate only the soils actually used
    selected_soils_dict = {soil_key: SOILS[soil_key] for soil_key in selected_soils}

    # ==============================================================================
    # 7) SELECT WHICH CASE TO PLOT
    # ==============================================================================
    plot_soil_key = "clay"
    plot_antecedent = "avg"
    plot_storm_duration_min = 60
    plot_storm_total_depth_mm = 80.0
    plot_storm_pattern_kind = "center_peaked"

    print("Running methods:", selected_methods)
    print("Running soils:", selected_soils)
    print("Running antecedent states:", selected_antecedents)
    print("Comparison mode:", experiment.comparison_mode)
    print("EFV bottom boundary:", experiment.efv_bottom_bc)
    print("Storm durations [min]:", storm_durations_min)
    print("Storm total depths [mm]:", storm_total_depths_mm)
    print("Storm pattern kinds:", storm_pattern_kinds)

    # ==============================================================================
    # 8) CALIBRATION
    # ==============================================================================
    #
    # Horton calibration:
    #   - uses the constant capacity-controlled storm
    #
    # SCS-CN calibration:
    #   - uses the selected symmetric SCS reference storm
    #
    # Philip calibration:
    #   - uses the strong constant Philip calibration storm
    #
    # Green-Ampt and EFV use direct soil parameters and are not calibrated here.
    calibrated_horton_params = None
    calibrated_scs_params = None
    calibrated_philip_params = None

    if experiment.comparison_mode == "surrogate":
        calibrated_horton_params = build_calibrated_horton_parameters(
            reference_storm_df=calibration_storm_df,
            soils=selected_soils_dict,
            dt_min=dt_min,
        )

        calibrated_scs_params = build_calibrated_scs_parameters(
            reference_storm_df=reference_peaked_storm_df,
            soils=selected_soils_dict,
            dt_min=dt_min,
        )

        calibrated_philip_params = build_calibrated_philip_parameters(
            reference_storm_df=philip_calibration_storm_df,
            soils=selected_soils_dict,
            dt_min=dt_min,
        )

        print("\n=== Calibrated Horton parameters (using constant calibration storm) ===")
        for soil_key, soil_params in calibrated_horton_params.items():
            print(f"\nSoil: {soil_key}")
            for antecedent, pars in soil_params.items():
                print(
                    f"  {antecedent}: "
                    f"f0 = {pars['f0_mm_h']:.2f} mm/h, "
                    f"fc = {pars['fc_mm_h']:.2f} mm/h, "
                    f"k = {pars['k_h_inv']:.3f} 1/h, "
                    f"RMSE = {pars['rmse_cum_infil_mm']:.3f} mm"
                )

        print("\n=== Calibrated SCS-CN parameters (using selected SCS reference storm) ===")
        for soil_key, pars in calibrated_scs_params.items():
            print(
                f"{soil_key}: "
                f"CN_I = {pars['CN_I']:.2f}, "
                f"CN_II = {pars['CN_II']:.2f}, "
                f"CN_III = {pars['CN_III']:.2f}, "
                f"fit_error = {pars['fit_abs_error_mm']:.3f} mm"
            )

        print("\n=== Calibrated Philip sorptivity parameters (using EFV + strong constant storm) ===")
        for soil_key, soil_params in calibrated_philip_params.items():
            print(f"\nSoil: {soil_key}")
            for antecedent, pars in soil_params.items():
                print(
                    f"  {antecedent}: "
                    f"S = {pars['S_mm_h05']:.3f} mm/h^0.5, "
                    f"points_used = {pars['n_points_used']}"
                )

    # ==============================================================================
    # 9) RUN FACTORIAL EXPERIMENTS
    # ==============================================================================
    all_summary_rows = []
    all_method_results = {}
    all_hydro_results = {}

    for soil_key in selected_soils:
        soil = SOILS[soil_key]

        for antecedent in selected_antecedents:
            for storm_duration_min in storm_durations_min:
                for storm_total_depth_mm in storm_total_depths_mm:
                    for storm_pattern_kind in storm_pattern_kinds:

                        # ----------------------------------------------------------
                        # Build the storm for this scenario
                        # ----------------------------------------------------------
                        storm_df = build_storm_from_pattern(
                            storm_duration_min=storm_duration_min,
                            total_depth_mm=storm_total_depth_mm,
                            dt_min=dt_min,
                            pattern=storm_pattern_kind,
                        )
                        
                        storm_desc = storm_descriptors(
                            storm_df=storm_df,
                            soil=soil,
                            antecedent=antecedent,
                        )
                        
                        # Store all methods for this specific scenario
                        scenario_method_results = {}
                        scenario_hydro_results = {}
                        scenario_metrics = {}

                        # ----------------------------------------------------------
                        # Run all selected methods for this scenario
                        # ----------------------------------------------------------
                        for method in selected_methods:
                            method_df, hydro_df, metrics = run_one_method_and_route(
                                method_name=method,
                                storm_df=storm_df,
                                soil=soil,
                                soil_key=soil_key,
                                antecedent=antecedent,
                                dt_min=dt_min,
                                catchment=catchment,
                                experiment=experiment,
                                calibrated_horton_params=calibrated_horton_params,
                                calibrated_scs_params=calibrated_scs_params,
                                calibrated_philip_params=calibrated_philip_params,
                            )

                            method_label = method_df["method"].iloc[0]

                            run_key = (
                                f"{method_label} | {soil.name} | {antecedent} | "
                                f"{storm_duration_min}min | {storm_total_depth_mm:.1f}mm | "
                                f"{storm_pattern_kind}"
                            )

                            # Global storage for later access / plotting
                            all_method_results[run_key] = method_df
                            all_hydro_results[run_key] = hydro_df

                            # Scenario-local storage for EFV comparison
                            scenario_method_results[method_label] = method_df
                            scenario_hydro_results[method_label] = hydro_df
                            scenario_metrics[method_label] = metrics

                        # ----------------------------------------------------------
                        # Compare all methods against EFV for this scenario
                        # ----------------------------------------------------------
                        if "EFV" not in scenario_method_results:
                            raise ValueError("EFV must be included in selected_methods for reference comparison.")

                        ref_method_df = scenario_method_results["EFV"]
                        ref_hydro_df = scenario_hydro_results["EFV"]

                        for method_label, method_df in scenario_method_results.items():
                            hydro_df = scenario_hydro_results[method_label]
                            metrics = scenario_metrics[method_label]

                            comparison_metrics = compare_to_reference(
                                ref_method_df=ref_method_df,
                                test_method_df=method_df,
                                ref_hydro_df=ref_hydro_df,
                                test_hydro_df=hydro_df,
                            )

                            row = {
                                "method": method_label,
                                "soil_key": soil_key,
                                "soil": soil.name,
                                "antecedent": antecedent,
                                "storm_duration_min": storm_duration_min,
                                "storm_total_depth_mm": storm_total_depth_mm,
                                "storm_pattern_kind": storm_pattern_kind,
                                "comparison_mode": experiment.comparison_mode,
                                "efv_bottom_bc": experiment.efv_bottom_bc,
                                "cum_infil_mm": float(method_df["cum_infil_mm"].iloc[-1]),
                                "cum_excess_mm": float(method_df["cum_excess_mm"].iloc[-1]),
                                "peak_flow_m3_s": metrics["peak_flow_m3_s"],
                                "time_to_peak_min": metrics["time_to_peak_min"],
                                "runoff_volume_m3": metrics["runoff_volume_m3"],
                                "method_runtime_s": metrics["method_runtime_s"],
                                **comparison_metrics,
                                **storm_desc,
                            }

                            if method_label == "Horton":
                                row["f0_mm_h"] = float(method_df["f0_mm_h"].iloc[0])
                                row["fc_mm_h"] = float(method_df["fc_mm_h"].iloc[0])
                                row["k_h_inv"] = float(method_df["k_h_inv"].iloc[0])

                            if method_label == "SCS-CN":
                                row["CN_used"] = float(method_df["CN_used"].iloc[0])

                            if method_label == "Philip":
                                row["philip_S_mm_h05"] = float(method_df["philip_S_mm_h05"].iloc[0])

                            all_summary_rows.append(row)

    summary_df = pd.DataFrame(all_summary_rows)

    # ==============================================================================
    # 10) DIAGNOSTIC FOR THE SELECTED PLOT CASE
    # ==============================================================================
    print("\n================ DIAGNOSTIC: SELECTED CASE =================")

    selected_case_df = summary_df[
        (summary_df["soil_key"] == plot_soil_key) &
        (summary_df["antecedent"] == plot_antecedent) &
        (summary_df["storm_duration_min"] == plot_storm_duration_min) &
        (summary_df["storm_total_depth_mm"] == plot_storm_total_depth_mm) &
        (summary_df["storm_pattern_kind"] == plot_storm_pattern_kind)
    ].copy()

    print(selected_case_df)

    # ==============================================================================
    # 11) PRINT SUMMARIES
    # ==============================================================================
    print("\n=== Catchment routing summary ===")
    print(f"Catchment name: {catchment.name}")
    print(f"Area [km²]: {catchment.area_km2}")
    print(f"Slope [m/m]: {catchment.slope_m_m:.4f}")
    print(f"tc [min]: {catchment.tc_min:.2f}")

    print("\n=== Experiment settings ===")
    print("Comparison mode:", experiment.comparison_mode)
    print("EFV bottom boundary:", experiment.efv_bottom_bc)
    print("Methods:", selected_methods)
    print("Soils:", selected_soils)
    print("Antecedent states:", selected_antecedents)
    print("Storm durations [min]:", storm_durations_min)
    print("Storm total depths [mm]:", storm_total_depths_mm)
    print("Storm pattern kinds:", storm_pattern_kinds)

    if experiment.comparison_mode == "surrogate":
        print("\n=== Horton calibration storm summary ===")
        print(calibration_storm_df)

        print("\n=== SCS-CN calibration storm summary ===")
        print(reference_peaked_storm_df)

        print("\n=== Philip calibration storm summary ===")
        print(philip_calibration_storm_df)

    print("\n=== Comparison summary ===")
    print(summary_df)

    # ==============================================================================
    # 12) BUILD DICTIONARIES FOR PLOTTING THE SELECTED CASE
    # ==============================================================================
    method_plot_dict = {}
    hydro_plot_dict = {}

    target_soil_name = SOILS[plot_soil_key].name

    selected_storm_df = build_storm_from_pattern(
        storm_duration_min=plot_storm_duration_min,
        total_depth_mm=plot_storm_total_depth_mm,
        dt_min=dt_min,
        pattern=plot_storm_pattern_kind,
    )

    for run_key, df in all_method_results.items():
        if (
            f"| {target_soil_name} | {plot_antecedent} | "
            f"{plot_storm_duration_min}min | {plot_storm_total_depth_mm:.1f}mm | "
            f"{plot_storm_pattern_kind}"
        ) in run_key:
            method_name = df["method"].iloc[0]
            method_plot_dict[method_name] = df

    for run_key, df in all_hydro_results.items():
        if (
            f"| {target_soil_name} | {plot_antecedent} | "
            f"{plot_storm_duration_min}min | {plot_storm_total_depth_mm:.1f}mm | "
            f"{plot_storm_pattern_kind}"
        ) in run_key:
            method_name = run_key.split(" | ")[0]
            hydro_plot_dict[method_name] = df

    # ==============================================================================
    # 13) PLOTS
    # ==============================================================================
    if method_plot_dict:
        plot_storm(
            selected_storm_df,
            title=(
                f"Storm: {plot_storm_duration_min} min, "
                f"{plot_storm_total_depth_mm:.1f} mm, "
                f"{plot_storm_pattern_kind}"
            ),
        )

        plot_excess_comparison(
            method_plot_dict,
            title=(
                f"Excess rainfall comparison - {target_soil_name} - {plot_antecedent} - "
                f"{plot_storm_duration_min} min - {plot_storm_total_depth_mm:.1f} mm - "
                f"{plot_storm_pattern_kind}"
            ),
        )

        plot_cumulative_infiltration(
            method_plot_dict,
            title=(
                f"Cumulative infiltration - {target_soil_name} - {plot_antecedent} - "
                f"{plot_storm_duration_min} min - {plot_storm_total_depth_mm:.1f} mm - "
                f"{plot_storm_pattern_kind}"
            ),
        )

        plot_hydrographs(
            hydro_plot_dict,
            title=(
                f"Hydrograph comparison - {target_soil_name} - {plot_antecedent} - "
                f"{plot_storm_duration_min} min - {plot_storm_total_depth_mm:.1f} mm - "
                f"{plot_storm_pattern_kind}"
            ),
        )
    else:
        print(
            f"\nNo runs found for the selected plot case: "
            f"{plot_soil_key}, {plot_antecedent}, "
            f"{plot_storm_duration_min} min, "
            f"{plot_storm_total_depth_mm:.1f} mm, "
            f"{plot_storm_pattern_kind}."
        )
    
    # ==============================================================================
    # 14) OPTIONAL EXPORTS
    # ==============================================================================
    # Save in the same folder as this script, using a safe fallback if the CSV is open.
    
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "comparison_summary.csv"
    
    try:
        summary_df.to_csv(csv_path, index=False)
        print(f"\ncomparison_summary.csv saved successfully at:\n{csv_path}")
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = script_dir / f"comparison_summary_{timestamp}.csv"
        summary_df.to_csv(fallback_path, index=False)
        print(
            "\nWARNING: comparison_summary.csv was locked or open in another program.\n"
            f"Saved instead as:\n{fallback_path}"
        )