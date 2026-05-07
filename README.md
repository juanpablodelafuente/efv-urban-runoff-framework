# EFV Urban Runoff Framework

Python scripts for testing how infiltration/loss representation affects runoff generation and routed hydrographs in urban flood modelling workflows.

The repository contains two main scripts:

1. `run_infiltration_comparison.py`
   - Controlled single-soil / single-HRU comparison.
   - Compares EFV, Green-Ampt, Horton, Philip and SCS-CN under the same storms and the same unit-hydrograph routing.
   - Main output: `comparison_summary.csv`.

2. `run_multi_hru_luc_cc.py`
   - Multi-HRU land-use and climate-stress scenario framework.
   - Uses `run_infiltration_comparison.py` as the stable infiltration/routing engine.
   - Tests land-use composition changes and climate/storm-stress changes by aggregating HRU-specific excess rainfall and routing the catchment-average excess.
   - Main outputs are saved in `multi_hru_outputs/`.

These scripts are intended as a transparent research framework, not as a calibrated operational flood-forecasting model.

---

## Scientific purpose

The workflow supports two linked questions:

1. **Method comparison:** How differently do simplified infiltration/loss methods reproduce EFV-based runoff generation under controlled storm, soil and antecedent conditions?
2. **Scenario application:** How does an EFV-based runoff response change under land-use composition and climate/storm-stress scenarios when full CityCAT coupling is not implemented?

The key rule in both scripts is that the routing setup is held fixed. Therefore, differences in routed hydrographs mainly reflect differences in runoff generation, not changes in routing.

---

## Repository contents

```text
efv-urban-runoff-framework/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── run_infiltration_comparison.py
├── run_multi_hru_luc_cc.py
│
└── multi_hru_outputs/
    └── .gitkeep
```

---

## Installation

A clean conda environment is recommended.

```bash
conda create -n efv_runoff python=3.11
conda activate efv_runoff
pip install -r requirements.txt
```

The scripts require only:

```text
numpy
pandas
matplotlib
```

---

## How to run

### Option 1: from terminal

From the repository folder:

```bash
python run_infiltration_comparison.py
python run_multi_hru_luc_cc.py
```

### Option 2: from Spyder

1. Open the repository folder as the working directory.
2. Open `run_infiltration_comparison.py` or `run_multi_hru_luc_cc.py`.
3. Run the full file.

Important: `run_multi_hru_luc_cc.py` must stay in the same folder as `run_infiltration_comparison.py`, because Script 2 imports the functions and classes from Script 1.

---

# Script 1: `run_infiltration_comparison.py`

## Purpose

This script compares infiltration/loss methods for homogeneous soil scenarios. It runs factorial combinations of:

- method,
- soil,
- antecedent condition,
- storm duration,
- storm total depth,
- storm temporal pattern.

For each case it computes rainfall excess, routes it through the same unit hydrograph, and compares each method against EFV.

## Main output

The script saves:

```text
comparison_summary.csv
```

This CSV contains event-scale metrics such as:

- cumulative infiltration,
- cumulative excess rainfall,
- peak flow,
- time to peak,
- runoff volume,
- method runtime,
- EFV-relative infiltration/excess/hydrograph metrics,
- storm descriptors such as mean intensity, peak intensity and intensity / Ks ratios.

## Adjustable variables in Script 1

Edit these inside the `if __name__ == "__main__":` block.

### 1. Time step

```python
dt_min = 10
```

Meaning: rainfall pulse and routing time step in minutes.

Use a value that divides all selected storm durations exactly. For example, with `dt_min = 10`, durations such as 60, 120 and 240 min are valid.

---

### 2. Experiment configuration

```python
experiment = ExperimentConfig(
    comparison_mode="structural",
    efv_bottom_bc="free_drainage",
    reference_method="efv",
)
```

Meanings:

- `comparison_mode`: normally keep as `"structural"`. This uses direct physically consistent parameter assignments rather than calibration.
- `efv_bottom_bc`: EFV lower boundary. Current options are `"free_drainage"` and `"no_flow"`.
- `reference_method`: normally keep as `"efv"`.

Recommended thesis setting:

```python
comparison_mode="structural"
efv_bottom_bc="free_drainage"
```

---

### 3. Storm-factor matrix

```python
storm_durations_min = [60, 120, 240]
storm_total_depths_mm = [80.0, 125.0, 150.0]
storm_pattern_kinds = ["constant", "center_peaked", "front_loaded", "back_loaded"]
```

Meanings:

- `storm_durations_min`: storm durations tested in minutes.
- `storm_total_depths_mm`: total event depths tested in mm.
- `storm_pattern_kinds`: temporal distribution of rainfall within the event.

Important: these lists are combined factorially. For example, `60 min` is run with `80`, `125` and `150 mm`, not only with the first depth.

Available storm patterns in Script 1:

```text
constant
center_peaked
front_loaded
back_loaded
```

---

### 4. Catchment and routing parameters

```python
catchment = CatchmentScenario(
    name="Conceptual urban catchment",
    area_km2=27.0,
    flow_length_m=2500.0,
    hmax_m=100.0,
    hmin_m=60.0,
    baseflow_m3_s=0.0,
)
```

Meanings:

- `area_km2`: catchment area used to scale the unit hydrograph response.
- `flow_length_m`: representative main flow path length.
- `hmax_m` and `hmin_m`: used to estimate catchment slope.
- `baseflow_m3_s`: constant baseflow added to routed hydrographs. Keep `0.0` if focusing only on direct runoff.

---

### 5. Method selection

```python
selected_methods = ["ga", "horton", "scs", "philip", "efv"]
```

Available method keys:

```text
ga       -> Green-Ampt
horton   -> Horton
scs      -> SCS Curve Number
philip   -> Philip
efv      -> EFV
```

Keep `"efv"` included if you want EFV-relative comparison metrics.

---

### 6. Soil and antecedent-condition selection

```python
selected_soils = ["compacted", "loam", "gi", "clay"]
selected_antecedents = ["dry", "avg", "wet"]
```

Available soil keys:

```text
compacted  -> compacted urban soil
't loam     -> urban loam
gi         -> green infrastructure / engineered soil
clay       -> urban clay
```

Available antecedent conditions:

```text
dry
avg
wet
```

The soil definitions are stored in the `SOILS` dictionary. You can change hydraulic properties there, including:

- `Ks_mm_h`,
- `theta_s`,
- `theta_r`,
- `theta_i_dry`, `theta_i_avg`, `theta_i_wet`,
- `psi_f_mm`,
- `CN_II`,
- Philip sorptivity values,
- Horton parameters,
- EFV van Genuchten parameters.

---

### 7. Selected diagnostic plot case

```python
plot_soil_key = "gi"
plot_antecedent = "wet"
plot_storm_duration_min = 120
plot_storm_total_depth_mm = 150
plot_storm_pattern_kind = "back_loaded"
```

Meaning: this does not control all runs. It only selects which already-run scenario will be plotted in the console/plot pane.

---

# Script 2: `run_multi_hru_luc_cc.py`

## Purpose

This script extends the first workflow to a conceptual Multi-HRU catchment. Instead of treating the catchment as one homogeneous soil, it represents the catchment as a mixture of HRUs:

- connected impervious surface,
- compacted pervious urban soil,
- loam pervious soil,
- GI / engineered soil,
- clay pervious soil.

For each land-use composition and climate scenario, the script computes HRU-specific excess rainfall, multiplies it by HRU area fraction, sums the catchment-average excess series, and routes the result through the same unit hydrograph logic.

## Main outputs

The script saves outputs in:

```text
multi_hru_outputs/
```

Main CSV files:

```text
multi_hru_summary.csv
multi_hru_excess_series.csv
multi_hru_hydrographs.csv
multi_hru_hru_contributions.csv
MAIN_<method>_scenario_summary_table.csv
```

Main figures:

- selected method across climate scenarios for one fixed land-use composition,
- selected method across land-use compositions for one fixed climate scenario,
- peak-flow change vs baseline,
- runoff-volume change vs baseline.

Optional figures:

- method comparison for one fixed land-use + climate scenario,
- EFV-relative method-bias plots.

## Adjustable variables in Script 2

Edit these inside the `if __name__ == "__main__":` block.

---

### 1. Time step

```python
dt_min = 10
```

Meaning: rainfall pulse and routing time step in minutes.

---

### 2. Experiment configuration

```python
experiment = ExperimentConfig(
    comparison_mode="structural",
    efv_bottom_bc="free_drainage",
    reference_method="efv",
)
```

Normally keep:

```python
comparison_mode="structural"
efv_bottom_bc="free_drainage"
```

---

### 3. Catchment and routing parameters

```python
catchment = CatchmentScenario(
    name="Conceptual urban catchment",
    area_km2=27.0,
    flow_length_m=2500.0,
    hmax_m=100.0,
    hmin_m=60.0,
    baseflow_m3_s=0.0,
)
```

Meanings are the same as in Script 1.

---

### 4. Methods to run

```python
selected_methods = ["ga", "horton", "scs", "philip", "efv"]
```

Script 2 still runs all methods so EFV-relative method bias remains available. Keep `"efv"` included.

---

### 5. Main plotting method

```python
main_plot_method_key = "efv"
```

Meaning: the main LUC/CC scenario plots will focus on this method.

Recommended thesis setting:

```python
main_plot_method_key = "efv"
```

You can change it to:

```text
ga
horton
scs
philip
efv
```

---

### 6. Land-use composition scenarios

```python
selected_compositions = [
    "baseline_urban",
    "densified_urban",
    "gi_retrofit",
    "degraded_gi",
    "clay_dominated",
]
```

Available composition keys:

```text
baseline_urban   -> reference mixed urban catchment
densified_urban  -> more connected impervious area
gi_retrofit      -> more GI and less connected runoff
degraded_gi      -> lower GI/storage performance
clay_dominated   -> more low-permeability pervious area
```

The actual HRU fractions are defined in the `COMPOSITIONS` dictionary. For each HRU you can adjust:

```python
area_fraction
hru_type
soil_key
depression_storage_mm
runoff_coefficient
```

Meanings:

- `area_fraction`: fraction of catchment area assigned to that HRU. Fractions must sum to `1.0`.
- `hru_type`: `"impervious"` or `"pervious"`.
- `soil_key`: used only for pervious HRUs. Must match a key in `SOILS`.
- `depression_storage_mm`: event-scale local storage before generated excess is routed.
- `runoff_coefficient`: used for impervious HRUs. `1.0` means all rainfall after storage becomes runoff; values below `1.0` represent additional local loss/disconnection.

---

### 7. Climate/storm-stress scenarios

```python
selected_climates = [
    "present",
    "intensified_depth",
    "intensified_peakiness",
    "shorter_more_intense",
    "wetter_antecedent",
]
```

Available climate keys:

```text
present                  -> reference storm
intensified_depth        -> larger storm depth
intensified_peakiness    -> same depth, more concentrated peak
shorter_more_intense     -> same depth over shorter duration
wetter_antecedent        -> same storm with wet antecedent condition
```

The scenarios are defined in `CLIMATE_SCENARIOS`. For each scenario you can adjust:

```python
storm_duration_min
total_depth_mm
pattern
antecedent
custom_depths_mm
```

Meanings:

- `storm_duration_min`: storm duration in minutes.
- `total_depth_mm`: total event rainfall depth.
- `pattern`: temporal rainfall pattern.
- `antecedent`: `"dry"`, `"avg"`, or `"wet"`.
- `custom_depths_mm`: optional list of rainfall depths per time step. Use this if you want a fully custom hyetograph.

Available standard patterns from Script 1:

```text
constant
center_peaked
front_loaded
back_loaded
```

Additional Script 2 pattern:

```text
sharp_center_peaked
```

---

### 8. Baseline scenario for percentage-change metrics

```python
baseline_composition_key = "baseline_urban"
baseline_climate_key = "present"
```

Meaning: percentage changes in peak flow, volume and excess are computed relative to this baseline scenario.

---

### 9. Main plot controls

```python
main_fixed_composition_for_climate_plot = "baseline_urban"
main_fixed_climate_for_land_use_plot = "present"
```

Meanings:

- `main_fixed_composition_for_climate_plot`: holds land use fixed and compares climate scenarios.
- `main_fixed_climate_for_land_use_plot`: holds climate fixed and compares land-use compositions.

---

### 10. Optional method-comparison plots

```python
make_optional_method_comparison_plots = True
optional_comparison_composition_key = "baseline_urban"
optional_comparison_climate_key = "present"
```

Meaning: these plots compare methods for one fixed scenario. They are diagnostic, not the main LUC/CC story.

Set this to `False` if you only want main scenario plots.

---
