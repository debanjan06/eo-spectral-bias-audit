# Data Inventory and Provenance

This directory contains the multi-modal datasets used to train and audit the AgriSight models. The data is divided into raw meteorological time-series and processed spatial patches.

---

## Directory Structure

```
data/
├── raw/                        # Original meteorological CSV files
└── processed/                  # Pre-processed tensors for model input
    ├── australia_patches/
    ├── california_patches/
    └── punjab_patches/
```

---

## Data Sources

**1. Spatial Data (Satellite Imagery)**

- Source: Sentinel-2 Level-2A (Bottom-of-Atmosphere reflectance)
- Bands: 4-channel input (Red, Green, Blue, Near-Infrared)
- Resolution: 10m per pixel
- Format: `.npy` (NumPy binary) tensors scaled to `(4, 32, 32)`

**2. Tabular Data (Meteorological)**

- Source: Regional weather stations and calculated indices
- Features:
  - `NDVI` — Normalized Difference Vegetation Index
  - `SAVI` — Soil Adjusted Vegetation Index
  - `EVI` — Enhanced Vegetation Index
  - `Temp_max` — Maximum Daily Temperature
  - `Rainfall` — Daily accumulation in mm
  - `Humidity` — Relative percentage

---

## Regional Profiles

| Region | Usage | Characteristics |
| :--- | :--- | :--- |
| California Central Valley | Training / Baseline | Mediterranean climate, high-intensity irrigation, high biomass |
| Western Australia | Stress Test (OOD) | Arid environment, high soil albedo, low vegetation density |
| Punjab Wheat Belt | Stress Test (OOD) | Continental climate, distinct phenological cycles compared to California |

---

## Pre-processing Pipeline

All raw imagery is processed through `src/data_pipeline/satellite_collector.py`, which performs the following steps:

1. Cloud masking (QA60 band filtering)
2. Normalization of reflectance values to `[0, 1]`
3. Patch extraction at 32×32 pixel centroids
4. Co-registration with meteorological timestamps

---

## Usage Note

Due to file size constraints, the `.npy` patches included in this repository are representative samples. For access to the full multi-terabyte dataset used in the final research, please refer to the contact information in the root README.
