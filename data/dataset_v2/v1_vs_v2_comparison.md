# Dataset V1 vs Dataset V2 Comparison

## Dataset V1

- Rows: 10000
- Columns in cleaned Review 1 dataset: 13
- Crop classes: 4
- Valid features used: N, P, K, pH, EC, OC, S, Zn, Fe, Cu, Mn, B
- Main limitation: valid features show extremely weak class separation for Crop_Type.
- Best Review 1 baseline accuracy: 0.2580

## Dataset V2: CropRec-BD v1

- Source: https://data.mendeley.com/datasets/dtf278skpw/1
- DOI: 10.17632/dtf278skpw.1
- License: CC BY 4.0
- Master dataset file: final_crops_data.csv
- Rows: 2892
- Columns: 5
- Crop classes: 9
- Target column: Crop Name
- Valid pre-plant features: Soil_Moisture, Humidity, Temperature
- Unknown/needs documentation: Soil
- Missing values: 0
- Duplicate rows: 1
- Infinite values: 0
- Constant columns: None
- Expected advantages: real public DOI-backed dataset, more crop classes than V1, and stronger crop-feature association in downloaded data.
- Known risks: only 4 input columns in the actual file, one ambiguous/redundant Soil column, one duplicate row, and class imbalance.

## Preliminary Decision

Dataset V2 should be classified as B - REQUIRES CLEANING/FEATURE REFINEMENT before model experiments.
