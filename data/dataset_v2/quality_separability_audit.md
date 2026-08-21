# Dataset V2 Quality and Separability Audit

## Generated Files

- crop_feature_summary.csv
- feature_overlap_by_crop.csv
- numeric_granularity_analysis.csv
- crop_specific_ranges.csv
- duplicate_rows_report.csv
- soil_redundancy_report.json
- label_quality_report.json
- class_balance_report.json
- provenance_audit.json

## Decision

V2 REQUIRES DATA CLEANING

Initial baseline feature set after cleaning decisions: Soil_Moisture, Humidity, Temperature.

Reason: Dataset V2 is promising and separable, but the redundant Soil column, one exact duplicate row, label spelling, and class imbalance should be handled in a documented cleaned copy before baseline modeling.
