# Dataset V2 Robustness Summary

## Cross-validation
- Mean Macro F1: 0.9896
- Std Macro F1: 0.0073
- Mean Accuracy: 0.9909

## Ablation
- Full feature Macro F1: 0.9929
- Without Soil_Moisture Macro F1: 0.3918
- Macro F1 drop without Soil_Moisture: 0.6011

## Feature Importance
- Soil_Moisture normalized importance: 0.9531

## Assessment
- B. Stable but heavily dependent on Soil_Moisture
- The model demonstrates strong performance under the evaluated dataset distribution, but the evidence shows strong dependence on Soil_Moisture. Do not claim 99% real-world farm generalization.
