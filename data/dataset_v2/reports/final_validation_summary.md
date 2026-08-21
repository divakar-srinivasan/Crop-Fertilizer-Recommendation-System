# Dataset V2 Final Validation Summary

This file consolidates existing Dataset V2 experimental evidence for project completion and paper preparation. It does not introduce new model development, retraining, hyperparameter tuning, dataset changes, label changes, or test split changes.

## Dataset

- Raw rows before cleaning: 2892
- Cleaned rows after duplicate removal: 2891
- Features: Soil_Moisture, Humidity, Temperature
- Target: Crop Name
- Classes: 9 (Banana, Jute, Maize, Mango, Pineapple, Potato, Strawberry, Sugarcane, Wheat)
- Missing rows removed: 0
- Exact duplicate rows removed from cleaned copy: 1
- Label correction: `Sugercane` standardized to `Sugarcane` for 180 rows
- Train/test split: 2312 train rows, 579 test rows, stratified 80/20 split
- Random state: 42
- Class imbalance note: test support ranges from 36 samples (Pineapple) to 122 samples (Maize)

## Model

- Models compared: Random Forest, LightGBM, XGBoost, CatBoost
- Selected model: CatBoost
- Selection criterion: primary `Macro_F1`, secondary `Weighted_F1`, tertiary `Accuracy`
- Locked artifact: `models/dataset_v2/crop_model_v2.pkl`
- Label encoder: `models/dataset_v2/label_encoder_v2.joblib`

## Evaluation

Held-out test metrics for the selected CatBoost model:

- Accuracy: 99.31%
- Weighted precision: 99.32%
- Weighted recall: 99.31%
- Weighted F1: 99.31%
- Macro precision: 99.19%
- Macro recall: 99.40%
- Macro F1: 99.29%
- Correct predictions on test set: 575 / 579

Final model comparison:

| Model | Accuracy | Weighted F1 | Macro F1 |
| --- | --- | --- | --- |
| Random Forest | 98.27% | 98.27% | 97.81% |
| LightGBM | 98.79% | 98.79% | 98.59% |
| XGBoost | 99.31% | 99.31% | 99.29% |
| CatBoost | 99.31% | 99.31% | 99.29% |

5-fold cross-validation used only the training split:

- Mean accuracy: 99.09%
- Accuracy standard deviation: 0.58%
- Mean macro F1: 98.96%
- Macro F1 standard deviation: 0.73%
- Test set used during cross-validation: false

## Robustness

Feature ablation results:

| Configuration | Features | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- |
| A_All_Features | Soil_Moisture, Humidity, Temperature | 99.31% | 99.29% | 99.31% |
| B_No_Soil_Moisture | Humidity, Temperature | 46.11% | 39.18% | 43.54% |
| C_No_Temperature | Soil_Moisture, Humidity | 99.31% | 99.29% | 99.31% |
| D_No_Humidity | Soil_Moisture, Temperature | 99.14% | 99.05% | 99.14% |

Feature importance for the selected CatBoost model:

| Feature | Importance | Normalized Importance |
| --- | --- | --- |
| Soil_Moisture | 95.3105 | 95.31% |
| Temperature | 2.7514 | 2.75% |
| Humidity | 1.9381 | 1.94% |

Stability assessment:

- Category: B. Stable but heavily dependent on Soil_Moisture
- Macro F1 drop without Soil_Moisture: 60.11 percentage points
- Recommended interpretation: The model demonstrates strong performance under the evaluated dataset distribution, but the evidence shows strong dependence on Soil_Moisture. Do not claim 99% real-world farm generalization.

## Class-Wise Performance

| Crop | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| Banana | 100.00% | 98.28% | 99.13% | 58 |
| Jute | 100.00% | 97.96% | 98.97% | 49 |
| Maize | 100.00% | 98.36% | 99.17% | 122 |
| Mango | 98.08% | 100.00% | 99.03% | 51 |
| Pineapple | 97.30% | 100.00% | 98.63% | 36 |
| Potato | 98.48% | 100.00% | 99.24% | 65 |
| Strawberry | 98.89% | 100.00% | 99.44% | 89 |
| Sugarcane | 100.00% | 100.00% | 100.00% | 36 |
| Wheat | 100.00% | 100.00% | 100.00% | 73 |

## Explainability

- Method: local SHAP explanation using `shap.TreeExplainer` with the locked CatBoost model
- Explained features: Soil_Moisture, Humidity, Temperature
- Application display: feature contribution table and bar chart for the predicted crop
- Interpretation boundary: SHAP estimates feature contributions to the model prediction; it does not prove causality or prove the prediction is agronomically correct.

## Application

The Streamlit app provides the final user workflow:

1. User enters Soil Moisture, Humidity, and Temperature.
2. The locked CatBoost model predicts a crop.
3. The app displays confidence and a probability distribution across all nine crops.
4. The app displays a local SHAP explanation for the predicted crop.

## Prediction Verification

Existing prediction verification examples:

| Test Row | Soil Moisture | Humidity | Temperature | Actual Crop | Predicted Crop | Correct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 44 | 55.56 | 29.58 | Wheat | Wheat | True |
| 2 | 80 | 80.38 | 20.12 | Jute | Jute | True |
| 3 | 65 | 65.56 | 14.96 | Banana | Banana | True |
| 4 | 91 | 86.38 | 28.6 | Maize | Maize | True |
| 5 | 38 | 63.5 | 26.56 | Wheat | Wheat | True |

## Final Experiment Table

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Model comparison | CatBoost selected; Accuracy 99.31%, Macro F1 99.29% | CatBoost had the strongest selected metric profile on the fixed held-out split. |
| 5-fold CV | Accuracy 99.09% +/- 0.58% | Performance was stable across stratified folds on training data only. |
| Ablation | All features 99.31%; Humidity + Temperature 46.11% | Removing Soil_Moisture greatly reduced performance, showing heavy dataset dependence on that feature. |
| Feature importance | Soil_Moisture 95.31%, Temperature 2.75%, Humidity 1.94% | Soil_Moisture is the dominant predictive feature in this dataset. |
| Per-class evaluation | Minimum class F1 98.63% for Pineapple; Sugarcane and Wheat F1 100.00% | Held-out class-wise performance is high, but supports are imbalanced. |

## Limitations

1. Dataset V2 is relatively small with 2891 cleaned samples.
2. Class imbalance exists; test support ranges from 36 to 122 samples per class.
3. Soil_Moisture has unusually strong crop separation in this dataset.
4. Model performance is dataset-dependent.
5. Results should not automatically be interpreted as real-world farm generalization.
6. External validation on an independent field dataset has not been performed.

## Reproducibility Trace

- Raw dataset SHA-256: `cfd15c93086a02991bcda90869537b8390794e98c6acc5c5e13f0b7c6c11afee`
- X_train SHA-256: `988e8bb1102d3314595f75ea662882c0e85d59b35a70d0f68ca51ecb22053322`
- X_test SHA-256: `c32b5560f02b1be83c7b7214c37c441212a5af137a81911138adf03f605890d5`
- y_train SHA-256: `d31b29d641fcee5b519f7a87bde201a1ae264d6aa52779144dba993a50ffc1aa`
- y_test SHA-256: `a7abf3c25c8d48de473477b4e7fd4409e55a4b39c8da6f80f52f87bee1d4ad1a`
- Label encoder SHA-256: `706722a5bfdd18e7b40a86c18198c148b035f1ea55a3f10db047879ef51efee8`
- Locked model SHA-256: `76bf21ceed70955b66bb0b11c74f920d9b4cf5186d4f2894a64cd58313c51b4e`
- Feature order verified: true
- Label encoding verified: true
- Test set used during training: false
- Test labels used for hyperparameter tuning: false

## Paper-Ready Figures

Existing figures:

- Confusion matrix: `images/dataset_v2/model_evaluation/catboost_confusion_matrix.png`
- Model comparison chart: `images/dataset_v2/model_evaluation/model_comparison.png`
- Crop distribution: `images/dataset_v2/crop_distribution.png`

Consolidation-created figures:

- Feature importance: `images/dataset_v2/model_evaluation/feature_importance.png`
- Ablation result: `images/dataset_v2/model_evaluation/feature_ablation_results.png`
- SHAP explanation example: `images/dataset_v2/model_evaluation/shap_explanation_example.png`

## Source Files Used

- `data/dataset_v2/reports/model_comparison.csv`
- `data/dataset_v2/reports/per_class_metrics.csv`
- `data/dataset_v2/reports/confusion_matrices.json`
- `data/dataset_v2/reports/cross_validation_summary.json`
- `data/dataset_v2/reports/feature_ablation_results.csv`
- `data/dataset_v2/reports/feature_importance_v2.csv`
- `data/dataset_v2/reports/prediction_test.csv`
- `data/dataset_v2/reports/preprocessing_summary.json`
- `data/dataset_v2/reports/robustness_summary.json`
- `data/dataset_v2/reports/test_set_sanity_report.json`

## Final Status

Dataset V2 final experimental evidence is consolidated and traceable to existing project artifacts.
