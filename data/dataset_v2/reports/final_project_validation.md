# Final Project Validation

This report freezes the completed Dataset V2 crop recommendation workflow for final presentation and paper preparation. It does not introduce new model development, retraining, dataset changes, label changes, feature changes, or hyperparameter tuning.

## Project Implementation

- Dataset: CropRec-BD Dataset V2 stored at `data/dataset_v2/final_crops_data.csv`.
- Cleaned dataset: 2,891 cleaned samples after removing one exact duplicate from the cleaned copy.
- Features: `Soil_Moisture`, `Humidity`, `Temperature`.
- Classes: 9 crop classes: Banana, Jute, Maize, Mango, Pineapple, Potato, Strawberry, Sugarcane, Wheat.
- Preprocessing: missing-value checks, duplicate handling, label correction from `Sugercane` to `Sugarcane`, stratified train/test split.
- Models compared: Random Forest, LightGBM, XGBoost, CatBoost.
- Best model: CatBoost selected by Macro F1, then Weighted F1, then Accuracy.
- Evaluation: fixed held-out test split plus stratified 5-fold cross-validation on training data only.
- Streamlit app: `app/app.py`.
- Confidence: shown from `predict_proba()` for the predicted crop.
- Probability distribution: all 9 crop probabilities are displayed.
- SHAP: local SHAP explanation using `shap.TreeExplainer` with the locked CatBoost model.

## Validation

- Functional testing: passed on real rows from `data/dataset_v2/cleaned/X_test.csv`.
- Case 1: Soil_Moisture 44, Humidity 55.56, Temperature 29.58 produced Wheat with 96.62% confidence.
- Case 2: Soil_Moisture 80, Humidity 80.38, Temperature 20.12 produced Jute with 98.55% confidence.
- Case 3: invalid NaN input was rejected by the Dataset V2 validation layer with `ConfidencePredictionInputError`.
- Probability checks: each tested valid case returned 9 probabilities, and probability sums were 1.0.
- SHAP checks: each tested valid case returned 3 finite feature contributions with no NaN or infinite values.
- Streamlit testing: widget-level app test passed with no application exceptions.
- Streamlit server: `streamlit run app/app.py` returned HTTP 200 locally.
- Regression testing: Review 1 prediction still works and Review 2 confidence/SHAP functionality still works.
- Error handling: missing artifact, invalid input, prediction-shape, and SHAP explanation errors are handled through user-facing Streamlit messages.
- Portability: runtime application code uses project-relative paths based on `Path(__file__)`.
- Reproducibility: requirements, hashes, feature order, random state, model artifact, and encoder artifact were verified.

## Experimental Evidence

| Evidence Item | Location | Status |
| --- | --- | --- |
| Model comparison | `data/dataset_v2/reports/model_comparison.csv` | Available |
| Final metric table | `data/dataset_v2/reports/final_results.csv` | Available |
| Per-class metrics | `data/dataset_v2/reports/per_class_metrics.csv` | Available |
| Confusion matrices | `data/dataset_v2/reports/confusion_matrices.json` | Available |
| Cross-validation | `data/dataset_v2/reports/cross_validation_summary.json` | Available |
| Ablation | `data/dataset_v2/reports/feature_ablation_results.csv` | Available |
| Feature importance | `data/dataset_v2/reports/feature_importance_v2.csv` | Available |
| Prediction examples | `data/dataset_v2/reports/prediction_test.csv` | Available |
| Preprocessing | `data/dataset_v2/reports/preprocessing_summary.json` | Available |
| Robustness | `data/dataset_v2/reports/robustness_summary.json` | Available |
| Consolidated validation summary | `data/dataset_v2/reports/final_validation_summary.md` | Available |

Final verified metrics:

- Held-out Dataset V2 test accuracy: 99.31%.
- Held-out Dataset V2 Macro F1: 99.29%.
- 5-fold cross-validation accuracy: 99.09% +/- 0.58%.
- All-feature ablation accuracy: 99.31%.
- Humidity + Temperature ablation accuracy: 46.11%.
- CatBoost feature importance: Soil_Moisture 95.31%, Temperature 2.75%, Humidity 1.94%.

Important interpretation:

- The model achieved 99.31% accuracy on the held-out Dataset V2 test split.
- The result is dataset-dependent and strongly influenced by the crop-specific Soil_Moisture distribution.
- SHAP estimates feature contributions to the model prediction; it does not prove causality.

## Limitations

1. Dataset V2 is relatively small with 2,891 cleaned samples.
2. Class imbalance exists across the 9 crop classes.
3. Soil_Moisture has unusually strong crop separation in this dataset.
4. Model performance is dataset-dependent.
5. Results should not automatically be interpreted as real-world farm generalization.
6. External validation on an independent field dataset has not been performed.
7. The application is a crop recommendation prototype and does not include fertilizer recommendation, IoT integration, cloud deployment, or new data collection.

## Reproducibility Freeze

- Random state: 42 where applicable.
- Feature order: `Soil_Moisture`, `Humidity`, `Temperature`.
- Raw dataset SHA-256: `cfd15c93086a02991bcda90869537b8390794e98c6acc5c5e13f0b7c6c11afee`.
- X_train SHA-256: `988e8bb1102d3314595f75ea662882c0e85d59b35a70d0f68ca51ecb22053322`.
- X_test SHA-256: `c32b5560f02b1be83c7b7214c37c441212a5af137a81911138adf03f605890d5`.
- y_train SHA-256: `d31b29d641fcee5b519f7a87bde201a1ae264d6aa52779144dba993a50ffc1aa`.
- y_test SHA-256: `a7abf3c25c8d48de473477b4e7fd4409e55a4b39c8da6f80f52f87bee1d4ad1a`.
- Locked model SHA-256: `76bf21ceed70955b66bb0b11c74f920d9b4cf5186d4f2894a64cd58313c51b4e`.
- Label encoder SHA-256: `706722a5bfdd18e7b40a86c18198c148b035f1ea55a3f10db047879ef51efee8`.

## GitHub Readiness

- `venv/`, `.venv/`, `__pycache__/`, `.env`, `*.log`, `*.tmp`, and `catboost_info/` are ignored.
- Runtime source files, reports, images, datasets, and locked model artifacts are not ignored.
- Model `.pkl` files are intentionally kept in the repository because the final Streamlit application must run without retraining.
- Non-runtime historical Review 1 report metadata contains local absolute paths. These files are not used by the final Dataset V2 application.
- No commit or push was performed during final validation.

## Final Status

READY FOR FINAL PRESENTATION.

READY FOR PAPER PREPARATION.

The project is frozen at the completed Dataset V2 CatBoost crop recommendation workflow with confidence-aware prediction, 9-class probability output, and local SHAP explanation.
