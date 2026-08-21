# Dataset V2 Preprocessing Summary

## Objective
Prepare the first Dataset V2 baseline split from CropRec-BD v1 without modifying the raw CSV or training models.

## Raw Dataset Status
- Source file: `data\dataset_v2\final_crops_data.csv`
- Rows before cleaning: 2892
- Columns before cleaning: 5
- Raw SHA-256 before: `cfd15c93086a02991bcda90869537b8390794e98c6acc5c5e13f0b7c6c11afee`
- Raw SHA-256 after: `cfd15c93086a02991bcda90869537b8390794e98c6acc5c5e13f0b7c6c11afee`
- Raw file unchanged: True

## Cleaning Operations
- Selected features: Soil_Moisture, Humidity, Temperature
- Target column: `Crop Name`
- Excluded column: `Soil`
- Duplicate rows before cleaning: 1
- Duplicate rows removed from cleaned copy: 1
- Rows after duplicate removal: 2891
- Missing-like values before row removal: 0
- Rows removed for missing values: 0
- Label standardization: {'Sugercane': 'Sugarcane'}

## Feature Validation

| Feature | Min | Max | Mean | Median | Std | Unique Values |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Soil_Moisture | 31.0000 | 99.0000 | 72.8291 | 80.0000 | 19.4105 | 69 |
| Humidity | 30.6900 | 95.5600 | 78.1917 | 82.5000 | 13.3206 | 756 |
| Temperature | 11.8400 | 34.3000 | 25.5239 | 27.6500 | 4.5826 | 1185 |

## Crop Classes
Banana, Jute, Maize, Mango, Pineapple, Potato, Strawberry, Sugarcane, Wheat

## Label Mapping

| Crop | Encoded Label |
| --- | ---: |
| Banana | 0 |
| Jute | 1 |
| Maize | 2 |
| Mango | 3 |
| Pineapple | 4 |
| Potato | 5 |
| Strawberry | 6 |
| Sugarcane | 7 |
| Wheat | 8 |

## Train/Test Split
- Test size: 0.2
- Random state: 42
- Stratified: True
- X_train shape: [2312, 3]
- X_test shape: [579, 3]
- y_train shape: [2312]
- y_test shape: [579]
- Train class distribution: {'Banana': 231, 'Jute': 198, 'Maize': 485, 'Mango': 204, 'Pineapple': 145, 'Potato': 259, 'Strawberry': 354, 'Sugarcane': 144, 'Wheat': 292}
- Test class distribution: {'Banana': 58, 'Jute': 49, 'Maize': 122, 'Mango': 51, 'Pineapple': 36, 'Potato': 65, 'Strawberry': 89, 'Sugarcane': 36, 'Wheat': 73}

## Output Files
- `data\dataset_v2\cleaned\croprec_bd_clean.csv`
- `data\dataset_v2\cleaned\X_train.csv`
- `data\dataset_v2\cleaned\X_test.csv`
- `data\dataset_v2\cleaned\y_train.csv`
- `data\dataset_v2\cleaned\y_test.csv`
- `data\dataset_v2\cleaned\label_encoder.joblib`
- `data\dataset_v2\reports\preprocessing_summary.json`
- `data\dataset_v2\reports\preprocessing_summary.md`

## Validation Results
- raw_sha_preserved: True
- cleaned_rows_equal_2891: True
- cleaned_has_three_features_plus_target: True
- soil_excluded_from_cleaned_dataset: True
- nine_classes_present: True
- train_rows_equal_2312: True
- test_rows_equal_579: True
- X_train_shape: [2312, 3]
- X_test_shape: [579, 3]
- y_train_shape: [2312]
- y_test_shape: [579]
- feature_order_preserved: True
- no_nan_values: True
- no_infinite_feature_values: True
- label_encoder_class_count: 9
- label_encoder_mapping_recorded: True
- primary_output_files_exist: True
- all_output_files_exist: True

## Errors/Warnings
- None

## Final Status
DATASET V2 CLEANING COMPLETED
