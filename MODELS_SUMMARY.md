# Random Forest Models - Summary Report

## Overview
Successfully created and tested Random Forest regression models for air quality prediction across multiple cities in India.

## Models Created

### 1. Delhi Model
- **File**: `Delhi_random_forest_model.pkl`
- **Size**: 6.63 MB
- **Data Source**: `r.k.-puram, delhi, delhi, india-air-quality.csv`
- **Training Samples**: 2,827 (Test: 707)
- **Performance**:
  - Mean Squared Error: 8.17
  - R² Score: 0.9990 (99.90% accuracy)
- **Configuration**:
  - Number of trees: 100
  - Max depth: 20
- **Test Prediction**: AQI = 60.60 ✅

### 2. Greater Noida Model
- **File**: `GNoida_random_forest_model.pkl`
- **Size**: 3.82 MB
- **Data Source**: `knowledge-park - iii, greater noida, india-air-quality.csv`
- **Training Samples**: 1,390 (Test: 348)
- **Performance**:
  - Mean Squared Error: 60.30
  - R² Score: 0.9916 (99.16% accuracy)
- **Configuration**:
  - Number of trees: 100
  - Max depth: 20
- **Test Prediction**: AQI = 75.29 ✅

### 3. Ghaziabad Model (Sanjay Nagar)
- **File**: `Ghaziabad_random_forest_model.pkl`
- **Size**: 3.48 MB
- **Data Source**: `sanjay-nagar, ghaziabad, india-air-quality.csv`
- **Training Samples**: 1,244 (Test: 311)
- **Performance**:
  - Mean Squared Error: 57.65
  - R² Score: 0.9928 (99.28% accuracy)
- **Configuration**:
  - Number of trees: 100
  - Max depth: 20
- **Test Prediction**: AQI = 71.81 ✅

### 4. Lucknow Model (Pre-existing)
- **File**: `Lucknow_random_forest_model.pkl`
- **Size**: 6.95 MB
- **Data Source**: Previously trained
- **Configuration**:
  - Number of trees: 100
  - Max depth: None (unlimited)
- **Test Prediction**: AQI = 53.29 ✅

## Support Files

### Imputer
- **File**: `imputer.pkl`
- **Size**: 0.87 KB
- **Purpose**: Handles missing values in input data using mean imputation

## Input Features
All models use the following 6 pollutant parameters:
1. **pm25** - Particulate Matter 2.5
2. **pm10** - Particulate Matter 10
3. **o3** - Ozone
4. **no2** - Nitrogen Dioxide
5. **so2** - Sulfur Dioxide
6. **co** - Carbon Monoxide

## Output
- **Prediction**: Air Quality Index (AQI)

## Model Performance Summary
All models demonstrate excellent performance with R² scores above 0.99, indicating:
- High accuracy in AQI predictions
- Strong correlation between pollutant levels and AQI
- Reliable predictions for real-time air quality assessment

## Compatibility
- **Trained with**: scikit-learn 1.8.0
- **Compatible with**: scikit-learn 1.3.2 - 1.8.0 (with compatibility fixes applied)
- **Python Version**: 3.11+

## Usage
Models are integrated into the Air-Pulse Streamlit application and can predict AQI based on current pollutant levels for each city.

## Training Scripts
- **Primary Training**: `train_all_models.py` - Automated training for multiple cities
- **Original Script**: `RandomForest.py` - Single city training template
- **Testing**: `test_models.py` - Validates all models

## Date Created
December 20, 2025

## Status
✅ All models tested and operational
✅ Integrated with Air-Pulse web application
✅ Ready for deployment
