# URBAN ENVIRONMENTAL ANOMALY DETECTION

## 1. ABSTRACT

This project applies unsupervised machine learning to identify unusual daily environmental conditions using air-pollution and weather measurements from the Beijing Multi-Site Air Quality dataset. Hourly observations from 12 monitoring stations are cleaned and aggregated into daily station-level and city-level features. A StandardScaler transforms the 23-feature city-level representation, and an Isolation Forest model flags observations that differ from patterns learned during training. PCA and K-Means are included as supplementary dimensionality-reduction and clustering methods. A Streamlit application presents historical anomalies, temporal and seasonal trends, station-level PM2.5 comparisons, feature-level descriptive comparisons, manual predictions, and CSV downloads.

## 2. PROBLEM STATEMENT

Environmental monitoring produces large volumes of measurements across pollutants, weather variables, stations, and time. Reviewing each measurement manually can make unusual combinations or periods difficult to identify. This project develops an unsupervised approach to flag unusual daily environmental profiles and explore their temporal and spatial patterns.

## 3. OBJECTIVES

1. Prepare hourly pollution and weather observations for analysis.
2. Engineer daily features from pollutant and weather variables.
3. Apply unsupervised machine-learning techniques to identify unusual observations.
4. Examine when anomalies occur across years, months, and seasons.
5. Compare PM2.5 measurements across the 12 monitoring stations.
6. Provide an interactive application for analysis, manual prediction, and data export.

## 4. DATASET

The project uses the Beijing Multi-Site Air Quality dataset covering March 2013 through February 2017. The source dataset contains 420,768 hourly records across 12 monitoring stations. The processed city-level daily dataset contains 1,461 days and 23 engineered environmental features, excluding the date field.

The features include daily mean, maximum, and standard deviation for pollutant concentrations, together with daily weather summaries such as mean temperature, pressure, dew point, total rainfall, and mean wind speed.

## 5. METHODOLOGY

### 5.1 Data preparation and feature engineering
The station CSV files are loaded, date/time fields are processed, and hourly observations are aggregated into daily summaries. Station-level summaries are combined to create a city-level daily representation. Missing-value handling and feature ordering must remain consistent with the training notebook and application.

### 5.2 Feature scaling
A fitted StandardScaler transforms the 23 features before Isolation Forest prediction. The saved scaler and model must receive the same feature names, order, and preprocessing assumptions used during training.

### 5.3 Isolation Forest
Isolation Forest is the primary anomaly-detection model. Its prediction convention is `-1` for an anomaly and `1` for a normal observation. The current dashboard baseline reports 73 anomalies among 1,461 city-days, equivalent to 5.00%. This is a model outcome, not an externally validated rate of unsafe air quality.

### 5.4 PCA and K-Means
PCA and K-Means provide supplementary analysis of dimensionality reduction and clustering. The saved PCA model retained 13 components for approximately 95.29% of variance in the prior analysis workflow. K-Means used two clusters. These methods should not be described as the source of the manual prediction label; that prediction uses the saved scaler and Isolation Forest.

### 5.5 Dashboard
The Streamlit interface provides Overview, Historical Anomalies, Temporal Trends, Station Analysis, Feature Insights, Manual Prediction, and Download Reports.

## 6. OBSERVED RESULTS

The current application displays:
- 1,461 city-days analyzed
- 73 anomalous city-days and 1,388 normal city-days
- 5.00% anomaly rate
- 12 monitoring stations and 420,768 source hourly records
- Higher average PM2.5 on anomalous city-days at all 12 stations in the displayed comparison
- The highest displayed relative station-level PM2.5 increase at Wanshouxigong (268.33%)
- The highest displayed seasonal anomaly rate in winter (13.02%) and the lowest in summer (1.09%)

These are descriptive results from the current processing and model configuration. They do not establish causality or independently verify regulatory air-quality status.

## 7. LIMITATIONS

1. Unsupervised anomaly labels do not provide ground-truth validation by themselves.
2. The model's output depends on training data, feature engineering, scaling, and contamination settings.
3. Seasonal and station comparisons are descriptive; they do not establish causal relationships.
4. A manual profile is only meaningful when its values use the expected units, daily aggregation definitions, and feature order.
5. Reference air-quality guidelines have specific averaging periods and should not be treated as universal minimum/maximum input limits.

## 8. CONCLUSION

The project demonstrates an end-to-end workflow for unsupervised environmental anomaly detection, from hourly data preparation and feature engineering to Isolation Forest prediction and interactive historical analysis. The Streamlit application makes the model outputs easier to inspect through temporal, station-level, and feature-level views and provides downloadable processed data. Further work could include validating alerts against independent environmental thresholds or expert-reviewed events, improving missing-data diagnostics, and evaluating model stability across alternative contamination settings.

## 9. TECHNOLOGIES USED

Python, pandas, NumPy, scikit-learn, joblib, Matplotlib, Streamlit, and Jupyter Notebook.
