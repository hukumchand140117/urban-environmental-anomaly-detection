# Urban Environmental Anomaly Detection

An unsupervised machine-learning project that identifies unusual daily environmental conditions using pollution and weather measurements from the Beijing Multi-Site Air Quality dataset.

## Project objective

Detect anomalous environmental conditions using unsupervised machine learning, examine pollution and weather measurements associated with detected anomalies, and analyze temporal and station-level patterns across 12 monitoring stations.

## Dataset

- Source: Beijing Multi-Site Air Quality dataset
- Monitoring stations: 12
- Time period: March 2013–February 2017
- Hourly records in the source dataset: 420,768
- City-level daily observations: 1,461
- Engineered daily features: 23 (excluding the date column)

The dataset contains particulate pollutants, gaseous pollutants, and weather variables. The project aggregates hourly observations into daily station-level and city-level features.

## Machine-learning methods

- **Isolation Forest:** primary model used to flag unusual city-level daily profiles. The model returns `-1` for an anomaly and `1` for a normal observation.
- **StandardScaler:** scales the 23 features before Isolation Forest prediction.
- **PCA:** dimensionality reduction used in the analysis workflow.
- **K-Means:** supplementary clustering analysis; it is not the model used for the manual prediction page.

The saved Isolation Forest was trained on the standardized 23 original features. The PCA model is not applied in the manual prediction pipeline.

## Current baseline results

The current dashboard displays:
- 1,461 city-days analyzed
- 73 anomalous city-days
- 1,388 normal city-days
- 5.00% anomaly rate
- 12 monitoring stations
- 420,768 hourly source records

These figures should be rechecked if preprocessing or model files are changed. The anomaly rate reflects the configured model and is not an official air-quality classification.

## Streamlit dashboard

The application includes seven pages:

1. **Overview** — dataset metrics and anomaly summary.
2. **Historical Anomalies** — date/classification filters and anomaly records.
3. **Temporal Trends** — yearly, monthly, and seasonal patterns.
4. **Station Analysis** — PM2.5 comparisons across monitoring stations.
5. **Feature Insights** — descriptive comparisons between normal and anomalous days.
6. **Manual Prediction** — prediction for a manually entered 23-feature daily profile.
7. **Download Reports** — CSV exports for historical results and processed daily datasets.

## Project structure

```text
City Anomaly Project/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── TESTING.md
├── .gitignore
├── dataset/
│   └── original station CSV files
├── models/
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── pca.pkl
│   └── kmeans_model.pkl
└── notebooks/
    └── Environmental_Anomaly_Detection.ipynb
```

Keep the actual filenames in your project if they differ from this example. Do not commit your `venv/` directory.

## Setup on Windows

Open a terminal in the project directory.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell prevents environment activation, you can run the environment's Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

## Interpretation and limitations

- An anomaly is a model-relative flag, not proof that a day is unsafe or that a pollutant violated a legal standard.
- WHO guideline values shown in the interface are reference benchmarks with specific averaging periods; they are not the thresholds used by Isolation Forest.
- The decision score is not a probability. Lower scores indicate observations farther into the model's anomalous region.
- Feature comparisons are descriptive associations, not causal explanations or model-specific feature attributions.
- Manual prediction evaluates a single entered profile. Historical analysis evaluates the processed dataset.
- Results depend on data cleaning, aggregation, feature order, scaler, and model versions. Use the same preprocessing pipeline for future inputs.

## Technologies

Python, pandas, NumPy, scikit-learn, joblib, Matplotlib, and Streamlit.
