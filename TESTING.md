# Testing and Validation Checklist

## A. Dashboard smoke test

- [x] Overview page opens.
- [x] Historical Anomalies page opens and displays filtered results.
- [x] Temporal Trends page displays yearly, monthly, and seasonal charts/tables.
- [x] Station Analysis page displays the 12-station comparison.
- [x] Feature Insights page displays descriptive feature comparisons.
- [x] Manual Prediction returns a label and decision score for the default input profile.
- [x] Download Reports page displays CSV export controls.
- [x] User reports that “Download all historical anomaly results” works.
- [x] User reports that “Download station-level daily dataset” works.

## B. Observed baseline

| Check | Expected/current displayed value |
|---|---:|
| Hourly source records | 420,768 |
| City-level daily observations | 1,461 |
| Monitoring stations | 12 |
| Engineered features | 23 |
| Anomalous city-days | 73 |
| Normal city-days | 1,388 |
| Anomaly rate | 5.00% |

## C. Recommended final checks

- [ ] Download and open the anomalous-days-only CSV.
- [ ] Download and open the city-level daily dataset CSV.
- [ ] Confirm CSV headers and non-empty data.
- [ ] Restart the app from a fresh terminal using `streamlit run app.py`.
- [ ] Confirm all required model files are present under `models/`.
- [ ] Confirm the original station CSVs are present under `dataset/`.
- [ ] Confirm the notebook and exported model files are included in the final project folder.
- [ ] Check that `requirements.txt` installs all imports used by `app.py`.

## D. Interpretation safeguards

- Isolation Forest `-1` means anomalous and `1` means normal.
- The decision score is not a probability.
- An anomaly label is not an official air-quality category or proof of a health hazard.
- Feature differences are descriptive and do not prove causation.
- The scaler, feature order, and daily aggregation must match the model-training pipeline.
