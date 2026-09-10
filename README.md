# EV Battery Failure Prediction

A data analysis and machine learning project predicting EV battery failure from
telemetry data, using Python, Pandas, Seaborn, and scikit-learn.

## Problem Statement

Using 20,000 EV battery telemetry records, can we predict which batteries are
likely to fail, and what actually drives that failure?

## Dataset

- **Source:** [EV Battery Health Prediction Dataset (20K)](https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k) — Kaggle
- **Size:** 20,000 rows, 70 columns
- **Target:** `battery_failure` (binary: 0 = healthy, 1 = failed) — ~93% healthy / ~7% failed
- **Note:** This dataset is synthetically generated and may not fully reflect
  real-world battery behavior. No large-scale public dataset of real EV battery
  failures currently exists openly, which is why a synthetic dataset was used here.

## Approach

1. **Data cleaning** — handled ~3–5% missing values per column across 66 of 69
   columns using median imputation (numeric) and mode imputation (categorical).
   Dropped `state_of_health` after finding it 99% correlated with
   `battery_health_percent`.
2. **Exploratory Data Analysis** — univariate distributions (histograms, count
   plots), bivariate comparisons against the target (box plots), and a full
   correlation analysis.
3. **Statistical validation** — t-tests on the top predictors confirmed the
   observed differences between healthy and failed batteries are statistically
   significant (p < 0.0001).
4. **Modeling** — trained and compared Logistic Regression and Random Forest
   (including a class-weight-balanced variant) to predict failure.

## Key Findings

Battery failure is primarily driven by **thermal stress and physical
degradation** — not by how the vehicle is driven day-to-day.

**Top predictors (by correlation with `battery_failure`):**
1. Thermal runaway risk (0.38)
2. Capacity loss percentage (0.31)
3. Internal resistance (0.28)
4. Operating temperature (0.26–0.28)
5. Prior fault history and BMS warning count (0.16–0.19)

Driving-behavior features (aggressive acceleration, braking, charging habits)
showed almost no relationship to failure — a useful negative finding that rules
out an entire category of intuitive-but-wrong hypotheses.

These findings were confirmed across four independent checks: visual comparison
(box plots), correlation analysis, formal statistical testing, and the trained
model's own learned feature weights.

## Model Results

| Metric (class 1 = failed) | Logistic Regression (scaled) | Random Forest | Random Forest (balanced) |
|---|---|---|---|
| Recall | **0.69** | 0.33 | 0.51 |
| Precision | 0.83 | 0.76 | 0.64 |
| F1-score | **0.75** | 0.46 | 0.57 |
| Accuracy | 0.97 | 0.95 | 0.95 |

**Best model:** Logistic Regression (scaled) — highest recall and F1-score for
detecting actual battery failures. This ran counter to the usual assumption
that Random Forest is the stronger default choice, highlighting why testing
multiple models matters rather than assuming.

Recall was prioritized over raw accuracy because the target is imbalanced
(~93/7) — a model predicting "healthy" every time would already score ~93%
accuracy while being useless for the actual goal of catching failures.

## Limitations

- Dataset is synthetically generated, not real-world telemetry
- Median/mode imputation introduced a minor, identified artifact (a spike at
  the imputed value) in a few distributions — flagged and accounted for during
  EDA rather than mistaken for a real pattern
- `predicted_remaining_life_cycles` showed a moderate correlation with the
  target and was investigated for potential data leakage before deciding how
  to treat it

## Project Structure

```
ev-battery-failure-prediction/
├── data/
│   └── ev_battery_failure.csv
├── notebooks/
│   └── 01_eda.ipynb
├── INSIGHTS.md
└── README.md
```

## How to Run

1. Clone this repository
2. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scipy scikit-learn jupyter
   ```
3. Launch Jupyter:
   ```
   jupyter notebook
   ```
4. Open `notebooks/01_eda.ipynb` and run all cells

## Tools Used

Python, Pandas, Matplotlib, Seaborn, SciPy, scikit-learn, Jupyter Notebook
