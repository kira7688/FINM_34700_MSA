# Final Project Results Summary: Bike Sharing

## Research Question

How well do calendar and weather variables explain and predict hourly bike-sharing demand, and what multivariate demand regimes appear in the data?

## Dataset

Dataset: `bike_sharing_hourly.csv`

Rows: 17,379 hourly observations from January 1, 2011 to December 31, 2012.

Target variables:

- Regression target: `cnt`, total hourly rentals.
- Classification target: `high_demand`, whether hourly demand is above the sample median.

Leakage controls:

- `casual` and `registered` are excluded when predicting `cnt` or `high_demand` because `cnt = casual + registered`.
- `instant`, `dteday`, `cnt`, and `high_demand` are also excluded from predictor sets.
- Predictors are calendar variables (`season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`), weather situation, and continuous weather variables (`temp`, `atemp`, `hum`, `windspeed`).

## Descriptive Findings

Demand has strong hourly structure. Working days show commuter peaks, especially around morning and evening commute hours, while non-working days have a flatter pattern.

Demand varies by season and weather. Warmer seasons and better weather conditions generally have higher rentals. Humidity and temperature are associated with demand and appear repeatedly in model importance.

The binary `high_demand` label is balanced: about 49.9% of observations are high demand.

## Multivariate Structure Results

PCA on processed calendar/weather predictors:

| Metric | Value |
|---|---:|
| PC1 + PC2 explained variance | 0.399 |
| PCs needed for 80% variance | 14 |

Interpretation: PCA finds real structure, but the predictor space is not reducible to only one or two dimensions. Calendar categories, hour patterns, and weather variables each contribute different pieces of variation.

k-means on PCA scores:

| Metric | Value |
|---|---:|
| Best k by silhouette | 2 |
| Best silhouette | 0.182 |
| Adjusted Rand index vs high-demand label | 0.094 |

The k=2 cluster profile separates a lower-demand/cooler regime from a higher-demand/warmer regime:

| Cluster | Rows | Mean Count | Median Count | High-Demand Rate | Mean Temp |
|---|---:|---:|---:|---:|---:|
| 0 | 8,469 | 129.1 | 87.0 | 0.342 | 0.328 |
| 1 | 8,910 | 246.8 | 209.0 | 0.649 | 0.657 |

GMM on PCA scores:

| Metric | Value |
|---|---:|
| Best k by BIC | 8 |
| Silhouette at selected k | 0.016 |

Interpretation: GMM finds finer probabilistic subgroups, but geometric separation is weak. For the report, k-means is easier to explain, while GMM is useful as a comparison showing that probabilistic substructure does not necessarily mean cleanly separated clusters.

## Count Prediction Results

Regression target: `cnt`

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Random forest | 51.96 | 33.01 | 0.921 |
| Lasso regression | 102.97 | 75.91 | 0.691 |
| Linear regression | 102.97 | 75.97 | 0.691 |
| Ridge regression | 102.98 | 75.97 | 0.691 |
| Decision tree | 105.07 | 77.55 | 0.678 |
| PCR regression | 151.75 | 113.81 | 0.329 |
| Mean baseline | 185.26 | 144.67 | 0.000 |

Best model: random forest.

Interpretation: random forest strongly outperforms the linear/regularized/PCR models, which means the relationship between predictors and demand is nonlinear. The likely nonlinearities are hour-by-working-day effects, season/weather interactions, and temperature/humidity thresholds.

## High-Demand Classification Results

Classification target: `high_demand`

| Model | Accuracy | Balanced Accuracy | F1 | ROC AUC |
|---|---:|---:|---:|---:|
| Random forest | 0.924 | 0.923 | 0.924 | 0.978 |
| Logistic regression | 0.886 | 0.886 | 0.888 | 0.958 |
| Shrinkage LDA | 0.879 | 0.879 | 0.883 | 0.954 |
| Decision tree | 0.807 | 0.807 | 0.811 | 0.867 |
| Majority baseline | 0.501 | 0.500 | 0.000 | 0.500 |

Best model: random forest.

Interpretation: logistic regression and LDA are strong, which means a large part of the high-demand boundary is close to linear after encoding calendar variables. Random forest is still best because it captures nonlinear and interaction effects more effectively.

## Feature Importance

Top count-prediction features from random forest permutation importance:

| Feature | Importance |
|---|---:|
| Hour 17 | 0.237 |
| Hour 8 | 0.225 |
| Hour 18 | 0.197 |
| Feels-like temperature | 0.138 |
| Humidity | 0.098 |
| Temperature | 0.084 |
| Working day | 0.067 |
| Hour 7 | 0.056 |
| Hour 19 | 0.054 |
| Year | 0.045 |

Interpretation: the strongest predictors are commute-hour indicators, temperature/feels-like temperature, humidity, working-day status, and year. This supports a demand-cycle interpretation rather than a purely weather-only interpretation.

## Multivariate Response Check

Supplementary responses: `casual` and `registered`

| Model | Casual R2 | Registered R2 | Average R2 |
|---|---:|---:|---:|
| Random forest multi-output | 0.848 | 0.931 | 0.890 |
| PLS multivariate regression | 0.579 | 0.688 | 0.634 |

CCA between calendar/weather predictors and rider-type responses:

| Canonical Pair | Canonical Correlation |
|---|---:|
| 1 | 0.829 |
| 2 | 0.722 |

Interpretation: the predictor block is strongly associated with both casual and registered demand. Registered demand is easier to predict than casual demand, likely because commuter behavior is more regular.

## Recommended Final Report Argument

Bike Sharing is the best final dataset because it gives a clean and defensible comparison between multivariate structure and supervised prediction. PCA and clustering reveal demand regimes, but their cluster-label agreement is modest. Supervised methods, especially random forests, predict demand very strongly. This contrast is exactly what the course project asks for: different multivariate and statistical-learning methods answer different questions.

Main conclusion: calendar and weather variables explain hourly bike demand very well, but the best prediction requires nonlinear modeling. Random forests are the best technique for this dataset, while PCA and clustering are best used for exploratory structure and visualization.

## Limitations

The analysis is predictive and associational, not causal. Weather and calendar variables are related to demand, but the models do not prove causal effects.

The train/test split is random, so it measures general predictive accuracy across similar observations, not a strict future-forecasting setup.

The raw date is excluded to avoid memorizing individual dates. The year variable is retained because the original dataset includes it as a calendar predictor, but it may partly capture system growth from 2011 to 2012.

Cluster results depend on preprocessing and PCA representation. The weak GMM silhouette suggests that demand regimes overlap rather than forming sharply separated Gaussian groups.

## Files Produced

- `final_project_bike_sharing_prediction.ipynb`
- `bike_project_results/final_project_summary_table.csv`
- `bike_project_results/final_regression_results.csv`
- `bike_project_results/final_classification_results.csv`
- `bike_project_results/cluster_model_metrics.csv`
- `bike_project_results/kmeans_cluster_profile.csv`
- `bike_project_results/random_forest_regression_importance.csv`
- `bike_project_results/random_forest_classification_importance.csv`
- `bike_project_results/figures/`
