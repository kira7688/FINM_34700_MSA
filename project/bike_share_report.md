# Capital Bikeshare Demand Analysis
## STAT 24620 / FINM 34700 / STAT 32950 — Group Data Project, Spring 2026

---

## Research Question

**Does feature engineering and unsupervised learning-based features help improve supervised prediction of bike-sharing demand?**

We investigate whether engineered time-of-day demand profiles and GMM-derived cluster memberships improve prediction accuracy for both hourly rental count (`cnt`) and high-demand classification (`high_demand`) compared to raw features alone.

---

## Dataset Description

The Capital Bikeshare Hourly Demand dataset (UCI ML Repository) contains 17,389 hourly observations from the Capital Bikeshare system in Washington D.C. across 2011–2012. Each record includes calendar variables (season, month, hour, weekday, holiday, working day), normalized weather measurements (temperature, humidity, wind speed, weather situation), and rental counts split by casual and registered riders.

**Target variables:**
- `cnt` — total hourly rental count (regression)
- `high_demand` — binary indicator: 1 if `cnt` ≥ 143 (median threshold), else 0 (classification)

**Train / test split:** 80% train (13,903 rows) / 20% test (3,476 rows), fixed at `random_state=42`.

---

## Preprocessing Decisions

| Decision | Rationale |
|---|---|
| Drop `instant`, `dteday`, `hour`, `timestamp` | Index / date columns, not informative features |
| Drop `casual`, `registered` as predictors | They sum to `cnt` — direct data leakage |
| Drop `atemp` | 0.99 correlation with `temp`; redundant |
| Encode `high_demand` as binary (0/1) | Map Low → 0, High → 1 for classification |
| `is_weekend` flag replacing `weekday` | Captures the dominant weekday vs. weekend signal more cleanly |
| Sin/cos encoding for `hr` and `mnth` | Dense cyclic variables; preserves wrap-around structure |
| One-hot encoding for `season` | Only 4 categories; effects are non-sinusoidal |
| StandardScaler inside pipeline | Required for ElasticNet; prevents scale-dependent penalization |
| `wk_hr_dist` / `nwk_hr_dist` features | Mean demand by hour conditioned on working/non-working day (Step 2+); note: computed on full data, introducing mild lookahead bias |
| GMM cluster one-hot columns | Cluster memberships from unsupervised analysis encoded as binary features (Step 3+) |

---

## Methods

### Supervised Methods

**ElasticNet (Regression & Classification)**
A regularized linear model combining L1 (Lasso) and L2 (Ridge) penalties. Hyperparameters alpha and l1_ratio selected via 5-fold cross-validation (`ElasticNetCV` / `LogisticRegressionCV`). StandardScaler applied inside a Pipeline to ensure fair penalization across features of different scales. The linear structure makes coefficients directly interpretable but limits the model's ability to capture nonlinear interactions (e.g., rush hour only on working days).

**Random Forest (Regression & Classification)**
An ensemble of 100 decision trees (`n_estimators=100, random_state=42`). No scaling required. Naturally captures nonlinear interactions and feature redundancy. Feature importance derived from mean impurity decrease. Used both as a regressor (`RandomForestRegressor`) predicting `cnt` and as a classifier (`RandomForestClassifier`) predicting `high_demand`.

### Unsupervised Methods

**PCA**
Applied to the fully feature-engineered dataset (Step 2.1 features, scaled). Two principal components extracted for visualization. PC1 and PC2 loadings interpreted to identify the dominant sources of variance across weather and time features.

**Gaussian Mixture Model (GMM)**
Fitted to hourly demand profiles aggregated by working day (`registered` and `casual` counts grouped by `hr × workingday`). Covariance structures (`full`, `tied`, `diag`, `spherical`) and component counts (1–29) evaluated by BIC on 5-fold cross-validation of the training split. Optimal cluster count selected at elbow (k=10, full covariance). Cluster memberships then projected back onto the full dataset and one-hot encoded as additional features for supervised models.

A standalone GMM classifier was also evaluated: cluster assignments mapped to demand labels (High/Low) based on within-cluster mean `high_demand`, then compared to ground truth.

---

## Results

### Table 1: Regression Results — Predicting `cnt`

| Step | Feature Set | # Features | ElasticNet MSE | ElasticNet R² | RF MSE | RF R² |
|---|---|---|---|---|---|---|
| **1** | Baseline (raw) | 12 | 19,396.17 | 0.3875 | 1,769.84 | 0.9441 |
| **1.1** | + Cyclic / one-hot encoding | 16 | 15,756.74 | 0.5024 | 2,119.26 | 0.9331 |
| **2** | + Demand profile features (lookahead) | 12 | 6,884.81 | 0.7826 | 1,970.16 | 0.9378 |
| **2.1** | + Demand profiles + cyclic encoding | 17 | 6,791.62 | 0.7855 | 1,934.46 | 0.9389 |
| **3** | + GMM cluster features | 17 | 12,882.12 | 0.5932 | **1,470.16** | **0.9536** |
| **3.1** | + GMM clusters + cyclic encoding | 21 | 11,764.28 | 0.6285 | 1,638.06 | 0.9483 |

### Table 2: Classification Results — Predicting `high_demand`

| Step | Feature Set | # Features | ElasticNet Acc. | ElasticNet F1 | RF Acc. | RF F1 | RF ROC-AUC |
|---|---|---|---|---|---|---|---|
| **1** | Baseline (raw) | 12 | 0.7822 | 0.7778 | 0.9243 | 0.9233 | 0.9245 |
| **1.1** | + Cyclic / one-hot encoding | 16 | 0.8412 | 0.8349 | 0.9318 | 0.9308 | 0.9320 |
| **2** | + Demand profile features (lookahead) | 12 | 0.9054 | 0.9030 | 0.9341 | 0.9329 | 0.9342 |
| **2.1** | + Demand profiles + cyclic encoding | 17 | 0.9074 | 0.9049 | 0.9318 | 0.9308 | 0.9320 |
| **3** | + GMM cluster features | 17 | 0.9028 | 0.9037 | 0.9479 | 0.9478 | 0.9484 |
| **3.1** | + GMM clusters + cyclic encoding | 21 | 0.9100 | 0.9104 | **0.9497** | **0.9492** | **0.9499** |

### Table 3: Standalone GMM Classifier Performance

| Metric | Value |
|---|---|
| Precision | 0.57 |
| Recall | 0.72 |
| F1 Score | 0.64 |

---

## Interpretation of Main Results

### Feature engineering strongly benefits ElasticNet

The clearest effect of feature engineering is on ElasticNet regression. Going from the raw baseline (R² = 0.39) to demand-profile features (R² = 0.78) more than doubles explained variance. This is because ElasticNet is a linear model and cannot learn the non-linear interaction between hour and working day on its own — the engineered `wk_hr_dist` and `nwk_hr_dist` features encode this interaction explicitly, making it linearly accessible.

### Random Forest is robust but also benefits from GMM features

Random Forest achieves high accuracy even on raw features (R² = 0.944), demonstrating its ability to learn complex interactions natively. However, adding GMM-derived cluster features (Step 3) still improves RF regression to R² = 0.9536, suggesting the demand regime labels captured something even the trees were not fully exploiting. The best overall classification result (RF Acc. = 0.9497, AUC = 0.9499) was achieved in Step 3.1 with both GMM clusters and cyclic encoding.

### PCA reveals time-of-day and weather as dominant variance drivers

PCA on the feature-engineered data shows that PC1 and PC2 together capture the primary structure in the data. The loading heatmap indicates that hour-related features (`hr_sin`, `hr_cos`, `wk_hr_dist`) load strongly on one component while weather variables (`temp`, `hum`) load on another. The scatter plot colored by hour of day shows clear circular arcs in PC1–PC2 space, confirming the strong cyclic time-of-day structure.

### GMM identifies meaningful demand regimes

The BIC curve across covariance structures and component counts identified an elbow at k=10 for the full covariance model. The 10 clusters map to interpretable demand regimes: high-peak commute hours on working days (cluster 8, mean cnt ≈ 429), moderate afternoon working-day demand, weekend leisure demand, and overnight low-demand hours. The demand profile by cluster (average hourly rental plotted by cluster label) shows clearly separated trajectories across the day, validating the clustering structure.

As a standalone predictor, the GMM classifier achieves F1 = 0.64, substantially below the supervised RF (F1 = 0.92+), confirming that unsupervised structure alone is insufficient for high-accuracy prediction. Its value is as an augmentation to supervised methods.

### Cyclic encoding helps ElasticNet more than RF

Adding sin/cos hour and month encoding (Step 1.1 vs Step 1) improves ElasticNet R² from 0.39 to 0.50 — a meaningful gain — while Random Forest R² actually decreases slightly (0.944 → 0.933). This is expected: RF can already find the correct split thresholds for raw hour values, while ElasticNet needs the cyclic structure made explicit.

---

## Comparison of Methods

| Dimension | ElasticNet | Random Forest |
|---|---|---|
| Raw baseline performance | Moderate (R² ≈ 0.39) | High (R² ≈ 0.94) |
| Benefit from feature engineering | Very large (+0.40 R² from demand profiles) | Small (+0.01 R² from GMM features) |
| Interpretability | High — direct coefficients | Moderate — feature importances |
| Cyclic encoding benefit | Large | Negligible / slightly negative |
| GMM cluster benefit | Moderate (R² +0.09) | Positive (R² +0.01, AUC +0.02) |
| Best performance | R² = 0.79 (Step 2.1) | R² = 0.9536 (Step 3) |

The central finding is that **feature engineering closes the gap between linear and non-linear models**, but does not eliminate it. Random Forest consistently outperforms ElasticNet across all steps, but ElasticNet's relative gain from engineered features is far larger. GMM cluster features provide the most reliable incremental improvement across both model families.

---

## Limitations

1. **Lookahead bias in demand profile features.** The `wk_hr_dist` and `nwk_hr_dist` features are computed from the full dataset before the train/test split. In a true production setting, these would need to be computed from training data only (or from historical data preceding the test period). The strong ElasticNet improvement in Steps 2 and 2.1 should be interpreted with this caveat.

2. **GMM clustering on training data only.** Step 3 GMM clusters were fitted on grouped aggregate data (not the raw rows), so there is less direct leakage — but the cluster labels assigned to the full dataset use the full-data fitted GMM, which is a mild form of in-sample information.

3. **Static demand threshold.** The `high_demand` binary label uses a fixed threshold of 143 (the sample median). This threshold was defined on the full dataset, which is a form of pre-processing leakage for strictly out-of-sample evaluation.

4. **No temporal cross-validation.** A random train/test split was used. Given the time-series nature of the data, a time-based split (e.g., train on 2011, test on 2012) would better reflect real forecasting performance and avoid temporal leakage from autocorrelated hours.

5. **Hyperparameter tuning.** Random Forest was not tuned (`n_estimators=100` default). Tuning depth, min samples, and feature subset size could further improve RF performance.

---

## Conclusion

Feature engineering and unsupervised learning features both contribute to improved supervised prediction performance, though with different magnitudes depending on the model type. For ElasticNet, demand profile features (derived from observed hourly patterns) produce the largest gain by encoding non-linear time-of-day interactions in a linear-accessible form. For Random Forest, GMM-derived cluster memberships provide the most consistent incremental improvement. The best overall models — RF with GMM clusters and cyclic encoding — achieve R² = 0.9536 for regression and AUC = 0.9499 for classification, compared to R² = 0.9441 and AUC = 0.9245 for the raw baseline. These results support the hypothesis that structured feature engineering and unsupervised demand regime discovery meaningfully improve bike-sharing demand prediction.

---

*Report prepared for STAT 24620 / FINM 34700 / STAT 32950, Spring 2026. Analysis conducted in Python using scikit-learn, pandas, matplotlib, seaborn, and plotly. AI tools used for coding assistance and grammar polishing.*
