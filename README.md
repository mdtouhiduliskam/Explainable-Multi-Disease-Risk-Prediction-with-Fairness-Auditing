# Explainable Multi-Disease Risk Prediction with Fairness Auditing on US Population Health Survey Data

---

## Overview

This repository contains the full implementation of a unified pipeline for chronic disease risk prediction that integrates:

- **8 classifiers** — from logistic regression to a stacking ensemble
- **4 explainability methods** — SHAP, Permutation Feature Importance (PFI), LIME, and TabNet attention
- **Multi-dimensional fairness auditing** — across sex, age, income, and education
- **4 chronic disease targets** — Diabetes, Heart Disease, Stroke, and Hypertension

All experiments are run on the [CDC BRFSS 2015 dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (253,680 respondents) under 10-fold stratified cross-validation with within-fold SMOTE to prevent data leakage.

---

## Key Results at a Glance

| Disease | Best AUC-ROC | Recall (Default) | Recall (Youden) | Max EOD |
|---|---|---|---|---|
| Diabetes | 0.819 | 0.220 | 0.777 | 0.009 |
| Heart Disease | 0.839 | 0.134 | 0.810 | 0.015 |
| Stroke | 0.811 | 0.010 | 0.773 | 0.009 |
| Hypertension | 0.799 | 0.662 | 0.753 | 0.006 |

> **Core finding:** All four disease models produce near-zero fairness disparity (max EOD = 0.015) — but for three diseases this equity reflects uniform under-detection, not good performance. Youden's J threshold calibration resolves this without sacrificing demographic equity.

---

---

## Diseases & Targets

Each disease is formulated as a binary classification task using self-reported diagnosis flags from the BRFSS survey:

| Disease | Target Variable | Positive Rate | Imbalance Ratio |
|---|---|---|---|
| Diabetes | `Diabetes_binary` | 15.8% | 1 : 5.3 |
| Heart Disease | `HeartDiseaseorAttack` | 9.4% | 1 : 9.6 |
| Stroke | `Stroke` | 4.1% | 1 : 23.6 |
| Hypertension | `HighBP` | 42.9% | 1 : 1.3 |

When predicting a given disease, all other disease indicator columns are **excluded from the input features** to prevent label leakage via comorbidity signaling.

---

## Models

| Model | Category | Notes |
|---|---|---|
| Logistic Regression | Linear baseline | L2 regularization, interpretable coefficients |
| K-Nearest Neighbors | Instance-based | k=7, uniform weights |
| Random Forest | Bagging ensemble | 200 trees, max depth 12 |
| XGBoost | Gradient boosting | Optuna-tuned per disease |
| LightGBM | Gradient boosting | Optuna-tuned per disease |
| CatBoost | Gradient boosting | Ordered boosting, default hyperparameters |
| TabNet | Deep tabular model | Attention-based, ante-hoc explainability |
| **Stacking Ensemble** | Meta-learner | XGBoost + LightGBM + CatBoost → Logistic Regression |

---

## Explainability Methods

| Method | Type | Scope |
|---|---|---|
| **SHAP** | Post-hoc | Global + local; game-theoretic Shapley values via TreeExplainer |
| **Permutation Feature Importance** | Post-hoc | Global; AUC-ROC degradation over 10 shuffle repeats |
| **LIME** | Post-hoc | Local; sparse linear surrogate for high/low-risk patient cases |
| **TabNet Attention** | Ante-hoc | Global; aggregated sparsemax masks across decision steps |

Cross-method agreement is quantified using **Jaccard similarity** on top-5 feature sets. Key finding: SHAP–TabNet agreement is strongest (mean J = 0.774), while SHAP–PFI divergence for stroke (J = 0.429 vs. cross-disease mean 0.607) flags threshold collapse — the model has learned internal structure but produces no positive predictions at the default threshold.

---

## Fairness Audit

Fairness is evaluated along four demographic dimensions:

| Dimension | Subgroups |
|---|---|
| Sex | Female, Male |
| Age | Young (18–44), Middle-aged (45–64), Senior (65+) |
| Income | Low (< $25K), Medium ($25–50K), High (> $50K) |
| Education | Low (no HS diploma), Medium (HS graduate), High (some college or above) |

**Metrics reported:**
- **Demographic Parity Difference (DPD):** Maximum difference in positive prediction rates across subgroups
- **Equalized Odds Difference (EOD):** Maximum difference in true positive rates (recall) across subgroups

Computed using [Fairlearn](https://fairlearn.org/) `MetricFrame` on cross-validated out-of-fold predictions.

---

## Threshold Calibration

Youden's J statistic is used to select the optimal classification threshold per disease:

$$t^* = \arg\max_{t} \left[ \text{Sensitivity}(t) + \text{Specificity}(t) - 1 \right]$$

| Disease | Default Threshold | Youden Threshold | Recall (Default → Optimized) |
|---|---|---|---|
| Diabetes | 0.500 | 0.120 | 0.220 → 0.777 |
| Heart Disease | 0.500 | 0.065 | 0.134 → 0.810 |
| Stroke | 0.500 | 0.030 | 0.010 → 0.773 |
| Hypertension | 0.500 | 0.408 | 0.662 → 0.753 |

Maximum EOD remains below **0.04** for all diseases after threshold calibration.

---


### Requirements

```
scikit-learn==1.3.*
xgboost==2.0.*
lightgbm==4.1.*
catboost==1.2.*
pytorch-tabnet==4.1.*
shap==0.43.*
lime==0.2.*
fairlearn==0.9.*
imbalanced-learn==0.11.*
optuna==3.4.*
torch
pandas
numpy
matplotlib
seaborn
```

---

## Dataset

This project uses the **CDC BRFSS 2015** cleaned dataset, curated by Alex Teboul on Kaggle:

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)

Place the downloaded CSV files in the `data/` directory before running notebooks.

> The dataset contains 253,680 complete survey responses and 22 variables (18 input features + 4 disease targets) with no missing values.

---

## Reproducibility

All experiments use a **fixed random seed of 42**. Experiments were run on Kaggle's GPU-accelerated infrastructure (NVIDIA Tesla T4). To reproduce results exactly:

1. Set `RANDOM_SEED = 42` in `src/utils.py`
2. Use the same library versions listed in `requirements.txt`
3. Run notebooks in order (`01` → `07`)

Optuna hyperparameter optimization uses 25 trials with 5-fold inner cross-validation on a 50,000-sample subset of the training data per disease target.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@article{hasan2025multidisease,
  title     = {Explainable Multi-Disease Risk Prediction with Fairness Auditing on US Population Health Survey Data},
  author    = {Hasan, Ekramul and Rahaman, Mustafizur and Asha, Nurtaz Begum and Rahat, SK Rakib Ul Islam and Amin, Md Al and Shakil, Mostafizur Rahman and Islam, Md Touhidul},
  journal   = {IEEE Access},
  year      = {2025},
  doi       = {10.1109/ACCESSXXX}
}
```

---

## Authors

| Author | Affiliation |
|---|---|
| Ekramul Hasan | Westcliff University, Irvine, CA, USA |
| Mustafizur Rahaman | Westcliff University, Irvine, CA, USA |
| Nurtaz Begum Asha | Westcliff University, Irvine, CA, USA |
| SK Rakib Ul Islam Rahat | International American University, Los Angeles, CA, USA |
| Md Al Amin | International American University, Los Angeles, CA, USA |
| Mostafizur Rahman Shakil | Westcliff University, Irvine, CA, USA |
| **Md Touhidul Islam** *(Corresponding)* | TU Dortmund, Germany — mdtouhidul.islam@tu-dortmund.de |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.