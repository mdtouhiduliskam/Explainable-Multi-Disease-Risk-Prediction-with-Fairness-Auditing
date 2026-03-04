import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, json, gc, copy, pickle
from datetime import datetime, timedelta
from collections import OrderedDict
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                               GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, matthews_corrcoef,
                              confusion_matrix, roc_curve, brier_score_loss,
                              average_precision_score, classification_report)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.base import clone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

import shap
import lime.lime_tabular
import dice_ml

from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference,
                                MetricFrame)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

from pytorch_tabnet.tab_model import TabNetClassifier
import torch



# ============================================================
# CELL 5: OPTUNA HYPERPARAMETER TUNING (~45-60 min)
# ============================================================
phase = 'P02_optuna_tuning'
phase_start = time.time()

if RM.is_phase_done(phase):
    print(f"⏭️ {phase} already done, loading cached params...")
    tuned_params = RM.master['tuned_params']
else:
    print("🔧 PHASE 2: Optuna Hyperparameter Tuning")
    print("   Estimated time: 45-60 min\n")
    
    TUNE_SAMPLE = 50000
    TUNE_TRIALS = 25
    
    tuned_params = {}
    
    for disease_name, target_col in DISEASES.items():
        print(f"\n  📌 {disease_name}")
        
        disease_features = [c for c in FEATURE_COLS if c != target_col]
        idx = np.random.choice(len(df), min(TUNE_SAMPLE, len(df)), replace=False)
        X_tune = df.iloc[idx][disease_features].values
        y_tune = df.iloc[idx][target_col].values
        
        # --- Tune XGBoost ---
        def xgb_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            model = XGBClassifier(**params, random_state=42, eval_metric='logloss',
                                   use_label_encoder=False, n_jobs=-1)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, vl in skf.split(X_tune, y_tune):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tune[tr])
                X_vl = sc.transform(X_tune[vl])
                model.fit(X_tr, y_tune[tr])
                scores.append(roc_auc_score(y_tune[vl], model.predict_proba(X_vl)[:, 1]))
            return np.mean(scores)
        
        print(f"    Tuning XGBoost ({TUNE_TRIALS} trials)...", end=' ', flush=True)
        study_xgb = optuna.create_study(direction='maximize')
        study_xgb.optimize(xgb_objective, n_trials=TUNE_TRIALS)
        print(f"AUC={study_xgb.best_value:.4f}")
        
        # --- Tune LightGBM ---
        def lgbm_objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            }
            model = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, vl in skf.split(X_tune, y_tune):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tune[tr])
                X_vl = sc.transform(X_tune[vl])
                model.fit(X_tr, y_tune[tr])
                scores.append(roc_auc_score(y_tune[vl], model.predict_proba(X_vl)[:, 1]))
            return np.mean(scores)
        
        print(f"    Tuning LightGBM ({TUNE_TRIALS} trials)...", end=' ', flush=True)
        study_lgbm = optuna.create_study(direction='maximize')
        study_lgbm.optimize(lgbm_objective, n_trials=TUNE_TRIALS)
        print(f"AUC={study_lgbm.best_value:.4f}")
        
        # --- Tune CatBoost ---
        def cat_objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 400),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }
            model = CatBoostClassifier(**params, random_state=42, verbose=0)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for tr, vl in skf.split(X_tune, y_tune):
                sc = StandardScaler()
                X_tr = sc.fit_transform(X_tune[tr])
                X_vl = sc.transform(X_tune[vl])
                model.fit(X_tr, y_tune[tr])
                scores.append(roc_auc_score(y_tune[vl], model.predict_proba(X_vl)[:, 1]))
            return np.mean(scores)
        
        print(f"    Tuning CatBoost ({TUNE_TRIALS} trials)...", end=' ', flush=True)
        study_cat = optuna.create_study(direction='maximize')
        study_cat.optimize(cat_objective, n_trials=TUNE_TRIALS)
        print(f"AUC={study_cat.best_value:.4f}")
        
        tuned_params[disease_name] = {
            'xgb': study_xgb.best_params,
            'xgb_best_auc': study_xgb.best_value,
            'lgbm': study_lgbm.best_params,
            'lgbm_best_auc': study_lgbm.best_value,
            'cat': study_cat.best_params,
            'cat_best_auc': study_cat.best_value,
        }
    
    RM.master['tuned_params'] = tuned_params
    RM.save_phase(phase, tuned_params)
    
RM.log_time(phase, time.time() - phase_start)
print(f"\n✅ Phase 2 complete")
