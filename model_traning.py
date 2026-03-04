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
# CELL 6: FULL MODEL TRAINING — ALL MODELS × ALL DISEASES (~2-3 hours)
# ============================================================
phase = 'P03_model_training'
phase_start = time.time()

print("🏋️ PHASE 3: Full Model Training (10-fold CV)")
print(f"   4 diseases × 7 models × {N_FOLDS} folds")
print(f"   Estimated time: 2-3 hours\n")

for disease_name, target_col in DISEASES.items():
    print(f"\n{'='*70}")
    print(f"🏥 {disease_name}")
    print(f"{'='*70}")
    
    disease_features = [c for c in FEATURE_COLS if c != target_col]
    X = df[disease_features].values
    y = df[target_col].values
    
    print(f"   N={len(y):,} | Pos={y.sum():,} ({y.mean()*100:.1f}%)")
    
    # Get tuned params
    tp = tuned_params[disease_name]
    xgb_p = tp['xgb']
    lgbm_p = tp['lgbm']
    cat_p = tp['cat']
    
    # Define model factories (lambdas for fresh instances each fold)
    model_configs = {
        'Logistic_Regression': lambda: LogisticRegression(max_iter=1000, random_state=42),
        'KNN': lambda: KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        'Random_Forest': lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost_Tuned': lambda: XGBClassifier(
            **xgb_p, random_state=42, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1),
        'LightGBM_Tuned': lambda: LGBMClassifier(
            **lgbm_p, random_state=42, verbose=-1, n_jobs=-1),
        'CatBoost_Tuned': lambda: CatBoostClassifier(
            **cat_p, random_state=42, verbose=0),
    }
    
    for model_name, model_fn in model_configs.items():
        start = time.time()
        mean_m, std_m, fold_m = run_cv_with_saving(
            X, y, model_fn, disease_name, model_name, n_splits=N_FOLDS
        )
        elapsed = time.time() - start
        
        print(f"   {model_name:25s} | AUC:{mean_m['AUC_ROC']:.4f}±{std_m['AUC_ROC']:.4f} "
              f"| F1:{mean_m['F1']:.4f} | MCC:{mean_m['MCC']:.4f} | {elapsed:.0f}s")
    
    gc.collect()

RM.save_phase(phase, {'status': 'complete', 'diseases': list(DISEASES.keys())})
RM.log_time(phase, time.time() - phase_start)
print(f"\n✅ Phase 3 complete")