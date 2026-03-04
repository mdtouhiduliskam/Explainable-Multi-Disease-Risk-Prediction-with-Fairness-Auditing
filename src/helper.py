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
# CELL 4: CORE HELPER FUNCTIONS
# ============================================================

def evaluate_model(y_true, y_pred, y_prob):
    """Compute ALL metrics we need for the paper."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'Specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'AUC_ROC': float(roc_auc_score(y_true, y_prob)),
        'AUC_PR': float(average_precision_score(y_true, y_prob)),
        'MCC': float(matthews_corrcoef(y_true, y_pred)),
        'Brier': float(brier_score_loss(y_true, y_prob)),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }


def run_cv_with_saving(X, y, model_fn, disease, model_name, n_splits=10):
    """
    Stratified K-fold CV with SMOTE inside folds.
    Saves each fold result in real-time.
    Returns everything needed for paper.
    """
    # Skip if already done
    if RM.is_model_done(disease, model_name):
        print(f"    ⏭️ {model_name} already done, skipping")
        cached = RM.master['model_results'][f"{disease}__{model_name}"]
        return cached['mean_metrics'], cached['std_metrics'], cached['fold_metrics']
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []
    fold_times = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_start = time.time()
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # SMOTE inside fold (NO DATA LEAKAGE)
        if y_train.mean() < 0.4:
            try:
                sm = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
                X_train, y_train = sm.fit_resample(X_train, y_train)
            except:
                pass
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        model = model_fn()
        model.fit(X_train_s, y_train)
        
        y_pred = model.predict(X_val_s)
        y_prob = model.predict_proba(X_val_s)[:, 1]
        
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())
        
        fold_m = evaluate_model(y_val, y_pred, y_prob)
        fold_m['fold'] = fold
        fold_m['train_size'] = len(y_train)
        fold_m['val_size'] = len(y_val)
        fold_m['fold_time'] = time.time() - fold_start
        fold_metrics.append(fold_m)
        fold_times.append(fold_m['fold_time'])
    
    # Aggregate
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1', 
                   'AUC_ROC', 'AUC_PR', 'MCC', 'Brier']
    
    mean_m = {k: float(np.mean([fm[k] for fm in fold_metrics])) for k in metric_keys}
    std_m = {k: float(np.std([fm[k] for fm in fold_metrics])) for k in metric_keys}
    
    # Save in real-time
    RM.save_model_result(
        disease, model_name, mean_m, std_m, fold_metrics,
        y_true=np.array(all_y_true),
        y_pred=np.array(all_y_pred),
        y_prob=np.array(all_y_prob)
    )
    
    return mean_m, std_m, fold_metrics


print("✅ Helper functions defined")
