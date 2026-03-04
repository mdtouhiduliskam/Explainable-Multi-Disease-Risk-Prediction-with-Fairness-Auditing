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
# CELL 3: LOAD DATASET + EDA (~2 min)
# ============================================================
phase = 'P01_data_loading'
phase_start = time.time()

df = pd.read_csv("/kaggle/input/datasets/alexteboul/diabetes-health-indicators-dataset/diabetes_012_health_indicators_BRFSS2015.csv")


 

# Binary diabetes target
df['Diabetes_binary'] = (df['Diabetes_012'] >= 1).astype(int)

# Config
DISEASES = {
    'Diabetes': 'Diabetes_binary',
    'HeartDisease': 'HeartDiseaseorAttack',
    'Stroke': 'Stroke',
    'Hypertension': 'HighBP'
}

TARGET_COLS = ['Diabetes_012', 'Diabetes_binary', 'HeartDiseaseorAttack', 'Stroke', 'HighBP']
FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS]
N_FOLDS = 10
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)

# Dataset statistics
dataset_meta = {
    'total_samples': int(len(df)),
    'total_features': len(FEATURE_COLS),
    'feature_names': FEATURE_COLS,
    'diseases': {},
    'missing_values': int(df.isnull().sum().sum()),
    'class_distributions': {}
}

for name, col in DISEASES.items():
    pos = int(df[col].sum())
    neg = int(len(df) - pos)
    dataset_meta['diseases'][name] = {
        'target_column': col,
        'positive': pos,
        'negative': neg,
        'positive_rate': round(pos / len(df), 4),
        'imbalance_ratio': f"1:{neg // max(pos, 1)}"
    }
    print(f"  {name}: {pos:,} positive ({pos/len(df)*100:.1f}%) | Ratio 1:{neg//max(pos,1)}")

# Feature statistics
feature_stats = []
for col in FEATURE_COLS:
    feature_stats.append({
        'feature': col,
        'mean': round(float(df[col].mean()), 4),
        'std': round(float(df[col].std()), 4),
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'unique': int(df[col].nunique()),
        'missing': int(df[col].isnull().sum())
    })

dataset_meta['feature_statistics'] = feature_stats

# Comorbidity matrix
comorbidity = df[list(DISEASES.values())].corr().round(4).to_dict()
dataset_meta['comorbidity_matrix'] = comorbidity

RM.master['metadata'] = dataset_meta
RM.save_phase(phase, dataset_meta, also_csv=pd.DataFrame(feature_stats))
RM.log_time(phase, time.time() - phase_start)

print(f"\n📊 Dataset: {len(df):,} rows × {len(FEATURE_COLS)} features")
print(f"✅ Phase 1 complete")
