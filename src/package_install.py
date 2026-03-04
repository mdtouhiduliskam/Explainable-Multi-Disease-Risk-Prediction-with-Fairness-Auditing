
import subprocess, sys

# Install in correct order — numpy first to avoid conflicts
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy>=2.0"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
    "shap", "lime", "dice-ml", "fairlearn", "optuna",
    "pytorch-tabnet", "imbalanced-learn", "lightgbm", 
    "catboost", "scikit-posthocs"
])

print("✅ All packages installed")
print("⚠️  NOW RESTART KERNEL: Runtime → Restart Session")
print("   Then SKIP this cell and run from Cell 2 onwards")

# %%
# ============================================================
# CELL 2: IMPORTS + RESULT MANAGER (Real-time JSON saving)
# ============================================================
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

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from scipy import stats
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except:
    HAS_POSTHOCS = False

print("✅ All imports loaded")