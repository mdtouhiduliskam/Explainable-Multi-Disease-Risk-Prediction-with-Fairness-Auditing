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
# RESULT MANAGER — saves everything in real-time as JSON
# ============================================================
class ResultManager:
    """
    Saves all experiment results to JSON in real-time.
    If Kaggle crashes, you lose NOTHING. Just reload and continue.
    """
    
    def __init__(self, base_dir='experiment_results'):
        self.base_dir = base_dir
        self.dirs = {
            'root': base_dir,
            'tables': f'{base_dir}/tables',
            'figures': f'{base_dir}/figures',
            'models': f'{base_dir}/models',
            'checkpoints': f'{base_dir}/checkpoints',
            'ablation': f'{base_dir}/ablation',
            'fairness': f'{base_dir}/fairness',
            'xai': f'{base_dir}/xai',
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        
        # Master results file — everything goes here
        self.master_file = f'{base_dir}/MASTER_RESULTS.json'
        self.master = self._load_or_create_master()
        
        self.start_time = time.time()
        print(f"📁 Results directory: {base_dir}/")
        print(f"📋 Master file: {self.master_file}")
    
    def _load_or_create_master(self):
        """Load existing results or create new."""
        if os.path.exists(self.master_file):
            with open(self.master_file, 'r') as f:
                master = json.load(f)
            print(f"🔄 Loaded existing results: {len(master.get('completed_phases', []))} phases done")
            return master
        return {
            'experiment_name': 'FairXRisk_Framework',
            'created_at': datetime.now().isoformat(),
            'dataset': 'CDC_BRFSS_2015',
            'completed_phases': [],
            'model_results': {},
            'xai_results': {},
            'fairness_results': {},
            'ablation_results': {},
            'statistical_tests': {},
            'tuned_params': {},
            'timing': {},
            'metadata': {}
        }
    
    def save_master(self):
        """Save master file (call after every phase)."""
        self.master['last_updated'] = datetime.now().isoformat()
        self.master['total_runtime_seconds'] = time.time() - self.start_time
        with open(self.master_file, 'w') as f:
            json.dump(self.master, f, indent=2, default=str)
    
    def save_phase(self, phase_name, data, also_csv=None):
        """Save a phase result as separate JSON + update master."""
        # Save individual JSON
        filepath = f"{self.dirs['checkpoints']}/{phase_name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save CSV if provided
        if also_csv is not None and isinstance(also_csv, pd.DataFrame):
            also_csv.to_csv(f"{self.dirs['tables']}/{phase_name}.csv", index=False)
        
        # Mark phase complete
        if phase_name not in self.master['completed_phases']:
            self.master['completed_phases'].append(phase_name)
        
        self.save_master()
        elapsed = time.time() - self.start_time
        print(f"  💾 Saved: {phase_name} (elapsed: {timedelta(seconds=int(elapsed))})")
    
    def save_model_result(self, disease, model_name, mean_metrics, std_metrics, 
                          fold_metrics, y_true=None, y_pred=None, y_prob=None):
        """Save individual model result in real-time."""
        key = f"{disease}__{model_name}"
        
        result = {
            'disease': disease,
            'model': model_name,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'fold_metrics': fold_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store predictions for later use (as lists for JSON)
        if y_true is not None:
            pred_file = f"{self.dirs['models']}/{key.replace(' ', '_')}_predictions.npz"
            np.savez_compressed(pred_file, 
                                y_true=y_true, y_pred=y_pred, y_prob=y_prob)
            result['predictions_file'] = pred_file
        
        self.master['model_results'][key] = result
        self.save_master()
    
    def save_xai(self, disease, method, data):
        """Save XAI results."""
        key = f"{disease}__{method}"
        self.master['xai_results'][key] = data
        
        filepath = f"{self.dirs['xai']}/{key.replace(' ', '_')}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.save_master()
    
    def save_fairness(self, disease, data):
        """Save fairness results."""
        self.master['fairness_results'][disease] = data
        filepath = f"{self.dirs['fairness']}/{disease.replace(' ', '_')}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.save_master()
    
    def is_phase_done(self, phase_name):
        """Check if a phase is already completed (for resume)."""
        return phase_name in self.master.get('completed_phases', [])
    
    def is_model_done(self, disease, model_name):
        """Check if specific model is already trained."""
        key = f"{disease}__{model_name}"
        return key in self.master.get('model_results', {})
    
    def log_time(self, phase_name, elapsed):
        """Log timing for a phase."""
        self.master['timing'][phase_name] = {
            'seconds': elapsed,
            'human': str(timedelta(seconds=int(elapsed)))
        }
        self.save_master()
import shutil
resume_file = "MASTER_RESULTS .json"
if os.path.exists(resume_file):
    os.makedirs('experiment_results', exist_ok=True)
    shutil.copy(resume_file, 'experiment_results/MASTER_RESULTS.json')
    print("✅ Resume file copied!")

# Initialize
RM = ResultManager('experiment_results')
print("✅ Result Manager initialized")