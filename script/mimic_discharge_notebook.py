#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version: 2.30
Author: Tim Frenzel

# Hospital Discharge classification
In this project, I'll build several ML and DL models to predict a patient's discharge disposition. By using lab results, medications, and comorbidities from the MIMIC IV dataset, I aim to guide clinicians toward safer, more efficient discharge decisions that support patient well-being and improve use of resources.

**Workflow**:
1. Perform **EDA** to understand key variables, missing data, and distributions.
2. Build and tune **traditional ML models** (Logistic Regression, XGBoost, LightGBM).
3. Implement a **Graph Neural Network** (RGCN) with a knowledge-graph approach, incorporating a 70/15/15 (train/val/test) node-level split, standard scaling, deeper RGCN layers, and early stopping.

**Goal**: 
Achieve robust classification performance and clinical insights for discharge planning.
"""

import os
import sys
import time
import logging
import warnings
import argparse
import re
import multiprocessing
import pickle
from datetime import datetime

# Data handling imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, f1_score, confusion_matrix,
                            classification_report, precision_score, recall_score,
                            average_precision_score, roc_curve)
from sklearn.impute import SimpleImputer

# XGBoost & LightGBM
from xgboost import XGBClassifier
import lightgbm as lgbm

# For oversampling
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# For explanation
import shap

# For GNN model
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import RGCNConv

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook")

# =============================================================================
# ARGUMENT PARSING & CONFIGURATION
# =============================================================================

parser = argparse.ArgumentParser(description='Hospital Discharge Classification')
parser.add_argument('--data-path', type=str, default='mimic_features.parquet',
                   help='Path to the MIMIC features parquet file')
parser.add_argument('--random-state', type=int, default=42,
                   help='Random seed for reproducibility')
parser.add_argument('--debug', action='store_true',
                   help='Run in debug mode with smaller dataset')
parser.add_argument('--no-eda', action='store_true',
                   help='Skip exploratory data analysis')
parser.add_argument('--no-ml', action='store_true',
                   help='Skip traditional ML model training')
parser.add_argument('--no-gnn', action='store_true',
                   help='Skip GNN model training')
parser.add_argument('--missing-threshold', type=float, default=0.7,
                   help='Threshold for removing columns with too many missing values (0.0-1.0)')
parser.add_argument('--use-gpu', action='store_true',
                   help='Use GPU acceleration for compatible models')
parser.add_argument('--use-smote', action='store_true',
                   help='Use SMOTE oversampling for class imbalance')
parser.add_argument('--corr-threshold', type=float, default=0.9,
                   help='Correlation threshold above which features are dropped')

args = parser.parse_args()

# Global configurations
DEBUG_MODE = args.debug
RANDOM_STATE = args.random_state
MISSING_THRESHOLD = args.missing_threshold
USE_GPU = args.use_gpu
USE_SMOTE = args.use_smote
CORR_THRESHOLD = args.corr_threshold
DATA_PATH = args.data_path

# CPU configuration - use 75% of available logical cores
N_JOBS = max(1, int(multiprocessing.cpu_count() * 0.75)) if not DEBUG_MODE else 2

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HospitalDischarge")

# Enable GPU if available and requested
if USE_GPU:
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("GPU not available, using CPU")
    except:
        device = torch.device('cpu')
        logger.info("Error checking GPU, defaulting to CPU")
else:
    device = torch.device('cpu')
    logger.info("Using CPU by user choice")

# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_and_prepare_data(file_path, missing_threshold=0.7):
    """
    Load and prepare dataset for analysis and modeling.
    - Creates 'disposition_binary' column (0=HOME, 1=NON-HOME) if not present
    - Optionally drops columns with too many missing values
    """
    logger.info(f"Loading dataset from: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    data = pd.read_parquet(file_path)
    original_shape = data.shape
    logger.info(f"Original data shape: {original_shape[0]:,} rows × {original_shape[1]:,} columns")

    # Create disposition binary label if needed
    if 'disposition_binary' not in data.columns:
        if 'discharge_disposition' in data.columns:
            data['disposition_binary'] = np.where(
                data["discharge_disposition"].str.contains("HOME", case=False, na=False),
                0, 1
            )
            logger.info("Created 'disposition_binary' from 'discharge_disposition'")
        else:
            logger.warning("No 'discharge_disposition' column; creating dummy disposition_binary")
            data['disposition_binary'] = 0
    
    # Log class distribution
    label_counts = data['disposition_binary'].value_counts(dropna=False)
    logger.info(f"disposition_binary distribution: {dict(label_counts)}")
    
    pos_count = (data['disposition_binary'] == 1).sum()
    neg_count = (data['disposition_binary'] == 0).sum()
    if pos_count > 0:
        imbalance_ratio = neg_count / pos_count
        logger.info(f"Class imbalance ratio (home:non-home) ~ {imbalance_ratio:.2f}:1")

    # Remove columns with excessive missing values
    if missing_threshold < 1.0:
        missing_percent = data.isnull().mean()
        cols_to_drop = missing_percent[missing_percent > missing_threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Removing {len(cols_to_drop)} columns with > {missing_threshold*100:.0f}% missing values")
            data = data.drop(columns=cols_to_drop)
    
    logger.info(f"Prepared data shape: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
    
    return data

# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

def perform_eda(data):
    """
    Perform exploratory data analysis on the dataset.
    Visualizes key distributions and relationships in the data.
    """
    logger.info("Starting exploratory data analysis...")
    
    # Overview of columns
    all_cols = data.columns.tolist()
    logger.info(f"Total columns: {len(all_cols)}")
    
    # Group columns by type
    demographic_cols = ["patient_id", "gender", "birth_date", "race", "ethnicity", "age_at_admission"]
    encounter_cols = [
        "encounter_id", "class", "status", "service_type", "admission_source",
        "discharge_disposition", "encounter_types", "length_of_stay",
        "admission_season", "start_time", "end_time"
    ]
    disposition_col = "disposition_binary"
    condition_cols = ["conditions", "condition_count"]
    med_summary_cols = ["med_count", "has_labs"]
    lab_cols = [c for c in data.columns if c.startswith("lab_") and any(x in c for x in ["_min", "_max", "_last"])]
    med_binary_cols = [c for c in data.columns if c.startswith("med_") and not c.startswith("med_count")]
    
    # Show column organization
    logger.info(f"Demographic columns: {[c for c in demographic_cols if c in data.columns]}")
    logger.info(f"Encounter columns: {[c for c in encounter_cols if c in data.columns]}")
    logger.info(f"Disposition column: {disposition_col}")
    logger.info(f"Condition columns: {[c for c in condition_cols if c in data.columns]}")
    logger.info(f"Medication summary: {[c for c in med_summary_cols if c in data.columns]}")
    logger.info(f"Lab columns count: {len([c for c in lab_cols if c in data.columns])}")
    logger.info(f"Medication binary columns count: {len([c for c in med_binary_cols if c in data.columns])}")

    # Missing data analysis
    missing_series = data.isnull().sum()
    missing_pct = (missing_series / len(data) * 100).round(2)
    logger.info("===== Missing Data Overview =====")
    logger.info(missing_pct.sort_values(ascending=False).head(15))

    # Basic statistics
    logger.info("===== Basic Statistic Distributions =====")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    desc = data[numeric_cols].describe().T
    logger.info(desc)

    # Visualizations
    # 1. Target distribution
    plt.figure(figsize=(5,5))
    val_counts = data[disposition_col].value_counts(dropna=False)
    labels = ["HOME","NON-HOME"]
    values = [val_counts.get(0,0), val_counts.get(1,0)]
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140,
            colors=["#66b3ff","#ff9999"], pctdistance=0.8)
    centre_circle = plt.Circle((0,0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title("Disposition (HOME vs. NON-HOME)")
    plt.show()

    # 2. Age distribution
    if "age_at_admission" in data.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data["age_at_admission"].dropna(),
                     bins=25, kde=True, color="#3498db")
        plt.title("Age at Admission Distribution")
        plt.xlabel("Age (years)")
        plt.ylabel("Encounter Count")
        plt.show()

    # 3. Length of stay distribution
    if "length_of_stay" in data.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data["length_of_stay"].dropna(), 
                     bins=30, color="darkorange", kde=True)
        plt.title("Length of Stay Distribution")
        plt.xlabel("LOS (days)")
        plt.ylabel("Frequency")
        plt.show()

    # 4. Condition count distribution
    if "condition_count" in data.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data["condition_count"].dropna(),
                     bins=20, kde=False, color="purple", edgecolor='k')
        plt.title("Condition Count per Encounter")
        plt.xlabel("Number of Conditions")
        plt.ylabel("Encounter Count")
        plt.show()

    # 5. Medications per encounter
    if "med_count" in data.columns:
        plt.figure(figsize=(7,5))
        sns.histplot(data["med_count"].dropna(),
                     bins=20, kde=False, color="#8e44ad", edgecolor='k')
        plt.title("Distinct Medications per Encounter")
        plt.xlabel("Number of Distinct Medications")
        plt.ylabel("Encounter Count")
        plt.show()

    # 6. Age vs disposition
    if "age_at_admission" in data.columns:
        plt.figure(figsize=(6,5))
        disp_df = data.copy()
        disp_df["disp_label"] = disp_df[disposition_col].map({0:"HOME",1:"NON-HOME"})
        sns.boxplot(data=disp_df, x="disp_label", y="age_at_admission", palette="pastel")
        plt.title("Age at Admission by Disposition")
        plt.xlabel("Disposition")
        plt.ylabel("Age (Years)")
        plt.show()

    # 7. Length of stay vs disposition
    if "length_of_stay" in data.columns:
        plt.figure(figsize=(6,5))
        sns.boxplot(data=disp_df, x="disp_label", y="length_of_stay", palette="Set1")
        plt.title("Length of Stay by Disposition")
        plt.xlabel("Disposition")
        plt.ylabel("LOS (Days)")
        plt.show()

    # 8. Condition count vs disposition 
    if "condition_count" in data.columns:
        group_condition = data.groupby(disposition_col)['condition_count'].mean()
        logger.info("\nMean condition_count by disposition_binary:")
        logger.info(group_condition)

        plt.figure(figsize=(5,4))
        mean_vals = group_condition.values
        x_labels = ["HOME","NON-HOME"]
        sns.barplot(x=x_labels, y=mean_vals, palette=["#2ecc71","#e74c3c"])
        plt.title("Avg # of Conditions by Disposition")
        plt.xlabel("Disposition")
        plt.ylabel("Mean Condition Count")
        plt.show()

    # 9. Top medications usage
    if med_binary_cols:
        usage_rates = {}
        for col in med_binary_cols:
            usage_rates[col] = data[col].mean() * 100
        usage_series = pd.Series(usage_rates).sort_values(ascending=False)
        top_10_meds = usage_series.head(10)

        plt.figure(figsize=(7,5))
        sns.barplot(x=top_10_meds.values,
                    y=top_10_meds.index,
                    palette="ch:rot=-.5,s=.75")
        plt.title("Top 10 Medication Columns (by usage %)")
        plt.xlabel("Usage Rate (%)")
        plt.ylabel("Medication Code Column")
        plt.show()

    logger.info("Exploratory data analysis completed")

# =============================================================================
# ML MODEL DEVELOPMENT
# =============================================================================

def patient_level_train_test_split(data, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets at the patient level.
    Ensures patients don't appear in both training and testing sets.
    """
    if 'patient_id' not in data.columns:
        logger.warning("patient_id column not found; using random encounter-level split.")
        return train_test_split(data, test_size=test_size, random_state=random_state)
    
    patient_ids = data['patient_id'].unique()
    logger.info(f"Total unique patients: {len(patient_ids)}")
    
    train_patients, test_patients = train_test_split(
        patient_ids, test_size=test_size, random_state=random_state
    )
    
    train_data = data[data['patient_id'].isin(train_patients)]
    test_data = data[data['patient_id'].isin(test_patients)]
    
    logger.info(f"Train Encounters: {len(train_data)}, Test Encounters: {len(test_data)}")
    return train_data, test_data

def remove_correlated_features(X, threshold=0.9):
    """
    Remove one of each pair of features whose correlation > threshold.
    """
    if X.shape[1] < 2:
        return X
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return X
    
    corr_matrix = X[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} correlated features (r>{threshold})")
        X = X.drop(columns=to_drop, errors='ignore')
    return X

def prepare_features(data, outcome_col='disposition_binary', debug=False,
                     max_categories=10, corr_threshold=0.9):
    """
    Prepare features for ML model training:
    - Exclude potential data leaks
    - Remove highly correlated features
    - Handle categorical features with too many categories
    """
    data = data.copy()
    
    if outcome_col not in data.columns:
        logger.error(f"Outcome column '{outcome_col}' not found. Creating dummy label=0.")
        data[outcome_col] = 0

    # Columns to exclude (potential data leaks or metadata)
    exclude_cols = [
        'readmission_30day','next_encounter_id','next_start_time','days_to_next_encounter',
        'readmission_90day','is_index_30day','is_readmission_30day','prev_encounter_id',
        'prev_end_time','encounter_seq','discharge_disposition','patient_id','encounter_id',
        'start_time','end_time','birth_date'
    ]
    
    # Also exclude the outcome column
    if outcome_col in data.columns:
        exclude_cols.append(outcome_col)
    
    # Keep only columns that exist in the data
    exclude_cols = [c for c in exclude_cols if c in data.columns]
    
    # Debug mode => keep fewer columns
    if debug:
        # keep just some numeric or categorical columns as example
        keep_cols = ['gender','age_at_admission','condition_count','med_count','length_of_stay']
        available = [c for c in keep_cols if c in data.columns and c not in exclude_cols]
        # everything else is excluded in debug
        exclude_cols.extend([c for c in data.columns if c not in available and c not in exclude_cols])
    
    # Extract features and target
    X = data.drop(columns=exclude_cols, errors='ignore')
    y = data[outcome_col]
    
    # Remove highly correlated features
    X = remove_correlated_features(X, threshold=corr_threshold)
    
    # Handle categorical columns with too many categories
    cat_cols = X.select_dtypes(include=['object','category']).columns
    for c in cat_cols:
        if X[c].nunique() > max_categories:
            logger.info(f"Dropping column '{c}', categories={X[c].nunique()} > max_categories={max_categories}")
            X.drop(columns=[c], inplace=True)

    logger.info(f"Final feature set: {X.shape[1]} features")
    return X, y

def build_pipeline(model_type, use_gpu=False, random_state=42, use_smote=False):
    """
    Build a scikit-learn pipeline with preprocessing and the specified model.
    Optional SMOTE oversampling for class imbalance.
    """
    # Define preprocessors for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])
    
    # Combined preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, make_column_selector(dtype_include=['object','category'])),
            ('num', numeric_transformer, make_column_selector(dtype_include=np.number))
        ],
        remainder='drop',
        sparse_threshold=0.3
    )
    
    # Configure weighting approach based on class imbalance
    if use_smote:
        # with SMOTE, no separate weighting
        xgb_scale_pos_weight = 1.0
        lgbm_class_weight = None
        lr_class_weight = None
    else:
        # approximate ratios
        xgb_scale_pos_weight = 3.0
        lgbm_class_weight = 'balanced'
        lr_class_weight = 'balanced'
    
    # Build the model
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            max_iter=1000,
            solver='saga',
            random_state=random_state,
            class_weight=lr_class_weight
        )
    elif model_type == 'xgboost':
        tree_method = 'gpu_hist' if use_gpu else 'hist'
        model = XGBClassifier(
            objective='binary:logistic',
            random_state=random_state,
            tree_method=tree_method,
            scale_pos_weight=xgb_scale_pos_weight,
            n_jobs=N_JOBS,
            learning_rate=0.03,
            n_estimators=500,
            max_depth=6,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    elif model_type == 'lightgbm':
        device = 'gpu' if use_gpu else 'cpu'
        model = lgbm.LGBMClassifier(
            objective='binary',
            random_state=random_state,
            device=device,
            class_weight=lgbm_class_weight,
            n_jobs=N_JOBS,
            boosting_type='dart',
            drop_rate=0.1,
            learning_rate=0.03,
            n_estimators=500,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline with or without SMOTE
    if use_smote:
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
    
    return pipeline

def get_hyperparameters(model_type, debug=False):
    """
    Get hyperparameter search space for model tuning.
    Simplified space in debug mode.
    """
    if debug:
        if model_type == 'logistic_regression':
            return {'classifier__C': [0.1, 1.0]}
        elif model_type == 'xgboost':
            return {'classifier__n_estimators': [50],
                    'classifier__max_depth': [3],
                    'classifier__learning_rate': [0.1]}
        elif model_type == 'lightgbm':
            return {'classifier__n_estimators': [50],
                    'classifier__learning_rate': [0.1]}
    else:
        if model_type == 'logistic_regression':
            return {
                'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
                'classifier__penalty': ['l1','l2']
            }
        elif model_type == 'xgboost':
            return {
                'classifier__n_estimators': [200, 300, 500],
                'classifier__max_depth': [4, 6, 8],
                'classifier__learning_rate': [0.01, 0.03, 0.05],
                'classifier__subsample': [0.7,0.8,0.9],
                'classifier__colsample_bytree': [0.7,0.8,0.9],
                'classifier__min_child_weight': [1,2,3],
                'classifier__gamma': [0, 0.1, 0.2],
                'classifier__reg_alpha': [0,0.1,1.0],
                'classifier__reg_lambda': [0.1,1.0,10.0]
            }
        elif model_type == 'lightgbm':
            return {
                'classifier__n_estimators': [200, 300, 500],
                'classifier__learning_rate': [0.01, 0.03, 0.05],
                'classifier__feature_fraction': [0.7,0.8,0.9],
                'classifier__bagging_fraction': [0.7,0.8,0.9],
                'classifier__drop_rate': [0.1,0.2,0.3],
                'classifier__reg_alpha': [0,0.1,1.0],
                'classifier__reg_lambda': [0.1,1.0,10.0]
            }
    return {}

def hyperparameter_tuning(pipeline, X_train, y_train, param_grid, model_name, 
                          patient_ids, debug=False, random_state=42):
    """
    Perform hyperparameter tuning with RandomizedSearchCV.
    Use GroupKFold to prevent data leakage at patient level.
    """
    cv_folds = 3 if not debug else 2
    n_iter = 10 if not debug else 3
    cv = GroupKFold(n_splits=cv_folds)
    groups = patient_ids if patient_ids is not None else np.arange(len(X_train))
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        n_jobs=N_JOBS if not debug else 1,
        verbose=1,
        random_state=random_state
    )
    search.fit(X_train, y_train, groups=groups)
    
    logger.info(f"\nBest parameters for {model_name}:")
    for param, value in search.best_params_.items():
        logger.info(f"  {param}: {value}")
    logger.info(f"Best CV score for {model_name}: {search.best_score_:.4f}")
    
    return search

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model on test data.
    Calculates and visualizes various metrics.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    avg_prec = average_precision_score(y_test, y_pred_proba)
    
    logger.info(f"\n{model_name} performance (discharge class):")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 score: {f1:.4f}")
    logger.info(f"  Average Precision: {avg_prec:.4f}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"  Confusion Matrix:\n{conf_matrix}")
    
    cls_report = classification_report(y_test, y_pred)
    logger.info(f"  Classification Report:\n{cls_report}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name} (Discharge Classification)')
    plt.legend()
    plt.show()
    
    # Store metrics
    metrics = {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_prec': avg_prec,
        'confusion_matrix': conf_matrix,
        'classification_report': cls_report,
        'fpr': fpr,
        'tpr': tpr
    }
    return metrics

def bootstrap_evaluate(model, X_test, y_test, n_iterations=30, random_state=42):
    """
    Bootstrap evaluation for confidence intervals of metrics.
    """
    rng = np.random.RandomState(random_state)
    aucs, f1_scores, avg_precs = [], [], []
    
    if isinstance(X_test, pd.DataFrame):
        X_test_values = X_test.values
    else:
        X_test_values = X_test
    if isinstance(y_test, pd.Series):
        y_test_values = y_test.values
    else:
        y_test_values = y_test
    
    for _ in range(n_iterations):
        indices = rng.randint(0, len(X_test_values), size=len(X_test_values))
        if isinstance(X_test, pd.DataFrame):
            X_boot = X_test.iloc[indices]
        else:
            X_boot = X_test_values[indices]
        if isinstance(y_test, pd.Series):
            y_boot = y_test.iloc[indices]
        else:
            y_boot = y_test_values[indices]
        
        y_pred_proba = model.predict_proba(X_boot)[:, 1]
        y_pred = model.predict(X_boot)
        
        aucs.append(roc_auc_score(y_boot, y_pred_proba))
        f1_scores.append(f1_score(y_boot, y_pred))
        avg_precs.append(average_precision_score(y_boot, y_pred_proba))
    
    results = {
        'auc': {
            'mean': np.mean(aucs),
            'ci_lower': np.percentile(aucs, 2.5),
            'ci_upper': np.percentile(aucs, 97.5)
        },
        'f1': {
            'mean': np.mean(f1_scores),
            'ci_lower': np.percentile(f1_scores, 2.5),
            'ci_upper': np.percentile(f1_scores, 97.5)
        },
        'avg_precision': {
            'mean': np.mean(avg_precs),
            'ci_lower': np.percentile(avg_precs, 2.5),
            'ci_upper': np.percentile(avg_precs, 97.5)
        }
    }
    return results

def generate_shap_plots(pipeline, X, model_name):
    """
    Generate and display SHAP explanation plots.
    Shows feature importance and value impact.
    """
    logger.info(f"Generating SHAP plots for {model_name} (discharge classification)...")
    
    preprocessor = None
    classifier = None
    
    step_names = [s[0] for s in pipeline.steps]
    if 'preprocessor' in step_names:
        preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    X_processed = preprocessor.transform(X)
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    
    model_type = type(classifier).__name__.lower()
    
    try:
        if 'xgb' in model_type or 'xgboost' in model_type:
            explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
        elif 'lgbm' in model_type or 'lightgbm' in model_type:
            explainer = shap.TreeExplainer(classifier, feature_perturbation="tree_path_dependent")
        elif 'logisticregression' in model_type:
            explainer = shap.LinearExplainer(classifier, X_processed, feature_dependence="independent")
        else:
            logger.warning(f"SHAP: fallback to KernelExplainer for {model_type}")
            explainer = shap.KernelExplainer(classifier.predict_proba, X_processed[:100])
    except Exception as e:
        logger.warning(f"Error setting up SHAP explainer for {model_name}: {e}")
        return
    
    try:
        # Get smaller sample if dataset is large
        X_sample = X_processed[:500] if X_processed.shape[0] > 500 else X_processed
        shap_values = explainer(X_sample)
        
        if isinstance(shap_values.values, list) and len(shap_values.values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
    except Exception as e:
        logger.warning(f"Error computing SHAP values for {model_name}: {e}")
        return
    
    feature_names = []
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary Plot - {model_name} (Discharge)")
    plt.tight_layout()
    plt.show()
    
    # Bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {model_name} (Discharge)")
    plt.tight_layout()
    plt.show()
    
    logger.info(f"SHAP plots displayed for {model_name}.")

def train_ml_models(data, debug=False):
    """
    Train and evaluate traditional ML models.
    """
    logger.info("Starting traditional ML model training...")
    
    # Split the data (patient-level)
    train_data, test_data = patient_level_train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)
    
    # Prepare features
    X_train, y_train = prepare_features(train_data, 
                                      outcome_col='disposition_binary',
                                      debug=debug,
                                      max_categories=10,
                                      corr_threshold=CORR_THRESHOLD)
    X_test, y_test = prepare_features(test_data, 
                                    outcome_col='disposition_binary',
                                    debug=debug,
                                    max_categories=10,
                                    corr_threshold=CORR_THRESHOLD)
    
    logger.info(f"Training features: {X_train.shape[1]}")
    
    # Extract patient groups for cross-validation, if available
    train_patient_ids = None
    if 'patient_id' in train_data.columns:
        train_patient_ids = train_data['patient_id'].values
    
    # Build pipelines
    pipeline_lr = build_pipeline('logistic_regression', use_gpu=False,
                               random_state=RANDOM_STATE, use_smote=USE_SMOTE)
    pipeline_xgb = build_pipeline('xgboost', use_gpu=USE_GPU,
                                random_state=RANDOM_STATE, use_smote=USE_SMOTE)
    pipeline_lgbm = build_pipeline('lightgbm', use_gpu=USE_GPU,
                                 random_state=RANDOM_STATE, use_smote=USE_SMOTE)
    
    # Hyperparameter tuning
    logger.info("Tuning Logistic Regression (discharge class)...")
    lr_params = get_hyperparameters('logistic_regression', debug=debug)
    search_lr = hyperparameter_tuning(pipeline_lr, X_train, y_train, lr_params,
                                    'Logistic Regression', train_patient_ids,
                                    debug, RANDOM_STATE)
    best_lr = search_lr.best_estimator_
    
    logger.info("Tuning XGBoost (discharge class)...")
    xgb_params = get_hyperparameters('xgboost', debug=debug)
    search_xgb = hyperparameter_tuning(pipeline_xgb, X_train, y_train, xgb_params,
                                     'XGBoost', train_patient_ids,
                                     debug, RANDOM_STATE)
    best_xgb = search_xgb.best_estimator_
    
    logger.info("Tuning LightGBM (discharge class)...")
    lgbm_params = get_hyperparameters('lightgbm', debug=debug)
    search_lgbm = hyperparameter_tuning(pipeline_lgbm, X_train, y_train, lgbm_params,
                                      'LightGBM', train_patient_ids,
                                      debug, RANDOM_STATE)
    best_lgbm = search_lgbm.best_estimator_
    
    # Evaluate on test set
    logger.info("Evaluating final models on the test set (discharge classification)...")
    lr_metrics = evaluate_model(best_lr, X_test, y_test, "Logistic Regression")
    xgb_metrics = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
    lgbm_metrics = evaluate_model(best_lgbm, X_test, y_test, "LightGBM")
    
    # Bootstrap evaluation
    logger.info("Performing bootstrap evaluation for each model (disposition).")
    lr_bootstrap = bootstrap_evaluate(best_lr, X_test, y_test, n_iterations=30, random_state=RANDOM_STATE)
    xgb_bootstrap = bootstrap_evaluate(best_xgb, X_test, y_test, n_iterations=30, random_state=RANDOM_STATE)
    lgbm_bootstrap = bootstrap_evaluate(best_lgbm, X_test, y_test, n_iterations=30, random_state=RANDOM_STATE)
    
    # Compare ROC curves
    plt.figure(figsize=(10,8))
    plt.plot(lr_metrics['fpr'], lr_metrics['tpr'], label=f"LR (AUC={lr_metrics['roc_auc']:.3f})")
    plt.plot(xgb_metrics['fpr'], xgb_metrics['tpr'], label=f"XGB (AUC={xgb_metrics['roc_auc']:.3f})")
    plt.plot(lgbm_metrics['fpr'], lgbm_metrics['tpr'], label=f"LGBM (AUC={lgbm_metrics['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison (Discharge Class)')
    plt.legend()
    plt.show()
    
    # SHAP Analysis
    logger.info("Generating SHAP explanations for feature importance...")
    generate_shap_plots(best_lr, X_test, "Logistic Regression")
    generate_shap_plots(best_xgb, X_test, "XGBoost")
    generate_shap_plots(best_lgbm, X_test, "LightGBM")
    
    # Print summary
    logger.info("\n===== ML Model Training Summary =====")
    
    # Compare models
    best_model_data = max([
        ('LR', lr_metrics['roc_auc']),
        ('XGB', xgb_metrics['roc_auc']),
        ('LGBM', lgbm_metrics['roc_auc'])
    ], key=lambda x: x[1])
    logger.info(f"Best performing model: {best_model_data[0]} (AUC: {best_model_data[1]:.4f})")
    
    # LR results
    logger.info("Logistic Regression:")
    logger.info(f"  AUC: {lr_metrics['roc_auc']:.4f} (95% CI: {lr_bootstrap['auc']['ci_lower']:.4f}-{lr_bootstrap['auc']['ci_upper']:.4f})")
    logger.info(f"  F1 Score: {lr_metrics['f1']:.4f} (95% CI: {lr_bootstrap['f1']['ci_lower']:.4f}-{lr_bootstrap['f1']['ci_upper']:.4f})")
    logger.info(f"  Precision: {lr_metrics['precision']:.4f}")
    logger.info(f"  Recall: {lr_metrics['recall']:.4f}")
    
    # XGB results
    logger.info("XGBoost:")
    logger.info(f"  AUC: {xgb_metrics['roc_auc']:.4f} (95% CI: {xgb_bootstrap['auc']['ci_lower']:.4f}-{xgb_bootstrap['auc']['ci_upper']:.4f})")
    logger.info(f"  F1 Score: {xgb_metrics['f1']:.4f} (95% CI: {xgb_bootstrap['f1']['ci_lower']:.4f}-{xgb_bootstrap['f1']['ci_upper']:.4f})")
    logger.info(f"  Precision: {xgb_metrics['precision']:.4f}")
    logger.info(f"  Recall: {xgb_metrics['recall']:.4f}")
    
    # LGBM results
    logger.info("LightGBM:")
    logger.info(f"  AUC: {lgbm_metrics['roc_auc']:.4f} (95% CI: {lgbm_bootstrap['auc']['ci_lower']:.4f}-{lgbm_bootstrap['auc']['ci_upper']:.4f})")
    logger.info(f"  F1 Score: {lgbm_metrics['f1']:.4f} (95% CI: {lgbm_bootstrap['f1']['ci_lower']:.4f}-{lgbm_bootstrap['f1']['ci_upper']:.4f})")
    logger.info(f"  Precision: {lgbm_metrics['precision']:.4f}")
    logger.info(f"  Recall: {lgbm_metrics['recall']:.4f}")
    
    logger.info("Traditional ML model training completed")
    
    # Return the best models for potential further use
    return {
        'logistic_regression': best_lr,
        'xgboost': best_xgb,
        'lightgbm': best_lgbm
    }

# =============================================================================
# GRAPH NEURAL NETWORK (GNN) MODEL
# =============================================================================

class HomogeneousData:
    """ Container for homogeneous graph data with multi-relational edges. """
    def __init__(self):
        self.x = None          # (num_nodes, feature_dim)
        self.y = None          # (num_nodes,) label
        self.edge_index = None # (2, num_edges)
        self.edge_type = None  # (num_edges,) integer for each relation
        self.node_type = None  # (num_nodes,) integer for node type
        self.num_nodes = 0
        self.num_edges = 0

        # 70/15/15 => train_mask, val_mask, test_mask
        self.train_mask = None
        self.val_mask   = None
        self.test_mask  = None

class DeeperRGCN(nn.Module):
    """
    RGCN with three relational GCN layers + dropout.
    Larger hidden dimension (128) for more model capacity.
    """
    def __init__(self, num_relations, in_channels, hidden_dim=128, dropout=0.4):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim,  hidden_dim, num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_dim,  hidden_dim, num_relations=num_relations)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # binary classification

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_type)
        x = torch.relu(x)
        x = self.dropout(x)

        out = self.fc(x)
        return out.view(-1)

def manual_to_homogeneous(hetero_data):
    """
    Convert HeteroData with node types [Encounter, Condition, Medication]
    into a HomogeneousData object for RGCN.
    Edges: 
      0 => Encounter->Condition
      1 => Encounter->Medication
      2 => Encounter->Encounter (follows)
    Node Types:
      0 => Encounter, 1 => Condition, 2 => Medication
    """
    num_enc = hetero_data['Encounter'].x.size(0)
    num_con = hetero_data['Condition'].x.size(0)
    num_med = hetero_data['Medication'].x.size(0)
    logger.info(f"Encounter nodes: {num_enc}, Condition nodes: {num_con}, Medication nodes: {num_med}")

    offset_enc = 0
    offset_con = num_enc
    offset_med = num_enc + num_con
    total_nodes= num_enc + num_con + num_med

    # Node features
    feat_dim = hetero_data['Encounter'].x.size(1) if num_enc>0 else 4
    x_enc = hetero_data['Encounter'].x if num_enc>0 else torch.zeros((0, feat_dim))
    x_con = torch.zeros((num_con, feat_dim), dtype=torch.float)
    x_med = torch.zeros((num_med, feat_dim), dtype=torch.float)
    x_all = torch.cat([x_enc, x_con, x_med], dim=0)

    # Node labels
    y_enc = hetero_data['Encounter'].y.clone() if num_enc>0 else torch.full((0,), -1, dtype=torch.long)
    y_con = torch.full((num_con,), -1, dtype=torch.long)
    y_med = torch.full((num_med,), -1, dtype=torch.long)
    y_all = torch.cat([y_enc, y_con, y_med], dim=0)

    # Node types
    ntype_enc = torch.full((num_enc,), 0, dtype=torch.long)
    ntype_con = torch.full((num_con,), 1, dtype=torch.long)
    ntype_med = torch.full((num_med,), 2, dtype=torch.long)
    node_type_all = torch.cat([ntype_enc, ntype_con, ntype_med], dim=0)

    # Edges & edge types
    def get_edges(hd, etuple):
        if etuple in hd.edge_types:
            return hd[etuple].edge_index
        else:
            return torch.empty((2,0), dtype=torch.long)

    e_enc_con = get_edges(hetero_data, ('Encounter','has_condition','Condition'))
    e_enc_med = get_edges(hetero_data, ('Encounter','has_medication','Medication'))
    e_enc_enc = get_edges(hetero_data, ('Encounter','follows','Encounter'))

    # E->C => type=0
    e1_src_global = e_enc_con[0] + offset_enc
    e1_dst_global = e_enc_con[1] + offset_con
    e1_type       = torch.full((e_enc_con.size(1),), 0, dtype=torch.long)

    # E->M => type=1
    e2_src_global = e_enc_med[0] + offset_enc
    e2_dst_global = e_enc_med[1] + offset_med
    e2_type       = torch.full((e_enc_med.size(1),), 1, dtype=torch.long)

    # E->E => type=2
    e3_src_global = e_enc_enc[0] + offset_enc
    e3_dst_global = e_enc_enc[1] + offset_enc
    e3_type       = torch.full((e_enc_enc.size(1),), 2, dtype=torch.long)

    # Combine
    edge_src_all = torch.cat([e1_src_global, e2_src_global, e3_src_global], dim=0)
    edge_dst_all = torch.cat([e1_dst_global, e2_dst_global, e3_dst_global], dim=0)
    edge_type_all= torch.cat([e1_type, e2_type, e3_type], dim=0)
    edge_index   = torch.stack([edge_src_all, edge_dst_all], dim=0)

    data = HomogeneousData()
    data.x = x_all
    data.y = y_all
    data.node_type = node_type_all
    data.edge_index = edge_index
    data.edge_type  = edge_type_all
    data.num_nodes  = total_nodes
    data.num_edges  = edge_index.size(1)
    return data

def split_encounter_nodes(data, split_ratio=(0.7,0.15,0.15), random_state=42):
    """
    Splits the Encounter nodes into train, val, test.
    ratio = (train%, val%, test%).
    """
    is_enc = (data.node_type == 0)
    enc_indices = is_enc.nonzero(as_tuple=True)[0].cpu().numpy()

    rng = np.random.RandomState(random_state)
    rng.shuffle(enc_indices)

    n = len(enc_indices)
    n_train = int(n * split_ratio[0])
    n_val   = int(n * split_ratio[1])
    
    # The remainder goes to test
    train_idx = enc_indices[:n_train]
    val_idx   = enc_indices[n_train : n_train+n_val]
    test_idx  = enc_indices[n_train+n_val :]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    logger.info(f"Encounter nodes => {n} total.")
    logger.info(f"Train: {train_mask.sum().item()}, Val: {val_mask.sum().item()}, Test: {test_mask.sum().item()}")
    return data

def evaluate_model_gnn(data, model, mask):
    """
    Return raw predictions, labels, etc. for a given mask (train/val/test).
    """
    with torch.no_grad():
        model.eval()
        out  = model(data.x, data.edge_index, data.edge_type)
        probs= torch.sigmoid(out)
    sel_probs  = probs[mask]
    sel_labels = data.y[mask]
    return sel_probs.cpu().numpy(), sel_labels.cpu().numpy()

def find_best_threshold(probs, labels):
    """
    Find threshold in [0,1] that maximizes F1 on (val) data.
    Tests 101 thresholds from 0.0 to 1.0 in increments of 0.01.
    """
    best_thresh = 0.5
    best_f1     = 0.0
    
    # Search for optimal threshold
    for t in np.linspace(0, 1, 101):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds) if (labels==1).sum() > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

def train_gnn_model(data, model, epochs=40, lr=0.01, patience=5):
    """
    Full training loop with early stopping:
      - uses train_mask for gradient updates
      - uses val_mask for threshold/f1 check
      - stops if val AUC doesn't improve for `patience` consecutive epochs
    """
    # Set up imbalance weighting
    train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
    y_train = data.y[train_nodes].float()
    num_pos = (y_train == 1).sum().item()
    num_neg = (y_train == 0).sum().item()

    if num_pos < 1:
        logger.warning("No positive samples in training => purely negative classification.")
        pos_weight = 1.0
    else:
        pos_weight = (num_neg / num_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_auc = 0.0
    best_state   = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training step
        model.train()
        optimizer.zero_grad()

        out = model(data.x, data.edge_index, data.edge_type)
        out_train = out[data.train_mask]
        labels_train = data.y[data.train_mask].float()

        loss = criterion(out_train, labels_train)
        loss.backward()
        optimizer.step()

        # Evaluate on validation set
        val_probs, val_labels = evaluate_model_gnn(data, model, data.val_mask)
        if (val_labels==1).sum() < 1:
            val_auc = 0.5
            val_f1  = 0.0
        else:
            val_auc = roc_auc_score(val_labels, val_probs)
            # default threshold=0.5 for logging
            val_preds = (val_probs >= 0.5).astype(int)
            val_f1 = f1_score(val_labels, val_preds)

        # Check if improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch}/{epochs} | Train Loss={loss.item():.4f} | Val AUC={val_auc:.4f} F1={val_f1:.4f}")

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"No improvement in val AUC for {patience} epochs. Stopping early.")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model

def visualize_knowledge_graph(G, max_nodes=300):
    """
    Visualize a knowledge graph, sampling nodes if too large.
    """
    # Optionally do a subgraph for visualization if too large
    if G.number_of_nodes() > max_nodes:
        # sample some from each type
        enc_list = [n for n,d in G.nodes(data=True) if d.get('ntype') == 'Encounter']
        con_list = [n for n,d in G.nodes(data=True) if d.get('ntype') == 'Condition']
        med_list = [n for n,d in G.nodes(data=True) if d.get('ntype') == 'Medication']
        
        np.random.shuffle(enc_list)
        np.random.shuffle(con_list)
        np.random.shuffle(med_list)
        
        sub_enc = enc_list[: max_nodes//3]
        sub_con = con_list[: max_nodes//3]
        sub_med = med_list[: max_nodes//3]
        sub_nodes = sub_enc + sub_con + sub_med

        leftover = max_nodes - len(sub_nodes)
        if leftover > 0:
            other = list(set(G.nodes()) - set(sub_nodes))
            np.random.shuffle(other)
            sub_nodes.extend(other[:leftover])

        subG = G.subgraph(sub_nodes)
    else:
        subG = G

    # Set node colors based on type
    color_map = []
    for node in subG.nodes(data=True):
        if node[1].get('ntype') == 'Encounter':
            color_map.append('blue')
        elif node[1].get('ntype') == 'Condition':
            color_map.append('red')
        elif node[1].get('ntype') == 'Medication':
            color_map.append('green')
        else:
            color_map.append('gray')

    # Visualize
    pos = nx.spring_layout(subG, seed=42, k=0.5)
    plt.figure(figsize=(10,8))
    nx.draw_networkx(subG, pos=pos,
                     node_size=80, node_color=color_map,
                     with_labels=False, edge_color='gray',
                     arrows=True, arrowstyle='->')
    plt.title("Knowledge Graph (E=blue, C=red, M=green)")
    plt.show()

from torch_geometric.data import HeteroData

def build_and_train_gnn(data, debug=False):
    """
    Build a heterogeneous graph from the data and train a GNN model.
    """
    logger.info("Starting GNN model building and training...")
    
    # Possibly downsample if debug
    if debug:
        data = data.sample(n=min(3000, len(data)), random_state=RANDOM_STATE)
        logger.info(f"Debug mode => {len(data)} rows for demonstration.")

    # Check for condition and medication codes
    if 'icd_code' not in data.columns:
        logger.warning("icd_code not found => creating random codes.")
        data['icd_code'] = np.random.choice(['I10','E11','I25','5859','K21'], size=len(data))
    if 'med_code' not in data.columns:
        logger.warning("med_code not found => creating random codes.")
        data['med_code'] = np.random.choice(['Metformin','Lisinopril','Aspirin','Insulin'], size=len(data))

    # Build heterogeneous graph data
    hetero_data = HeteroData()

    # Numeric features + standard scaling
    base_features = [
        'age_at_admission','length_of_stay','condition_count','med_count'
    ]
    # Filter only existing columns
    final_feats = [c for c in base_features if c in data.columns]
    if not final_feats:
        logger.warning("No numeric features found, using placeholder features.")
        data['placeholder_feature'] = np.random.normal(size=len(data))
        final_feats = ['placeholder_feature']

    # Fill missing values and scale features
    data[final_feats] = data[final_feats].fillna(0)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(data[final_feats])

    x_enc = torch.tensor(scaled_values, dtype=torch.float).to(device)
    y_enc = torch.tensor(data['disposition_binary'].values, dtype=torch.long).to(device)

    hetero_data['Encounter'].x = x_enc
    hetero_data['Encounter'].y = y_enc

    # Create node indices
    data = data.reset_index(drop=True)
    data['enc_idx'] = data.index

    # Condition nodes
    conds = data['icd_code'].dropna().unique().tolist()
    cond2idx = {c:i for i,c in enumerate(conds)}
    x_cond = torch.zeros((len(conds), x_enc.size(1)), dtype=torch.float).to(device)
    hetero_data['Condition'].x = x_cond

    # Medication nodes
    meds = data['med_code'].dropna().unique().tolist()
    med2idx = {m:i for i,m in enumerate(meds)}
    x_med = torch.zeros((len(meds), x_enc.size(1)), dtype=torch.float).to(device)
    hetero_data['Medication'].x = x_med

    # E->C edges
    e1_src, e1_dst = [], []
    for i, row in data.iterrows():
        if pd.isna(row['icd_code']):
            continue
        enc = row['enc_idx']
        cid = cond2idx[row['icd_code']]
        e1_src.append(enc)
        e1_dst.append(cid)
    e1_src = torch.tensor(e1_src, dtype=torch.long).to(device)
    e1_dst = torch.tensor(e1_dst, dtype=torch.long).to(device)
    hetero_data['Encounter','has_condition','Condition'].edge_index = torch.stack([e1_src, e1_dst])

    # E->M edges
    e2_src, e2_dst = [], []
    for i, row in data.iterrows():
        if pd.isna(row['med_code']):
            continue
        enc = row['enc_idx']
        mid = med2idx[row['med_code']]
        e2_src.append(enc)
        e2_dst.append(mid)
    e2_src = torch.tensor(e2_src, dtype=torch.long).to(device)
    e2_dst = torch.tensor(e2_dst, dtype=torch.long).to(device)
    hetero_data['Encounter','has_medication','Medication'].edge_index = torch.stack([e2_src, e2_dst])

    # E->E (follows) edges
    if 'patient_id' not in data.columns:
        data['patient_id'] = np.random.randint(0, 300 if not debug else 50, size=len(data))

    e3_list = []
    for pid, group in data.groupby('patient_id'):
        sub = group.sort_values('enc_idx')
        idx_list = sub['enc_idx'].tolist()
        for i in range(len(idx_list)-1):
            s, t = idx_list[i], idx_list[i+1]
            e3_list.append((s, t))
            e3_list.append((t, s))  # bidirectional
    
    if e3_list:
        e3_src = torch.tensor([s for s,t in e3_list], dtype=torch.long).to(device)
        e3_dst = torch.tensor([t for s,t in e3_list], dtype=torch.long).to(device)
        hetero_data['Encounter','follows','Encounter'].edge_index = torch.stack([e3_src, e3_dst])

    # Visualize knowledge graph
    logger.info("Creating knowledge graph visualization...")
    G = nx.MultiDiGraph()
    
    # Add nodes
    n_enc = hetero_data['Encounter'].x.size(0)
    for i in range(n_enc):
        node_id = f"E_{i}"
        G.add_node(node_id, ntype='Encounter')

    n_con = hetero_data['Condition'].x.size(0)
    for i in range(n_con):
        node_id = f"C_{i}"
        G.add_node(node_id, ntype='Condition')

    n_med = hetero_data['Medication'].x.size(0)
    for i in range(n_med):
        node_id = f"M_{i}"
        G.add_node(node_id, ntype='Medication')

    # Add edges
    if ('Encounter','has_condition','Condition') in hetero_data.edge_types:
        e1 = hetero_data['Encounter','has_condition','Condition'].edge_index
        for i in range(e1.size(1)):
            s_enc = f"E_{e1[0,i].item()}"
            t_con = f"C_{e1[1,i].item()}"
            G.add_edge(s_enc, t_con, relation='has_condition')

    if ('Encounter','has_medication','Medication') in hetero_data.edge_types:
        e2 = hetero_data['Encounter','has_medication','Medication'].edge_index
        for i in range(e2.size(1)):
            s_enc = f"E_{e2[0,i].item()}"
            t_med = f"M_{e2[1,i].item()}"
            G.add_edge(s_enc, t_med, relation='has_medication')

    if ('Encounter','follows','Encounter') in hetero_data.edge_types:
        e3 = hetero_data['Encounter','follows','Encounter'].edge_index
        for i in range(e3.size(1)):
            s_enc = f"E_{e3[0,i].item()}"
            t_enc = f"E_{e3[1,i].item()}"
            G.add_edge(s_enc, t_enc, relation='follows')

    logger.info(f"Knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Visualize the graph
    visualize_knowledge_graph(G, max_nodes=300 if not debug else 100)
    
    # Convert to homogeneous graph for RGCN
    homo_data = manual_to_homogeneous(hetero_data)
    logger.info(f"Homogeneous graph: {homo_data.num_nodes} nodes, {homo_data.num_edges} edges")

    # Create 70/15/15 split
    homo_data = split_encounter_nodes(homo_data, split_ratio=(0.7,0.15,0.15), random_state=RANDOM_STATE)

    # Build & train the GNN model
    max_rel = int(homo_data.edge_type.max().item())+1 if homo_data.num_edges>0 else 3
    in_channels = homo_data.x.size(1)
    
    model = DeeperRGCN(num_relations=max_rel, in_channels=in_channels, hidden_dim=128, dropout=0.4)
    model = model.to(device)
    
    logger.info("Starting GNN model training...")
    model = train_gnn_model(homo_data, model, epochs=40, lr=0.01, patience=5)

    # Find best threshold on validation set
    val_probs, val_labels = evaluate_model_gnn(homo_data, model, homo_data.val_mask)
    if (val_labels==1).sum() < 1:
        best_thresh = 0.5
        logger.warning("No positives in validation set => cannot optimize threshold.")
    else:
        best_thresh, best_val_f1 = find_best_threshold(val_probs, val_labels)
        logger.info(f"Best threshold on validation set: {best_thresh:.3f}, F1={best_val_f1:.4f}")

    # Evaluate on test set with best threshold
    test_probs, test_labels = evaluate_model_gnn(homo_data, model, homo_data.test_mask)
    if (test_labels==1).sum() < 1:
        test_auc = 0.5
        test_f1 = 0.0
    else:
        test_auc = roc_auc_score(test_labels, test_probs)
        test_preds = (test_probs >= best_thresh).astype(int)
        test_f1 = f1_score(test_labels, test_preds)

    logger.info(f"Final TEST performance => AUC={test_auc:.4f}, F1={test_f1:.4f} (threshold={best_thresh:.3f})")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    plt.figure(figsize=(5,4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - GNN Model (Discharge)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Home(0)', 'Non-home(1)'], rotation=45)
    plt.yticks(tick_marks, ['Home(0)', 'Non-home(1)'])
    thresh = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]),
                     horizontalalignment="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # ROC curve
    if (test_labels==1).sum() > 0:
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'GNN Model (AUC={test_auc:.3f})')
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - GNN Model (Discharge Classification)')
        plt.legend()
        plt.show()

    # Print summary
    logger.info("\n===== GNN Model Training Summary =====")
    logger.info("Strategies:")
    logger.info("  1) 70/15/15 node-level split (train/val/test)")
    logger.info("  2) Standard scaling of numeric features")
    logger.info("  3) Deeper & wider RGCN (3 layers, 128 hidden, 0.4 dropout)")
    logger.info("  4) Threshold search + early stopping")
    logger.info(f"Graph structure => {homo_data.num_nodes} nodes, {homo_data.num_edges} edges")
    logger.info(f"Best threshold on validation: {best_thresh:.3f}")
    logger.info(f"Final TEST => AUC={test_auc:.4f}, F1={test_f1:.4f}")
    
    logger.info("GNN model training completed")
    
    return model

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function that orchestrates the entire workflow:
    1. Data loading and preparation
    2. Exploratory data analysis
    3. Traditional ML model training
    4. GNN model building and training
    """
    start_time = time.time()
    
    logger.info("===== Hospital Discharge Classification =====")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Using {N_JOBS} CPU cores for processing")
    logger.info(f"GPU acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    logger.info(f"SMOTE oversampling: {'Enabled' if USE_SMOTE else 'Disabled'}")
    logger.info(f"Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    
    # 1. Load and prepare data
    data = load_and_prepare_data(DATA_PATH, missing_threshold=MISSING_THRESHOLD)
    
    # 2. Perform exploratory data analysis if requested
    if not args.no_eda:
        perform_eda(data)
    
    # 3. Train traditional ML models if requested
    if not args.no_ml:
        ml_models = train_ml_models(data, debug=DEBUG_MODE)
    
    # 4. Build and train GNN model if requested
    if not args.no_gnn:
        gnn_model = build_and_train_gnn(data, debug=DEBUG_MODE)
    
    # Calculate and display total runtime
    end_time = time.time()
    runtime = end_time - start_time
    logger.info(f"\nTotal runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    return 0

if __name__ == "__main__":
    main()