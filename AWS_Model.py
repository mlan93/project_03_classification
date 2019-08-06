#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Note: this is a supporting python file used for modeling in AWS. While the majority of the project was
# completed locally due to issues running the AWS instance, gridsearch was performed on learning rates for
# xgboost.

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
    
with open('df_M.pkl', 'rb') as f:
    df_M = pickle.load(f)
    
# Clean dataframe
X = df_M.drop(['UNIQUE_ID', 'MIGRATE1', 'POVERTY_BIN', 'BPL', 'EDUC', 'MARST'], axis = 1)
y = df_M[['POVERTY_BIN']]
X_init, X_te, y_init, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_init, y_init, test_size=0.25, random_state=42, stratify=y_init)

# Assign weight columns
WT_init = X_init['PERWT'].copy()

# Scale all X_values
std = StandardScaler()

X_sets = [X_init]
X_scale = []

for df in X_sets:
    df_sc = df.copy()
    df_scale_cols = df.drop(['PERWT'], axis = 1)
    scaler = std.fit_transform(df_scale_cols.values)
    df_sc.loc[:,list(df_scale_cols.columns)] = scaler
    X_scale.append(df_sc)

X_init = X_scale[0]

# XGBClassifier gridsearch
xgb_model = XGBClassifier(random_state=42)

params = {'learning_rate': [0.03, 0.05, 0.07]}

# Learning rates compared: 0.03, 0.05, 0.07, 0.1

xgb_grid = GridSearchCV(xgb_model, param_grid = params, scoring='precision', cv = 3, n_jobs = -1)

xgb_grid.fit(X_init.drop(['index','PERWT'], axis = 1), y_init.values.ravel(),              sample_weight = WT_init.values.ravel())

with open('xgb_grid.pkl', 'wb') as f:
    pickle.dump(xgb_grid, f)

