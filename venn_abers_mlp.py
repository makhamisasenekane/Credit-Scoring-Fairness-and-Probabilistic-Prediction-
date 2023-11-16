#!/usr/bin/python3

# code for classification using Randomforest and MLP

import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#from sklearn.calibration import CalibrationDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import calibration as cal

from venn_abers import VennAbersCalibrator, VennAbers

np.random.seed(42)

'''
X, y = make_classification(
    n_samples=100000, n_features=20, n_informative=2, n_redundant=2, random_state=1
)

train_samples = 1000  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=100000 - train_samples,
)
'''

data_path  = 'german_data.csv'
df= pd.read_csv(data_path)


# check the colmns and shape
print(df.columns)
print(df.shape)

# Further EDA
print(df.duplicated().sum())
print(df.isna().sum())
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Correlation Matrix
correlation_matrix = df.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(correlation_matrix,annot=True,square=True, linewidths=.5,cmap=plt.cm.Reds)
plt.show()

X = df.drop(columns=['approval'])
y = df['approval']
print("Original:", Counter(y))
sm = SMOTE(random_state=42)
X, y = sm.fit_resample(X, y)
print("After sampling:", Counter(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=0)


print(X_train.shape)
print(y_train.shape)


# underlying classifier
clf = MLPClassifier(random_state=42).fit(X_train, y_train)
clf.fit(X_train, y_train)
clf_prob = clf.predict_proba(X_test)

# Inductive Venn-ABERS calibration (IVAP)
va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, shuffle=False)
va.fit(X_train, y_train)
va_inductive_prob = va.predict_proba(X_test)

# Cross Venn-ABERS calibration (CVAP)
va = VennAbersCalibrator(estimator=clf, inductive=False, n_splits=2)
va.fit(X_train, y_train)
va_cv_prob = va.predict_proba(X_test)



log_losses=[]
log_losses.append(log_loss(y_test, clf_prob))
log_losses.append(log_loss(y_test, va_inductive_prob))
log_losses.append(log_loss(y_test, va_cv_prob))
#log_losses.append(log_loss(y_test, va_prefit_prob))
print("Log_losses:", log_losses)


nb = MLPClassifier(random_state=42).fit(X_train, y_train)
nb_vennabers =  VennAbersCalibrator(estimator=nb, inductive=False, n_splits=3, precision=4)

metrics_list = []

clf = MLPClassifier(random_state=42).fit(X_train, y_train)
clf =  VennAbersCalibrator(estimator=nb, inductive=True, n_splits=3, precision=4) #Venn-Abers predictor (IVAP)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:,1]
l_loss = log_loss(y_test, y_prob)
brier_loss = brier_score_loss(y_test, y_prob)
calibration_error = cal.get_calibration_error(y_prob, y_test)
y_pred = clf.predict(X_test)
metrics_list.append([brier_loss, l_loss, calibration_error])
print("loss metrics (IVAP):", metrics_list)

metrics_list = []

clf = MLPClassifier(random_state=42).fit(X_train, y_train)
clf =  VennAbersCalibrator(estimator=nb, inductive=False, n_splits=3, precision=4) #Venn-Abers predictor (CVAP)
clf.fit(X_train, y_train)
y_prob = clf.predict_proba(X_test)[:,1]
l_loss = log_loss(y_test, y_prob)
brier_loss = brier_score_loss(y_test, y_prob)
calibration_error = cal.get_calibration_error(y_prob, y_test)
y_pred = clf.predict(X_test)
metrics_list.append([brier_loss, l_loss, calibration_error])
print("loss metrics (CVAP):", metrics_list)

