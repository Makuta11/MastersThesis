#%%
import os, sys, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.svm import SVC
from src.dataloader import *
from sklearn.pipeline import make_pipeline
from src.generate_kernel import compute_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.validation_score import val_scores
from src.utils import decompress_pickle, compress_pickle

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Subject split
#user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_train = np.array([4, 11, 16, 18, 25, 26, 29, 30])
user_test = np.array([5, 28])
user_val = np.array([5])

# Data loading with test-train splits 
print("Loading Dataset")
TT.tick()
gamma = "scale"
type_kern = "rbf"
settings = {'kernel': 'rbf', 'gamma': 'scale'}
kernel, test_idx, val_idx, train_idx, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True, kernel=type_kern)
labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")
print(f"It took {TT.tock()} seconds to load the data")

au = 15
for i, c in enumerate([1]):
    i = aus.index(au)
    #print(f'Fitting SVC for: {au}')

    # Initialize SVC from sklearn library
    clf = SVC(C=c, kernel='precomputed')

    # Redefine labels to classify single AU at a time
    trainlab = labels_train.iloc[:,i]
    trainlab[trainlab > 0] = 1
    testlab = labels_test.iloc[:,i]
    testlab[testlab > 0] = 1
    
    # Fit SVC to the training kernel
    #TT.tick()
    #print("Starting fit...")
    clf.fit(kernel[train_idx][:,train_idx], trainlab)
    #print(f'Time to fit: {TT.tock()} \n')

    # Predict classes for the test kernel
    #TT.tick()
    #print("Starting prediction...")
    y_pred = clf.predict(kernel[test_idx][:,train_idx])
    #y_pred_train = clf.predict(kernel[train_idx][:,train_idx])
    #print(f'It took {TT.tock()} to make predictions')

    # Compare sums
    #print(f'train vs test sum: {y_pred.sum()} & {y_pred_train}')

    # Calculate f1_scores
    print("Starting f1-score calculation")
    #print(f'\nTest scores on AU{au} identification:\n{val_scores(y_pred, testlab)}')
    print(f'\nTest scores on AU{au} C:{c} identification:\n{classification_report(testlab, y_pred, target_names=["not active", "active"])}')

    # Save the SVC model for specific AU
    if sys.platform == 'linux':
        compress_pickle(f"/work3/s164272/data/models/KSVM_NORM_{gamma}_{type_kern}_AU{au}", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

