import os, sys, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.dataloader import *
from src.utils import decompress_pickle, compress_pickle

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Subject split
user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_test = np.array([3,5,7,12,13])
user_val = np.array([5])

# Data loading with test-train splits 
print("Loading Dataset")
type_kern = "rbf"
kernel, test_idx, val_idx, train_idx, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True, kernel=type_kern)
labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")

# Initialize PCA transformation pipeline with standardization to shift columns to zero-mean
transform = make_pipeline(StandardScaler(copy=False, with_mean=True, with_std=False), KernelPCA(kernel='precomputed', n_components=300))

# Compute PCA on kernel
print("Transforming kernelized data")
TT.tick()
data_transformed = transform.fit_transform(kernel)
print(f'It took {TT.tock()} seconds to perform PCA on the kernel')

# Loop through specified action units and use LDA for classification
for i, au in enumerate([1]):

    # Convert labels to binary
    trainlab = labels_train.iloc[:,i]
    trainlab[trainlab > 0] = 1
    testlab = labels_test.iloc[:,i]
    testlab[testlab > 0] = 1

    # Initialize and fit LDA
    clf = LDA(shrinkage='auto').fit(data_transformed[train_idx], trainlab)

    # Predict classes for test data
    y_pred = clf.predict(data_transformed[test_idx])

    # Calculate f1_scores
    print("Starting f1-score calculation")
    print(f'\nTest scores on AU{au} identification:\n{classification_report(testlab, y_pred, target_names=["not active", "active"])}')

    # Save the SVC model for specific AU
    if sys.platform == 'linux':
        compress_pickle(f"/work3/s164272/data/models/KLDA_{type_kern}_AU{au}", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)


