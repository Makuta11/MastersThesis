#%%
import os, sys, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from src.dataloader import *
from sklearn.metrics import f1_score
from sklearn.decomposition import KernelPCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.validation_score import val_scores
from src.utils import decompress_pickle, compress_pickle

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Subject split
user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_test = np.array([3,5,7,12,13])
user_val = np.array([5])

# Data loading with test-train splits 
print("Loading Dataset")
t = time.time()
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")
print(f"It took {time.time() - t} seconds to load the data")

from sklearn.svm import SVC
clf = SVC(kernel='rbf', gamma="auto", class_weight="balanced")
trainlab = labels_train.iloc[:,0]
trainlab[trainlab > 0] = 1
testlab = labels_test.iloc[:,0]
testlab[testlab > 0] = 1

TT.tick()
clf.fit(data_train, trainlab)
print(f'Time to fit: {TT.tock()}')

print("Starting prediction...")
t3 = time.time()
y_pred = clf.predict(data_test)
print(f'It took {time.time() - t3} to make predictions')

print(np.unique(y_pred))
print("\nStarting f1-score calculation")

# Calculate f1_scores
print(f'\nScores on AU identification:\n{val_scores(y_pred, testlab)}')

# Save the test model
if sys.platform == 'linux':
    compress_pickle("/work3/s164272/models/KSVM_clf", clf)
else:
    compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

if __name__ == "__main__":
    print("starting script")
    #main(True)

