import os, sys, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score

from src.dataloader import *
from src.utils import decompress_pickle, compress_pickle

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]
au = 20

# Subject split
#user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_train = np.array([28, 7, 29])
user_val = np.array([5])

#users_list = pickle.load(open("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/subsets", 'rb'))
#users = users_list[f'AU{au}']
#user_train, user_val = train_test_split(users, test_size=0.2, random_state=42) 
user_test = user_val

# Data loading with test-train splits 
print("Loading Dataset")
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")

# Loop through specified action units and use AdaBoost for classification
for i, au in enumerate([au]):
    # Change index
    i = aus.index(au)
    
    # Convert labels to binary
    trainlab = labels_train.iloc[:,i]
    trainlab[trainlab > 0] = 1
    testlab = labels_test.iloc[:,i]
    testlab[testlab > 0] = 1
    print(np.sum(testlab))

    # Initialize and fit AdaBoost
    clf = AdaBoostClassifier(n_estimators=5000, random_state=0).fit(data_train, trainlab)

    # Predict classes for test data
    y_pred = clf.predict(data_test)

    # Calculate f1_scores
    print("Starting f1-score calculation")
    print(f'\nTest scores on AU{au} identification:\n{classification_report(testlab, y_pred, target_names=["not active", "active"])}')

    # Save the SVC model for specific AU
    if sys.platform == 'linux':
        compress_pickle(f"/work3/s164272/data/models/ADA_AU{au}", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)


