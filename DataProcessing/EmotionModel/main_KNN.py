# Imports
import os, sys, time, pickle

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

from src.dataloader import *
from src.validation_score import val_scores
from src.utils import decompress_pickle, compress_pickle

def main(bool):

    # AUs in dataset
    aus = [1,2,4,5,6,9,12,15,17,20,25,26]

    # Subject split
    user_train = np.array([1,2,4,6,8,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
    user_test = np.array([3,5,7,9,12,13])
    user_val = np.array([5])

    # Data loading with test-train splits 
    print("Loading Dataset")
    t = time.time()
    data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
    labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")
    print(f"It took {time.time() - t} seconds to load the data")
    
    print("Starting fit")
    t1 = time.time()
    clf = MultiOutputClassifier(KNeighborsClassifier(), n_jobs=24).fit(data_train, labels_train)
    print(f"Model fit in {time.time() - t1} seconds") 

    # Clear memory space
    del data_train, labels_train

    print("Starting prediction...")
    t2 = time.time()
    y_pred = clf.predict(data_test)
    print(f'It took {time.time() - t2} to make predictions')

    # Clear memory space
    del data_test
    
    print(np.unique(y_pred))
    print("\nStarting f1-score calculation")

    # Convert to bool raveled bool array for AU identification
    y_pred_ones = y_pred.ravel()
    y_pred_ones[y_pred_ones >= 1] = 1
    labels_test_ones = labels_test.to_numpy().ravel()
    labels_test_ones[labels_test_ones >= 1] = 1

    # Insert label names to differentiate precision socres between labels
    predAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(y_pred_ones)]
    trueAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(labels_test_ones)]

    # Calculate f1_scores
    print(f'\nScores on AU identification:\n{val_scores(predAU, trueAU)}')

    for i, au in enumerate(aus):
        print(f"f1-score for intensity of AU{au}:")
        print(f'{val_scores(labels_test.iloc[:,i].to_numpy(), y_pred[:,i])}')

    # Save the test model
    if sys.platform == 'linux':
        compress_pickle("/work3/s164272/models/KNN_clf", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

if __name__ == "__main__":
    print("starting script")
    main(True)
