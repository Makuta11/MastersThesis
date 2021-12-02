import os, sys, time, pickle

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from src.validation_score import val_scores
from src.utils import decompress_pickle, compress_pickle

def main(bool):

    # AUs in dataset
    aus = [1,2,4,5,6,9,12,15,17,20,25,26]

    print("Loading Dataset")
    t = time.time()
    if sys.platform == 'linux':
        data = np.load('/work3/s164272/data/Features/face_space_dict_disfa_large_subset.npy', allow_pickle=True)
        #data = decompress_pickle("/work3/s164272/data/Features/face_space_dict_disfa_large1.pbz2")
        labels = decompress_pickle("/work3/s164272/data/Features/disfa_labels_large1.pbz2")
    else:
        data = np.load("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_large_subset.npy", allow_pickle=True)
        labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/disfa_labels_large1.pbz2")

    # Unfold dict inside 0-dimensional array (caused by np.save/np.load)
    if data.shape == ():
        data = data.tolist()

    # Convert from dictionary to np array
    data_list = list(data.items())
    data_arr = np.array(data_list)
    
    # Collect bad inputs
    bad_idx = []
    ln = data_arr[0,1].shape[0]
    for i, arr in enumerate(data_arr[:,1]):
        try:
            if len(arr) != ln:
                bad_idx.append(i)
        except:
            bad_idx.append(i)

    print(f'There were a total of {len(bad_idx)} bad indexes')

    # Remove bad inputs
    data_arr = np.delete(data_arr, bad_idx, axis=0)

    # Construct final data arrays
    X = np.vstack(data_arr[:,1])
    X = np.nan_to_num(X)
    y = labels.drop(columns="ID").to_numpy()
    y = np.delete(y, bad_idx, axis = 0)
    print(f"Data loaded in {time.time() - t} seconds")

    # Test train split
    X_test, y_test = X[:4840*5], y[:4840*5]
    X_train, y_train = X[4840*5:], y[4840*5:]
    print(f'Train size = {X_train.shape}\nTrain size lab = {y_train}\nTest size = {X_test.shape} \nTest size lab = {y_test}')

    # Clear memory space
    del X, y, data, data_arr, data_list, bad_idx
    
    print("Starting fit")
    t1 = time.time()
    forest =  RandomForestClassifier(random_state = 1)
    clf = MultiOutputClassifier(forest, n_jobs = -1).fit(X_train, y_train)
    print(f"Model fit in {time.time() - t1} seconds") 
    # Clear memory space
    del X_train, y_train 

    print("Starting prediction...")
    t2 = time.time()
    y_pred = clf.predict(X_test)
    print(f'It took {time.time() - t2} to make predictions')

    del X_test
    
    print(np.unique(y_pred))
    print("\nStarting f1-score calculation")

    y_pred_ones = y_pred
    y_pred_ones[y_pred_ones >= 1] = 1
    print(f'\nScores on AU identification:\n{val_scores(y_pred_ones, y_test)}')

    for i, au in enumerate(aus):
        print(f"f1-score for intensity of AU{au}:")
        print(f'{f1_score(y_test[:,i], y_pred[:,i], average = None, zero_division = 1)}')

    # Save the test model
    if sys.platform == 'linux':
        compress_pickle("/work3/s164272/RF_clf", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

if __name__ == "__main__":
    print("starting script")
    main(True)
