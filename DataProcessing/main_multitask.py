import os, sys, time, torch, pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.model import *
from src.dataloader import *
from sklearn.model_selectiom import KFold
from src.generate_feature_vector import decompress_pickle, compress_pickle

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

def main(bool):
    print("Loading Dataset")
    t = time.time()

    # subject split
    # users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
    user_train_val = np.array([2,3,4,6,7,8,10,11,16,17,18,21,23,24,25,26,27,28,30,32])
    user_test = np.array([1,5,9,12,13,29,31])
    
    data, labels, train_idx, test_idx = load_data(user_train_val, user_test)

    train_val_dataset = ImageTensorDatasetMultitask(data[train_idx], labels[train_idx])
    test_dataset = ImageTensorDatasetMultitask(data[test_idx], labels[test_idx])

    train_val_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    bad_idx = []
    data_list = list(data.items())
    data_arr = np.array(data_list)
    
    # Collect bad inputs
    ln = data_arr[0,1].shape[0]
    for i, arr in enumerate(data_arr[:,1]):
        try:
            if len(arr) != ln:
                bad_idx.append(i)
        else:
            bad_idx.append(i)

    # Delete bad inputs
    data_arr = np.delete(data_arr, bad_idx, axis=0)

    # Construct final data arrays
    X = np.vstack(data_arr[:,1])
    y = labels.drop(columns="ID").to_numpy()
    
    print(f"Data loaded in {time.time() - t} seconds")

    print("Starting fit")
    t1 = time.time()
    clf = MultiOutputClassifier(KNeighborsClassifier(), n_jobs=-1).fit(X,y)
    clf.predict(X[-2:])
    print(f"Model fit in {time.time() - t1} seconds")

    print("This is a prediction")
    clf.predict(X[-5:])
    print(y[-5:])

    print("Activate debugger for inspection of data")

    # Save the test model
    compress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/knearest_test", clf)


if __name__ == "__main__":
    print("starting script")
    main(True)

