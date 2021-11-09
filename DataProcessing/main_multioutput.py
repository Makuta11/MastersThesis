import os
import sys
import time
import numpy as np
import pandas as pd

from src.generate_feature_vector import decompress_pickle, compress_pickle
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

def main(bool):
    print("Loading Dataset")
    t = time.time()
    data = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2")
    labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels_test.pbz2")

    bad_idx = []
    data_list = list(data.items())
    data_arr = np.array(data_list)
    
    # Collect bad inputs
    ln = data_arr[0,1].shape[0]
    for i, arr in enumerate(data_arr[:,1]):
        try:
            if len(arr) != ln:
                bad_idx.append(i)
        except:
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

