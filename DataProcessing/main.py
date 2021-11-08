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
    data = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa.pbz2")
    labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels.pbz2")

    data_list = np.vstack(np.array(list(data.items()))[:,1])
    y = labels.drop["ID"].to_numpy()

    #import pdb; pdb.set_trace()
    
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
    #import pdb;pdb.set_trace()


if __name__ == "__main__":
    print("starting script")
    main(True)

