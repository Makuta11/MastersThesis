#%%
import os
import sys
import time
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification
from src.generate_feature_vector import decompress_pickle, compress_pickle

def main(bool):
    print("Loading Dataset")
    t = time.time()
    if sys.platform == 'linux':
        data = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa.pbz2")
        labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels.pbz2")
    else:
        data = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2")
        labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/disfa_labels_test.pbz2")

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

    print(f'There were a total of {len(bad_idx)} bad indexes')

    # Delete bad inputs
    data_arr = np.delete(data_arr, bad_idx, axis=0)

    # Construct final data arrays
    X = np.vstack(data_arr[:,1])
    X = np.nan_to_num(X)
    y = labels.drop(columns="ID").to_numpy()
    y = np.delete(y, bad_idx, axis = 0)

    print(f"Data loaded in {time.time() - t} seconds")

    print("Starting fit")
    t1 = time.time()
    X_text, y_test = X[:1616], y[:1616]
    X_train, y_train = X[1616:], y[1616:]
    clf = MultiOutputClassifier(KNeighborsClassifier(), n_jobs=-1).fit(X_train, y_train)
    print(f"Model fit in {time.time() - t1} seconds")
    
    print("Starting prediction")
    t2 = time.time()
    y_pred = clf.predict(X_test)
    print(f'It took {t2 - time.time()} to make predictions')

    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print(f'F1-score (micro) of the classifier was: {f1_micro}')
    print(f'F1-score (macro) of the classifier was: {f1_macro}')

    # Save the test model
    if sys.platform == 'linux':
        compress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/knearest_test", clf)
    else:
        compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

if __name__ == "__main__":
    print("starting script")
    main(True)


# %%
