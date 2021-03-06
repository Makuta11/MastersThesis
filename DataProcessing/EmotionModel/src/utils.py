import pickle, bz2, os, torch, fnmatch

import numpy as np
import pandas as pd

from glob import glob
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.class_weight import compute_class_weight

def decompress_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def compress_pickle(title: str, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def get_class_weights_AU(labels_train):
    class_weights_AU = []
    for element in np.array(labels_train.sum()[:-1]):
        if element == 0:
            class_weights_AU.append(torch.tensor(10**5))
        else:
            class_weights_AU.append(torch.tensor(labels_train.shape[0]/element))
    class_weights_AU = torch.tensor(class_weights_AU)
    return class_weights_AU

def get_class_weights_AU_int(labels_train):
    class_weights_int = []
    for col in labels_train.drop(columns="ID").columns:
        tmp = compute_class_weight(class_weight='balanced', classes=np.unique(labels_train[col]), y=labels_train[col].to_numpy())
        tmpd = {}
        for i, key in enumerate(np.array(labels_train[col].value_counts().axes)[0]):
                tmpd[key] = tmp[i]
        for k in [1,2,3,4,5]:
            if k not in tmpd.keys():
                tmpd[k] = 10**5
        class_weights_int.append([tmpd[1], tmpd[2], tmpd[3], tmpd[4], tmpd[5]])
    class_weights_int = np.array(class_weights_int)
    class_weights_int  = torch.tensor(class_weights_int, dtype=torch.float)

    return class_weights_int

def fetch_video_npy(dataDir):
    """
    Takes a directory continaing data files and returns a list of all .csv files in the directory and sub-directories

    Args:
        dataDir (str): root directory containing outputs from the experiments

    Returns:
        list: a list containing the filenames of all experiment files saved as a .csv
    """
    all_rest_files = [file
                    for path, subdir, files in os.walk(dataDir)
                    for file in glob(os.path.join(path, "*.npy"))
                    if "video" in file]

    return all_rest_files


