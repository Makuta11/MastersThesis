import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

def decompress_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def compress_pickle(title: str, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_data(user_train_val, user_test):
    dataset = decompress_pickle(f'/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa.pbz2')
    labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels.pbz2")

    # Initialize parameters
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
    labels = labels.drop(bad_idx)

    # Construct final data arrays
    data_arr = np.vstack(data_arr[:,1])

    labels_test = pd.concat([labels[(labels.ID==te)] for te in user_test])
    labels_train_val = pd.concat([labels[(labels.ID==tr)] for tr in user_train_val])

    test_idx = list(labels_test.index)
    train_idx = list(labels_train.index)

    data_test = data_arr[test_idx, 1]
    data_train = data_arr[train_idx, 1]

    return data_test, data_train, labels_test.reset_index(drop=True), labels_train.reset_index(drop=True)

class ImageTensorDatasetMultitask(data.Dataset):
    
    def __init__(self, data, labels):
        self.user_id = df["ID"]
        self.label = df.drop["ID"]
        self.label_AU = self.label[(self.label >= 1)] = 1
        self.data = data

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, key):
        data = self.data[key]
        return slef.data[key], self.label[key]
        