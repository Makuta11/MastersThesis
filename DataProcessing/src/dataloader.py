import torch, bz2, sys
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import pandas as pd

def decompress_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def compress_pickle(title: str, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_data(user_train, user_val, user_test):
    if sys.platform == "linux":
        dataset = decompress_pickle(f'/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa.pbz2')
        labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels.pbz2")
    else:
        dataset = decompress_pickle(f'/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2')
        labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/disfa_labels_test.pbz2")

    # Initialize parameters
    bad_idx = []
    data_list = list(dataset.items())
    data_arr = np.array(data_list)
    
    print(f'data: {data_arr.shape} - labels: {labels.shape}')

    # Collect bad inputs
    ln = data_arr[0,1].shape[0]
    for i, arr in enumerate(data_arr[:,1]):
        try:
            if len(arr) != ln:
                bad_idx.append(i)
        except:
            bad_idx.append(i)

    # Delete bad inputs from labels. Since indexes are pulled from labels, we do not have to remove it from the data list
    labels = labels.drop(bad_idx)

    print(f'data: {data_arr.shape} - labels: {labels.shape}')

    # Construct final data arrays
    data_arr = np.vstack(data_arr[:,1])
    data_arr = np.nan_to_num(data_arr)

    if sys.platform == "linux":
        labels_test = pd.concat([labels[(labels.ID==te)] for te in user_test])
        labels_val = pd.concat([labels[(labels.ID==va)] for va in user_val])
        labels_train = pd.concat([labels[(labels.ID==tr)] for tr in user_train])
    else:
        labels_test = labels.iloc[:75]
        labels_val = labels.iloc[75:200]
        labels_train = labels.iloc[200:]

    test_idx = list(labels_test.index)
    val_idx = list(labels_val.index)
    train_idx = list(labels_train.index)

    print(f'data_arr shape: {data_arr.shape}')
    print(f'test: {test_idx}')
    print(f'val: {val_idx}')
    print(f'train: {train_idx}')

    data_test = data_arr[test_idx, :]
    data_val = data_arr[val_idx, :]
    data_train = data_arr[train_idx, :]

    return data_test, data_val, data_train, labels_test.reset_index(drop=True), labels_val.reset_index(drop=True), labels_train.reset_index(drop=True)

class ImageTensorDatasetMultitask(data.Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.user_id = labels["ID"]
        self.labels = labels.drop(columns = "ID")
        
    def __len__(self):
        return len(self.labels)

    def __nf__(self):
        return len(self.data[0])
    
    def __getitem__(self, key):
        data = self.data[key]
        
        # Multi-label classification labels
        """ Think this is a mistake. I might try later
        AUs = list(self.labels.iloc[key][self.labels.iloc[key] != 0].index)
        if len(AUs) == 0:
            AUs = 0
        """
        AUs = self.labels.iloc[key]
        AUs[AUs != 0] = 1
        
        # One hot encode AU_intensities
        AU_int = np.zeros((12,5))
        for i, lab in enumerate(self.labels.iloc[key]):
            AU_int[i][lab] = 1

        return self.data[key], AUs.values, [AU_int[0],AU_int[1],AU_int[2],AU_int[3],AU_int[4],AU_int[5],AU_int[6],AU_int[7],AU_int[8],AU_int[9],AU_int[10],AU_int[11]]
        