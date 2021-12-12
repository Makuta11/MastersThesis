import torch, bz2, sys, pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from src.utils import decompress_pickle

def load_data(user_train, user_val, user_test, subset = None, kernel = None):
    if subset:
        if sys.platform == "linux":
            # Big dataload on hpc
            dataset = np.load('/work3/s164272/data/Features/face_space_dict_disfa_large_subset.npy', allow_pickle=True)
            labels = decompress_pickle("/work3/s164272/data/Features/disfa_labels_large1.pbz2")
            misses = np.load('/work3/s164272/data/Features/misses_disfa_large_subset.npy', allow_pickle=True)
            # Unfold dict inside 0-dimensional array (caused by np.save/np.load)
            dataset = dataset.tolist()
        else:
            # Small testing dataload for local machine
            #dataset = decompress_pickle(f'/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/face_space_dict_disfa_test_subset.npy')
            dataset = np.load("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/face_space_dict_disfa_large_subset_300_test.npy", allow_pickle=True)
            labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/disfa_labels_large1.pbz2")
            misses = np.load('/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/misses_disfa_large_subset.npy', allow_pickle=True)
            # Unfold dict inside 0-dimensional array (caused by np.save/np.load)
            labels = labels[:4840*6]
            dataset = dataset.tolist()
    else:
        if sys.platform == "linux":
            # Big dataload on hpc
            dataset = decompress_pickle('/work3/s164272/data/Features/face_space_dict_disfa_large1.pbz2')
            labels = decompress_pickle("/work3/s164272/data/Features/disfa_labels_large1.pbz2")
            misses = decompress_pickle('/work3/s164272/data/Features/misses_disfa_large1.pbz2')
        else:
            # Small testing dataload for local machine
            #dataset = decompress_pickle(f'/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2')
            dataset = np.load("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/face_space_dict_disfa_large_subset.npy", allow_pickle=True)
            labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/disfa_labels_large1.pbz2")

    # Remove missed dataindexes from labels (currently not any so its not implemented)
    if len(misses) > 0:
        pass

    # Initialize parameters
    bad_idx = []
    if type(dataset) == dict:
        data_list = list(dataset.items())
        data_arr = np.array(data_list)
    else:
        data_arr = dataset

    # Collect bad inputs
    ln = data_arr[0,1].shape[0]
    for i, arr in enumerate(data_arr[:,1]):
        try:
            if len(arr) != ln:
                bad_idx.append(i)
        except:
            bad_idx.append(i)

    # Delete bad inputs from labels and array
    labels = labels.drop(bad_idx).reset_index(drop=True)
    data_arr = np.delete(data_arr, bad_idx, axis=0)

    # Construct final data arrays
    data_arr = np.vstack(data_arr[:,1])
    data_arr = np.nan_to_num(data_arr)

    # Check for small dataload on local system
    if sys.platform == "linux":
        labels_test = pd.concat([labels[(labels.ID==te)] for te in user_test])
        labels_train = pd.concat([labels[(labels.ID==tr)] for tr in user_train])
        
        # Test if validation set is given
        if len(user_val) == 0:
            labels_val = np.array([])
        else:
            labels_val = pd.concat([labels[(labels.ID==va)] for va in user_val])
    else:
        labels_test = labels.iloc[:1]
        labels_val = labels.iloc[0:4840*2]
        labels_train = labels.iloc[4840*2:int(4840*6)]

    # Extract test-val-train indexes
    test_idx = list(labels_test.index)
    val_idx = list(labels_val.index)
    train_idx = list(labels_train.index)

    # Slice data
    data_test = data_arr[test_idx, :]
    data_val = data_arr[val_idx, :]
    data_train = data_arr[train_idx, :]

    # For kernel we only need the indexes since the kernel is precomputed and we will be slicing into that instead
    if kernel:
        if sys.platform == "linux":
            kernel = np.load(f"/work3/s164272/data/assests/{kernel}_kernel.npy")
        else:
            raise Exception("Not implemented on local platform (needs 127GB memory)")
        
        # Clear some memory space since data is not needed (indexing into kernel for test and train)
        del data_test, data_val, data_train
        
        return kernel, test_idx, val_idx, train_idx, labels_test.reset_index(drop=True), labels_val.reset_index(drop=True), labels_train.reset_index(drop=True)
    else:
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
        # Multi-label classification labels
        AUs = np.array(self.labels.iloc[key])
        AUs[AUs != 0] = 1
        
        """CrossEntropyLoss does not take one-hot-encoded labels
            # One hot encode AU_intensities
            AU_int = np.zeros((12,5))
            for i, lab in enumerate(self.labels.iloc[key]):
                AU_int[i][lab] = 1
        """
        
        return self.data[key], AUs, self.labels.iloc[key].values
        #return self.data[key], AUs.values, torch.Tensor(np.array([AU_int[0],AU_int[1],AU_int[2],AU_int[3],AU_int[4],AU_int[5],AU_int[6],AU_int[7],AU_int[8],AU_int[9],AU_int[10],AU_int[11]]))

class ImageTensorDatasetMultiLabel(data.Dataset):
    
    def __init__(self, data, labels):
        self.data = data
        self.user_id = labels["ID"]
        self.labels = labels.drop(columns = "ID")
        
    def __len__(self):
        return len(self.labels)

    def __nf__(self):
        return len(self.data[0])
    
    def __getitem__(self, key):
        # Multi-label classification labels
        AUs = np.array(self.labels.iloc[key])
        AUs[AUs != 0] = 1
        
        return self.data[key], AUs

def main():
    pass

if __name__ == "__main__":
    main()