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

def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint_test-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

def load_data(user_train_val, user_test):
    if sys.platform == "linux":
        dataset = decompress_pickle(f'/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2')
        labels = decompress_pickle("/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/disfa_labels_test.pbz2")
    else:
        dataset = decompress_pickle(f'/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/face_space_dict_disfa_test.pbz2')
        labels = decompress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles/disfa_labels_test.pbz2")

    # Initialize parameters
    bad_idx = []
    data_list = list(dataset.items())
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
    labels = labels.drop(bad_idx)

    # Construct final data arrays
    data_arr = np.vstack(data_arr[:,1])
    data_arr = np.nan_to_num(data_arr)

    if sys.platform == "linux":
        labels_test = pd.concat([labels[(labels.ID==te)] for te in user_test])
        labels_train_val = pd.concat([labels[(labels.ID==tr)] for tr in user_train_val])
    else:
        labels_test = labels.iloc[:100]
        labels_train_val = labels.iloc[100:]

    test_idx = list(labels_test.index)
    train_idx = list(labels_train_val.index)

    data_test = data_arr[test_idx, :]
    data_train = data_arr[train_idx, :]

    return data_test, data_train, labels_test.reset_index(drop=True), labels_train_val.reset_index(drop=True)

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
        