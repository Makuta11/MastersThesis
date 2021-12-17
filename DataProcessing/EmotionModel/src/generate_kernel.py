import os, sys

import numpy as np
import pandas as pd

from pelutils import TT
from src.utils import decompress_pickle
from sklearn.metrics.pairwise import rbf_kernel

def compute_kernel(X, settings: dict):
    print("Computing kernel")
    if settings['kernel'] == 'linear':
        K = np.dot(X,X.T) # linear kernel
    elif settings['kernel'] == 'rbf':
        K = rbf_kernel(X, gamma=None) # gamma defaults to 1/n_features
    else:
        print('Invalid kernel option, terminating')
        exit()
    print('.. finished')
    return K

def load_data_for_kernel(subset = None):
    if subset:
        if sys.platform == "linux":
            # Big dataload on hpc
            dataset = np.load('/work3/s164272/data/Features/shape_space_dict_disfa_large_subset_300.npy', allow_pickle=True)
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
    return data_arr

def save_kernel(K, name):
    if sys.platform == "linux":
        save_path = "/work3/s164272/data/assests"
    else:
        raise Exception("Please do not try to compute this locally!")
    
    np.save(f'{save_path}/{name}_kernel.npy', K)

def main():
    # Load dataset
    print("Loading data")
    TT.tick()
    dataset = load_data_for_kernel(subset=True)
    print(f"Data loaded in {TT.tock()} seconds")

    # Define settings for kernel
    settings = {'kernel': 'rbf'}

    # Compute kernel
    print("Computing kernel")
    TT.tick()
    K = compute_kernel(dataset, settings)
    print(f"Kernal generated in {TT.tock()} seconds")

    # Save kernel to assests folder
    print("Saving kernel")
    TT.tick()
    save_kernel(K, settings['kernel'])
    print(f"Kernel saved in {TT.tock()} seconds")

if __name__ == "__main__":
    main()