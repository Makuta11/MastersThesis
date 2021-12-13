#%%
import os
import cv2
import ssl
import sys
import time
import pickle
import numpy as np
import pandas as pd
import threading
from glob import glob
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count, Process
from concurrent.futures import ThreadPoolExecutor
from utils import compress_pickle, decompress_pickle
import urllib.request

bad_idx = {
    "30": np.array([[939, 962],[1406,1422],[2100,2132],[2893,2955]]),
    "29": np.array([[4090,4543]]),
    "28": np.array([[1875,1885],[4571,4690]]),
    "27": np.array([[3461,3494],[4738,4785]]),
    "25": np.array([[4596,4662],[4816,4835]]),
    "23": np.array([[1021,1049],[3378,3557],[3584,3668],[4547,4621],[4741,4772],[4825,4840]]),
    "21": np.array([[574,616],[985,1164],[1190,1205],[1305,1338],[1665,1710],[1862,2477],[2554,4657],[4710,4722]]),
    "11": np.array([[4529,4533],[4830,4840]]),
    "9": np.array([[1736,1808],[1851,1885]]),
    "6": np.array([[1349,1405]]),
    "4": np.array([[4541,4555]]),
    "2": np.array([[800,826]]),
    "1": np.array([[398,429],[3190,3243]])
    }

def generate_AU_df(folder):
    key = folder[-5:]
    dkey = int(key[-3:])
    aus = [1,2,4,5,6,9,12,15,17,20,25,26]
    df = pd.DataFrame()
    for au in aus:
        tmp = pd.read_csv(f'{folder}/{key}_au{au}.txt', header=None)
        df[f'AU{au}'] = tmp[1].iloc[:4841]
    df["ID"] = dkey
    return [dkey, df]

if __name__ == '__main__':

    if sys.platform == "linux":
        AU_path = "/zhome/08/3/117881/MastersThesis/data/DISFA/ActionUnit_Labels"
        pickles_path = "/work3/s164272/data/Features"
    else:
        AU_path = "/Users/DG/Documents/PasswordProtected/DISFA/ActionUnit_Labels"
        pickles_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles"

    # Generate label dataframes
    print("Label generation started....")
    subj_list = sorted(os.listdir(AU_path))
    lab_list = Parallel(n_jobs = -1, verbose = 10)(delayed(generate_AU_df)(f'{AU_path}/{file}') for file in subj_list)
    print("Dictionary combination started....")
    
    lab_array = np.array(lab_list)
    
    df = pd.DataFrame()

    for i, frame in enumerate(lab_array):
        tmp = frame[1].iloc[::6,:]
        df = df.append(tmp)
    
    df = df.reset_index(drop=True)

    compress_pickle(f'{pickles_path}/disfa_labels', df)

# %%
