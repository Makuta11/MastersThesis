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
from generate_feature_vector import compress_pickle, decompress_pickle
import urllib.request

def generate_AU_df(folder):
    key = folder[-5:]
    dkey = int(key[-3:])
    aus = [1,2,4,5,6,9,12,15,17,20,25,26]
    df = pd.DataFrame()
    for au in aus:
        tmp = pd.read_csv(f'{folder}/{key}_au{au}.txt', header=None)
        df[f'AU{au}'] = tmp[1].iloc[:4840]
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
    lab_list = Parallel(n_jobs = -1, verbose = 10)(delayed(generate_AU_df)(f'{AU_path}/{file}') for file in sorted(os.listdir(AU_path)))
    print("Dictionary combination started....")
    
    lab_array = np.array(lab_list)
    
    df = pd.DataFrame()

    for i, frame in enumerate(lab_array):
        tmp = frame[1].iloc[::6,:]
        df = df.append(tmp)
    
    df = df.reset_index(drop=True)

    compress_pickle(f'{pickles_path}/disfa_labels', df)

# %%
