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
import urllib.request

def fetch_vid_files(vid_dir):
    files = [file
            for path, subdir, files in os.walk(vid_dir)
            for file in glob(os.path.join(path, "*.avi"))]
    return files

def fetch_AU_files(AU_dir):
    files = [file
            for path, subdir, files in os.walk(AU_dir)
            for file in glob(os.path.join(path, "*.txt"))]
    return files

def convert_vid_to_array(file):
    key = int(file[-12:-9])
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    count = 0
    while success:
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        success, image = vidcap.read()
        count += 1
    print(f'{count} frames read')
    return {key: {img: frames, labels: labs}}

def generate_AU_df(folder):
    key = folder[68:73]
    dkey = int(key[-3:])
    aus = [1,2,4,5,6,9,12,15,17,20,25,26]
    df = pd.DataFrame()
    for au in aus:
        tmp = pd.read_csv(f'{folder[:67]}/{key}_au{au}.txt', header=None)
        df[f'AU{au}'] = tmp[1]

    return {dkey: df}

if __name__ == '__main__':

    if sys.platform == "linux":
        video_path = "/zhome/08/3/117881/MastersThesis/data/DISFA/Video_RightCamera"
        AU_path = "/zhome/08/3/117881/MastersThesis/data/DISFA/ActionUnit_Labels"
        pickles_path = "/zhome/08/3/117881/MastersThesis/Data Processing/pickles"
    else:
        video_path = "/Users/DG/Documents/PasswordProtected/DISFA/Video_RightCamera"
        AU_path = "/Users/DG/Documents/PasswordProtected/DISFA/ActionUnit_Labels"
        pickles_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/Data Processing/pickles"

    # Fetch paths
    vid_list = fetch_vid_files(video_path)
    au_list = fetch_AU_files(AU_path)
    
    # Initialize parameters
    img_dict = dict()
    labs_dict = dict()
    misses_img = []
    misses_labs = []

    # Generate label dataframes
    print("Label generation started....")
    dictionary_list_lab = Parallel(n_jobs = -1, verbose = 10)(delayed(generate_AU_df)(file) for file in au_list)
    print("Dictionary combination started....")
    for d in dictionary_list_lab:
            try:
                labs_dict.update(d)
            except:
                misses_labs.append(d)

    # Generate image dictionaries
    print("Generation started....")
    dictionary_list = Parallel(n_jobs = -1, verbose = 10)(delayed(convert_vid_to_array)(file) for file in vid_list)
    print("Dictionary combination started....")
    for d in dictionary_list:
            try:
                img_dict.update(d)
            except:
                misses_img.append(d)

    compress_pickle(f'{pickles_path}/disfa_img_dict', img_dict)
    compress_pickle(f'{pickles_path}/disfa_img_misses', misses_img)
    compress_pickle(f'{pickles_path}/disfa_labs_dict', labs_dict)
    compress_pickle(f'{pickles_path}/disfa_labs_misses', misses_labs)
    
    # %%
