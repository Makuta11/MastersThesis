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

def convert_vid_to_array(file):
    key = file[-12:-9]
    dataDir = f"/zhome/08/3/117881/MastersThesis/data/DISFA/ImageDir/SN{key}_"
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    count = 0
    while success:
        success, image = vidcap.read()
        if count%6 == 0:
            cv2.imwrite(dataDir + '{0:05}'.format(count) + ".jpg", image)
        count += 1

if __name__ == '__main__':

    if sys.platform == "linux":
        video_path = "/zhome/08/3/117881/MastersThesis/data/DISFA/Video_RightCamera"
        pickles_path = "/zhome/08/3/117881/MastersThesis/Data Processing/pickles"
    else:
        video_path = "/Users/DG/Documents/PasswordProtected/DISFA/Video_RightCamera"
        pickles_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/Data Processing/pickles"

    # Fetch paths
    vid_list = sorted(fetch_vid_files(video_path))

    # Generate image dictionaries
    print("Generation started....")
    Parallel(n_jobs = -1, verbose = 10)(delayed(convert_vid_to_array)(file) for file in vid_list)
    print("Generation done!!!")

# %%
