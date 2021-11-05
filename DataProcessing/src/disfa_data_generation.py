#%%
import os
import cv2
import ssl
import wget
import time
import pickle
import requests
import numpy as np
import pandas as pd
import xlwings as xw
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
    return {key: frames}

video_path = "/Users/DG/Documents/PasswordProtected/DISFA/Video_RightCamera"
AU_path = "/Users/DG/Documents/PasswordProtected/DISFA/ActionUnit_Labels"
vid_list = fetch_vid_files(video_path)


dictionary_list = Parallel(n_jobs = 5, verbose = 5)(delayed(convert_vid_to_array)(file) for file in vid_list)




# %%
