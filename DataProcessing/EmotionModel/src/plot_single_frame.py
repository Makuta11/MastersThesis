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
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, Process
from utils import compress_pickle, decompress_pickle


subject = "13"
session = "01"

data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-1_video_0.0.npy", allow_pickle=True)

#%%


plt.imshow(data[100], cmap="gray")

np.save("/Users/DG/Documents/PasswordProtected/speciale_outputs/mikkel.npy", data[10])
# %%
