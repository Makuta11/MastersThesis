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

from local_fun import get_landmarks_mp

def plot_mp_landmarks(landmarks, contors = None, annotate=False, img_dir=None):
    _, ax = plt.subplots(figsize=(15,18))
    if type(img_dir) == str:
        img = Image.open(img_dir)
        ax.imshow(img)
    elif img_dir is not None:   
        img = img_dir
        ax.imshow(img, cmap="gray")
    else:
        ax.invert_yaxis()
    if annotate == True:
        for i, landmark in enumerate(landmarks):
            ax.scatter(landmark[0],landmark[1],color='#39ff14', s =0.5)
            if i%409 == 0 or i%185 == 0:
                ax.annotate(str(i), (landmark[0], landmark[1]), fontsize=5)
    if contors:
        landmarks = np.array(landmarks)
        for i, contor in enumerate(contors):
            x, y, _ = zip(*landmarks[np.array(contor)])
            ax.scatter(np.array(x), np.array(y),color="#39ff14", s = 0.5)

def plot_img_w_landmarks(image):
    landmarks, _ = get_landmarks_mp(image)
    subset_contours = list(np.load("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/subset_contors.npy", allow_pickle=True))
    plot_mp_landmarks(landmarks, contors = subset_contours, annotate= False, img_dir=image)

#%%
# Load data
subject = "01"
session = "01"
task = "1"
data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-{task}_video_0.0.npy", allow_pickle=True)
df = pd.read_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_collective")

#%%
# Determine frames
mask = (df.Session == int(session)) & (df.Task == int(task)) & (df.ID == int(subject))

# Emotion codings
happiness = (df.AU12 == 1) & (df.AU6 == 1)
sadness = (df.AU4 == 1) & (df.AU15 == 1)
disgust = (df.AU9 == 1) & (df.AU15 == 1) & (df.AU17 == 1)
surprise = (df.AU2 == 1) & (df.AU5 == 1) & (df.AU26 == 1)
anger = (df.AU4 == 1) & (df.AU5 == 1) # There are the only ones we have labeled
fear = (df.AU2 == 1) & (df.AU4 == 1) & (df.AU5 == 1) & (df.AU20 == 1) & (df.AU26 == 1)

# Choose emotions to look for
emotions = (happiness) | (sadness) | (surprise)

# Find and frames
frames = df[(mask) & (happiness)]["Vid_idx"]
frames = [x*6 for x in frames]

# Plot frames
plt.figure(figsize=(15,12))
marks = False

for i in frames:
    plt.figure(figsize=(10,8))
    if marks: 
        plot_img_w_landmarks(data[i])
    else:
        plt.imshow(data[i], cmap="gray")
    plt.show()

# %%
#np.save("/Users/DG/Documents/PasswordProtected/speciale_outputs/mikkel.npy", data[10])

# %%
