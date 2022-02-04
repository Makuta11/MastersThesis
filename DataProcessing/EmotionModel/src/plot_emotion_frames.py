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
subject = "13"
session = "02"
task = "1"
try:
    data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-{task}_video_0.0.npy", allow_pickle=True)
except:
    data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-{task}_video_1.0.npy", allow_pickle=True)

#%%
df = pd.read_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_collective")

#%%

# %%
# Determine frames
mask = (df.Session == int(session)) & (df.Task == int(task)) & (df.ID == int(subject))

# Emotional categories
ems = ["happiness", "sadness", "disgust", "surprise", "anger", "fear", "AU12"]

# Emotion codings
happiness = (df.AU12 == 1) & (df.AU6 == 1)
sadness = (df.AU4 == 1) & (df.AU15 == 1) & (df.AU1 == 1) & (df.AU17 == 1)
disgust = (df.AU9 == 1) & (df.AU15 == 1) & (df.AU17 == 1) & (df.AU4 == 1) #missing 10
surprise = (df.AU2 == 1) & (df.AU1 == 1) & (df.AU26 == 1) & (df.AU5 == 1)
anger = (df.AU4 == 1) & (df.AU17== 1) & (df.AU5 == 1) #Missing 7 and 23
fear = (df.AU2 == 1) & (df.AU4 == 1) & (df.AU1 == 1) & (df.AU5 == 1) & (df.AU20 == 1) & (df.AU26 == 1)

# Find and frames
frames = []
for em_mask in [happiness, sadness, disgust, surprise, anger, fear]:
    frame = df[(mask) & (em_mask)]["Vid_idx"]
    frame = [x*6 for x in frame]
    if len(frame) > 3:
        frames.append(frame[:3])
    else:
        frames.append(frame)

frames = np.array(frames).ravel()

# Plot frames
plt.figure(figsize=(15,12))
marks = False

ems = ["happiness", "sadness", "disgust", "surprise", "anger", "fear", "AU12"]
for l, ls in enumerate(frames):
    print(f"Images for {ems[l]}")
    for frame in ls:
        plt.figure(figsize=(10,8))
        if l == 6:
            break
        if marks: 
            plot_img_w_landmarks(data[frame])
        else:
            plt.imshow(data[frame], cmap="gray")
        plt.show()

# %%


#def avg_perf_scores(df, stim, nback, session = None, IDprint = None):

stat_dict = {"ID": [], "Stim": [], "Period": [],"Task": [], "Nback": [], "happiness": [], "sadness": [], "disgust": [], "surprise": [], "anger": [], "fear": [], "AU12": []}

for stim in [1,2]:
    for nback in range(1,4):
        for session in [1,2]:
            mask = (df.Stim == stim) & (df.Nback == nback)

            for ID in df.ID.unique():
                for em in ems:
                    stat_dict[em].append(df[(mask) & (df.Task == 1) & (df.ID == ID)][em].sum())
                    stat_dict[em].append(df[(mask) & (df.Task == 2) & (df.ID == ID)][em].sum())
                
                # append to dictionary
                stat_dict["ID"].append(ID)
                stat_dict["ID"].append(ID)
                stat_dict["Stim"].append(stim - 1)
                stat_dict["Stim"].append(stim - 1)
                stat_dict["Period"].append(session)
                stat_dict["Period"].append(session)
                stat_dict["Task"].append(1)
                stat_dict["Task"].append(2)
                stat_dict["Nback"].append(nback)
                stat_dict["Nback"].append(nback)

stat_dict_2 = {"ID": [], "Stim": [], "Period": [], "Nback": [], "happiness": [], "sadness": [], "disgust": [], "surprise": [], "anger": [], "fear": [], "AU12": []}

for stim in [1,2]:
    for nback in range(1,4):
        for session in [1,2]:
            mask = (df.Stim == stim) & (df.Nback == nback)

            for ID in df.ID.unique():
                for em in ems:
                    t1 = df[(mask) & (df.Task == 1) & (df.ID == ID)][em].sum()
                    t2 = df[(mask) & (df.Task == 2) & (df.ID == ID)][em].sum()
                    stat_dict_2[em].append(t2-t1)
                
                # append to dictionary
                stat_dict_2["ID"].append(ID)
                stat_dict_2["Stim"].append(stim - 1)
                stat_dict_2["Period"].append(session)
                stat_dict_2["Nback"].append(nback)

df_emotion_4obs = pd.DataFrame(stat_dict)
df_emotion_diff = pd.DataFrame(stat_dict_2)

df_emotion_4obs.to_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_em_4obs")
df_emotion_diff.to_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_em_diff")
# %%
