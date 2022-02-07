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
import matplotlib.gridspec as gridspec
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count, Process
from utils import compress_pickle, decompress_pickle

from local_fun import get_landmarks_mp

#%%
df = pd.read_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_collective")

def get_slice(land):
    xmax,xmin = int(np.array(land[0])[:,0].max()) + 20, int(np.array(land[0])[:,0].min()) - 20
    ymax,ymin = int(np.array(land[0])[:,1].max()) + 20, int(np.array(land[0])[:,1].min())
    return [ymin,ymax,xmin,xmax]
#%%
# Load data
for subject in ["01", "02", "03", "09", "10", "13", "15", "17", "18", "20"]:
    for session in ["01"]:
        for task in ["1"]:
            try: # loads from file where no mistakes were made
                data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-{task}_video_0.0.npy", allow_pickle=True)
            except:# loads from file where recording had to be restarted
                data = np.load(f"/Users/DG/Documents/PasswordProtected/speciale_outputs/{subject}_{session}/{subject}_ses:{session}_N-Back-{task}_video_1.0.npy", allow_pickle=True)

            # Determine frames
            mask = (df.Session == int(session)) & (df.Task == int(task)) & (df.ID == int(subject))
            # Emotional categories
            ems = ["happiness", "sadness", "disgust", "surprise", "anger", "fear"]

            # Find and frames
            frames_dict= dict()
            img_number = 1
            for em_mask in ["happiness", "sadness", "disgust", "surprise", "anger", "fear"]:
                frames = []
                frame = df[(mask) & (df[em_mask] == 1)]["Vid_idx"]
                frame = [x*6 for x in frame]
                if len(frame) >= img_number:
                    frames.append(frame[:img_number])
                elif len(frame) < img_number:
                    for l in range(img_number):
                        try:
                            frames.append(frame[l])
                        except:
                            frames.append(0)
                frames_dict[em_mask] = np.array(frames).ravel()

            #frames = np.array(frames).ravel()
            #%%
            fig_factor = 2
            fig = plt.figure(tight_layout=True, figsize=(8,4))
            gs = fig.add_gridspec(nrows=1, ncols=6, left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)

            for i, key in enumerate(frames_dict):
                for j, frame in enumerate(frames_dict[key]):
                    ax = fig.add_subplot(gs[j,i])
                    if frame == 0:
                        ax.imshow(np.zeros([300,300]), cmap="gray")
                    else:
                        try:
                            land = get_landmarks_mp(data[frame])
                            r_list = get_slice(land)
                            img = data[frame][r_list[0]:r_list[1],r_list[2]:r_list[3]]
                            plot_img = cv2.resize(img, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
                            ax.imshow(plot_img, cmap="gray")
                        except:
                            ax.imshow(np.zeros([300,300]), cmap="gray")
                    ax.tick_params(axis=u'both', which=u'both',length=0)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if j == 0:
                        ax.set_title(ems[i])
                    fig.subplots_adjust(wspace=0)

            fig.savefig(f"/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/emotion_grids/grid_{subject}_{session}_{task}")

            #%%
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


    # %%

    # %%
