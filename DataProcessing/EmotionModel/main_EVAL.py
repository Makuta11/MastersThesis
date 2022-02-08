import os, sys, time, torch, pickle, tqdm, datetime

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models import *
from src.dataloader import *
from src.train_functions import *
from src.validation_score import *
from src.utils import decompress_pickle, compress_pickle
from src.utils import get_class_weights_AU, get_class_weights_AU_int

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_multilabel_classification

def single_subj_frame(ID, session, task, idx):
    """
        This fuction generates a DataFrame for a single ID, session and task
    """
    
    stimKey = [1,1,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0]

    if session == 1 and stimKey[ID - 1] == 1:
        stim = [2]*900
    elif session == 2 and stimKey[ID - 1] == 0:
        stim = [2]*900
    else:
        stim = [1]*900

    Nback = np.append([[1]*300, [2]*300], [3]*300)

    data = {
        'Vid_idx': np.array([np.arange(idx[0],idx[0]+300),np.arange(idx[1],idx[1]+300),np.arange(idx[2],idx[2]+300)]).ravel(),
        'ID': [ID]*900,
        'Session': [session]*900,
        'Task': [task]*900,
        'Stim': stim, 
        'Nback': Nback
    }

    return pd.DataFrame(data)

# Video parameters
frame_rate = 5  #1/s
test_duration = 60 #s

# Models dir
model_dir = "DataProcessing/EmotionModel/src/assets/models/" # for loading in pre-trained models

# Import timestamps for n-back tests
df_timestamps = pd.read_csv("DataProcessing/EmotionModel/src/assets/df_timestamps")

# Network Parameters for model reinitialization
FC_HIDDEN_DIM_1 = 2**8
FC_HIDDEN_DIM_2 = 2**6
FC_HIDDEN_DIM_3 = 2**5
FC_HIDDEN_DIM_4 = 2**4
DROPOUT_RATE = 0.5
DATA_SHAPE = 3571

# Action Unit to investigate
num_intensities = 2
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Load data
data_dir = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/video_features_real/"

# Create collective DataFrame
df_collective = pd.DataFrame()

for file in os.listdir(data_dir):

    if "misses" in file:
        continue

    # load data
    data = load_data_for_eval(f'{data_dir}{file}')

    # Extract tracability features from data
    ID = int(file[-18:-16])
    session = int(file[-15:-13])
    task = int(file[-5])

    # Compute starting index of 1- 2- and 3- backs
    mask = (df_timestamps.ID == ID) & (df_timestamps.Session == session) & (df_timestamps.Task == task) 
    idx_1back = int((df_timestamps[(mask)]["1NbackOffset"] / 1000) * frame_rate)
    idx_2back = int((df_timestamps[(mask)]["2NbackOffset"] / 1000) * frame_rate)
    idx_3back = int((df_timestamps[(mask)]["3NbackOffset"] / 1000) * frame_rate)

    # Extract only data which was recorded during the n-back test
    data_nbacks = np.vstack([data[idx_1back:idx_1back + frame_rate * test_duration], data[idx_2back:idx_2back + frame_rate * test_duration], data[idx_3back:idx_3back + frame_rate * test_duration]]) 

    # Device determination - allows for same code with and without access to CUDA (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization
    model = SingleClassNetwork(DATA_SHAPE, num_intensities, FC_HIDDEN_DIM_1, FC_HIDDEN_DIM_2, FC_HIDDEN_DIM_3, FC_HIDDEN_DIM_4, DROPOUT_RATE).to(device)

    # Generate frame for placing AU Responses
    df_tmp = single_subj_frame(ID, session, task, [idx_1back, idx_2back, idx_3back])

    # Perform classifications for one AU at a time
    for AU_model in os.listdir(model_dir):
        
        # AU key for dataframe head
        key = AU_model[:-3]

        # Load model if script is run for evaluating trained model
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(f'{model_dir}{AU_model}'), strict = False)
        else:
            model.load_state_dict(torch.load(f'{model_dir}{AU_model}', map_location=device))

        model.eval()
        out = model(torch.tensor(data_nbacks).float().to(device)).argmax(axis=1).numpy()

        df_tmp[key] = out

    df_collective = pd.concat([df_collective, df_tmp], ignore_index=True)

# Generate Emotion Columns TODO add AU1 sadness, surprise, and Fear
df_collective["happiness"] = df_collective.AU12 * df_collective.AU6
df_collective["sadness"] = df_collective.AU1 * df_collective.AU4 * df_collective.AU6 * df_collective.AU15 * df_collective.AU17
df_collective["disgust"] = df_collective.AU4 * df_collective.AU9 * df_collective.AU17
df_collective["surprise"] = df_collective.AU1 * df_collective.AU2 * df_collective.AU5 * df_collective.AU25 * df_collective.AU26
df_collective["anger"] = df_collective.AU4 * df_collective.AU17 # Missing some labels
df_collective["fear"] = df_collective.AU1 * df_collective.AU2 * df_collective.AU4 * df_collective.AU5 * df_collective.AU20 * df_collective.AU25 * df_collective.AU26 


df_collective.to_csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_collective")

# Produce specific dataframes from df_collective to be statistically analyzed in Rstudio

print("Done!!!")