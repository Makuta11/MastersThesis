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

# Model path
model_path = "DataProcessing/EmotionModel/src/assets/models/" # for loading in pre-trained models

# Action Unit to investigate
num_intensities = 2
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Load data
data_dir = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/pickles/video_features/01_01_N-Back-1.npy"
data = load_data_for_eval(data_dir)

# Network Parameters
FC_HIDDEN_DIM_1 = 2**8
FC_HIDDEN_DIM_2 = 2**6
FC_HIDDEN_DIM_3 = 2**5
FC_HIDDEN_DIM_4 = 2**4
DROPOUT_RATE = 0.5
DATA_SHAPE = 6211

# Device determination - allows for same code with and without access to CUDA (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model initialization
model = SingleClassNetwork(DATA_SHAPE, num_intensities, FC_HIDDEN_DIM_1, FC_HIDDEN_DIM_2, FC_HIDDEN_DIM_3, FC_HIDDEN_DIM_4, DROPOUT_RATE).to(device)

# Load model if script is run for evaluating trained model
if not train:
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path), strict = False)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

if evaluate:
    # Test model performance on given dataloaders
    for i, dataloaders in enumerate([train_dataloader, val_dataloader]):
        AU_scores = get_single_predictions(model, au_idx, dataloaders, device)





