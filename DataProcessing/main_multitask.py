import os, sys, time, torch, pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.models import *
from src.dataloader import *
from src.train_functions import *
from sklearn.model_selection import KFold
from src.generate_feature_vector import decompress_pickle, compress_pickle

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

# users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

print("Loading Dataset")
t = time.time()

# Data parameters
num_AU = 12
num_intensities = 5 # 4 levels and an inactive level
batch_size = 32

# Subject split
user_train_val = np.array([2,3,4,6,7,8,10,11,16,17,18,21,23,24,25,26,27,28,30,32])
user_test = np.array([1,5,9,12,13,29,31])

# Data loading
data_test, data_train, labels_test, labels_train = load_data(user_train_val, user_test)

train_val_dataset = ImageTensorDatasetMultitask(data_train, labels_train)
test_dataset = ImageTensorDatasetMultitask(data_test, labels_test)

train_val_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Network Parameters
FC_HIDDEN_DIM_1 = 2**10
FC_HIDDEN_DIM_2 = 2**12
FC_HIDDEN_DIM_3 = 2**12
FC_HIDDEN_DIM_4 = 2**10
FC_HIDDEN_DIM_5 = 2**8

# Training Parameters
EPOCHS = 100
SAVE_FREQ = 10
DROPOUT_RATE = 0
LEARNING_RATE = 1e-3
DATA_SHAPE = train_val_dataset.__nf__()

# Logging Parameters
save_path = "checkpoints/"
logdir = 'logs/'

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Multitask(DATA_SHAPE, num_AU, num_intensities, FC_HIDDEN_DIM_1, FC_HIDDEN_DIM_2, FC_HIDDEN_DIM_3, 
                FC_HIDDEN_DIM_4, FC_HIDDEN_DIM_5, DROPOUT_RATE).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #TODO weight decay
criterion = MultiTaskLossWrapper(model, task_num= 12 + 1)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

os.makedirs(save_path, exist_ok=True)
#logger = SummaryWriter(logdir)

# Run training
loss_collect, model = train_model(model, optimizer, criterion, EPOCHS, train_val_dataloader, device, save_path=save_path, save_freq=SAVE_FREQ)

# Save text files with loss
np.savetxt('loss_collect_test.txt', loss_collect)
