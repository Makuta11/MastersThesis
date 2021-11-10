import os, sys, time, torch, pickle, tqdm

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models import *
from src.dataloader import *
from src.train_functions import *
from src.generate_feature_vector import decompress_pickle, compress_pickle

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

# users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

print("Loading Dataset")
t = time.time()

# Data parameters
num_AU = 12
num_intensities = 5 # 4 levels and an inactive level
batch_size = 32

# Subject split
user_train = np.array([2,3,4,6,7,8,10,11,16,17,18,21,23,24,25,26,27,28,30,32])
user_val = np.array([3,7,11,9,31])
user_test = np.array([1,5,12,13,29])

# Data loading
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test)

train_dataset = ImageTensorDatasetMultitask(data_train, labels_train)
val_dataset = ImageTensorDatasetMultitask(data_val, labels_val)
test_dataset = ImageTensorDatasetMultitask(data_test, labels_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Network Parameters
FC_HIDDEN_DIM_1 = 2**10
FC_HIDDEN_DIM_2 = 2**12
FC_HIDDEN_DIM_3 = 2**12
FC_HIDDEN_DIM_4 = 2**10
FC_HIDDEN_DIM_5 = 2**8

# Training Parameters
EPOCHS = 10
SAVE_FREQ = 10
DROPOUT_RATE = 0
LEARNING_RATE = 1e-3
DATA_SHAPE = train_dataset.__nf__()

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
model, loss_collect, val_loss_collect = train_model(model, optimizer, criterion, EPOCHS, train_dataloader, val_dataloader, device, save_path=save_path, save_freq=SAVE_FREQ)

# Save train, val loss
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
ax.semilogy(np.arange(EPOCHS), loss_collect, color="blue", linewidth="3")
ax.semilogy(np.arange(EPOCHS), val_loss_collect, color="orange", linewidth="3")
ax.set_title("Training & Validation loss")
ax.set_xlabel("Epochs")
plt.savefig("logs/train_val_loss.png", dpi=300)

# Save text files with loss
np.savetxt('loss_collect_test.txt', loss_collect)
