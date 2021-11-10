import os, sys, time, torch, pickle

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#from src.models import *
from src.dataloader import *
from sklearn.model_selection import KFold
from src.generate_feature_vector import decompress_pickle, compress_pickle

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

# users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

print("Loading Dataset")
t = time.time()

# Data parameters
BATCH_SIZE = 32

# Subject split
user_train_val = np.array([2,3,4,6,7,8,10,11,16,17,18,21,23,24,25,26,27,28,30,32])
user_test = np.array([1,5,9,12,13,29,31])

# Data loading
data_test, data_train, labels_test, labels_train = load_data(user_train_val, user_test)

train_val_dataset = ImageTensorDatasetMultitask(data_train, labels_train)
test_dataset = ImageTensorDatasetMultitask(data_test, labels_test)

train_val_dataloader = torch.utils.data.DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=-1)

# Network Parameters
FC_HIDDEN_DIM_1 = 2**10
FC_HIDDEN_DIM_2 = 2**12
FC_HIDDEN_DIM_3 = 2**12
FC_HIDDEN_DIM_4 = 2**10
FC_HIDDEN_DIM_5 = 2**8

# Training Parameters
EPOCHS = 100
LEARNING_RATE = 1e-3
SAVE_FREQ = 10

save_path = "checkpoints/"
logdir = 'logs/'


# Train setup
model.train()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

os.makedirs(save_path, exist_ok=True)
logger = SummaryWriter(logdir)

# Run training
epoch = 0
iteration = 0
while True:
    batch_losses = []
    for imgs, targets, _ in train_val_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()

        model_result = model(imgs)
        loss = criterion(model_result, targets.type(torch.float))

        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()

        logger.add_scalar('train_loss', batch_loss_value, iteration)
        batch_losses.append(batch_loss_value)
        with torch.no_grad():
            result = calculate_metrics(model_result.cpu().numpy(), targets.cpu().numpy())
            for metric in result:
                logger.add_scalar('train/' + metric, result[metric], iteration)