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
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.datasets import make_multilabel_classification

# users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

# Debugging/model test parameter
train = True
evaluate = True
model_path = ""

# Data parameters
aus = [1,2,4,5,6,9,12,15,17,20,25,26]
num_AU = 12
num_intensities = 6 # 5 different intensity levels (when active)

# Subject split
user_train = np.array([1,2,4,6,8,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_val = np.array([3,5,7,9,12,13])
user_test = np.array([5])

# Data loading
print("Loading Dataset")
t = time.time()
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
print(f"It took {time.time() - t} seconds to load the data")

# Train, Val, Test split
train_dataset = ImageTensorDatasetMultitask(data_train, labels_train)
val_dataset = ImageTensorDatasetMultitask(data_val, labels_val)
test_dataset = ImageTensorDatasetMultitask(data_test, labels_test)

# Create collective loss figure across validation loops
plt.style.use('fivethirtyeight')
fig_tot, ax_tot = plt.subplots(figsize=(10,12))

# CV test on bactch size
for k, BATCH_SIZE in enumerate([64]):

    # Place in dataloaders for ease of retrieval
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 2)

    #Calcualte class weights within AUs - for cross entropy loss
    class_weights_AU = get_class_weights_AU(labels_train)
    class_weights_int = get_class_weights_AU_int(labels_train)

    # Clear up memory space
    del data_test, data_train, data_val

    # Network Parameters (subject to change)
    #FC_HIDDEN_DIM_1 = 2**8
    #FC_HIDDEN_DIM_2 = 2**6
    #FC_HIDDEN_DIM_3 = 2**5
    #FC_HIDDEN_DIM_4 = 2**9 # currently not used
    #FC_HIDDEN_DIM_5 = 2**4 # dimension in final fc_layers for classification

    FC_HIDDEN_DIM_1 = 2**10
    FC_HIDDEN_DIM_2 = 2**8
    FC_HIDDEN_DIM_3 = 2**6
    FC_HIDDEN_DIM_4 = 2**5
    FC_HIDDEN_DIM_5 = 2**4

    # Training Parameters
    if sys.platform == "linux":
        EPOCHS = 600
    else:
        EPOCHS = 30
    SAVE_FREQ = 10
    DATA_SHAPE = train_dataset.__nf__()

    # Logging Parameters
    today = str(datetime.datetime.now())
    if sys.platform == "linux":
        os.makedirs(f"/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/logs/{today[:19]}")
        os.makedirs(f"/work3/s164272/{today[:19]}")
        save_path = f"/work3/s164272/{today[:19]}"
    else:
        save_path = "localOnly"
        logdir = 'logs'
        os.makedirs(f'{save_path}/{today[:19]}')

    # CV testing for LR, DR, and WD
    for i, LEARNING_RATE in enumerate([1e-5]):
        for j, DROPOUT_RATE in enumerate([0.5]):
            for k, WEIGHT_DECAY in enumerate([1e-3]):
                
                # Name for saving the model
                name = f'B:{BATCH_SIZE}_DR:{DROPOUT_RATE}_LR:{LEARNING_RATE}_WD:{WEIGHT_DECAY}'

                # Device determination - allows for same code with and without access to CUDA (GPU)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                # Model initialization
                model = Multitask(DATA_SHAPE, num_AU, num_intensities, FC_HIDDEN_DIM_1, FC_HIDDEN_DIM_2, FC_HIDDEN_DIM_3, 
                                FC_HIDDEN_DIM_4, FC_HIDDEN_DIM_5, DROPOUT_RATE).to(device)
                
                # Load model if script is run for evaluating trained model
                if not train:
                    if torch.cuda.is_available():
                        model.load_state_dict(torch.load(model_path), strict = False)
                    else:
                        model.load_state_dict(torch.load(model_path, map_location=device))
                
                # Initialize model with loss wrapper for multitask learning
                model = MultiTaskLossWrapper(model, task_num = 13, cw_AU = class_weights_AU.to(device), cw_int = class_weights_int.to(device))
                
                # Optimization parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
                #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150], gamma = 0.1)

                if train:
                    # Run training
                    model, loss_collect, val_loss_collect, sigma_collect = train_multitask_model(model, optimizer, EPOCHS, train_dataloader, val_dataloader, device, save_path=save_path, save_freq=SAVE_FREQ, name=name, scheduler=None)

                    # Plot each individual figure
                    plt.style.use('fivethirtyeight')
                    fig, ax = plt.subplots(figsize=(10,12))
                    ax.semilogy(np.arange(EPOCHS), loss_collect, color="blue", linewidth="3", label="train_loss")
                    ax.semilogy(np.arange(EPOCHS), val_loss_collect, color="orange", linewidth="3", label="val_loss", linestyle="dashed")
                    ax.set_title(f"BS:{BATCH_SIZE}, LR:{LEARNING_RATE}, DR:{DROPOUT_RATE}")
                    ax.set_xlabel("Epochs")
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    # Plot each sigma figure
                    plt.style.use('fivethirtyeight')
                    fig_uw, ax_uw = plt.subplots(figsize=(10,12))
                    for i, au in enumerate(np.append(["AUs"], aus)):
                        if i == 0:
                            ax_uw.plot(np.arange(EPOCHS), np.exp(-sigma_collect[:,i]) , color = "magenta", linewidth="3", label=f"AUs Overall")
                        elif i >= 7:
                            ax_uw.plot(np.arange(EPOCHS), np.exp(-sigma_collect[:,i]) , linewidth="3", label=f"AU{au}")
                        else:
                            ax_uw.plot(np.arange(EPOCHS), np.exp(-sigma_collect[:,i])  , linewidth="3", label=f"AU{au}", linestyle="dashed")
                    ax_uw.set_title(f"Uncertainty Weights - BS:{BATCH_SIZE}, LR:{LEARNING_RATE}, DR:{DROPOUT_RATE}")
                    ax_uw.set_xlabel("Epochs")
                    ax_uw.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                    # Plot on collective figure
                    ax_tot.semilogy(np.arange(EPOCHS), loss_collect, linewidth="3", label = f"train_Dr:{DROPOUT_RATE}_Lr:{LEARNING_RATE}")
                    ax_tot.semilogy(np.arange(EPOCHS), val_loss_collect, linewidth="3", label = f"val_Dr:{DROPOUT_RATE}_Lr:{LEARNING_RATE}", linestyle="dashed")
                    ax_tot.set_title(f"BS:{BATCH_SIZE}, LR:{LEARNING_RATE}, DR:{DROPOUT_RATE}")
                    ax_tot.set_xlabel("Epochs")
                    ax_tot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    
                    # Create output directory for images
                    if sys.platform == 'linux':
                        fig.savefig(f"logs/{today[:19]}/TrVal_fig_{name}.png", dpi=128, bbox_inches='tight')
                        fig_uw.savefig(f"logs/{today[:19]}/UW_fig_{name}.png", dpi=128, bbox_inches='tight')
                    else:
                        fig.savefig(f"{save_path}/{today[:19]}/TrVal_fig_{name}.png", dpi=128, bbox_inches='tight')
                        fig_uw.savefig(f"{save_path}/{today[:19]}/UW_fig_{name}.png", dpi=128, bbox_inches='tight')
                    
                if evaluate:
                    # Test model performance on given dataloaders
                    for dataloader in [train_dataloader, val_dataloader]:
                        AU_scores, intensity_scores = get_predictions_multitask(model, dataloader, device)

                        # Print scores
                        print(f'n\{name}:')
                        print(f'\nScores on AU identification:\n{val_scores(AU_scores[0], AU_scores[1])}')
                        print("\nScores on AU intensities")
                        for au in aus:
                            pred = intensity_scores[f'AU{au}']["pred"]
                            true = intensity_scores[f'AU{au}']["true"]
                            if len(true) > 0 or len(pred) > 0:
                                print(f'AU{au}:\n{val_scores(true,pred)}')
                
                # Clear up memory and reset individual figures
                del model, loss_collect, val_loss_collect, fig, ax

# Save collective plot
if sys.platform == 'linux':
    fig_tot.savefig(f"logs/{today[:19]}/TrVal_fig_tot_{name}.png", dpi=128, bbox_inches='tight')
else:
    fig_tot.savefig(f"{save_path}/{today[:19]}/TrVal_fig_tot_{name}.png", dpi=128, bbox_inches='tight')
