import os, sys, time, torch, pickle, tqdm, datetime, optuna

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
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import make_multilabel_classification

# Define search space for hyperparameter tuning
class Objective(object):
    def __init__(self, today):
        self.today = today

    def __call__(self, trial):

        params = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-7, 1e-5),
                #'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                #'batch_size': trial.suggest_int('batch_size', 16, 256),
                'dropout_rate': trial.suggest_uniform("dropout_rate", 0.42, 0.5),
                'weight_decay': trial.suggest_loguniform("weight_decay", 1e-4, 1e-2)
                }
        
        # users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

        # Debugging/model test parameter
        train = True
        evaluate = True
        model_path = "/work3/s164272/2021-12-15_17:21:19/checkpoint_test-100-B:64_DR:0.5_LR:1e-06_WD:0.001Net256x64x32x16.pt" # for loading in pre-trained models

        # Data parameters
        aus = [1,2,4,5,6,9,12,15,17,20,25,26]
        num_intensities = 2

        # Subject split
        #user_train = np.array([1,2,4,6,8,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
        user_train = np.array([2, 3, 5, 12, 16, 23, 31, 32])
        #user_val = np.array([3,5,7,9,12,13])
        user_val = np.array([25, 28])
        user_test = np.array([25, 28])

        # Data loading
        print("Loading Dataset")
        data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)

        # Generate Train, Val, Test datasets
        train_dataset = ImageTensorDatasetMultiLabel(data_train, labels_train)
        val_dataset = ImageTensorDatasetMultiLabel(data_val, labels_val)
        test_dataset = ImageTensorDatasetMultiLabel(data_test, labels_test)

        # Action Unit to investigate
        au = 20
        au_idx = aus.index(au)

        # CV test on bactch size
        for k, BATCH_SIZE in enumerate([64]):#[params['batch_size']]):

            # Place in dataloaders for ease of retrieval
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Calcualte truncated class weights for training data
            labtrain = labels_train.iloc[:,au_idx]
            labtrain[labtrain > 0] = int(1)
            cw_int = torch.tensor(compute_class_weight(class_weight='balanced', classes=np.unique(labtrain), y=labtrain)).float()

            # Clear up memory space
            del data_test, data_train, data_val

            # Network Parameters (subject to change)
            FC_HIDDEN_DIM_1 = 2**8
            FC_HIDDEN_DIM_2 = 2**6
            FC_HIDDEN_DIM_3 = 2**5
            FC_HIDDEN_DIM_4 = 2**4

            # Training Parameters
            if sys.platform == "linux":
                EPOCHS = 200
            else:
                EPOCHS = 5
            SAVE_FREQ = 10
            DATA_SHAPE = train_dataset.__nf__()

            # CV testing for LR, DR, and WD
            for i, LEARNING_RATE in enumerate([params['learning_rate']]):
                for j, DROPOUT_RATE in enumerate([params['dropout_rate']]):
                    for k, WEIGHT_DECAY in enumerate([params['weight_decay']]):
                        
                        # Name for saving the model
                        name = f'AU{au}_B:{BATCH_SIZE}_DR:{round(DROPOUT_RATE,2)}_LR:{LEARNING_RATE}_WD:{round(WEIGHT_DECAY,4)}  Net{FC_HIDDEN_DIM_1}x{FC_HIDDEN_DIM_2}x{FC_HIDDEN_DIM_3}x{FC_HIDDEN_DIM_4}'

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
                        
                        # Initialize criterion for multi-label loss
                        criterion = nn.CrossEntropyLoss(weight = cw_int.to(device))
                        
                        # Optimization parameters
                        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay= WEIGHT_DECAY)
                        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [150], gamma = 0.1)

                        if train:
                            # Run training
                            model, loss_collect, val_loss_collect = train_single_model(model, au_idx, optimizer, criterion, EPOCHS, train_dataloader, val_dataloader, device, save_path=save_path, save_freq=SAVE_FREQ, name=name, scheduler=None)
                        
                        if evaluate:
                            # Test model performance on given dataloaders
                            for dataloaders in [train_dataloader, val_dataloader]:
                                AU_scores = get_single_predictions(model, au_idx, dataloaders, device)

                                # Print scores
                                print(f'{name}:')
                                print(f'\nTest scores on AU{au} identification:\n{classification_report(AU_scores[1], AU_scores[0], target_names=["not active", "active"])}')
                                print(f'F1_score{f1_score(AU_scores[1], AU_scores[0])}')

                            # Plot each individual figure
                            plt.style.use('fivethirtyeight')
                            fig, ax = plt.subplots(figsize=(10,12))
                            ax.semilogy(np.arange(EPOCHS), loss_collect, color="blue", linewidth="3", label="train_loss")
                            ax.semilogy(np.arange(EPOCHS), val_loss_collect, color="orange", linewidth="3", label="val_loss", linestyle="dashed")
                            ax.set_title(f'{name}    f1:{round(f1_score(AU_scores[1], AU_scores[0])*100, 2)}')
                            ax.set_xlabel("Epochs")
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                            
                            # Create output directory for images
                            if sys.platform == 'linux':
                                fig.savefig(f"logs/{self.today[:19]}/TrVal_fig_{name}_f1:{round(f1_score(AU_scores[1], AU_scores[0]),2)}.png", dpi=128, bbox_inches='tight')
                            else:
                                fig.savefig(f"{save_path}/{self.today[:19]}/TrVal_fig_{name}.png", dpi=128, bbox_inches='tight')
                            
                        
                        # Clear up memory and reset individual figures
                        del model, loss_collect, val_loss_collect, fig, ax
        
        return f1_score(AU_scores[1], AU_scores[0])

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

# Hyperparameter tuning with optuna
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(Objective(today), n_trials=30)

best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# plot
optuna.visualization.matplotlib.plot_intermediate_values(study)
plt.savefig(f"logs/{today[:19]}/optuna1.png", dpi=128, bbox_inches='tight')
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig(f"logs/{today[:19]}/optuna2.png", dpi=128, bbox_inches='tight')
optuna.visualization.matplotlib.plot_parallel_coordinate(study)
plt.savefig(f"logs/{today[:19]}/optuna3.png", dpi=128, bbox_inches='tight')
optuna.visualization.matplotlib.plot_param_importances(study)
plt.savefig(f"logs/{today[:19]}/optuna4.png", dpi=128, bbox_inches='tight')
