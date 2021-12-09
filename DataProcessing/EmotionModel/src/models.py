import torch

from torch import utils
from torch import nn

import torch.nn.functional as F
import torch.optim as optim

class MultiLabelClassifier(nn.Module):

    def __init__(self, data_shape, num_AU, fc_hidden_dim_1, fc_hidden_dim_2, fc_hidden_dim_3, fc_hidden_dim_4, dropout_prop):
        super(MultiLabelClassifier, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(data_shape),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = data_shape, out_features = fc_hidden_dim_1),
            nn.BatchNorm1d(fc_hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_1, out_features = fc_hidden_dim_2),
            nn.BatchNorm1d(fc_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_2, out_features = fc_hidden_dim_3),
            nn.BatchNorm1d(fc_hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_3, out_features = fc_hidden_dim_4),
            #nn.BatchNorm1d(fc_hidden_dim_4),
            #nn.ReLU(),
            #nn.Dropout(p = dropout_prop),
            #nn.Linear(in_features = fc_hidden_dim_4, out_features = fc_hidden_dim_5)
        )
        
        # Fully conencted layer for multi-label classification of AU being present in each image
        self.fc_layer_AU = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_4),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_4, out_features = num_AU)
        )
    
    def forward(self, data):
        batch_size = data.shape[0]

        # Body encorder for the netoworks
        X_shared = self.fc_layer(data)

        #fully connected layer for multi-label classification (12 possible targets)
        X = self.fc_layer_AU(X_shared)

        return X

class Multitask(nn.Module):

    def __init__(self, data_shape, num_AU, num_intensities, fc_hidden_dim_1, fc_hidden_dim_2,
                 fc_hidden_dim_3, fc_hidden_dim_4, fc_hidden_dim_5, dropout_prop):
        super(Multitask, self).__init__()

        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(data_shape),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = data_shape, out_features = fc_hidden_dim_1),
            nn.BatchNorm1d(fc_hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_1, out_features = fc_hidden_dim_2),
            nn.BatchNorm1d(fc_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_2, out_features = fc_hidden_dim_3),
            nn.BatchNorm1d(fc_hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_3, out_features = fc_hidden_dim_5),
            #nn.BatchNorm1d(fc_hidden_dim_4),
            # nn.ReLU(),
            # nn.Dropout(p = dropout_prop),
            # nn.Linear(in_features = fc_hidden_dim_4, out_features = fc_hidden_dim_5)
        )
        
        # Fully conencted layer for multi-label classification of AU being present in each image
        self.fc_layer_AU = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_AU)
        )

        # Fully connected layers for each AU Intensity determining network
        self.fc_layer_AU1 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU2 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU4 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU5 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU6 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU9 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU12 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU15 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU17 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU20 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU25 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )
        self.fc_layer_AU26 = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_intensities)
        )

    def forward(self, data):
        batch_size = data.shape[0]

        # Shared body encorder for the netowork
        X_shared = self.fc_layer(data)

        #fully connected layer for multi-label classification (level 1: 1 group, 12 targets)
        X = self.fc_layer_AU(X_shared)
        
        #fully connecetd layers for multi-class classification (level 2: 12 groups, 5 targets)
        AU_intensities= [F.softmax(self.fc_layer_AU1(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU2(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU4(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU5(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU6(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU9(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU12(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU15(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU17(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU20(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU25(X_shared), dim = 1),
                            F.softmax(self.fc_layer_AU26(X_shared), dim = 1)]

        return [X, AU_intensities]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_num, cw_AU, cw_int):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num

        # Class weights given as inputs to each lossfunction to counter class imbalance
        self.cw_AU = cw_AU
        self.cw_int = cw_int
        
        # Initialize log_sigmas for uncertainty weighting
        self.log_sigmas = nn.Parameter(torch.ones((task_num)))

    def forward(self, data, AUs, AU_intensities, device):
        
        # Retrieve output from the forward pass of the model
        out_AU, out_AU_intensities = self.model(data)

        # BCEWithLogitsLoss combines a sigmoid layer and binary cross entropy to a single class. This also allows for the addition of class weighting and increases numerical stability by taking advantage of the log-sum-exp trick
        bce = nn.BCEWithLogitsLoss(pos_weight = self.cw_AU)
        
        # Defined for experimental changes to loss function - (may not be used in final version - currently not in use)
        rel = nn.ReLU()

        # Calculate loss for the multi-label classification of identifying if AU is present in image
        AU_loss = bce(out_AU, AUs)
        loss_collect = 0.5 * torch.exp(-2*self.log_sigmas[0]) * AU_loss + (self.log_sigmas[0])
        #loss_collect = torch.exp(-self.log_sigmas[0]) * AU_loss + (self.log_sigmas[0])

        # Calculate loss for the intensity of the AUs present in the image
        for i, lab in enumerate(AU_intensities.permute(1,0)):
            
            # Find indexes that contain the AU so each intensity determining network is trained only on images containing containing each specific AU
            AU_idx = (lab >= 1).nonzero(as_tuple=True)[0]
            if len(AU_idx) > 0:
                au_tmp_loss = F.cross_entropy(out_AU_intensities[i][AU_idx], lab[AU_idx] - 1, weight = self.cw_int[i]) #Subtract one from label to end up with 5 classes [0,1,2,3,4]
                loss_collect += 0.5 * torch.exp(-2*self.log_sigmas[i])*au_tmp_loss + (self.log_sigmas[i])
                #loss_collect += torch.exp(-self.log_sigmas[i]) * au_tmp_loss + (self.log_sigmas[i])

        # The loss of the entire network is collected in loss_collect, while the learnable weights for each individual loss is stored in log_sigmas for plotting
        return loss_collect, self.log_sigmas.data.tolist()
        