import torch
from torch import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class Multitask(nn.Module):

    def __init__(self, data_shape, num_AU, num_intensities, fc_hidden_dim_1, fc_hidden_dim_2,
                 fc_hidden_dim_3, fc_hidden_dim_4, fc_hidden_dim_5, dropout_prop):
        super(Multitask, self).__init__()

        #TODO: add uncertainty weights (self.stop_weights = stop_weights)

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
            nn.BatchNorm1d(fc_hidden_dim_4),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_4, out_features = fc_hidden_dim_5)
        )
        
        self.fc_layer_AU = nn.Sequential(
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_AU)
        )

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

        self.sigm = nn.Sigmoid()


        """ Implement for loss on intensity
          self.fc_purpose = nn.Sequential(
              nn.BatchNorm1d(2 * lstm_hidden_dim * seq_length + 5),
              nn.ReLU(),
              nn.Dropout(p = dropout_prop),
              nn.Linear(in_features = 2 * lstm_hidden_dim * seq_length+ 5, out_features = fc_hidden_dim),
              nn.BatchNorm1d(fc_hidden_dim),
              nn.ReLU(),
              nn.Dropout(p = dropout_prop),
              nn.Linear(in_features = fc_hidden_dim, out_features = num_purpose)
          )

          self.fc_mode = nn.Sequential(
              nn.BatchNorm1d(2 * lstm_hidden_dim * seq_length + 5),
              nn.ReLU(),
              nn.Dropout(p = dropout_prop),
              nn.Linear(in_features = 2 * lstm_hidden_dim * seq_length + 5, out_features = fc_hidden_dim),
              nn.BatchNorm1d(fc_hidden_dim),
              nn.ReLU(),
              nn.Dropout(p = dropout_prop),
              nn.Linear(in_features = fc_hidden_dim, out_features = num_mode)
          )
        """

        
    def forward(self, data):
        batch_size = data.shape[0]

        #Shared architecture
        X_shared = self.fc_layer(data)

        #fully connected layer for multi-label classification (level 1: 1 group, 12 targets)
        X = self.fc_layer_AU(X_shared)
        
        #fully connecetd layers for multi-class classification (level 2: 12 groups, 4 targets)
        AU1 = self.fc_layer_AU1(X_shared)
        AU2 = self.fc_layer_AU2(X_shared)
        AU4 = self.fc_layer_AU4(X_shared)
        AU5 = self.fc_layer_AU5(X_shared)
        AU6 = self.fc_layer_AU6(X_shared)
        AU9 = self.fc_layer_AU9(X_shared)
        AU12 = self.fc_layer_AU12(X_shared)
        AU15 = self.fc_layer_AU15(X_shared)
        AU17 = self.fc_layer_AU17(X_shared)
        AU20 = self.fc_layer_AU20(X_shared)
        AU25 = self.fc_layer_AU25(X_shared)
        AU26 = self.fc_layer_AU26(X_shared)

        return [self.sigm(X),[AU1, AU2, AU4, AU5, AU6, AU9, AU12, AU15, AU17, AU20, AU25, AU26]]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_num = 1 + 12):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_sigmas = nn.Parameter(torch.zeros((task_num)))

    def forward(self, data, AUs, AU_intensities, device):

        out_AU, out_AU_intensities = data[0], data[1]
            
        # Calculate loss for the multi-label classification of identifying if AU is present in image
        AU_loss = F.binary_cross_entropy(out_AU, AUs)#, weight= self.model.stop_weights) #TODO: add class weight
        loss_collect = torch.exp(-self.log_sigmas[0])*AU_loss + self.log_sigmas[0] #TODO: add uncertainty weights

        # Calculate loss for the intensity of the AUs present in the image
        for i, lab in enumerate(AU_intensities.permute(1,0,2)):
            au_tmp_loss = F.cross_entropy(out_AU_intensities[i], lab) 
            loss_collect += torch.exp(-self.log_sigmas[i])*au_tmp_loss + self.log_sigmas[i] #TODO: add uncertainty weights
        """ 
            This needs to be added for monitoring the progression all individual losses
            else:
                au_tmp_loss = torch.Tensor(0).to(device)
        """

        return loss_collect, self.log_sigmas.data.tolist()