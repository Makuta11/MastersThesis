import torch
from torch import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class Multitask(nn.Module):

    def __init__(self, data, num_AU, num_intensities, fc_hidden_dim_1, fc_hidden_dim_2):
        super(Multitask, self).__init__()

        self.stop_weights = stop_weights

        self.fc_layer = nn.Sequential(
            nn.BatchNorm1d(data.shape[0]),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = data.shape[0], out_features = fc_hidden_dim_1),
            nn.BatchNorm1d(fc_hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_1, out_features = fc_hidden_dim_2)
            nn.BatchNorm1d(fc_hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_2, out_features = fc_hidden_dim_3)
            nn.BatchNorm1d(fc_hidden_dim_3),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_3, out_features = fc_hidden_dim_4)
            nn.BatchNorm1d(fc_hidden_dim_4),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_4, out_features = fc_hidden_dim_5)
            nn.BatchNorm1d(fc_hidden_dim_5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim_5, out_features = num_AU)
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

        #fully connected layer for multi-label classification (level 1: 1 group)
        X = self.fc_layer(data)
        
        #fulle connecetd layers for multi-class classification (level 2: 12 groups)
        
        intensity_list = [X_AU1 = self.fc_layer(data),
                          X_AU2 = self.fc_layer(data),
                          X_AU4 = self.fc_layer(data),
                          X_AU5 = self.fc_layer(data),
                          X_AU6 = self.fc_layer(data),
                          X_AU9 = self.fc_layer(data),
                          X_AU12 = self.fc_layer(data),
                          X_AU15 = self.fc_layer(data),
                          X_AU17 = self.fc_layer(data),
                          X_AU20 = self.fc_layer(data),
                          X_AU25 = self.fc_layer(data),
                          X_AU26 = self.fc_layer(data)]


        #TODO this needs to be extended to supper hierarchical multi-task multi-label learning
        return [self.sigm(X), intensity_list]

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_num = 1 + 12):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_sigmas = nn.Parameter(torch.zeros((task_num)))

    def forward(self, data, label, device):

        out_AU, out_AU_intensities = self.model(data)
            
        AU_loss = F.binary_cross_entropy(out_AU, label, weight= self.model.stop_weights) #TODO: add class weight
        loss_collect = torch.exp(-self.log_sigmas[0])*stop_loss + self.log_sigmas[0]/2

        for i, lab in enumerate(label):
          if lab != 0:
            au_tmp_loss = F.cross_entropy(out_AU_intensities[i], label[i]) 
            loss_collect += torch.exp(-self.log_sigmas[i])*au_tmp_loss + self.log_sigmas[0]/12
          else:
            au_tmp_loss = torch.Tensor(0).to(device)

        return loss_collect, mode_loss, purpose_loss, stop_loss, self.log_sigmas.data.tolist()