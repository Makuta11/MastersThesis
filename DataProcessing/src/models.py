import torch
from torch import utils
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class Multitask(nn.Module):

    def __init__(self, data, num_AU, num_intensities):
        super(Multitask, self).__init__()
        self.conv_hidden_dim = conv_hidden_dim
        self.stop_weights = stop_weights
        self.layer_img1 = nn.Sequential(
            nn.Conv3d(in_channels = 11, out_channels = self.conv_hidden_dim[0],
                                kernel_size = (1, 3, 3),
                                stride = (1, 1, 1), 
                                padding = (0, 1, 1)),
           nn.BatchNorm3d(self.conv_hidden_dim[0]),
           nn.ReLU(),  
           nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2)),
           nn.Dropout(p = dropout_prop)
        )

        self.layer_img2 = nn.Sequential(
            nn.Conv3d(in_channels = self.conv_hidden_dim[0], out_channels = self.conv_hidden_dim[1],
                                kernel_size = (1, 3, 3),
                                stride = (1, 1, 1), 
                                padding = (0, 1, 1)),
           nn.BatchNorm3d(self.conv_hidden_dim[1]),
           nn.ReLU(),  
           nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2)),
           nn.Dropout(p = dropout_prop)
        )

        self.layer_img3 = nn.Sequential(
            nn.Conv3d(in_channels = self.conv_hidden_dim[1], out_channels = self.conv_hidden_dim[2],
                                kernel_size = (1, 3, 3),
                                stride = (1, 1, 1), 
                                padding = (0, 1, 1)),
           nn.BatchNorm3d(self.conv_hidden_dim[2]),
           nn.ReLU(),  
           nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2)),
           nn.Dropout(p = dropout_prop)
        )


        t, h, w = conv3d_output_shape(img_shape, kernel_size = (1,2,2), 
                                      stride = (1,2,2), padding = (0,0,0))
        
        #t, h, w = conv3d_output_shape((t,h,w), kernel_size = (1,2,2), 
        #                              stride = (1,2,2), padding = (0,0,0))
        
        #t, h, w = conv3d_output_shape((t,h,w), kernel_size = (1,2,2), 
        #                              stride = (1,2,2), padding = (0,0,0))
        
        self.flatten_1 = nn.Flatten(start_dim = 2) 

        self.lstm_input = self.conv_hidden_dim[0] * h * w + 2

        self.batchnorm_conv = nn.BatchNorm1d(seq_length)
        self.batchnorm_seq = nn.BatchNorm1d(seq_length)

        self.lstm = nn.LSTM(input_size = self.lstm_input, hidden_size = lstm_hidden_dim,
                            batch_first = True, bidirectional = True, num_layers = 2,
                            dropout = dropout_prop)
        
        self.dropout = nn.Dropout(p = dropout_prop)
        self.flatten_2 = nn.Flatten(start_dim = 1)
        
        self.fc_stop = nn.Sequential(
            nn.BatchNorm1d(2 * lstm_hidden_dim * seq_length + 5),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = 2 * lstm_hidden_dim * seq_length + 5, out_features = fc_hidden_dim),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p = dropout_prop),
            nn.Linear(in_features = fc_hidden_dim, out_features = num_classes)
        )
        

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

        
    def forward(self, X_img, X_seq, X_tod):
        batch_size = X_img.shape[0]
        time = X_img.shape[1]

        #convolutional layers
        X_img = X_img.permute(0, 4, 1, 2, 3)
        X_img = self.layer_img1(X_img)

        X_img = X_img.permute(0, 2, 1, 3, 4)
        X_img = self.flatten_1(X_img)  

        X_collect = torch.cat((X_img, X_seq), dim = 2)
        X_collect = self.batchnorm_conv(X_collect)

        #conv lstm layers
        X_collect, _ = self.lstm(X_collect)
        X_collect = self.flatten_2(X_collect)
        X_collect = torch.cat((X_collect, X_tod), dim = 1)
        
        #final fully connected layer
        X_stop = self.fc_stop(X_collect)

        X_purpose = self.fc_purpose(X_collect)
        
        X_mode = self.fc_mode(X_collect)

        return F.softmax(X_stop, dim = 1), F.softmax(X_mode, dim = 1), F.softmax(X_purpose, dim = 1)

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, model, task_num = 3):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_sigmas = nn.Parameter(torch.zeros((task_num)))

    def forward(self, X_img, X_seq, X_tod, label, device):

        out_stop, out_mode, out_purpose = self.model(X_img, X_seq, X_tod)

        purpose_idx = (label[:,2]>=0)
        mode_idx = (label[:,1]>=0)

        stop_loss = F.cross_entropy(out_stop, label[:, 0], weight= self.model.stop_weights)
        loss_collect = torch.exp(-self.log_sigmas[0])*stop_loss + self.log_sigmas[0]/2

        if purpose_idx.sum() > 0:
          purpose_loss = F.cross_entropy(out_purpose[purpose_idx], label[purpose_idx, 2])
          loss_collect += torch.exp(-self.log_sigmas[2])*purpose_loss + self.log_sigmas[2]/2
        else:
          purpose_loss = torch.Tensor(0).to(device)

        if mode_idx.sum() > 0:
          mode_loss = F.cross_entropy(out_mode[mode_idx], label[mode_idx, 1])
          loss_collect += torch.exp(-self.log_sigmas[1])*mode_loss + self.log_sigmas[1]/2
        else:
          mode_loss = torch.Tensor(0).to(device)

        return loss_collect, mode_loss, purpose_loss, stop_loss, self.log_sigmas.data.tolist()