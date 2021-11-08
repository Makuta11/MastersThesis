import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np

def decompress_pickle(file: str):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data

def compress_pickle(title: str, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_data(user_train_val, user_test):
    #TODO: Make sure dictionary keys are accessible for training
    dataset = decompress_pickle(f'/zhome/08/3/117881/MastersThesis/DataProcessing/pickles/face_space_dict_disfa.pbz2')
    

class ImageTensorDatasetEpoch(data.Dataset):
    
    def __init__(self, image_data, label):
        self.user_id = df[]
        