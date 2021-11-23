import pickle, bz2
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from generate_feature_vector import decompress_pickle

tmp = decompress_pickle("/work3/s164272/data/Features/face_space_dict_disfa_large1.pbz2")
data_list = list(tmp.items())
data_arr = np.array(data_list)
data_arr = np.vstack(data_arr[:,1])
data_arr = np.nan_to_num(data_arr)

print(data_arr.shape)
