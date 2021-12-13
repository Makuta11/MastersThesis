import os, sys, time, pickle, umap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.dataloader import *
from src.utils import decompress_pickle, compress_pickle

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Subject split
user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_test = np.array([3,5,7,12,13])
user_val = np.array([5])

# Data loading
print("Loading Dataset")
t = time.time()
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
print(f"It took {time.time() - t} seconds to load the data")

permutation_list = [[data_test,labels_test]]

for i, data in enumerate(permutation_list):
    for i, au in enumerate([1]):

        # Convert labels to binary
        lab = data[1].iloc[:,i]
        lab[lab > 0] = 1

        ##### First Pipeline #####
        pipe = make_pipeline(SimpleImputer(strategy="mean"))
        X = pipe.fit_transform(data[0].copy())

        # Fit UMAP to processed data
        manifold = umap.UMAP().fit(X, lab)
        X_reduced_1 = manifold.transform(X)

        # Plot and save fig
        plt.style.use('fivethirtyeight')
        fig1, ax1 = plt.subplots(figsize=(10,12))
        ax1.scatter(X_reduced_1[:, 0], X_reduced_1[:, 1], c=lab, s=1)
        fig1.savefig(f"UMAP/test_UMAP_1_AU{au}.png", dpi=128, bbox_inches='tight')

        ##### Second Pipeline #####
        pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer(n_quantiles=lab.shape[0]))
        X = pipe.fit_transform(data[0].copy())

        # Fit UMAP to processed data
        manifold = umap.UMAP().fit(X, lab)
        X_reduced_2 = manifold.transform(X)

        # Plot and save fig
        plt.style.use('fivethirtyeight')
        fig1, ax1 = plt.subplots(figsize=(10,12))
        ax1.scatter(X_reduced_2[:, 0], X_reduced_2[:, 1], c=lab, s=1)
        fig1.savefig(f"UMAP/test_UMAP_2_AU{au}.png", dpi=128, bbox_inches='tight')