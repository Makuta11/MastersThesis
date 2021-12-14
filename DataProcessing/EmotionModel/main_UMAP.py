import os, sys, time, pickle, umap

import umap.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.dataloader import *
from src.utils import decompress_pickle, compress_pickle

def draw_umap(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    plt.title(title, fontsize=18)
    plt.savefig(f"UMAP/test_UMAP_1_n:{n_neighbors}_1_AU{au}.png", dpi=128, bbox_inches='tight')

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
    for i, au in enumerate(aus):
        i = aus.index(au)
        for numn in [1000]:
            for ncomp in [100]:
                # Convert labels to binary
                lab = data[1].iloc[:,i]
                lab[lab > 0] = 1

                ##### First Pipeline #####
                pipe = make_pipeline(SimpleImputer(strategy="mean"))
                X = pipe.fit_transform(data[0].copy())

                # Fit UMAP to processed data
                manifold = umap.UMAP(n_components=ncomp, n_neighbors=numn, min_dist=0.0).fit(X, lab)
                X_reduced_1 = manifold.transform(X)

                if ncomp == 2:
                    # Plot and save fig
                    umap.plot.points(manifold, labels = lab, theme="fire")
                    plt.savefig(f"UMAP/test_UMAP_1_n:{numn}_1_AU{au}.png", dpi=128, bbox_inches='tight')
                
                clf_LDA = LDA().fit(X_reduced_1[:4840*4], lab[:4840*4])
                clf_SVM = SVC(class_weight="balanced").fit(X_reduced_1[:4840*4], lab[:4840*4])

                y_pred_LDA = clf_LDA.predict(X_reduced_1[4840*4:])
                y_pred_SVM = clf_SVM.predict(X_reduced_1[4840*4:])

                print(f'Action Unit {au}:')
                print(f'LDA n:{numn}, c:{ncomp}, AU{au}\n{classification_report(lab[4840*4:], y_pred_LDA)}')
                print(f'SVM n:{numn}, c:{ncomp}, AU{au}\n{classification_report(lab[4840*4:], y_pred_SVM)}')

                """
                ##### Second Pipeline #####
                pipe = make_pipeline(SimpleImputer(strategy="mean"), QuantileTransformer())
                X = pipe.fit_transform(data[0].copy())

                # Fit UMAP to processed data
                manifold = umap.UMAP().fit(X, lab)
                #X_reduced_2 = manifold.transform(X)

                # Plot and save fig
                umap.plot.points(manifold, labels = lab, theme="fire")
                plt.savefig(f"UMAP/test_UMAP_2_n:{numn}_AU{au}.png", dpi=128, bbox_inches='tight')
                """
