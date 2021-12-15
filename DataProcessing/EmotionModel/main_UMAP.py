import os, sys, time, pickle, umap

import umap.plot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pelutils import TT
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, f1_score, confusion_matrix

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
user_test = np.array([3,7,12,13])
user_val = np.array([5])

# Data loading
print("Loading Dataset")
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)

permutation_list = [[data_test,labels_test]]

for i, data in enumerate(permutation_list):
    for i, au in enumerate([1]):
        for numn in [100]:
            for ncomp in [2]:
                # Convert labels to binary
                lab = data[1].iloc[:,i]
                lab[lab > 0] = 1

                lab_val = labels_val.iloc[:,i]
                lab_val[lab_val > 0] = 1
                ##### First Pipeline #####
                #pipe = make_pipeline(SimpleImputer(strategy="mean"))
                #X = pipe.fit_transform(data[0].copy())

                # Fit UMAP to processed data
                mapper = umap.UMAP(n_components=ncomp, n_neighbors=numn, random_state=42).fit(data[0])
                svc = SVC().fit(mapper.embedding_, lab)
                knn = KNeighborsClassifier().fit(mapper.embedding_, lab)

                test_embedding = mapper.transform(data_val)
                svc.score(mapper.transform(data_val), lab_val), knn.score(mapper.transform(data_val), lab_val)
                
                mapper = umap.UMAP(n_components=ncomp, n_neighbors=numn, random_state=42).fit(data[0])
                svc = SVC().fit(mapper.embedding_, lab)
                knn = KNeighborsClassifier().fit(mapper.embedding_, lab)

                test_embedding = mapper.transform(data_val)

                pred_svc = svc.predict(mapper.transform(data_val))
                pred_knn = knn.predict(mapper.transform(data_val))

                f1_svc = f1_score(lab_val, pred_svc, zero_division=1)
                f1_knn = f1_score(lab_val, pred_knn, zero_division=1)
                
                cm_svc = confusion_matrix(lab_val, pred_svc)
                cm_knn = confusion_matrix(lab_val, pred_knn)

                print(f'Scores for n_components:{ncomp} and n_neighbors:{numn} for AU{au}')
                print(f'SVC:')
                print(f1_svc)
                print(cm_svc)
                print(f'KNN')
                print(f1_knn)
                print(cm_knn)

                if ncomp == 2:
                    # Plot and save fig
                    fig, ax = plt.subplots(1, figsize=(14, 10))
                    plt.scatter(*mapper.embedding_.T, s=0.5, cmap="Dark2")
                    plt.scatter(*test_embedding.T, s=0.5, cmap="Pastel1")
                    plt.title(f'Test and Train embeddings; SVM:{round(f1_svc, 2)}, KNN:{round(f1_knn, 2)} ');
                    plt.savefig(f"UMAP/testset_AU{au}_nneig:{numn}_ncmop:{ncomp}.png", dpi=128, bbox_inches='tight')

                #clf_LDA = LDA().fit(X_reduced_1[:4840*4], lab[:4840*4])
                #clf_SVM = SVC(class_weight="balanced").fit(X_reduced_1[:4840*4], lab[:4840*4])

                #y_pred_LDA = clf_LDA.predict(X_reduced_1[4840*4:])
                #y_pred_SVM = clf_SVM.predict(X_reduced_1[4840*4:])

                #print(f'Action Unit {au}:')
                #print(f'LDA n:{numn}, c:{ncomp}, AU{au}\n{classification_report(lab[4840*4:], y_pred_LDA)}')
                #print(f'SVM n:{numn}, c:{ncomp}, AU{au}\n{classification_report(lab[4840*4:], y_pred_SVM)}')

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
