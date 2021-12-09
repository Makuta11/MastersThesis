#%%
import os, sys, time, pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kfda import Kfda
from src.dataloader import *
from sklearn.metrics import f1_score
from sklearn.decomposition import KernelPCA
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from src.validation_score import val_scores
from src.utils import decompress_pickle, compress_pickle

#def main(bool):

# AUs in dataset
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Subject split
user_train = np.array([1,2,4,6,8,9,10,11,16,17,18,21,23,24,25,26,27,28,29,30,31,32])
user_test = np.array([3,5,7,12,13])
user_val = np.array([5])

#%%

# Data loading with test-train splits 
print("Loading Dataset")
t = time.time()
data_test, data_val, data_train, labels_test, labels_val, labels_train = load_data(user_train, user_val, user_test, subset=True)
labels_test, labels_train, labels_val = labels_test.drop(columns="ID"), labels_train.drop(columns="ID"), labels_val.drop(columns="ID")
print(f"It took {time.time() - t} seconds to load the data")

#%%
<<<<<<< HEAD
#for ncom in [6000, 4000, 2000]:
    #for gam in [1e-5, 1e-3, 0.1]:
ncom = None
gam = 0.0001

print("Kernelizing data")
t1 = time.time()
dataset = np.vstack(data_train, data_test)
dataset_transform = KernelPCA(kernel='rbf', n_components = ncom, gamma = gam, eigen_solver="randomized").fit_transform(dataset)
data_test_transform_1 = KernelPCA(kernel='rbf', n_components = ncom, gamma = gam).fit_transform(data_test)
print(f"Data was kernelized in {time.time() - t1} seconds")  
=======
for ncom in [6200]:
    for gam in [0.001]:
        print("Kernelizing data")
        t1 = time.time()
        data_train_transform_1 = KernelPCA(kernel='rbf', n_components = ncom, n_jobs = -1).fit_transform(data_train)
        data_test_transform_1 = KernelPCA(kernel='rbf', n_components = ncom, n_jobs = -1).fit_transform(data_test)
        print(f"Data was kernelized in {time.time() - t1} seconds")  
>>>>>>> 20dc2843b5ce3db365efda8f92505b6df8458535

<<<<<<< HEAD
        #%%
        print("Starting LDA fit...")
        t2 = time.time()
        clf = MultiOutputClassifier(LDA(), n_jobs=-1).fit(data_train_transform_1, labels_train)
        print(f"Data fit in {time.time() - t2} seconds") 
=======
#%%
display = False
if display == True:
    # Create index for labels
    AU = {"Max Action Unit Intensity": labels_train.max(axis=1)}
    #AU[AU == 1] = "Action Unit Present"
    #AU[AU == 0] = "No Action Unit Present"
    AUnon = labels_train.sum(axis=1) == 0

    # Initialize figure
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')

    # Plot training data in kernel space
    ax.scatter(data_train_transform_1[AU, 0], data_train_transform_1[AU, 1], data_train_transform_1[AU, 2], c="red", label="Active AU")
    ax.scatter(data_train_transform_1[AUnon, 0], data_train_transform_1[AUnon, 1], data_train_transform_1[AUnon, 2], c="blue", label="No active AU")

    # Label axis and set title
    #ax.title(f"Kernel PCA for training data \n n_components: {ncom}, gamma: {gam}")
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.legend(loc='best')

#%%
print("Starting LDA fit...")
t2 = time.time()
clf = MultiOutputClassifier(LDA(), n_jobs=8).fit(data_train_transform_1, labels_train)
print(f"Data fit in {time.time() - t2} seconds") 
>>>>>>> 5ce4ef0aea485c32d8ca0357afac8f00da64aab4

#%%

print("Starting prediction...")
t3 = time.time()
y_pred = clf.predict(data_test_transform_1)
print(f'It took {time.time() - t3} to make predictions')

#%%

print(np.unique(y_pred))
print("\nStarting f1-score calculation")

# Convert to bool raveled bool array for AU identification
y_pred_ones = y_pred.ravel()
y_pred_ones[y_pred_ones >= 1] = 1
labels_test_ones = labels_test.to_numpy().ravel()
labels_test_ones[labels_test_ones >= 1] = 1

#%%
# Insert label names to differentiate precision socres between labels
predAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(y_pred_ones)]
trueAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(labels_test_ones)]

print(f'number of components: {ncom} \nGamme set to: {gam}')

# Calculate f1_scores
print(f'\nScores on AU identification:\n{val_scores(predAU, trueAU)}')

# Calculate f1_scores for intensities
for i, au in enumerate(aus):
    print(f'\nScores on AU{au} intensity: \n{val_scores(y_pred.iloc[:,i], labels_test[:,i])}')

#%%
"""
# Save the test model
if sys.platform == 'linux':
compress_pickle("/work3/s164272/models/KERN_clf", clf)
else:
compress_pickle("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/pickles", clf)

if __name__ == "__main__":
print("starting script")
    #main(True)
"""
