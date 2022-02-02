#%%
import os, glob, mne
from pydoc import describe
import pandas as pd
from libEEG import general, plots, features
from os.path import exists
import matplotlib.pyplot as plt
import warnings

dir_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/data/Clean"
EEG_files = os.listdir(dir_path)
print(EEG_files)

output_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/data/clean_2"

#%%
%matplotlib inline
%gui qt

# %%
for file in EEG_files:

    # Only look at stimulation periods
    if ("10_03" in file): #| ("_08" in file):

        # load file
        tmp = mne.io.read_raw_fif(f'{dir_path}/{file}', preload = True, verbose = False)

        # plot file - stops the loop for inspection
        tmp.plot(block= True)

        # save file to clean_2
        tmp.save(f'{output_path}/{file}', overwrite=True)

# %%
