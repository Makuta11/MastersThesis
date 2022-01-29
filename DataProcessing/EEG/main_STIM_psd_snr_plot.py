#%%
import mne, glob, os 

import pandas as pd
import matplotlib.pyplot as plt

from libEEG import general, features, plots

#%%
#%matplotlib inline
#%gui qt

# Get list of files:
dir_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/Clean"
EEG_files = os.listdir(dir_path)
EEG_files

raw_list = []
for file in EEG_files:
    if ("_03" in file) | ("_08" in file):# | ("_06" in file) | ("_09" in file):
            
        if "17" in file:
            continue
        
        print(file)
        raw_filt = mne.io.read_raw_fif(f'{dir_path}/{file}', preload = True, verbose = False)       

        # Delete the on annotations
        idx = [index for index, value in enumerate(list(raw_filt.annotations.description)) if value == 'On']
        raw_filt.annotations.delete(idx)

        raw_list.append(raw_filt.copy())

#Get PSD and SNR
df_psd, df_snr = features.PSD_SNR_df(raw_list).df_all_subjects(method='Welch',
                                                                win_size=60,
                                                                all_channels=True,
                                                                peak=40,
                                                                tmin =30,
                                                                tmax= 60*15 - 30
                                                                )
#create masks
group_a = ["01","02","09","10","18"]
group_b = ["03","13","15","17","20"]

for id in group_a:
    df_psd.loc[(df_psd.Subject == f'{id}_03'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_08'), 'Stimulus'] = "Continuous"
    df_snr.loc[(df_snr.Subject == f'{id}_03'), 'Stimulus'] = "ISF"
    df_snr.loc[(df_snr.Subject == f'{id}_08'), 'Stimulus'] = "Continuous"

for id in group_b:
    df_psd.loc[(df_psd.Subject == f'{id}_08'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_03'), 'Stimulus'] = "Continuous"
    df_snr.loc[(df_snr.Subject == f'{id}_08'), 'Stimulus'] = "ISF"
    df_snr.loc[(df_snr.Subject == f'{id}_03'), 'Stimulus'] = "Continuous"

mask_psd = (df_psd.Stimulus == "Continuous") | (df_psd.Stimulus == "ISF")
mask_snr = (df_snr.Stimulus == "Continuous") | (df_snr.Stimulus == "ISF")

#
save_path = "assets/psd_snr.png"
plots.plot_general_df(df_psd[mask_psd],df_snr[mask_snr], palette = "tab10", ymin=-20, save_path = save_path)

# %%
