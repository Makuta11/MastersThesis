#%%
import mne, glob, os 

import pandas as pd
import matplotlib.pyplot as plt

from libEEG import general, features, plots

#%%
#%matplotlib inline
#%gui qt

# Get list of files:
dir_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/data/Clean"
EEG_files = os.listdir(dir_path)
EEG_files

raw_list = []
for file in EEG_files:
    if ("_02" in file) | ("_05" in file) | ("_07" in file) | ("_10" in file):
        
        print(file)
        raw_filt = mne.io.read_raw_fif(f'{dir_path}/{file}', preload = True, verbose = False)       

        # Delete the on annotations
        idx = [index for index, value in enumerate(list(raw_filt.annotations.description)) if value == 'On']
        raw_filt.annotations.delete(idx)

        raw_list.append(raw_filt.copy())

#Get PSD and SNR
df_psd, _ = features.PSD_SNR_df(raw_list).df_all_subjects(method='Welch',
                                                                win_size=20,
                                                                all_channels=True,
                                                                peak=40,
                                                                tmin =1,
                                                                tmax= 59
                                                                )

#%%
bands = {
    'alpha': [8, 12],
    'delta': [1, 4],
    'theta': [4, 8],
    'beta1': [12,20],
    'beta2': [20, 30],
    'gamma': [30, 50]
}

df_bandpowers = pd.DataFrame()

for key in bands:
    df_tmp = features.BandPower(raw_list[0], tmin=1, tmax=29).bandpower_df(bands[key], window_sec=10)
    df_tmp = df_tmp.mean().to_frame().T
    df_tmp["band"] = key
    df_bandpowers = pd.concat([df_bandpowers, df_tmp], ignore_index=True)

#%%#create masks
group_a = ["01","02","09","10","18"]
group_b = ["03","13","15","17","20"]

for id in group_a:
    df_psd.loc[(df_psd.Subject == f'{id}_02'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_02'), 'Task'] = 1
    df_psd.loc[(df_psd.Subject == f'{id}_05'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_05'), 'Task'] = 2
    df_psd.loc[(df_psd.Subject == f'{id}_07'), 'Stimulus'] = "Continuous"
    df_psd.loc[(df_psd.Subject == f'{id}_07'), 'Task'] = 1
    df_psd.loc[(df_psd.Subject == f'{id}_10'), 'Stimulus'] = "Continuous"
    df_psd.loc[(df_psd.Subject == f'{id}_10'), 'Task'] = 2

for id in group_b:
    df_psd.loc[(df_psd.Subject == f'{id}_08'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_03'), 'Stimulus'] = "Continuous"

mask_psd = (df_psd.Stimulus == "Continuous") | (df_psd.Stimulus == "ISF")
mask_snr = (df_snr.Stimulus == "Continuous") | (df_snr.Stimulus == "ISF")

#
save_path = "assets/psd_snr.png"
plots.plot_general_df(df_psd[mask_psd],df_snr[mask_snr], palette = "tab10", ymin=-20, save_path = save_path)

# %%
