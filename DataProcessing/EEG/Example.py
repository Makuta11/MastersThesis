#%%
from libEEG import general, features, plots
import mne
import glob 
import os 


%matplotlib qt
%gui qt

#%%
# Single file import:
# raw = mne.io.read_raw_fif("Clean\\03_03_raw.fif")


#%% Get list of files:
EEG_files = glob.glob('Clean\*03_raw.fif') + glob.glob('Clean\*08_raw.fif')
EEG_files

raw_list = []
for file in EEG_files:
    print(file)
    raw_filt = mne.io.read_raw_fif(file, preload = True, verbose = False)       

    # Delete the BAD annotations
    idx = [index for index, value in enumerate(list(raw_filt.annotations.description)) if value == 'On']
    raw_filt.annotations.delete(idx)

    raw_list.append(raw_filt.copy())

#%% Get PSD and SNR
df_psd, df_snr = features.PSD_SNR_df(raw_list).df_all_subjects(method='Welch',
                                                                win_size=10,
                                                                all_channels=True,
                                                                peak=40,
                                                                tmin=8*60,
                                                                tmax = 10*60
                                                                )


#%% Save PSD and SNR
folder = 'PSD_and_SNR/'
if not os.path.exists(folder):
        os.makedirs(folder)
    
df_snr.to_csv(folder+'SNR'+'.csv', na_rep='NA')   
df_snr.to_pickle(folder+'SNR'+'.pkl')  

df_psd.to_pickle(folder+'PSD'+'.pkl')  

# %% Plot data frames -> None row are droped for plotting 
plots.plot_general_df(df_psd,df_snr, palette=None)