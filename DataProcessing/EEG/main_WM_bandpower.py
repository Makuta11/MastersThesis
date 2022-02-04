#%%
import mne, glob, os 

import pandas as pd
import seaborn as sns 
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from libEEG import general, features, plots

#%%
%matplotlib inline
%gui qt

# Get list of files:
dir_path = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/data/Clean"
EEG_files = os.listdir(dir_path)
EEG_files

raw_list = []
id_list = []
for file in sorted(EEG_files):
    if ("_02" in file) | ("_05" in file) | ("_07" in file) | ("_10" in file):

        if ("13_07" in file) | ("13_10" in file):
            continue
        
        print(file)
        id_list.append(file[:5])
        raw_filt = mne.io.read_raw_fif(f'{dir_path}/{file}', preload = True, verbose = False)       

        # Delete the on annotations
        idx = [index for index, value in enumerate(list(raw_filt.annotations.description)) if value == 'On']
        raw_filt.annotations.delete(idx)

        raw_list.append(raw_filt.copy())

#%%
#Get PSD and SNR
df_psd, df_snr = features.PSD_SNR_df(raw_list).df_all_subjects(method='Welch',
                                                                 win_size=20,
                                                                 all_channels=True,
                                                                 p_overlap=0.8,
                                                                 peak=40,
                                                                 tmin =1,
                                                                 tmax= 59
                                                                 )
df_psd["Nback"] = df_psd["Stimulus"]

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

for i, file in enumerate(raw_list):
    
    ID = int(file.filenames[0][-13:-11])

    if ID == 13:
        continue

    if ("_02" in id_list[i]):
        task = 1
        session = 1
    elif ("_05" in id_list[i]):
        task = 2
        session = 1
    elif ("_07" in id_list[i]):
        task = 1
        session = 2
    elif ("_10" in id_list[i]):
        task = 2
        session = 2

    for key in bands:
        df_tmp = features.BandPower(file, tmin=1, tmax=29).bandpower_df(bands[key], window_sec=10)
        df_tmp = df_tmp.mean().to_frame().T
        df_tmp["Band"] = key
        df_tmp["Task"] = task
        df_tmp["Session"] = session
        df_tmp["ID"] = ID

        if (session == 1) & (str(ID) in ["1","2","9","10","18"]):
            df_tmp["Stimulus"] = "ISF"
        elif (session == 1) & (str(ID) in ["3","13","15","17","20"]):
            df_tmp["Stimulus"] = "Continuous"
        elif (session == 2) & (str(ID) in ["3","13","15","17","20"]):
            df_tmp["Stimulus"] = "ISF"
        else:
            df_tmp["Stimulus"] = "Continuous"

        df_bandpowers = pd.concat([df_bandpowers, df_tmp], ignore_index=True)

    df_tmp.head()

df_bandpowers = df_bandpowers.drop(["Fail"], axis=1)

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
    df_psd.loc[(df_psd.Subject == f'{id}_02'), 'Stimulus'] = "Continuous"
    df_psd.loc[(df_psd.Subject == f'{id}_02'), 'Task'] = 1
    df_psd.loc[(df_psd.Subject == f'{id}_05'), 'Stimulus'] = "Continuous"
    df_psd.loc[(df_psd.Subject == f'{id}_05'), 'Task'] = 2
    df_psd.loc[(df_psd.Subject == f'{id}_07'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_07'), 'Task'] = 1
    df_psd.loc[(df_psd.Subject == f'{id}_10'), 'Stimulus'] = "ISF"
    df_psd.loc[(df_psd.Subject == f'{id}_10'), 'Task'] = 2

# Generate SNR power band differences matrix
df_bp_diff = pd.DataFrame()
df_bp_diff["1-back-diff"] = df_bandpowers[(df_bandpowers.Task == 2)]["1-back"].values - df_bandpowers[(df_bandpowers.Task == 1)]["1-back"].values
df_bp_diff["2-back-diff"] = df_bandpowers[(df_bandpowers.Task == 2)]["2-back"].values - df_bandpowers[(df_bandpowers.Task == 1)]["2-back"].values
df_bp_diff["3-back-diff"] = df_bandpowers[(df_bandpowers.Task == 2)]["3-back"].values - df_bandpowers[(df_bandpowers.Task == 1)]["3-back"].values
df_bp_diff["Band"] = df_bandpowers[(df_bandpowers.Task == 2)]["Band"].values
df_bp_diff["ID"] = df_bandpowers[(df_bandpowers.Task == 2)]["ID"].values
df_bp_diff["Session"] = df_bandpowers[(df_bandpowers.Task == 2)]["Session"].values
df_bp_diff["Stimulus"] = df_bandpowers[(df_bandpowers.Task == 2)]["Stimulus"].values


# %% 
def big_powerband_plot(nback):

    mask_psd_ISF = (df_psd.Nback == f"{nback}-back") & (df_psd.Stimulus == "ISF") 
    mask_psd_CON = (df_psd.Nback == f"{nback}-back") & (df_psd.Stimulus == "Continuous") 

    fig_factor = 2
    fig = plt.figure(tight_layout=True,figsize=[fig_factor*8.4,fig_factor*6.8])
    gs = gridspec.GridSpec(4, 6)

    df_plot_1 = pd.DataFrame(df_psd[mask_psd_CON].groupby(['Task','Freq','Subject']).mean().mean(axis = 1).reset_index())
    df_plot_1.rename(columns = {0: 'psd'}, inplace = True)
    df_plot_2 = pd.DataFrame(df_psd[mask_psd_ISF].groupby(['Task','Freq','Subject']).mean().mean(axis = 1).reset_index())
    df_plot_2.rename(columns = {0: 'psd'}, inplace = True)

    ax1 = fig.add_subplot(gs[0:2,0:3])
    sns.lineplot(data=df_plot_1, x="Freq", y="psd",hue='Task',ax = ax1,ci='sd',palette="tab10",legend=True)
    ax1.set_xlim([0,50])
    ax1.xaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0))
    ax1.set_ylim([-20,25])
    ax1.set_xlabel('Frequency [Hz]',fontsize=20)
    ax1.set_ylabel("PSD [dB]",fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.text(1, -19, "delta", rotation = 90)
    ax1.text(4.2, -19, "theta", rotation = 90)
    ax1.text(8, -19, "alpha", rotation = 90)
    ax1.text(12.2, -19, "beta1")
    ax1.text(20.2, -19, "beta2")
    ax1.text(30.2, -19, "gamma")
    for line in [1,4,8,12,20,30,50]:
        ax1.axvline(x=line, color = "k", linewidth=0.8, alpha=0.6)
    ax1.set_title('Continuous')
    ax1.legend(title="Task", labels=["Pre","Post"])

    ax2 = fig.add_subplot(gs[0:2,3:6])
    sns.lineplot(data=df_plot_2, x="Freq", y="psd",hue='Task',ax = ax2,ci='sd',palette="tab10",legend=True)
    ax2.set_xlim([0,50])
    ax2.xaxis.set_major_locator(ticker.IndexLocator(base=10, offset=0))
    ax2.set_ylim([-20,25])
    ax2.set_xlabel('Frequency [Hz]',fontsize=20)
    ax2.set_ylabel("PSD [dB]",fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.text(1, -19, "delta", rotation = 90)
    ax2.text(4.2, -19, "theta", rotation = 90)
    ax2.text(8, -19, "alpha", rotation = 90)
    ax2.text(12.2, -19, "beta1")
    ax2.text(20.2, -19, "beta2")
    ax2.text(30.2, -19, "gamma")
    for line in [1,4,8,12,20,30,50]:
        ax2.axvline(x=line, color = "k", linewidth=0.8, alpha=0.6)
    ax2.set_title('ISF')
    ax2.legend(title="Task", labels=["Pre","Post"])

    order = ["Continuous", "ISF"]
    pallete = ["#398BED", "#F3B532"]

    ax3 = fig.add_subplot(gs[2, 0:2])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax3,data = df_bp_diff[df_bp_diff.Band == "delta"],palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax3,data = df_bp_diff[df_bp_diff.Band == "delta"],color="k",order=order)
    ax3.set_ylim([-800,500])
    ax3.set_title('Delta Power Change')

    ax4 = fig.add_subplot(gs[2, 2:4])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax4,data = df_bp_diff[df_bp_diff.Band == "theta"], palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax4,data =df_bp_diff[df_bp_diff.Band == "theta"],color="k",order=order)
    ax4.set_ylim([-300,35])
    ax4.set_title('Theta Power Change')

    ax5 = fig.add_subplot(gs[2, 4:6])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax5,data = df_bp_diff[df_bp_diff.Band == "alpha"], palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax5,data = df_bp_diff[df_bp_diff.Band == "alpha"],color="k",order=order)
    ax5.set_ylim([-60,8])
    ax5.set_title('Alpha Power Change')

    ax6 = fig.add_subplot(gs[3, 0:2])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax6,data = df_bp_diff[df_bp_diff.Band == "beta1"], palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax6,data = df_bp_diff[df_bp_diff.Band == "beta1"],color="k",order=order)
    ax6.set_ylim([-30,20])
    ax6.set_title('Beta 1 Power Change')

    ax7 = fig.add_subplot(gs[3, 2:4])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax7,data = df_bp_diff[df_bp_diff.Band == "beta2"], palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax7,data = df_bp_diff[df_bp_diff.Band == "beta2"],color="k",order=order)
    ax7.set_ylim([-10,6])
    ax7.set_title('Beta 2 Power Change')

    ax8 = fig.add_subplot(gs[3, 4:6])
    sns.boxplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax8,data = df_bp_diff[df_bp_diff.Band == "gamma"], palette=pallete, showfliers = False,order=order)
    sns.swarmplot(x = 'Stimulus', y = f'{nback}-back-diff', ax=ax8,data = df_bp_diff[df_bp_diff.Band == "gamma"],color="k",order=order)
    ax8.set_ylim([-25,7])
    ax8.set_title('Gamma Power Change')

    for ax in [ax3,ax4,ax5,ax6,ax7,ax8]:
        ax.set_ylabel('PSD*Freq [DB*Hz]')

    plt.show()

big_powerband_plot(3)


# %% Instpect data for 40Hz peak
for sub in df_psd[(df_psd.Stimulus == "ISF")].Subject.unique():
    df_psd_mask = (df_psd.Subject == sub)

    fig_factor = 1
    fig = plt.figure(tight_layout=True,figsize=[fig_factor*8.4,fig_factor*6.8])
    ax = fig.add_subplot()
    df_plot_1 = pd.DataFrame(df_psd[df_psd_mask].groupby(['Stimulus','Freq','Subject']).mean().mean(axis = 1).reset_index())
    df_plot_1.rename(columns = {0: 'psd'}, inplace = True)
    sns.lineplot(data=df_plot_1, x="Freq", y="psd",ax = ax,ci='sd',palette="tab10",legend=True)
    ax.set_title(str(sub))

