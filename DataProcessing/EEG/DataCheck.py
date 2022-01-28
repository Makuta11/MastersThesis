#%%
import pandas as pd
import glob
import mne


#%%
files = glob.glob("Sub**/*.edf")
files

#%%
df = pd.DataFrame(columns = ['Date','Subject','Recording','fs','Duration','Event', 'N Annotations','True N annotations'])

for file in files:
    sub = file.split('\\')[-1].split('.edf')[0].split('_')[0]
    recording = int(file.split('\\')[-1].split('.edf')[0].split('_')[1])

    print(sub, recording)

    raw = mne.io.read_raw_edf(file, verbose = False)
    n_anno = len(raw.annotations)
    fs = raw.info['sfreq']
    date = raw.info['meas_date']
    duration = raw.__len__()/fs

    if recording in [1,6,4,9]:
        true_anno = 2
        event = 'RS'
    elif recording in [2,7,5,10]:
        true_anno = 2
        event = 'WM'
    elif recording in [3,8]:
        true_anno = 2
        event = 'Stim'

    df = df.append({'Date': date,
                    'Subject': sub,
                    'Recording': recording,
                    'fs': fs,
                    'Duration': duration,
                    'Event': event,
                    'N Annotations': n_anno,
                    'True N annotations': true_anno,
                    }, ignore_index=True)

df.to_csv('DataCheck.csv')