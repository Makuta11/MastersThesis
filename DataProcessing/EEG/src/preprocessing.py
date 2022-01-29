#%%
from pydoc import describe
import mne
import glob
import mne
import pandas as pd
from libEEG import general, plots, features
from os.path import exists
import warnings

# warnings.filterwarnings('error')
# Use to make set_annotation() to break the script

#%%
EEG_files = glob.glob('/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/**/**/**.edf')
for file in EEG_files:
    print(file)
# print()

df_WM_Triggers = pd.read_csv('/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EEG/df_timestamps', index_col = 0)

condition_dict = {'01': 'RS','02': 'WM','03': 'Stim','04': 'RS','05': 'WM',
                  '06': 'RS','07': 'WM','08': 'Stim','09': 'RS','10': 'WM',}

#%%
def get_annotations(raw, file):
    start_timestamp = raw.info['meas_date'].replace(tzinfo=None).timestamp()
    dur = len(raw)/raw.info['sfreq']

    if '_03' in file or '_08' in file: 
        # Stimulus recording
        if (dur - raw.annotations[-1]['onset']) < 30:
            onset = 0
            duration = raw.annotations.onset[0]
        else:
            onset = raw.annotations.onset[0]
            duration = 15*60
        description = 'Stim'

    elif '_01' in file or '_04' in file or '_06' in file or '_09' in file:
        # RS recording
        Session = {1: 1, 4: 1, 6: 2, 9: 2}
        Task = {1: 1, 4: 2, 6: 1, 9: 2}
        df_temp = df_WM_Triggers[( df_WM_Triggers.ID == int(sub.split('_')[0]) ) &
                        ( df_WM_Triggers.Session == Session[ int(sub.split('_')[1]) ] ) &
                        ( df_WM_Triggers.Task == Task[ int(sub.split('_')[1]) ] ) ]

        offset = df_temp['RestOffset'].iloc[0]/1000

        if len(raw.annotations) > 0 and (dur - raw.annotations[-1]['onset']) > 30:
            start = raw.annotations.onset[0] + offset
        else:
            start = df_temp['RestTimeStamp'].iloc[0]/1000 - start_timestamp
        
        onset = [start + i for i in range(0,600,30)]
        duration = [30,30]*10
        description = ['EO','EC']*10

    elif '_02' in file or'_05' in file or'_07' in file or'_10' in file:
        # WM recording
        Session = {2: 1, 5: 1, 7: 2, 10: 2}
        Task = {2: 1, 5: 2, 7: 1, 10: 2}
        df_temp = df_WM_Triggers[( df_WM_Triggers.ID == int(sub.split('_')[0]) ) &
                        ( df_WM_Triggers.Session == Session[ int(sub.split('_')[1]) ] ) &
                        ( df_WM_Triggers.Task == Task[ int(sub.split('_')[1]) ] ) ]
                        
        offsets = list(df_temp[['1NbackOffset','2NbackOffset','3NbackOffset']].iloc[0])

        if len(raw.annotations) > 0 and (dur - raw.annotations[-1]['onset']) > 30:
            start = raw.annotations.onset[0]
            onset = [start + (i/1000) for i in offsets]
        else:
            start = list(df_temp[['1BackStart','2BackStart','3BackStart']].iloc[0])
            onset = [(i/1000) - start_timestamp for i in start]
        
        duration = [60]*3
        description = ['1-back','2-back','3-back']

    return onset, duration, description

#%%  Load data
for file in EEG_files:
    sub = file.split('/')[-1].split('.edf')[0]
    output_path = 'clean_2/'+ sub +'_raw.fif'

    if not exists(output_path) and '17_01' not in file:
        check_data = False
        print('///////////// ' + condition_dict[sub.split('_')[1]] + ' /////////////' )
        print(sub)

        raw = mne.io.read_raw_edf(file, preload = True, verbose = False)        
        raw.info['subject_info'] = {'his_id': sub}
        raw.info['experimenter'] = 'CBJ'
        raw.info['line_freq'] = 50
        
        fs = raw.info['sfreq']
        dur = len(raw)/fs
        raw = general.set_standard_montage_ZETO(raw)

        original_annotations = raw.annotations.copy()
        print(original_annotations.onset)

        # raw.plot(block = True)

        # if len(raw.annotations) > 2 and (dur - raw.annotations[-1]['onset']) < 30:
        #     # Delete all but last two annotations 
        #     raw.plot(block = True)
        #     idx = list(range(0,len(raw.annotations)-2))
        #     raw.annotations.delete(idx)
        #     check_data = True
        
        # elif len(raw.annotations) > 2 :
        #     # Delete all but last annotations
        #     raw.plot(block = True)
        #     idx = list(range(0,len(raw.annotations)-1))
        #     raw.annotations.delete( idx )
        #     check_data = True

        if not len(raw.annotations) > 0 or (dur - raw.annotations[-1]['onset']) < 30 or 'part' in sub:
            check_data = True
        
        #% Filter, mark bad channels, and manually edit annotations.
        raw_filt = general.filter(raw,low_pass = 100)

        #% Interpolate
        raw_filt.interpolate_bads(reset_bads=True, verbose = False)

        #% Rereference (w. projection)
        raw_filt.set_eeg_reference(ref_channels='average', projection=True, verbose = False)
         
        ##%  Auto set annotations based on recording and/or trigger files
        onset, duration, description = get_annotations(raw_filt,file)

        experiment_annotations = mne.Annotations(onset = onset, 
                                                duration = duration, 
                                                description = description, 
                                                orig_time= raw_filt.annotations.orig_time)

        # raw_filt.set_annotations(annotations = original_annotations + experiment_annotations)

        try:
            raw_filt.set_annotations(annotations = original_annotations + experiment_annotations)
        except Warning:
            check_data = True
            raw_filt.set_annotations(annotations = original_annotations + experiment_annotations)

        #% Mark bads segments: 
        anno_path = 'bad_annotations' + raw_filt.info['subject_info']['his_id']+'.csv'
        try:
            print('Loading extra annotations...', anno_path)
            annotations = mne.read_annotations(anno_path)
            raw_filt.set_annotations(annotations) 
            print('Done!')
        except:
            print('No extra annotation is available')
            raw_filt = general.mark_bad_segments(raw_filt)

        if input('Save BAD annotations? (y) ').lower() == 'y':
            raw_filt.annotations.save(anno_path, overwrite=True)

        #% Re-set original annotations on top of bad annotations
        raw_filt.set_annotations(annotations=raw_filt.annotations + original_annotations, verbose = False)

        #% Check data
        if check_data:
            print('Check final data')
            raw_filt.plot(block=True)

        #% Save annotated data
        raw_filt.save(output_path, overwrite=True)

# %%
