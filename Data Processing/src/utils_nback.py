#%%
import os
import math
import pickle
import fnmatch
import numpy as np
import pandas as pd
from glob import glob
from rich.console import Console
from rich.table import Column, Table

def fetch_rest_csv(dataDir):
    """
    Takes a directory continaing data files and returns a list of all .csv files in the directory and sub-directories

    Args:
        dataDir (str): root directory containing outputs from the experiments

    Returns:
        list: a list containing the filenames of all experiment files saved as a .csv
    """
    all_rest_files = [file
                    for path, subdir, files in os.walk(dataDir)
                    for file in glob(os.path.join(path, "*.csv"))
                    if "Rest" in file]

    return all_rest_files

def fetch_rest_npy(dataDir):
    """
    Takes a directory continaing data files and returns a list of all .npy files in the directory and sub-directories

    Args:
        dataDir (str): root directory containing outputs from the experiments

    Returns:
        list: a list containing the filenames in directory and subdirectoyr of type .npy
    """
    all_RS_files = [file
                    for path, subdir, files in os.walk(dataDir)
                    for file in glob(os.path.join(path, "*.npy"))
                    if "RS" in file]

    return all_RS_files

def single_rest_frame(file1, file2, file3, file4):
    """
        This fuction generates a DataFrame from the .csv collected from pavlovia and .npy file generated from the dashboard.
        The DataFrame contains the following 
    """
    tmp1 = pd.read_csv(file1, usecols=["globalTimeStart","session","participant"])
    tmp2 = pickle.load(open(file2,'rb'))
    tmp3 = pd.read_csv(file3, usecols=["NbackStart1","Nback2start","NbackStart3"])
    tmp4 = pickle.load(open(file4, 'rb'))

    if "RS1" in file2:
        task = [1]
    else:
        task = [2]

    data = {
        'ID': [tmp1["participant"][0]],
        'Session': [tmp1["session"][0]],
        'Task': task,
        'RestTimeStamp': [tmp1["globalTimeStart"][0]],
        'RestOffset': [tmp1["globalTimeStart"][0] - round(tmp2[0]*1000,0)],
        '1BackStart': [tmp3["NbackStart1"].dropna().reset_index(drop=True, inplace=False)[0]],
        '2BackStart': [tmp3["Nback2start"].dropna().reset_index(drop=True, inplace=False)[0]],
        '3BackStart': [tmp3["NbackStart3"].dropna().reset_index(drop=True, inplace=False)[0]],
        '1NbackOffset': [tmp3["NbackStart1"].dropna().reset_index(drop=True, inplace=False)[0] - round(tmp4[0]*1000,0)],
        '2NbackOffset': [tmp3["Nback2start"].dropna().reset_index(drop=True, inplace=False)[0] - round(tmp4[0]*1000,0)],
        '3NbackOffset': [tmp3["NbackStart3"].dropna().reset_index(drop=True, inplace=False)[0] - round(tmp4[0]*1000,0)]
    }

    return pd.DataFrame(data).reset_index(drop=True, inplace=False)

def gen_timestamp_frame(dataDir):
    rest_list_csv = sorted(fetch_rest_csv(dataDir))
    rest_list_npy = sorted(fetch_rest_npy(dataDir))
    nback_list_csv = sorted(fetch_nback_csv(dataDir))
    nback_list_npy = sorted(fetch_nback_npy(dataDir))

    frames = []
    for file1, file2, file3, file4 in zip(rest_list_csv, rest_list_npy, nback_list_csv,nback_list_npy):
        frames.append(single_rest_frame(file1,file2,file3,file4))
    
    return pd.concat(frames).reset_index(drop=True, inplace=False)

def fetch_nback_csv(dataDir):
    """
        This function finds all .csv files in a direvtory and subdirectories which contains 'Nback' in its name
    """
    all_nback_files = [file
                    for path, subdir, files in os.walk(dataDir)
                    for file in glob(os.path.join(path, "*.csv"))
                    if "Nback" in file]
    return all_nback_files

def fetch_nback_npy(dataDir):
    """
        This function finds all .csv files in a direvtory and subdirectories which contains 'Nback' in its name
    """
    all_nback_files = [file
                    for path, subdir, files in os.walk(dataDir)
                    for file in glob(os.path.join(path, "*.npy"))
                    if "N-Back" in file and "time_stamps" in file]
    return all_nback_files

def single_subj_frame(file):
    """
        This fuction generates a DataFrame from a single .csv file with the data sorted for easy use and filtering.
    """
    
    stimKey = [1,1,0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,1,0,0]
    
    tmp = pd.read_csv(file).fillna(int(0))
    tmp = tmp.mask(tmp == "space", 1)

    if tmp["session"][0] == 1 and stimKey[tmp["participant"][0] - 1] == 1:
        stim = [1]*90
    elif tmp["session"][0] == 2 and stimKey[tmp["participant"][0] - 1] == 0:
        stim = [1]*90
    else:
        stim = [0]*90

    if "first" in tmp["expName"][0]:
        task = [1]*90
    else:
        task = [2]*90

    scores = list(np.append([tmp["key_resp_3.corr"][0:30]],[tmp["response_2.corr"][30:60]])) #redo 3back
    resp = list(np.append([tmp["key_resp_3.keys"][0:30],tmp["response_2.keys"][30:60]],tmp["key_resp_4.keys"][60:90]))
    respTime = list(np.append([tmp["key_resp_3.rt"][0:30],tmp["response_2.rt"][30:60]],tmp["key_resp_4.rt"][60:90]))
    Nback = np.append([[1]*30, [2]*30], [3]*30)

    #Implement correction to 3-Back test due to Pavlovia sync error
    corr_3_ans = [0]*30
    corr_3_ans[4] = 1;corr_3_ans[9] = 1;corr_3_ans[13] = 1;corr_3_ans[28] = 1

    corr_3_scores = []
    for i, x in enumerate(resp[60:90]):
        if corr_3_ans[i] == x:
            corr_3_scores.append(1)
        else:
            corr_3_scores.append(0)

    data = {
        'ID': tmp['participant'],
        'Session': tmp['session'],
        'Task': task,
        'Stim': stim, 
        'CorrAns': list(np.append([tmp['corrAns'][0:60]],[corr_3_ans])),
        'Scores': list(np.append([scores],[corr_3_scores])),
        'KeyRespons': resp, 
        'ResponsTime': respTime,
        'Nback': Nback
    }

    #pdb.set_trace()

    return pd.DataFrame(data)

def gen_nback_frame(dataDir):
    """
    Generate DataFrame with Nback results for all subjects.

    This function simply concatenates Nback DataFrames.
    
    Parameters
    ----------
    dataDir : str
        Directory where data is stored.
    
    Returns
    -------
    df : dataFrame
        ID: the randomized subject ID of the participant
        Session: the session of the data collected (1 or 2)
        Task: relating to berfor (1) or after (2) the stimulation period
        Stim: if the subject recieved 40Hz (1) or continuous (0) light during that session
        CorrAns: indicates the trials where the user should have pressed space
        KeyResponse: indicates the trials where the user pressed space
        ResponsTime: indicates the response time of the user (0 if no response)
        Nback: relates to the type of Nback (1, 2, or 3).
    """
    nbackList = fetch_nback_csv(dataDir)
    
    frames = []
    for file in nbackList:
        frames.append(single_subj_frame(file))
    
    return pd.concat(frames).reset_index(drop=True, inplace=False)

def avg_perf_score_change(df, stim, nback, session = None, IDprint = None):
    mask = (df.Stim == stim) & (df.Nback == nback)
    
    if session:
        mask = (mask) & (df.Session == session)
    
    if IDprint:
        ID_list = []
    
    scores_list = []
    for IDs in df[mask].ID.unique():
        tmp1 = df[(mask) & (df.Task == 1) & (df.ID == IDs)]["Scores"].sum()
        tmp2 = df[(mask) & (df.Task == 2) & (df.ID == IDs)]["Scores"].sum()
        scores_list.append(round(((tmp2 - tmp1)/tmp1)*100,2))
        if IDprint:
            ID_list.append(IDs)

    if IDprint:
        print(f'    {ID_list}')

    return scores_list, round(np.mean(scores_list),2)

def avg_perf_score(df, nback, task=None, stim=None, session=None):
    mask = (df.Nback == nback)

    if task:
        mask = (mask) & (df.Task == task)
    if stim is not None:
        mask = (mask) & (df.Stim == stim)
    if session:
        mask = (mask) & (df.Session == session)

    scores_list = []
    for IDs in df[mask].ID.unique():
        tmp = df[(mask) & (df.ID == IDs)]["Scores"].mean()*100
        scores_list.append(round(tmp,2))
    
    return scores_list, round(np.mean(scores_list), 2)

def train_effect(df): #TODO: estimate in R instead (this result here has no statistical rigor)
    tmp1 = round(df[(df.Session == 1) & (df.Task == 1)]["Scores"].sum()/df[(df.Session == 1) & (df.Task == 1)]["Scores"].shape[0],2)
    tmp2 = round(df[(df.Session == 1) & (df.Task == 2)]["Scores"].sum()/df[(df.Session == 1) & (df.Task == 2)]["Scores"].shape[0],2)
    tmp3 = round(df[(df.Session == 2) & (df.Task == 1)]["Scores"].sum()/df[(df.Session == 2) & (df.Task == 1)]["Scores"].shape[0],2)
    tmp4 = round(df[(df.Session == 2) & (df.Task == 2)]["Scores"].sum()/df[(df.Session == 2) & (df.Task == 2)]["Scores"].shape[0],2)

    print(tmp1,tmp2,tmp3,tmp4)

def collective_performace_change(df, full = None):
        
    nsubj_s1 = len(df[df.Session == 1].ID.unique())
    nsubj_s2 = len(df[df.Session == 2].ID.unique())

    if full:
        print("┌-------------┐")
        print("|   ID list   |")
        print("└-------------┘")
        tot_perf_list = []

    perf_list = []
    score_list = []
    for session in range(1,3):
        for nback in range(1,4):
            for stim in range(2):
                perf_list.append(avg_perf_score_change(df, stim=stim, nback=nback, session=session)[1])
                if full:
                    tot_perf_list.append(avg_perf_score_change(df, stim=stim, nback=nback, session=session, IDprint=True)[0])
                    for task in range(1,3):
                        score_list.append(avg_perf_score(df, nback=nback, task=task, stim=stim, session=session)[0])

    if full:
        print("")
        print("┌----------------------------┐")
        print("| Percent Change per Session |")
        print("└----------------------------┘")
        for i, score in enumerate(tot_perf_list):
            print(f'Session: {math.floor(i/6) + 1}. Nback: {math.floor(i/2)%3+1}. stim: {i%2}. Score:{score}. Var: {round(np.var(score),2)}')

        print("")
        print("┌-------------------------┐")
        print("| Accuracy Score per Task |")
        print("└-------------------------┘")
        for i, raw_scores in enumerate(score_list):
            print(f'Session: {math.floor(i/12) + 1}. Nback: {math.floor(i/4)%3+1}. stim: {math.floor(i/2)%2}. task:{i%2+1} Score:{raw_scores}.')

    print("\n"+
          "┌----------------------------------------┐")
    print("|  Percent Change in N-Back Performance  |")
    print("└----------------------------------------┘")
    print(f'    Session 1 (n={nsubj_s1})')
    print(f'        1-Back: \n'+
          f'            sham change: {perf_list[0]}% \n'+
          f'            stim change: {perf_list[1]}%')
    print(f'        2-Back: \n'+
          f'            sham change: {perf_list[2]}% \n'+
          f'            stim change: {perf_list[3]}%')
    print(f'        3-Back: \n'+
          f'            sham change: {perf_list[4]}% \n'+
          f'            stim change: {perf_list[5]}%\n')
    print(f'    Session 2 (n={nsubj_s2})')
    print(f'        1-Back: \n'+
          f'            sham change: {perf_list[6]}% \n'+
          f'            stim change: {perf_list[7]}%')
    print(f'        2-Back: \n'+
          f'            sham change: {perf_list[8]}% \n'+
          f'            stim change: {perf_list[9]}%')
    print(f'        3-Back: \n'+
          f'            sham change: {perf_list[10]}% \n'+
          f'            stim change: {perf_list[11]}%\n')
    print(f'    Across Sessions (not corrected for subj diff)')
    print(f'        1-Back: \n'+
          f'            sham change: {round(np.mean([perf_list[0], perf_list[6]]), 2)}% \n'+
          f'            stim change: {round(np.mean([perf_list[1], perf_list[7]]), 2)}%')
    print(f'        2-Back: \n'+
          f'            sham change: {round(np.mean([perf_list[2], perf_list[8]]), 2)}% \n'+
          f'            stim change: {round(np.mean([perf_list[3], perf_list[9]]), 2)}%')
    print(f'        3-Back: \n'+
          f'            sham change: {round(np.mean([perf_list[4], perf_list[10]]), 2)}% \n'+
          f'            stim change: {round(np.mean([perf_list[5], perf_list[11]]), 2)}%')

if __name__ == "__main__":
    dataDir = "/Users/DG/Documents/PasswordProtected/Speciale Outputs/"
    rest_list_csv = fetch_rest_csv(dataDir)
    rest_list_npy = fetch_rest_npy(dataDir)
    
    df = gen_nback_frame(dataDir)
    df_timestamps = gen_timestamp_frame(dataDir)

    collective_performace_change(df,full=True)

# %%
