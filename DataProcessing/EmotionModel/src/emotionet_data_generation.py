import os, ssl, wget, time, pickle, requests, threading

import numpy as np
import pandas as pd
import xlwings as xw
import urllib.request

from glob import glob
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

def fetch_img_files(urlDir):
    """Generates a list of file pathnames from directory and subdirectories containing .txt suffix

    Args:
        dataDir (str): root directory of data
    """
    files = [file
            for path, subdir, files in os.walk(urlDir)
            for file in glob(os.path.join(path, "*.txt"))]
    return files

def gen_df(urlDir):
    names = np.append(["URL1","URL2"],[str(x) for x in np.arange(1,61)]).astype(list)
    files = sorted(fetch_img_files(urlDir))
    frames = []
    for file in files:
        frames.append(pd.read_csv(file, error_bad_lines=False, delimiter="\t", names=names))
    return pd.concat(frames).reset_index(drop=True, inplace=False)

def download(link, filelocation):
    urllib.request.urlretrieve(link[:-1], filelocation)

def new_download_thread(link, filelocation):
    download_thread = threading.Thread(target=download, args=(link, filelocation))
    download_thread.start()

def main(urlDir, dataDir, pickleDir):
    if "labels" in os.listdir(pickleDir):
        df = pickle.load( open(f'{pickleDir}labels', 'rb'))
        print("loaded")
    else:    
        df = pd.read_excel(urlDir)
    t = time.time()
    """
    if not os.listdir(dataDir):
        print(f'Downloading {df.shape[0]}Images...')
        for i, file in enumerate(df.iloc[:,0]):
            time.sleep(0.005)
            try:
                new_download_thread(file,  dataDir + '{0:05}'.format(i) + ".jpg")
            except:
                try:
                    new_download_thread(df.iloc[i,1],  dataDir + '{0:05}'.format(i) + ".jpg")
                except:
                    print("e")
            if (i+1)%1001 == 0:
                print(f'{i} of {df.shape[0]} images downloaded at {round(i/(time.time() - t), 2)} images per second')
    """
    return df

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    urlDir = "/Users/DG/Documents/School/00. MASTER_THESIS/Data/EmotioNet/EmotioNet_FACS_aws_2020_24600.xlsx"
    dataDir = "/Users/DG/Documents/PasswordProtected/EmotioNetData/"
    pickleDir = "/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/DataProcessing/pickles/"
    df = main(urlDir, dataDir, pickleDir)
    path = f'{pickleDir}labels'
    pickle.dump(df, open(path, 'wb'))
