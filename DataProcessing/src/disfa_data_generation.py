import os
import ssl
import wget
import time
import pickle
import requests
import numpy as np
import pandas as pd
import xlwings as xw
import threading
from glob import glob
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor
import urllib.request

def fetch_vid_files(vidDir):
    files = [file
            for path, subdir, files in os.walk(urlDir)
            for file in glob(os.path.join(path, "*.txt"))]
    return files

