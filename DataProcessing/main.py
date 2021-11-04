import os
import mne
import sys
import numpy as np
import pandas as pd

from src.utils_nback import collective_performace_change, gen_nback_frame, gen_timestamp_frame

def main(bool):
    dataDir = "/Users/DG/Documents/PasswordProtected/Speciale Outputs/"
    df = gen_nback_frame(dataDir)
    TS = gen_timestamp_frame(dataDir)
    collective_performace_change(df,full=bool)

    return df, TS

if __name__ == "__main__":
    main(False)