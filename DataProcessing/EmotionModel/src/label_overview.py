import pandas as pd
import numpy as np
import pickle

# Load files to print
lab_dict = pickle.load(open("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/subsets", 'rb'))
lab = pickle.load(open("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/label_overview", "rb"))

# Print
print(f'Labels Overview:\n{lab}\n')
print(f'Classes Chosen with label:samples ratio >= 1:13')
for key in lab_dict:
    print(key, lab_dict[key])
