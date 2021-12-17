import pickle
import pandas as pd
import numpy as np
from utils import *

# Define User
users = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32])

# Define AUs
aus = [1,2,4,5,6,9,12,15,17,20,25,26]

# Import labels
labels = decompress_pickle("/work3/s164272/data/Features/disfa_labels_large1.pbz2")
labels[labels.iloc[:,:-2] > 0] = 1

# Initialize dataframe
data = dict()

for i, au in enumerate(aus):
    tmp = []
    for user in users:
        mask = labels["ID"] == user
        tmp.append(labels[mask].iloc[:,i].sum())
    data[f'AU{au}'] = tmp

# Create pandas DataFrame
df = pd.DataFrame(data=data)

# Set index to users
df.set_index([users], inplace=True)

# Display
print(df)

# Save df
pickle.dump(df, open("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/label_overview", 'wb'))

# Return max 10 to train each AU on
subsets = dict()
for col in df:
    tmp = []
    top = np.inf
    for i in range(10):
        mask = df[col] < top
        tmp.append(df[col][mask].idxmax())
        top = df[col][mask].max()
    subsets[col] = sorted(tmp)

print(subsets)
pickle.dump(subsets, open("/zhome/08/3/117881/MastersThesis/DataProcessing/EmotionModel/src/assets/subsets", "wb"))
