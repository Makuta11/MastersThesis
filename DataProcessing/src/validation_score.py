import numpy as np
import pandas as pd 

def val_scores(pred, label):

    unique_labels = np.unique(np.append(label,pred))
    collect = pd.DataFrame()
    for lab in unique_labels:
        tp = ((pred == lab) & (label == lab)).sum()
        fp = ((pred == lab) & (label!=lab)).sum()
        fn = ((pred != lab) & (label == lab)).sum()

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)

        f_score = 2*precision*recall/(precision + recall)
        
        temp = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1 score': f_score}, index = [lab])
        collect = collect.append(temp)
    
    return collect