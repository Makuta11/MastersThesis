import numpy as np
import pandas as pd 

def val_scores(pred, label):

    unique_labels = np.unique(label)
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

def get_predictions(model, val_data_loader):
    collect = dict()
    df = pd.DataFrame()

    model.eval()
    for i, x in enumerate(val_dataloader):
        data = x[0].float().to(device)
        AUs = x[1].float().to(device)
        AU_intensities = x[2].type(torch.LongTensor).to(device)

        out = model(data)

        AU_pred
        AU_int_pred