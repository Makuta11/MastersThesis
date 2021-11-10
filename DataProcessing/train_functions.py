import numpy as np
import pandas as pd
import os
import torch

def train_model(model, optimizer, criterion, num_epochs, train_dataloader, val_dataloader, device,
                scheduler = None):
    
    loss_collect = []
    val_loss_collect = []

    for epoch in range(num_epochs):
        running_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2].float().to(device)
            
            optimizer.zero_grad()
            out = model(data)
            del data
            torch.cuda.empty_cache()

            loss = criterion(out, AUs)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()

            del label, loss
            torch.cuda.empty_cache()
        
        loss_collect = np.append(loss_collect, running_loss/(i+1))
