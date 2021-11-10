import numpy as np
import pandas as pd
import os
import torch

def train_model(model, optimizer, criterion, num_epochs, train_dataloader, device,
                save_path, save_freq, scheduler = None):
    
    loss_collect = []
    val_loss_collect = []

    for epoch in range(num_epochs):
        
        # Initialilze loss and set model to train mode
        running_loss = 0
        model.train()
        
        for i, x in enumerate(train_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2]
            for elm in AU_intensities:
                elm.float().to(device)
                #AU_intensities = torch.cat((AU_intensities, elm.to(device)), axis=0)
            
            optimizer.zero_grad()
            out = model(data)
            del data
            torch.cuda.empty_cache()

            loss,_ = criterion(out, AUs, AU_intensities, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()

            del AUs, AU_intensities, loss
            torch.cuda.empty_cache()
        
        loss_collect = np.append(loss_collect, running_loss/(i+1))

        print(str(epoch + 1) + ' out of ' + str(num_epochs))
        print(loss_collect[epoch])
        if epoch % save_freq == 0:
            checkpoint_save(model, save_path, epoch)

    return loss_collect, model