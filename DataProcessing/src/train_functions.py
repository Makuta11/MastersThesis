import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

def checkpoint_save(model, save_path, epoch, name):
    f = os.path.join(save_path, 'checkpoint_test-{:03d}-{}.pth'.format(epoch + 1, name))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

def train_model(model, optimizer, criterion, num_epochs, train_dataloader, val_dataloader, device,
                save_path, save_freq, scheduler = None, name = None):
    
    loss_collect = []
    val_loss_collect = []

    for epoch in range(num_epochs):
        
        # Initialilze loss and set model to train mode
        running_loss = 0
        val_loss = 0
        model.train()
        
        for i, x in enumerate(train_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2].type(torch.LongTensor).to(device)
            #for elm in AU_intensities:
            #    elm.float().to(device)
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
        
        if scheduler:
           scheduler.step()

        loss_collect = np.append(loss_collect, running_loss/(i+1))

        # get validation loss
        model.eval()
        for i, x in enumerate(val_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2].type(torch.LongTensor).to(device)
            #for elm in AU_intensities:
            #    elm.float().to(device)

            out = model(data)
            
            del data
            torch.cuda.empty_cache()

            loss,_ = criterion(out, AUs, AU_intensities, device)
            val_loss += loss.detach().cpu().item()
            
            del AUs, AU_intensities, loss
            torch.cuda.empty_cache()
        
        if scheduler:
           scheduler.step()

        val_loss_collect = np.append(val_loss_collect, val_loss/(i+1))

        print(str(epoch + 1) + ' out of ' + str(num_epochs))
        print(loss_collect[epoch])
        print(val_loss_collect[epoch])
        if (epoch + 1) % save_freq == 0:
            checkpoint_save(model, save_path, epoch, name)

    return model, loss_collect, val_loss_collect

def get_predictions(model, test_dataloader):
        collect = dict()
        df = pd.DataFrame()

        for i, x in enumerate(test_dataloader):
            model.eval()
            out = model(x[0].float().to(device))

            AUs_true = x[1]
            AU_intensities = x[2]
