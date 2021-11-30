import os
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd

def checkpoint_save(model, save_path, epoch, name):
    f = os.path.join(save_path, 'checkpoint_test-{:03d}-{}.pt'.format(epoch + 1, name))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

def train_model(model, optimizer, num_epochs, train_dataloader, val_dataloader, device,
                save_path, save_freq, scheduler = None, name = None):
    
    loss_collect = []
    val_loss_collect = []
    
    # Collection array for uncertainty weights
    sigma_collect = np.zeros((num_epochs, 13))

    for epoch in range(num_epochs):
        
        # Initialilze loss and set model to train mode
        running_loss = 0
        val_loss = 0
        
        model.train()
        for i, x in enumerate(train_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2].type(torch.LongTensor).to(device)
            
            optimizer.zero_grad()

            loss = model(data, AUs, AU_intensities, device)
            loss[0].backward()
            optimizer.step()
            running_loss += loss[0].detach().cpu().item()

            del AUs, AU_intensities
            torch.cuda.empty_cache()
        
        if scheduler:
           scheduler.step()

        loss_collect = np.append(loss_collect, running_loss/(i+1))
        sigma_collect[epoch,:] = loss[1]

        # get validation loss
        model.eval()
        for i, x in enumerate(val_dataloader):
            data = x[0].float().to(device)
            AUs = x[1].float().to(device)
            AU_intensities = x[2].type(torch.LongTensor).to(device)

            loss = model(data, AUs, AU_intensities, device)
            val_loss += loss[0].detach().cpu().item()
            
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

    return model, loss_collect, val_loss_collect, sigma_collect

def get_predictions(model, test_dataloader, device):
    model.eval()
    with torch.no_grad():
        predAU = []
        trueAU = []
    
        aus = [1,2,4,5,6,9,12,15,17,20,25,26]
        intensities_dict = dict()
        
        for j, au in enumerate(aus):
            intensities_dict[f'AU{au}'] = dict()
            intensities_dict[f'AU{au}']["pred"] = []
            intensities_dict[f'AU{au}']["true"] = []

        for i, x in enumerate(test_dataloader):
    
            out = model.model(x[0].float().to(device))

            # Append logic bool array for predicted and true labels
            predAU.append(list(out[0].cpu().numpy().ravel() > 0)) 
            trueAU.append(list(x[1].cpu().numpy().ravel() > 0))

            # Append intensity prediction to dictionary organized for each AU
            for j, au in enumerate(aus):
                AU_idx = (x[2] >= 1).nonzero(as_tuple=True)
                if (j) in list(AU_idx[1]):
                    idx = list(AU_idx[1]).index(j)
                    if len(AU_idx) > 0:
                        intensities_dict[f'AU{au}']["pred"].append(np.argmax(out[1][AU_idx[1][idx]][AU_idx[0][idx]].cpu()).numpy() + 1)
                        intensities_dict[f'AU{au}']["true"].append(x[2][AU_idx[0][idx]][AU_idx[1][idx]].cpu().numpy())

        predAU = np.concatenate(predAU).ravel()
        trueAU = np.concatenate(trueAU).ravel()
        predAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(predAU)]
        trueAU = [0 if elem == False else aus[i%12] for i, elem in enumerate(trueAU)]

        #TODO make sure the output does not truncate to 15 - seems like it was a problem locally
        for j, au in enumerate(aus):
            intensities_dict[f'AU{au}']["pred"] = np.array(intensities_dict[f'AU{au}']["pred"]).ravel()
            intensities_dict[f'AU{au}']["true"] = np.array(intensities_dict[f'AU{au}']["true"]).ravel()
        
    return [predAU, trueAU], intensities_dict