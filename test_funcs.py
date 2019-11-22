import torch
import copy
from tqdm import tqdm
import numpy as np

def test_autoencoder_dataloader(device, model, dataloader_test, shapedata, mm_constant = 1000):
    model.eval()
    l1_loss = 0
    l2_loss = 0
    shapedata_mean = torch.Tensor(shapedata.mean).to(device)
    shapedata_std = torch.Tensor(shapedata.std).to(device)
    with torch.no_grad():
        for i, sample_dict in enumerate(tqdm(dataloader_test)):
            tx = sample_dict['points'].to(device)
            prediction = model(tx)  
            if i==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
                
            if dataloader_test.dataset.dummy_node:
                x_recon = prediction[:,:-1]
                x = tx[:,:-1]
            else:
                x_recon = prediction
                x = tx
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            
            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim=2)))*x.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return predictions, l1_loss, l2_loss

def test_ptg_autoencoder_with_pseudo(device, model, dataloader_test, shapedata, num_down, mm_constant = 1000):

    with torch.no_grad():
        model.eval()
        l1_loss = 0
        l2_loss = 0
        shapedata_mean = torch.Tensor(shapedata.mean).to(device)
        shapedata_std = torch.Tensor(shapedata.std).to(device)
        for b, sample_dict in enumerate(tqdm(dataloader_test)):
            pseudo_input = []
            for d in range(num_down):
                k = 'd{0}'.format(d)
                pseudo_temp = sample_dict[k]
                pseudo_input.append(pseudo_temp.view(-1,pseudo_temp.shape[2]).to(device))
                
            tx = sample_dict['points'].to(device)
            cur_bsize = tx.shape[0]
            prediction = model(tx, pseudo_input)
            
            if b==0:
                predictions = copy.deepcopy(prediction)
            else:
                predictions = torch.cat([predictions,prediction],0) 
                
            x_recon = prediction
            x = tx
                
            l1_loss+= torch.mean(torch.abs(x_recon-x))*x.shape[0]/float(len(dataloader_test.dataset))
            
            x_recon = (x_recon * shapedata_std + shapedata_mean) * mm_constant
            x = (x * shapedata_std + shapedata_mean) * mm_constant
            l2_loss+= torch.mean(torch.sqrt(torch.sum((x_recon - x)**2,dim = 2))) *x.shape[0]/float(len(dataloader_test.dataset))
            
        predictions = predictions.cpu().numpy()
        l1_loss = l1_loss.item()
        l2_loss = l2_loss.item()
    
    return predictions, l1_loss, l2_loss



# def test_autoencoder(device, model, loss_fn, test_data, bsize):
#     with torch.no_grad():
#         model.eval()
#         loss = 0
#         for b in tqdm(range(int(np.ceil(float(test_data.shape[0])/bsize)))):
#             test_data_temp = torch.from_numpy(test_data[b*bsize:min(b*bsize+bsize,test_data.shape[0])]).to(device)
#             prediction = model(test_data_temp)
#             if b==0:
#                 predictions = copy.deepcopy(prediction)
#             else:
#                 predictions = torch.cat([predictions,prediction],0)
#             loss+= loss_fn(prediction, test_data_temp)*prediction.shape[0]/test_data.shape[0]
#         predictions = predictions.cpu().numpy()
#     return predictions, loss