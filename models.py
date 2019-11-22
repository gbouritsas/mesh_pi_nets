import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd.function import Function
from torch.autograd import Variable
from torch_scatter import scatter_max
import numpy as np
import random

import pdb
import math
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# from pytorch_geometric_utils import orderings2batches

################################################### 
################# AUTOENCODER MAIN METHOD  #####################
################################################### 

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size,out_c,activation='elu',bias=True,device=None, 
                 injection = False, residual = False, num_points = None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device
        self.injection = injection
        self.residual = residual

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=False)  
#         self.conva2 = nn.Linear(in_c*spiral_size,out_c,bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
            bound = 1 / math.sqrt(in_c*spiral_size)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)
            
        if self.residual:
#             self.proj = nn.Linear(in_c,out_c,bias=False)
            self.convs2 = nn.Linear(in_c*spiral_size,out_c,bias=False)
            self.conva3 = nn.Linear(in_c*spiral_size,out_c,bias=False)
            self.convs3 = nn.Linear(out_c*spiral_size,out_c,bias=False)
            
        if self.injection:
#             self.normalizer4 =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)
            self.normalizer3 =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)
            self.normalizer2 =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)
#             self.normalizer1 =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)
#             self.normalizer =  nn.BatchNorm1d((num_points+1) * out_c, affine= True)

        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()
  
        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]

        if self.injection:
            if self.residual:
                #1
#                 s_x = self.proj(x).reshape(bsize*num_pts,-1)
#                 out_1 = self.conv(spirals)
#                 out_2 = out_1*s_x
#                 out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
#                 out_feat = self.normalizer2(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias       
                
#                 #2
#                 out_1 = self.proj(x).reshape(bsize*num_pts,-1)
#                 s_x = self.conv(spirals)
#                 out_2 = out_1*s_x
#                 out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
#                 out_feat = self.normalizer2(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias  

#                 #2nd order
#                 out_1 = self.conv1(spirals)
#                 s_x = self.conv(spirals)
#                 out_2 = out_1*s_x
#                 out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
#                 out_feat = self.normalizer(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias  

                #3rd order
                a_x2 = self.conva2(spirals)
                s_x2 = self.convs2(spirals)
                out_2 = a_x2*s_x2
                a_x2 = a_x2.reshape(bsize, num_pts * a_x2.shape[-1])
                out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
                out_2 = self.normalizer2(out_2 + a_x2).reshape(bsize*num_pts,-1)
                out_2 = out_2.view(bsize,num_pts,self.out_c)
                spirals2 = out_2[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*self.out_c)

                a_x3 = self.conva3(spirals)
                s_x3 = self.convs3(spirals2)
                out_3 = a_x3*s_x3
                out_3 = out_3.reshape(bsize, num_pts * out_2.shape[-1])
                out_feat = self.normalizer3(out_3).reshape(bsize*num_pts,-1) + a_x3 + self.bias  

            else:  
        
                #2nd order
#                 out_1 = self.conv(spirals)
#                 out_2 = out_1*out_1
#                 out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
#                 out_feat = self.normalizer(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias
                
                #3rd order
                out_1 = self.conv(spirals)
                out_2 = out_1*out_1
                out_3 = out_1*out_1*out_1
                out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
                out_3 = out_3.reshape(bsize, num_pts * out_3.shape[-1])
                out_feat = self.normalizer3(out_3).reshape(bsize*num_pts,-1) + \
                                    self.normalizer2(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias
                
                #4th order
                out_1 = self.conv(spirals)
                out_2 = out_1*out_1
                out_3 = out_1*out_1*out_1
                out_4 = out_1*out_1*out_1*out_1
                out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
                out_3 = out_3.reshape(bsize, num_pts * out_3.shape[-1])
                out_4 = out_4.reshape(bsize, num_pts * out_4.shape[-1])
                out_feat = self.normalizer4(out_4).reshape(bsize*num_pts,-1)+ \
                                      self.normalizer3(out_3).reshape(bsize*num_pts,-1) + \
                                      self.normalizer2(out_2).reshape(bsize*num_pts,-1) + \
                                      out_1 + self.bias
                
    #             out_1 = out_1.reshape(bsize, num_pts * out_1.shape[-1])
    #             out_feat = self.normalizer(out_1).reshape(bsize*num_pts,-1) + self.bias 
#                 out_2 = out_1*out_1
    #             out_3 = out_1*out_1*out_1
    #             out_4 = out_1*out_1*out_1*out_1
    #             out_feat = self.normalizer(out_2) + out_1
    #             out_feat = self.normalizer(out_2) + out_1 + self.bias 
#                 out_1 = out_1.reshape(bsize, num_pts * out_1.shape[-1])
#                 out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
    #             out_3 = out_3.reshape(bsize, num_pts * out_3.shape[-1])
    #             out_4 = out_4.reshape(bsize, num_pts * out_4.shape[-1])
    #             out_feat = self.normalizer4(out_4).reshape(bsize*num_pts,-1)+ \
    #                                   self.normalizer3(out_3).reshape(bsize*num_pts,-1) + \
    #                                   self.normalizer2(out_2).reshape(bsize*num_pts,-1) + \
    #                                   out_1 + self.bias
    #             out_feat = self.normalizer3(out_3).reshape(bsize*num_pts,-1) + \
    #                         self.normalizer2(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias
#                 out_feat = self.normalizer2(out_2).reshape(bsize*num_pts,-1)  + self.normalizer1(out_1).reshape(bsize*num_pts,-1) + self.bias
    #             out_feat = self.normalizer(out_2).reshape(bsize*num_pts,-1) + out_1
    #             out_feat = self.normalizer(out_2) + out_1
    #             out_2 = out_2.reshape(bsize, num_pts * out_2.shape[-1])
    #             out_feat = self.normalizer(out_2).reshape(bsize*num_pts,-1) + out_1 + self.bias 
    #             out_feat = out_2 + out_1 + self.bias 
    #             out_feat = self.normalizer2(out_2) + self.normalizer1(out_1) + self.bias 
    #             out_feat = out_1 + self.bias 
    #             out_feat = self.normalizer(out_feat)

        else:
            out_feat = self.conv(spirals) + self.bias 
            if self.residual:
                proj_x = self.proj(x)
                out_feat = out_feat + proj_x
            
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

    
class SpiralAutoencoder_extra_conv(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, 
                 spirals, D, U, device, activation = 'elu', injection = False, residual = True):
        super(SpiralAutoencoder_extra_conv,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        self.injection = injection
        self.residual = residual
        
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device, 
                                            injection = self.injection, residual = self.residual, 
                                            num_points = sizes[i]).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation, device=device, 
                                        injection = self.injection, residual = self.residual,
                                        num_points =  sizes[i]).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
#         if self.injection:
#             self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size, bias=False)   
#             self.fc_latent_enc_bias = Parameter(torch.Tensor(latent_size))
#             bound = 1 / math.sqrt((sizes[-1]+1)*input_size)
#             torch.nn.init.uniform_(self.fc_latent_enc_bias, -bound, bound)

#             self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0], bias = False)
#             self.fc_latent_dec_bias = Parameter(torch.Tensor((sizes[-1]+1)*filters_dec[0][0]))
#             bound = 1 / math.sqrt(latent_size)
#             torch.nn.init.uniform_(self.fc_latent_dec_bias, -bound, bound)
#         else:
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device, 
                                             injection = self.injection,  residual = self.residual,
                                             num_points =  sizes[-2-i]).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device, 
                                                 injection = self.injection, residual = self.residual,
                                                 num_points = sizes[-2-i]).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device, 
                                                 injection = self.injection, residual = self.residual,
                                                 num_points = sizes[-2-i]).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device, 
                                                 injection = self.injection,  residual = self.residual,
                                                 num_points = sizes[-2-i]).to(device))         
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device,
                                                 injection = self.injection,  residual = self.residual,
                                                 num_points = sizes[-2-i]).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        
#         if self.injection:
#             out_1 = self.fc_latent_enc(x)
#             out_2 = out_1*out_1
#             out = out_2 + out_1 + self.fc_latent_enc_bias
#         else:
        out = self.fc_latent_enc(x) 
        return out
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        
#         if self.injection:
#             out_1 = self.fc_latent_dec(z)
#             out_2 = out_1*out_1
#             x = out_2 + out_1 + self.fc_latent_dec_bias
#         else:
        x = self.fc_latent_dec(z) 
    
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x

    
    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x   


################################################### 
################# GAN MAIN METHOD #####################
################# CHECK FOR BUGS BEFORE UPLOADING #################
###################################################     
    
    
class Discriminator_extra_conv(nn.Module):
    def __init__(self, filters, sizes, spiral_sizes, spirals, D, device, activation = 'elu'):
        super(Discriminator,self).__init__()
        self.sizes = sizes
        self.spirals = spirals
        self.spiral_sizes = spiral_sizes
        self.filters = filters
        self.D = D
        self.activation = activation
        
        self.conv = []
        input_size = filters[0][0]
        for i in range(len(spiral_sizes)-1):
#             import pdb;pdb.set_trace()
            if filters[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters[1][i],\
                                              activation=self.activation, device=device).to(device))
                input_size = filters[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters[0][i+1],\
                          activation=self.activation, device=device).to(device))
            input_size = filters[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_out = nn.Linear((sizes[-1]+1)*input_size, 1)
    
    def forward(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[i](x,S[i].repeat(bsize,1,1))
            x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        return self.fc_out(x)
 
        
    
class Generator_extra_conv(nn.Module):
    def __init__(self, filters, latent_size, sizes, spiral_sizes, spirals, U, device, activation = 'elu'):
        super(Generator,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spiral_sizes = spiral_sizes
        self.spirals = spirals
        self.filters = filters
        self.U = U
        self.activation = activation

        self.fc_latent = nn.Linear(latent_size, (sizes[-1]+1)*filters[0][0])
        self.tanh = nn.Tanh()

        self.dconv = []
        input_size = filters[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters[0][i+1],\
                              activation=self.activation, device=device).to(device))
                input_size = filters[0][i+1]  
                
                if filters[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters[1][i+1],\
                                                  activation=self.activation, device=device).to(device))
                    input_size = filters[1][i]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters[0][i+1],\
                                  activation=self.activation, device=device).to(device))
                    input_size = filters[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters[1][i+1],\
                                              activation='identity', device=device).to(device))                  
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters[0][i+1],\
                                  activation='identity', device=device).to(device))
                    input_size = filters[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)       
        
        
    def forward(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
#         x = self.tanh(self.fc_latent(z))
        x = self.fc_latent(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[i](x,S[-2-i].repeat(bsize,1,1))
        return x     
    


################################################### 
################# AUTOENCODER LSTM-BASED  #####################
################################################### 


class SpiralLSTM(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', num_layers = 1, device=None, random_shift = False):
        super(SpiralLSTM,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.spiral_size = spiral_size
                
        self.device = device
        self.random_shift = random_shift
#         self.random_shift_value = random.randint(0,spiral_size)

        self.lstm = nn.LSTM(input_size = in_c, hidden_size = out_c, num_layers = num_layers, batch_first = True)
        self.lstm = init_lstm(self.lstm, out_c, forget_bias = 2)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_x):
        bsize, num_pts, feats = x.size()
        
        if self.random_shift:
            spirals_all = [[[[spiral_x_i[j][0]] + \
                             list(np.roll(spiral_x_i[j][1:], \
                                random.randint(0,len(spiral_x_i[j][1:])-1)))][0]\
                                    for j in range(len(spiral_x_i))]\
                                        for spiral_x_i in spiral_x]
        else:
            spirals_all = spiral_x 

        if isinstance(spirals_all[0],np.ndarray):
            spiral_adj = np.array(spirals_all)
        else:
            spiral_adj = np.zeros((len(spirals_all), len(spirals_all[0])+1, self.spiral_size)) - 1
            for i in range(len(spirals_all)):
                for j in range(len(spirals_all[i])):
                    spiral_adj[i,j,:len(spirals_all[i][j])] = spirals_all[i][j][: self.spiral_size]            
            
        if spiral_adj.shape[0]!=bsize:
            spiral_adj = torch.tensor(spiral_adj, device = self.device).repeat(bsize,1,1)
        else:
            spiral_adj = torch.tensor(spiral_adj, device = self.device)
        spiral_adj = spiral_adj.long()
          
        spirals_index = spiral_adj.view(bsize*num_pts*self.spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device = self.device).view(-1,1).repeat([1,num_pts*self.spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,self.spiral_size, feats) # [bsize*numpt, spiral*feats]

        out_all, (out_feat, cell_feat) = self.lstm(spirals)
#         out_feat = cell_feat.squeeze()
        out_feat = out_feat.squeeze()
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device = self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat
        
def init_lstm(lstm, lstm_hidden_size, forget_bias=2):
    for name,weights in lstm.named_parameters():
        if "bias_hh" in name:
            #weights are initialized 
            #(b_hi|b_hf|b_hg|b_ho), 
            weights[lstm_hidden_size:lstm_hidden_size*2].data.fill_(forget_bias)
        elif 'bias_ih' in name:
            #(b_ii|b_if|b_ig|b_io)
            pass
        elif "weight_hh" in name:
            torch.nn.init.orthogonal_(weights)
        elif 'weight_ih' in name:
            torch.nn.init.xavier_normal_(weights)
    return lstm  
    
    
class SpiralAutoencoder_LSTM(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, D, U, device, \
                 activation = 'elu', clusters = None, num_layers = 1, random_shift = False):
        super(SpiralAutoencoder_LSTM,self).__init__()
        
        self.latent_size = latent_size
        self.sizes = sizes; 
        self.spiral_sizes = spiral_sizes
        
        self.filters_enc = filters_enc; 
        self.filters_dec = filters_dec; 
        self.num_layers = num_layers
        
        self.D = D; 
        self.U = U
        self.device = device
        
        self.activation = activation
        self.clusters = clusters
        self.random_shift = random_shift
        
        self.lstm_enc = []
        input_size = filters_enc[0][0]
        
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.lstm_enc.append(SpiralLSTM(input_size, spiral_sizes[i], filters_enc[1][i],\
                                              activation = self.activation, num_layers = self.num_layers, device=device, \
                                              random_shift = self.random_shift).to(device))
                input_size = filters_enc[1][i]

            self.lstm_enc.append(SpiralLSTM(input_size, spiral_sizes[i], filters_enc[0][i+1],\
                                            activation=self.activation, num_layers = self.num_layers, device=device,\
                                            random_shift = self.random_shift).to(device))
            input_size = filters_enc[0][i+1]

        self.lstm_enc = nn.ModuleList(self.lstm_enc)   
        
        self.fc_latent_enc = nn.Linear((self.sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (self.sizes[-1]+1)*filters_dec[0][0])
        
        self.lstm_dec = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.lstm_dec.append(SpiralLSTM(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                activation=self.activation, num_layers = self.num_layers, device=device,\
                                                random_shift = self.random_shift).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.lstm_dec.append(SpiralLSTM(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],\
                                                activation=self.activation, num_layers = self.num_layers, device=device,\
                                                random_shift = self.random_shift).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.lstm_dec.append(SpiralLSTM(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                    activation=self.activation, num_layers = self.num_layers, device=device,\
                                                    random_shift = self.random_shift).to(device))
                    input_size = filters_dec[0][i+1]
                    self.lstm_dec.append(nn.Linear(input_size,filters_dec[1][i+1]).to(device))
                    input_size = filters_dec[1][i+1] 
                else:
                    self.lstm_dec.append(nn.Linear(input_size,filters_dec[0][i+1]).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.lstm_dec = nn.ModuleList(self.lstm_dec)

    def encode(self, x, spirals_x):
        bsize = x.size(0)
        S = spirals_x
        D = self.D
        clusters = self.clusters
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.lstm_enc[j](x,[s[i] for s in S])
            j+=1
            if self.filters_enc[1][i]:
                x = self.lstm_enc[j](x,[s[i] for s in S])
                j+=1
            if clusters is None:
                x = torch.matmul(D[i],x)
            else:
                cluster = torch.LongTensor(clusters[i]).view(1,-1,1)
                cluster = cluster.repeat(bsize,1,x.size(-1)).to(self.device)
                x,_ = scatter_max(x, cluster, dim=1)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self, z, spirals_x):
        bsize = z.size(0)
        S = spirals_x
        U = self.U
        
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            if isinstance(self.lstm_dec[j], nn.Linear):
                x = self.lstm_dec[j](x)
            else:
                x = self.lstm_dec[j](x,[s[-2-i] for s in S])
            j+=1
            if self.filters_dec[1][i+1]: 
                if isinstance(self.lstm_dec[j], nn.Linear):
                    x = self.lstm_dec[j](x)
                else:
                    x = self.lstm_dec[j](x,[s[-2-i] for s in S])
                j+=1
        return x

    
    def forward(self, x, spirals_x):
        bsize = x.size(0)
        z = self.encode(x, spirals_x)
        x = self.decode(z, spirals_x)
        return x 


################################################### 
################# Lim et al. Shifted Spiral Convolution #####################
################################################### 


class SpiralShiftConv(nn.Module):
    def __init__(self, in_c, spiral_size, out_c, activation='elu', bias=True, device=None, random_shift = False):
        super(SpiralShiftConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.spiral_size = spiral_size
        
        self.device = device
        self.random_shift = random_shift

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_x):
        bsize, num_pts, feats = x.size()
#         import pdb;pdb.set_trace()
        if self.random_shift:
            spirals_all = [[[[spiral_x_i[j][0]] + \
                             list(np.roll(spiral_x_i[j][1:], \
                                random.randint(0,len(spiral_x_i[j][1:])-1)))][0]\
                                    for j in range(len(spiral_x_i))]\
                                        for spiral_x_i in spiral_x]
            
        else:
            spirals_all = spiral_x
        if isinstance(spirals_all[0],np.ndarray):
            spiral_adj = np.array(spirals_all)
        else:
            spiral_adj = np.zeros((len(spirals_all), len(spirals_all[0])+1, self.spiral_size)) - 1
            for i in range(len(spirals_all)):
                for j in range(len(spirals_all[i])):
                    spiral_adj[i,j,:len(spirals_all[i][j])] = spirals_all[i][j][: self.spiral_size]

        if spiral_adj.shape[0]!=bsize:
            spiral_adj = torch.tensor(spiral_adj, device = self.device).repeat(bsize,1,1)
        else:
            spiral_adj = torch.tensor(spiral_adj, device = self.device)
        spiral_adj = spiral_adj.long()
        
        spirals_index = spiral_adj.view(bsize*num_pts*self.spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device = self.device).view(-1,1).repeat([1,num_pts*self.spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,self.spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device = self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat
        
class SpiralShiftAutoencoder_extra_conv(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, D, U, device, \
                 activation = 'elu', clusters = None, random_shift = False):
        super(SpiralShiftAutoencoder_extra_conv,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spiral_sizes = spiral_sizes
        
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        
        self.D = D
        self.U = U
        self.device = device
        
        self.activation = activation
        self.clusters = clusters
        self.random_shift = random_shift
        
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralShiftConv(input_size, spiral_sizes[i], filters_enc[1][i],\
                                                activation=self.activation, device=device, \
                                                random_shift = self.random_shift).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralShiftConv(input_size, spiral_sizes[i], filters_enc[0][i+1],\
                                            activation=self.activation, device=device, \
                                            random_shift = self.random_shift).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((self.sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (self.sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralShiftConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                activation=self.activation, device=device, \
                                                random_shift = self.random_shift).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralShiftConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],\
                                                    activation=self.activation, device=device, \
                                                    random_shift = self.random_shift).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralShiftConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                    activation=self.activation, device=device, \
                                                    random_shift = self.random_shift).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(SpiralShiftConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],\
                                                    activation='identity', device=device, \
                                                    random_shift = self.random_shift).to(device))                  
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(SpiralShiftConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                    activation='identity', device=device, \
                                                    random_shift = self.random_shift).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    def encode(self,x, spirals_x):
        bsize = x.size(0)
        S = spirals_x
        D = self.D
        clusters = self.clusters
        
        j = 0
        for i in range(len(self.spiral_sizes)-1):
#             import pdb;pdb.set_trace()
            x = self.conv[j](x,[s[i] for s in S])
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,[s[i] for s in S])
                j+=1
            if clusters is None:
                x = torch.matmul(D[i],x)
            else:
                cluster = torch.LongTensor(clusters[i]).view(1,-1,1)
                cluster = cluster.repeat(bsize,1,x.size(-1)).to(self.device)
                x,_ = scatter_max(x, cluster, dim=1)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self, z, spirals_x):
        bsize = z.size(0)
        S = spirals_x
        U = self.U
        
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,[s[-2-i] for s in S])
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x,[s[-2-i] for s in S])
                j+=1
        return x

    
    def forward(self, x, spirals_x):
        bsize = x.size(0)
        z = self.encode(x, spirals_x)
        x = self.decode(z, spirals_x)
        return x

    
################################################### 
################# AUTOENCODER PYTORCH GEOMETRIC IMPLEMENTATION  #####################
################# STILL CONTAINS BUGS  #####################
###################################################     
    
    
class SpiralAutoencoder_ptg(nn.Module):
    def __init__(self, filters_enc, filters_dec, 
                 latent_size, sizes, spiral_sizes, spirals,
                 D, U, device, activation = 'elu'):
        super(SpiralAutoencoder_ptg,self).__init__()
        from ordered_conv import ordered_conv
        
        self.E = None
        self.last_bsize = -1
            
        self.latent_size = latent_size
        self.sizes = sizes
        self.spiral_sizes = spiral_sizes
        self.spirals = spirals
        
        
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        
        self.D = D
        self.U = U
        self.device = device
        
        self.activation = activation
        
        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(ordered_conv(input_size, spiral_sizes[i], filters_enc[1][i],
                                                activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(ordered_conv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                            activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((self.sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (self.sizes[-1]+1)*filters_dec[0][0])
        
        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(ordered_conv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]  
                
                if filters_dec[1][i+1]:
                    self.dconv.append(ordered_conv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                    activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(ordered_conv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                    activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],\
                                                    activation='identity', device=device).to(device))                  
                    input_size = filters_dec[1][i+1] 
                else:
                    self.dconv.append(ordered_conv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],\
                                                    activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]                      
                    
        self.dconv = nn.ModuleList(self.dconv)

    def encode(self, x, edges):
#         import pdb;pdb.set_trace()       
        bsize = x.size(0)
        x = x.view(-1,x.shape[2])
        D = self.D
        j = 0
        
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,edges[i])
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,edges[i])
                j+=1
            x = torch.matmul(D[i], x.view(bsize,self.sizes[i],-1)).view(bsize*self.sizes[i+1],-1)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self, z, edges):
#         import pdb;pdb.set_trace()
        bsize = z.size(0)
        U = self.U
        x = self.fc_latent_dec(z)
        j=0
        
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i], x.view(bsize,self.sizes[-1-i],-1)).view(bsize*self.sizes[-2-i],-1)
            x = self.dconv[j](x, edges[-2-i])
            j+=1
            if self.filters_dec[1][i+1]: 
                x = self.dconv[j](x, edges[-2-i])
                j+=1
        return x

    
    def forward(self, x):
#         import pdb;pdb.set_trace()
        bsize = x.size(0)
        if self.E is None or self.last_bsize != bsize:
            self.E = orderings2batches(self.spirals, bsize, self.spiral_sizes, self.sizes, self.device)
            self.last_bsize = bsize
        edges = self.E
        z = self.encode(x, edges)
        x = self.decode(z, edges)
        return x


################################################### 
################# OLD/UNUSED CLASSES  #####################
###################################################   

class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, device, activation = 'elu'):
        super(SpiralAutoencoder,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation
        
        self.conv = []
        for i in range(len(spiral_sizes)-1):
            self.conv.append(SpiralConv(filters_enc[i], spiral_sizes[i], filters_enc[i+1],\
                                          activation=self.activation, device=device).to(device))
        self.conv = nn.ModuleList(self.conv)   
        
        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*filters_enc[-1], latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0])
        
        self.dconv = []
        for i in range(len(spiral_sizes)-2):
            self.dconv.append(SpiralConv(filters_dec[i],spiral_sizes[-2-i], filters_dec[i+1],\
                                          activation=self.activation, device=device).to(device))
        self.dconv.append(SpiralConv(filters_dec[-2], spiral_sizes[0], \
                                     filters_dec[-1], activation='identity', device=device).to(device))
        self.dconv = nn.ModuleList(self.dconv)

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[i](x,S[i].repeat(bsize,1,1))
            x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)
    
    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
        
        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[i](x,S[-2-i].repeat(bsize,1,1))
        return x

    
    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x



class Discriminator(nn.Module):
    def __init__(self, filters, sizes, spiral_sizes, spirals, D, device, activation = 'leaky_relu'):
        super(Discriminator,self).__init__()
        self.sizes = sizes
        self.spirals = spirals
        self.spiral_sizes = spiral_sizes
        self.filters = filters
        self.D = D
        self.activation = activation
        
        self.conv = []
        for i in range(len(spiral_sizes)-1):
            self.conv.append(SpiralConv(filters[i],spiral_sizes[i], filters[i+1],\
                                          activation=self.activation, device=device).to(device))
        self.conv = nn.ModuleList(self.conv)
        self.fc_out = nn.Linear((sizes[-1]+1)*filters[-1], 1)
    
    def forward(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[i](x,S[i].repeat(bsize,1,1))
            x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        return self.fc_out(x)
 
        
    
class Generator(nn.Module):
    def __init__(self, filters, latent_size, sizes, spiral_sizes, spirals, U, device, activation = 'leaky_relu'):
        super(Generator,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spiral_sizes = spiral_sizes
        self.spirals = spirals
        self.filters = filters
        self.U = U
        self.activation = activation
        

        self.fc_latent = nn.Linear(latent_size, (sizes[-1]+1)*filters[0])
        #self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
        self.dconv = []
        for i in range(len(spiral_sizes)-2):
            self.dconv.append(SpiralConv(filters[i],spiral_sizes[-2-i], filters[i+1],\
                                          activation=self.activation, device=device).to(device))
        self.dconv.append(SpiralConv(filters[-2],spiral_sizes[0], filters[-1],activation='identity',device=device).to(device))
        self.dconv = nn.ModuleList(self.dconv)
        
        
    def forward(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U
#         x = self.tanh(self.fc_latent(z))
        x = self.fc_latent(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[i](x,S[-2-i].repeat(bsize,1,1))
        return x    
    
  
    



class SpiralVAE(SpiralAutoencoder):

    def __init__(self,latent_size,sizes,spiral_sizes,spirals,D,U,device,beta=1.0,beta_mult=1.0):
        super(SpiralVAE,self).__init__(latent_size,sizes,spiral_sizes,spirals,D,U,device)
        self.beta = beta
        self.beta_mult = beta_mult
        #self.fc_latent_enc = nn.Linear((sizes[-1]+1)*32,self.latent_size*2)
        self.fc_latent_enc = nn.Linear(self.fc_latent_enc.weight.data.size(1),self.latent_size*2)

    def forward(self,x):
        bsize = x.size(0)
        muvar = self.encode(x)
        mu = muvar[:,:self.latent_size]
        logvar = muvar[:,self.latent_size:]
        # reparametrization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + (eps*std)
        x = self.decode(z)
        return x, mu, logvar