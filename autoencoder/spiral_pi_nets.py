import argparse

import sys
sys.path.append('../')
from lib import mesh_sampling
import numpy as np
import json
import os
import copy
from facemesh import FaceData
import time
import pickle
import trimesh

try:
    import psbody.mesh
    found = True
except ImportError:
    found = False
if found:
    from psbody.mesh import Mesh, MeshViewer, MeshViewers

from autoencoder_dataset import autoencoder_dataset
from torch.utils.data import DataLoader

from spiral_utils import get_adj_trigs, generate_spirals
from models import SpiralAutoencoder, SpiralAutoencoder_extra_conv

from test_funcs import test_autoencoder_dataloader
from train_funcs import train_autoencoder_dataloader


import torch
from tensorboardX import SummaryWriter

from sklearn.metrics.pairwise import euclidean_distances
meshpackage = 'trimesh'
root_dir = '/data/gb318/datasets/'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2list2int(v):
    return [int(c) for c in v.split(',')]

def str2ListOfLists2int(v):
    return [[[] if c==' ' else int(c) for c in vi.split(',')] for vi in v.split(',,')]

def str2list2float(v):
    return [float(c) for c in v.split(',')]

def str2list2bool(v):
    return [str2bool(c) for c in v.split(',')]

def str2ListOfLists2bool(v):
    return [[[] if c==' ' else str2bool(c) for c in vi.split(',')] for vi in v.split(',,')]
        
def loss_l1(outputs, targets):
    L = torch.abs(outputs - targets).mean()
    return L 

def main(args):
    
    torch.cuda.get_device_name(args['device_idx'])
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.benchmark = False
    np.random.seed(args['seed'])
    torch.set_num_threads(args['num_threads'])

    
    args['data'] = os.path.join(root_dir, args['dataset'], 'preprocessed', args['name'])
    
    if args['dataset'] == 'COMA':
        args['reference_mesh_file'] = os.path.join(root_dir, args['dataset'],
                                           'preprocessed/templates/template.obj')
        args['downsample_directory'] = os.path.join(root_dir, args['dataset'], 
                                            'preprocessed/templates',args['downsample_method'], args['downsample_config'])
    elif args['dataset'] == 'mein3d':
        args['reference_mesh_file'] = os.path.join(root_dir, args['dataset'],
                                                   'lsfm_mean','lsfm_mean_masked.obj')
        args['downsample_directory'] = os.path.join(root_dir, args['dataset'],'lsfm_mean','original', 
                                                    args['downsample_method'],args['downsample_config'])
    elif args['dataset'] == 'DFAUST':
        args['reference_mesh_file'] = os.path.join(root_dir, args['dataset'],'template/template.obj')
        args['downsample_directory'] = os.path.join(root_dir, args['dataset'],'template',
                                                    args['downsample_method'], args['downsample_config'])
    else:
        raise NotImplementedError

    
    args['results_folder'] = os.path.join(root_dir, args['dataset'],'results',
                                          'higher_order_'+ args['generative_model'], 
                                           args['downsample_method'], args['downsample_config'], 
                                           args['results_folder'])
    
    ## CREATE FOLDERS
    if args['generative_model'] == 'autoencoder':
        args['results_folder'] = os.path.join(args['results_folder'],'latent_'+str(args['nz']))

    if not os.path.exists(os.path.join(args['results_folder'])):
        os.makedirs(os.path.join(args['results_folder']))

    summary_path = os.path.join(args['results_folder'],'summaries',args['name'])
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)  

    checkpoint_path = os.path.join(args['results_folder'],'checkpoints', args['name'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    samples_path = os.path.join(args['results_folder'],'samples', args['name'])
    if not os.path.exists(samples_path):
        os.makedirs(samples_path)

    prediction_path = os.path.join(args['results_folder'],'predictions', args['name'])
    if not os.path.exists(prediction_path):
        os.makedirs(prediction_path)

    if not os.path.exists(args['downsample_directory']):
        os.makedirs(args['downsample_directory'])
        
    if args['hardcode_down_ref']:
        if args['dataset'] == 'COMA' and args['downsample_method'] == 'COMA_downsample':
            reference_points = [[3567,4051,4597],
                                [1010,1081,1170],
                                [256, 276, 295],
                                [11, 69, 74],
                                [17, 17, 17]]
        elif args['dataset'] == 'COMA' and args['downsample_method'] == 'meshlab_downsample' and\
                args['downsample_config'] == 'preserve_topology=True_preserve_boundary=False':
            reference_points = [[3567, 4051, 4597],
                                 [1105, 1214, 1241],
                                 [289, 310, 318],
                                 [70, 80, 85],
                                 [2, 19, 24]]
        else:
            raise NotImplementedError
    else:
        if args['dataset'] == 'COMA':
            reference_points = [[3567,4051,4597]] 
        elif args['dataset'] == 'mein3d' or args['dataset'] == 'mein3d_texture':
            reference_points = [[23822]] 
        elif args['dataset'] == 'DFAUST':
            reference_points = [[414]]
            
            
    ## INITIALIZE DATASET, DOWNSAMPLINGS AND GRAPH
    print("Loading data .. ")
    if not os.path.exists(args['data']+'/mean.npy') or not os.path.exists(args['data']+'/std.npy'):
        facedata = FaceData(nVal=args['nVal'], train_file=args['data']+'/train.npy',
                                 test_file=args['data']+'/test.npy', reference_mesh_file=args['reference_mesh_file'],
                                 pca_n_comp=args['nz'], normalization = args['normalization'],\
                                 meshpackage = meshpackage, load_flag = True)
        np.save(args['data']+'/mean.npy', facedata.mean)
        np.save(args['data']+'/std.npy', facedata.std)
    else:
        facedata = FaceData(nVal=args['nVal'], train_file=args['data']+'/train.npy',\
                            test_file=args['data']+'/test.npy', reference_mesh_file=args['reference_mesh_file'],\
                            pca_n_comp=args['nz'], normalization = args['normalization'],\
                            meshpackage = meshpackage, load_flag = False)
        facedata.mean = np.load(args['data']+'/mean.npy')
        facedata.std = np.load(args['data']+'/std.npy')
        facedata.n_vertex = facedata.mean.shape[0]
        facedata.n_features = facedata.mean.shape[1]

    if not os.path.exists(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl')):
        if facedata.meshpackage == 'trimesh':
            raise NotImplementedError
        print("Generating Transform Matrices ..")


        if args['downsample_method'] == 'COMA_downsample':
            M,A,D,U,F = mesh_sampling.generate_transform_matrices(facedata.reference_mesh, args['ds_factors'])
        elif args['downsample_method'] == 'meshlab_downsample':
            M,A,D,U,F = mesh_sampling.generate_transform_matrices_given_downsamples(facedata.reference_mesh,
                                                                                    args['downsample_directory'],
                                                                                    len(args['ds_factors']))
        else:
            raise NotImplementedError(downsample_method)

        with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'wb') as fp:
            M_verts_faces = [(M[i].v, M[i].f) for i in range(len(M))]
            pickle.dump({'M_verts_faces':M_verts_faces,'A':A,'D':D,'U':U,'F':F}, fp)
    else:
        print("Loading Transform Matrices ..")
        with open(os.path.join(args['downsample_directory'],'downsampling_matrices.pkl'), 'rb') as fp:
            downsampling_matrices = pickle.load(fp,encoding = 'latin1')

        M_verts_faces = downsampling_matrices['M_verts_faces']
        if facedata.meshpackage == 'mpi-mesh':
            M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
        elif facedata.meshpackage == 'trimesh':
            M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process = False)\
                 for i in range(len(M_verts_faces))]
        A = downsampling_matrices['A']
        D = downsampling_matrices['D']
        U = downsampling_matrices['U']
        F = downsampling_matrices['F']
        
    if not args['hardcode_down_ref']:
        print("Calculating reference points for downsampled versions..")
        for i in range(len(args['ds_factors'])):
            if facedata.meshpackage == 'mpi-mesh':
                dist = euclidean_distances(M[i+1].v, M[0].v[reference_points[0]])
            elif facedata.meshpackage == 'trimesh':
                dist = euclidean_distances(M[i+1].vertices, M[0].vertices[reference_points[0]])
            reference_points.append(np.argmin(dist,axis=0).tolist())
            
    ## COMPUTE ORDERING

    if facedata.meshpackage == 'mpi-mesh':
        sizes = [x.v.shape[0] for x in M]
    elif facedata.meshpackage == 'trimesh':
        sizes = [x.vertices.shape[0] for x in M]
#     if args['dummy_node']:
#         sizes = [size + 1 for size in sizes]
    Adj, Trigs = get_adj_trigs(A, F, facedata.reference_mesh, meshpackage = facedata.meshpackage)

    spirals_np, spiral_sizes, spirals = generate_spirals(args['step_sizes'], M, Adj, Trigs, \
                                                        reference_points = reference_points, \
                                                        dilation = args['dilation'], random = False, \
                                                        meshpackage = facedata.meshpackage, counter_clockwise = True)
    
    bU = []
    bD = []
    for i in range(len(D)):
        if args['dummy_node']:
            d = np.zeros((1,D[i].shape[0]+1,D[i].shape[1]+1))
            u = np.zeros((1,U[i].shape[0]+1,U[i].shape[1]+1))
            d[0,:-1,:-1] = D[i].todense()
            u[0,:-1,:-1] = U[i].todense()
            d[0,-1,-1] = 1
            u[0,-1,-1] = 1
        else:
            d = np.zeros((1,D[i].shape[0],D[i].shape[1]))
            u = np.zeros((1,U[i].shape[0],U[i].shape[1]))
            d[0,:,:] = D[i].todense()
            u[0,:,:] = U[i].todense()
        bD.append(d)
        bU.append(u)
        
#     for a in A:
#         a.setdiag(1)

    ### TRANSFER DATA TO GPU    

    if args['GPU']:
        device = torch.device("cuda:"+str(args['device_idx']) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    if args['dummy_node']:
        tspirals = [torch.from_numpy(s).long().to(device) for s in spirals_np]
    else:
        raise NotImplementedError
    tD = [torch.from_numpy(s).float().to(device) for s in bD]
    tU = [torch.from_numpy(s).float().to(device) for s in bU]
    
    
    ### INITIALIZE DATALOADERS, MODELS, OPTIMIZERS, SCHEDULERS
    # Building model, optimizer, and loss function

    if args['mode'] == 'train':
        dataset_train = autoencoder_dataset(root_dir = args['data'], points_dataset = 'train',
                                            facedata = facedata,
                                            normalization = args['normalization'], 
                                            dummy_node = args['dummy_node'])

        dataloader_train = DataLoader(dataset_train, batch_size=args['batch_size'],\
                                             shuffle = args['shuffle'], num_workers = args['num_workers'])

        dataset_val = autoencoder_dataset(root_dir = args['data'], points_dataset = 'val', 
                                          facedata = facedata,
                                          normalization = args['normalization'],
                                          dummy_node = args['dummy_node'])

        dataloader_val = DataLoader(dataset_val, batch_size=args['batch_size'],\
                                         shuffle = False, num_workers = args['num_workers'])

    
    dataset_test = autoencoder_dataset(root_dir = args['data'], points_dataset = args['test_set'],
                                       facedata = facedata,
                                       normalization = args['normalization'],
                                       dummy_node = args['dummy_node'])

    dataloader_test = DataLoader(dataset_test, batch_size=args['batch_size'],\
                                         shuffle = False, num_workers = args['num_workers'])
    
    
    if 'autoencoder' in args['generative_model']:
        model = SpiralAutoencoder_extra_conv(filters_enc = args['filter_sizes_enc'],   
                                             filters_dec = args['filter_sizes_dec'],
                                             latent_size=args['nz'],
                                             sizes=sizes,
                                             spiral_sizes=spiral_sizes,
                                             spirals=tspirals,
                                             D=tD, U=tU,device=device,
                                             injection = args['injection'],
                                             residual = args['residual'],
                                             order = args['order'],
                                             normalize = args['normalize'],
                                             model = args['model'],
                                             activation = args['activation']).to(device)
        

    if args['mode'] == 'train':
        optim = torch.optim.Adam(model.parameters(),lr=args['lr'],weight_decay=args['regularization'])
        if args['scheduler']:
            scheduler=torch.optim.lr_scheduler.StepLR(optim, args['decay_steps'],gamma=args['decay_rate'])
        else:
            scheduler = None

    if args['loss']=='l1':
        loss_fn = loss_l1
        
    ## PARAMS
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params)) 
    print(model)
    # print(M[4].v.shape)
    
    ### TRAINING LOOP
    if args['mode'] == 'train':
        writer = SummaryWriter(summary_path)
        with open(os.path.join(args['results_folder'],'checkpoints', args['name'] +'_params.json'),'w') as fp:
            saveparams = copy.deepcopy(args)
            json.dump(saveparams, fp)

        if args['resume']:
                print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file'])))
                checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),\
                                             map_location = device)
                start_epoch = checkpoint_dict['epoch'] + 1
                model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])
                optim.load_state_dict(checkpoint_dict['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint_dict['scheduler_state_dict'])
                print('Resuming from epoch %s'%(str(start_epoch)))     
        else:
            start_epoch = 0

        if args['generative_model'] == 'autoencoder':
            train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                         device, model, optim, loss_fn,
                                         bsize = args['batch_size'],
                                         start_epoch = start_epoch,
                                         n_epochs = args['num_epochs'],
                                         eval_freq = args['eval_frequency'],
                                         scheduler = scheduler,
                                         writer = writer,
                                         save_recons=args['save_recons'],
                                         facedata=facedata,
                                         metadata_dir=checkpoint_path, 
                                         samples_dir=samples_path,
                                         checkpoint_path = args['checkpoint_file'])
    elif args['mode'] == 'test':
        print('loading checkpoint from file %s'%(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar')))
        checkpoint_dict = torch.load(os.path.join(checkpoint_path,args['checkpoint_file']+'.pth.tar'),map_location=device)
        model.load_state_dict(checkpoint_dict['autoencoder_state_dict'])

        predictions, norm_l1_loss, l2_loss = test_autoencoder_dataloader(device, model, dataloader_test, 
                                                                         facedata, mm_constant = args['mm_constant'])    
        np.save(os.path.join(prediction_path,'predictions'), predictions)   

        print('autoencoder: normalized loss', norm_l1_loss)

        print('autoencoder: euclidean distance in mm=', l2_loss)
            
            
            
            
            
            
            
            

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()

    # model hyperparameters
    parser.add_argument('--generative_model', type=str, default= 'autoencoder')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='DFAUST')
    parser.add_argument('--downsample_method', type=str, default='COMA_downsample')
    parser.add_argument('--downsample_config', type=str, default='')
    parser.add_argument('--results_folder', type=str, default= 'temp')
    parser.add_argument('--hardcode_down_ref', type=str2bool, default=False)
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint')
    
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--eval_frequency', type=int, default=200)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_threads', type=int, default=1)
    
    
    parser.add_argument('--filter_sizes_enc', type=str2ListOfLists2int, default=[[3, 16, 16, 16, 32],[[],[],[],[],[]]])
    parser.add_argument('--filter_sizes_dec', type=str2ListOfLists2int, default=[[32, 32, 16, 16, 3],[[],[],[],[],[]]])
    parser.add_argument('--nz', type=int, default=16)
    parser.add_argument('--dummy_node', type=str2bool, default=True)
    parser.add_argument('--ds_factors', type=str2list2int, default=[4, 4, 4, 4])
    parser.add_argument('--step_sizes', type=str2list2int, default=[1, 1, 1, 1, 1])
    parser.add_argument('--dilation', type=str2list2int, default=None)
#     parser.add_argument('--dilation', type=str2bool, default=[2, 2, 2, 1, 1])

    parser.add_argument('--injection', type=str2bool, default= True)
    parser.add_argument('--order', type=int, default= 2)
    parser.add_argument('--model', type=str, default= 'full')
    parser.add_argument('--normalize', type=str, default= 'final')
    parser.add_argument('--residual', type=str2bool, default= False)
    parser.add_argument('--activation', type=str, default= 'elu')
    
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--regularization', type=float, default=5e-5)
    parser.add_argument('--scheduler', type=str2bool, default=True)
    parser.add_argument('--decay_rate', type=float, default=0.99)
    parser.add_argument('--decay_steps', type=int, default=1)
    parser.add_argument('--loss', type=str, default='l1')
    
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--nVal', type=int, default=100)
    parser.add_argument('--normalization', type=str2bool, default=True)
    parser.add_argument('--mm_constant', type=int, default=1000)
    
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--save_recons', type=str2bool, default= True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')
    
    parser.add_argument('--GPU', type=str2bool, default=True)
    parser.add_argument('--device_idx', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    main(vars(args))