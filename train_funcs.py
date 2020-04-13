def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
    
    
import os
import torch
import numpy as np
import random


if isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

def train_autoencoder_dataloader(dataloader_train, dataloader_val,
                                      device, model, optim, loss_fn, 
                                      bsize, start_epoch, n_epochs, eval_freq, scheduler = None,
                                      writer=None, save_recons=True, facedata=None,
                                      metadata_dir=None, samples_dir = None, checkpoint_path = None):
    if not facedata.normalization:
        facedata_mean = torch.Tensor(facedata.mean).to(device)
        facedata_std = torch.Tensor(facedata.std).to(device)
    
    total_steps = start_epoch*len(dataloader_train)

    for epoch in range(start_epoch, n_epochs):
        model.train()

        tloss = []
        for b, sample_dict in enumerate(tqdm(dataloader_train)):
            optim.zero_grad()
                
            tx = sample_dict['points'].to(device)
            cur_bsize = tx.shape[0]
            
            tx_hat = model(tx)
            loss = loss_fn(tx, tx_hat)

            loss.backward()
            optim.step()
            
            if facedata.normalization:
                tloss.append(cur_bsize * loss.item())
            else:
                with torch.no_grad():
                    if facedata.mean.shape[0]!=tx.shape[1]:
                        tx_norm = tx[:,:-1,:]
                        tx_hat_norm = tx_hat[:,:-1,:]
                    else:
                        tx_norm = tx
                        tx_hat_norm = tx_hat
                    tx_norm = (tx_norm - facedata_mean)/facedata_std
                    tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    tx_hat_norm = (tx_hat_norm -facedata_mean)/facedata_std
                    tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    loss_norm = loss_fn(tx_norm, tx_hat_norm)
                    tloss.append(cur_bsize * loss_norm.item())
            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b, sample_dict in enumerate(tqdm(dataloader_val)):

                tx = sample_dict['points'].to(device)
                cur_bsize = tx.shape[0]

                tx_hat = model(tx)               
                loss = loss_fn(tx, tx_hat)
                
                if facedata.normalization:
                    vloss.append(cur_bsize * loss.item())
                else:
                    with torch.no_grad():
                        if facedata.mean.shape[0]!=tx.shape[1]:
                            tx_norm = tx[:,:-1,:]
                            tx_hat_norm = tx_hat[:,:-1,:]
                        else:
                            tx_norm = tx
                            tx_hat_norm = tx_hat
                        tx_norm = (tx_norm - facedata_mean)/facedata_std
                        tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                        tx_hat_norm = (tx_hat_norm -facedata_mean)/facedata_std
                        tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                        loss_norm = loss_fn(tx_norm, tx_hat_norm)
                        vloss.append(cur_bsize * loss_norm.item())   

        if scheduler:
            scheduler.step()
            
        epoch_tloss = sum(tloss) / float(len(dataloader_train.dataset))
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        if len(dataloader_val.dataset) > 0:
            epoch_vloss = sum(vloss) / float(len(dataloader_val.dataset))
            writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
            print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        else:
            print('epoch {0} | tr {1} '.format(epoch,epoch_tloss))
        model = model.cpu()
  
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    facedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
                                                     msh, mesh_ind)
                mesh_ind = [0]
                msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                facedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
                                                 msh, mesh_ind)

    print('~FIN~')


def train_autoencoder(device, model, optim, loss_fn, \
                      train_data, valid_data, bsize,\
                      start_epoch, n_epochs, eval_freq, scheduler = None, \
                      writer=None, shuffle=True, save_recons=True, facedata=None,\
                      metadata_dir=None, samples_dir = None, checkpoint_path = None):

    total_steps = start_epoch * len(list(range(int(np.ceil(float(train_data.shape[0])/bsize)))))
    
    if not facedata.normalization:
        facedata_mean = torch.Tensor(facedata.mean).to(device)
        facedata_std = torch.Tensor(facedata.std).to(device)
    
    
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        if shuffle:
            np.random.shuffle(train_data)
        # train for epoch
        tloss = []
        for b in tqdm(range(int(np.ceil(float(train_data.shape[0])/bsize)))):
            optim.zero_grad()
            
            tx = torch.from_numpy(train_data[b*bsize:min(b*bsize+bsize,train_data.shape[0])]).to(device)
            tx_hat = model(tx)
            loss = loss_fn(tx, tx_hat)

            
            loss.backward()
            optim.step()
            
            if facedata.normalization:
                tloss.append(tx.shape[0] * loss.item())
            else:
                with torch.no_grad():
                    tx_norm = (tx[:,:-1,:] - facedata_mean)/facedata_std
                    tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    tx_hat_norm = (tx_hat[:,:-1,:] -facedata_mean)/facedata_std
                    tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                    
                    loss_norm = loss_fn(tx_norm, tx_hat_norm)
                    tloss.append(tx.shape[0] * loss_norm.item())
                    
            if writer and total_steps % eval_freq == 0:
                if facedata.normalization:
                    writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                else:
                    writer.add_scalar('loss/loss/data_loss',loss_norm.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b in range(valid_data.shape[0]//bsize):
                
                tx = torch.from_numpy(valid_data[b*bsize:min(b*bsize+bsize,valid_data.shape[0])]).to(device)
                tx_hat = model(tx)
                loss = loss_fn(tx,tx_hat)

                if facedata.normalization:
                    vloss.append(tx.shape[0] * loss.item())
                else:
                    with torch.no_grad():
                        tx_norm = (tx[:,:-1,:] - facedata_mean)/facedata_std
                        tx_norm = torch.cat((tx_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)

                        tx_hat_norm = (tx_hat[:,:-1,:] -facedata_mean)/facedata_std
                        tx_hat_norm = torch.cat((tx_hat_norm,torch.zeros(tx.shape[0],1,tx.shape[2]).to(device)),1)
                        
                        loss_norm = loss_fn(tx_norm, tx_hat_norm)
                        vloss.append(tx.shape[0] * loss_norm.item())                
                

        if scheduler:
            scheduler.step()
        
        epoch_tloss = sum(tloss) / float(train_data.shape[0])
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        epoch_vloss = sum(vloss) / float(valid_data.shape[0])
        writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
        print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        
        model = model.cpu()
  
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    facedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
                                                     msh, mesh_ind)
                mesh_ind = [0]
                msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                facedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
                                                 msh, mesh_ind)

    print('~FIN~')

def train_autoencoder_orderings(device, model, optim, loss_fn, \
                      train_data, train_spirals, valid_data, valid_spirals, bsize,\
                      start_epoch, n_epochs, eval_freq, random_shift = 'None', scheduler = None, \
                      writer=None, shuffle=True, save_recons=True, facedata=None,\
                      metadata_dir=None, samples_dir = None, checkpoint_path = None):

    total_steps = start_epoch * len(list(range(int(np.ceil(float(train_data.shape[0])/bsize)))))
    
    for epoch in range(start_epoch, n_epochs):
        model.train()
        
        spirals_all = []
        if random_shift == 'epoch':
            if len(train_spirals) == 1:
                spirals_all.append([[[[spiral_hierarchy[j][0]] + \
                                     list(np.roll(spiral_hierarchy[j][1:], \
                                        random.randint(0,len(spiral_hierarchy[j][1:])-1)))][0]\
                                            for j in range(len(spiral_hierarchy))]\
                                                for spiral_hierarchy in train_spirals[0]])
            else:
                raise NotImplementedError
        elif random_shift == 'batch' or random_shift == 'None':
            spirals_all = train_spirals
        else:
             raise NotImplementedError
        
        if shuffle:
#             import pdb;pdb.set_trace()
            perm = np.random.permutation(train_data.shape[0])
            train_data = train_data[perm]
            if len(spirals_all) == train_data.shape[0]:
                spirals_all = [spirals_all[i] for i in perm]
        tloss = []
        for b in tqdm(range(int(np.ceil(float(train_data.shape[0])/bsize)))):
            if random_shift == 'batch':
                spirals_all = []
                if len(train_spirals) == 1:
                    spirals_all.append([[[[spiral_hierarchy[j][0]] + \
                                         list(np.roll(spiral_hierarchy[j][1:], \
                                            random.randint(0,len(spiral_hierarchy[j][1:])-1)))][0]\
                                                for j in range(len(spiral_hierarchy))]\
                                                    for spiral_hierarchy in train_spirals[0]])
                else:
                    raise NotImplementedError
            else:
                pass     
            
            
            
            
            optim.zero_grad()
#             import pdb;pdb.set_trace()
            tx = torch.from_numpy(train_data[b*bsize:min(b*bsize+bsize,train_data.shape[0])]).to(device)
            if len(spirals_all) == 1:
                spirals_x = spirals_all
            else:
                spirals_x = spirals_all[b*bsize:min(b*bsize+bsize,len(spirals_all))]
            tx_hat = model(tx, spirals_x)
            loss = loss_fn(tx, tx_hat)

            
            loss.backward()
            optim.step()
            
            tloss.append(tx.shape[0] * loss.item())
            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('loss/loss/data_loss',loss.item(),total_steps)
                writer.add_scalar('training/learning_rate', optim.param_groups[0]['lr'],total_steps)
            total_steps += 1

        # validate
        model.eval()
        vloss = []
        with torch.no_grad():
            for b in range(valid_data.shape[0]//bsize):
                
                tx = torch.from_numpy(valid_data[b*bsize:min(b*bsize+bsize,valid_data.shape[0])]).to(device)
                if len(valid_spirals) == 1:
                    spirals_x = valid_spirals
                else:
                    spirals_x = valid_spirals[b*bsize:min(b*bsize+bsize,len(valid_spirals))]
                tx_hat = model(tx, spirals_x)
                loss = loss_fn(tx,tx_hat)
                
                vloss.append(tx.shape[0] * loss.item())

        if scheduler:
            scheduler.step()

        epoch_tloss = sum(tloss) / float(train_data.shape[0])
        writer.add_scalar('avg_epoch_train_loss',epoch_tloss,epoch)
        epoch_vloss = sum(vloss) / float(valid_data.shape[0])
        writer.add_scalar('avg_epoch_valid_loss', epoch_vloss,epoch)
        print('epoch {0} | tr {1} | val {2}'.format(epoch,epoch_tloss,epoch_vloss))
        
        model = model.cpu()
  
        torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'autoencoder_state_dict': model.state_dict(),
            'optimizer_state_dict' : optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))

        model = model.to(device)

        if save_recons:
            with torch.no_grad():
                if epoch == 0:
                    mesh_ind = [0]
                    msh = tx[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                    facedata.save_meshes(os.path.join(samples_dir,'input_epoch_{0}'.format(epoch)),
                                                     msh, mesh_ind)
                mesh_ind = [0]
                msh = tx_hat[mesh_ind[0]:1,0:-1,:].detach().cpu().numpy()
                facedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch)),
                                                 msh, mesh_ind)

    print('~FIN~')
    
    
    
    
##### Code obtained and modified from https://github.com/eriklindernoren/PyTorch-GAN#wasserstein-gan-gp, Copyright 2018 Erik Linder-Noren


def compute_gradient_penalty(discriminator, real, fake, device):
    if real.dim() == 3:

        alpha = torch.rand(real.size(0), 1, 1).to(device)
    else:
        alpha = torch.rand(real.size(0),1).to(device)
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    fake_ = torch.ones(real.size(0),1).requires_grad_(False).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan(device, discriminator, generator, optim_D, optim_G, \
               train_data, bsize, \
               start_epoch, n_epochs, eval_freq, scheduler_G, scheduler_D, \
               lambda_gp, n_critic, fixed_noise_size, generate_noise,\
               writer=None, shuffle=True, save_generations=True, save_renderings = False, facedata=None, \
               metadata_dir=None, samples_dir = None, checkpoint_path = None):

    total_steps = start_epoch * len(list(range(int(np.ceil(float(train_data.shape[0])/bsize)))))

    fixed_noise = generate_noise(fixed_noise_size, generator.latent_size)

    for epoch in range(start_epoch, n_epochs):
        if shuffle:
            np.random.shuffle(train_data)
        generator.train()
        discriminator.train()
        # train for epoch
        glosses = []
        dlosses = []
        gp_losses = []
        w_divergence_losses = []
        for b in tqdm(range(int(np.ceil(float(train_data.shape[0])/bsize)))):
            # critic
            for p in discriminator.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = False
            optim_D.zero_grad()

            # REAL
            tx = torch.from_numpy(train_data[b*bsize:min(b*bsize+bsize,train_data.shape[0])]).to(device)
            real_validity = discriminator(tx)
            # FAKE
            z = generate_noise(tx.size(0), generator.latent_size).to(device)
            gen_meshes = generator(z)
            fake_validity = discriminator(gen_meshes.detach())
            # GRADIENT PENALTY
            gradient_penalty = compute_gradient_penalty(discriminator,tx,gen_meshes.detach(),device)

            w_divergence_neg = torch.mean(fake_validity) - torch.mean(real_validity); gp_loss = lambda_gp * gradient_penalty
            dloss = w_divergence_neg + gp_loss

            dloss.backward()
            optim_D.step()

            dlosses.append(-tx.size(0) * dloss.item())
            w_divergence_losses.append(-tx.size(0) * w_divergence_neg.item())
            gp_losses.append(tx.size(0) * gp_loss.item())

            if total_steps % n_critic == 0: # only train generator when critic converges
                for p in discriminator.parameters():
                    p.requires_grad = False
                for p in generator.parameters():
                    p.requires_grad = True

                optim_G.zero_grad()
                z = generate_noise(bsize, generator.latent_size).to(device)
                gen_meshes = generator(z)
                fake_validity = discriminator(gen_meshes)
                #print(gen_meshes[0][0])

                gloss = -fake_validity.mean()
                gloss.backward()
                optim_G.step()
                glosses.append(gloss.item())

            if writer and total_steps % eval_freq == 0:
                writer.add_scalar('gan/loss/generator_loss',gloss.item(),total_steps)
                writer.add_scalar('gan/loss/critic_loss',-dloss.item(),total_steps)
                writer.add_scalar('gan/loss/gradient_penalty_loss',gp_loss.item(),total_steps)
                writer.add_scalar('gan/loss/w_divergence',-w_divergence_neg.item(),total_steps)

                writer.add_scalar('training/learning_rate_discriminator', optim_D.param_groups[0]['lr'],total_steps)
                writer.add_scalar('training/learning_rate_generator', optim_G.param_groups[0]['lr'],total_steps)
                writer.add_scalar('training/lambda_gp', lambda_gp,total_steps)
            total_steps += 1

        if scheduler_G:
            scheduler_G.step()
        if scheduler_D:
            scheduler_D.step()

        epoch_gloss = sum(glosses) / float(len(glosses))
        writer.add_scalar('avg_epoch_generator_loss', epoch_gloss,epoch)
        epoch_dloss = sum(dlosses) / float(train_data.shape[0])
        writer.add_scalar('avg_epoch_critic_loss',epoch_dloss,epoch)
        epoch_w_divergence_loss = sum(w_divergence_losses) / float(train_data.shape[0])
        writer.add_scalar('avg_epoch_w_divergence', epoch_w_divergence_loss,epoch)
        epoch_gp_loss = sum(gp_losses) / float(train_data.shape[0])
        writer.add_scalar('avg_epoch_gradient_penalty_loss',epoch_gp_loss,epoch)
        print('epoch {0} | gloss {1} | dloss {2}'.format(epoch,epoch_gloss,epoch_dloss))

        generator = generator.cpu(); discriminator = discriminator.cpu()
        
        torch.save({'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_generator_state_dict' : optim_G.state_dict(),
            'optimizer_discriminator_state_dict': optim_D.state_dict(),
            'scheduler_generator_state_dict': scheduler_G.state_dict(),
            'scheduler_discriminator_state_dict': scheduler_D.state_dict(),                    
        },os.path.join(metadata_dir, checkpoint_path+'.pth.tar'))
        
        if epoch % 10 == 0:
            torch.save({'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_generator_state_dict' : optim_G.state_dict(),
            'optimizer_discriminator_state_dict': optim_D.state_dict(),
            'scheduler_generator_state_dict': scheduler_G.state_dict(),
            'scheduler_discriminator_state_dict': scheduler_D.state_dict(),    
            },os.path.join(metadata_dir, checkpoint_path+'%s.pth.tar'%(epoch)))
            
        generator = generator.to(device); discriminator = discriminator.to(device)

        if save_generations:
            with torch.no_grad():
                generator.eval()
                gen_meshes = generator(fixed_noise.to(device))
                msh = gen_meshes[:,0:-1,:].detach().cpu().numpy()
                if not os.path.exists(os.path.join(samples_dir,'epoch_{0}'.format(epoch))):
                    os.makedirs(os.path.join(samples_dir,'epoch_{0}'.format(epoch)))
                facedata.save_meshes(os.path.join(samples_dir,'epoch_{0}'.format(epoch),\
                                     'epoch_{0}'.format(epoch)), msh, list(range(0,gen_meshes.size(0))))
        elif save_renderings:
            raise NotImplementedError

    print('~FIN~')





