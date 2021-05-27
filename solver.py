import os
import json
import time
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils.measureLips import spectralNorms
from utils.prune import *
from utils.common_utils import *
from Mypnufft_mc_func_cardiac import *

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.core.debugger import set_trace


class Solver():
    def __init__(self, module, opt):
        
        self.opt = opt
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")       
        
        self.prepare_dataset()
        self.writer = SummaryWriter(opt.ckpt_root)
        
        self.net = module.Net(opt).to(self.dev)
        self.net_input_set = get_input_manifold(opt.input_type, opt.num_cycle, self.Nfr).to(self.dev)
        
        p = get_params(opt.opt_over, self.net, self.net_input_set)
        
        if opt.input_type.endswith('mapping'):
            self.mapnet = module.MappingNet(opt).to(self.dev)            
            print('... adding params of mapping network ...')
            p += self.mapnet.parameters()
            self.net_input_set = self.mapnet(self.net_input_set).reshape((self.Nfr,1,
                                                    self.img_size//self.opt.up_factor, 
                                                    self.img_size//self.opt.up_factor))
            
        # Compute number of parameters          
        s  = sum([np.prod(list(pnb.size())) for pnb in p]);
        print ('# params: %d' % s)

        self.step = 0
        self.optimizer = torch.optim.Adam(p, lr=opt.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=opt.step_size, 
                                                    gamma=opt.gamma)
        if opt.isresume is not None:
            self.load(opt.isresume)
        
        self.loss_fn = nn.MSELoss()
        

        self.t1, self.t2 = None, None
        self.best_psnr, self.best_psnr_step = 0, 0
        self.best_ssim, self.best_ssim_step = 0, 0

        #flag for pruning 
        self.pruned = False 

    def fit(self):
        opt = self.opt
        batch_size = opt.batch_size   
        Nc = self.Nc
        Nfr = self.Nfr
        Nvec = self.Nvec
        Nfibo = opt.Nfibo
        coil = self.coil
        denc = self.denc
        net_input_set = self.net_input_set
        syn_radial_ri_ts = self.syn_radial_ri_ts    
        step = self.step
        self.t1 = time.time()
        while step < opt.max_steps:
            # randomly pick frames to train (batch, default = 1)
            idx_fr=np.random.randint(0, Nfr)
            idx_frs = range(min(idx_fr, Nfr-batch_size), min(idx_fr+batch_size, Nfr))

            net_input_z = torch.autograd.Variable(net_input_set[idx_frs,:,:,:],requires_grad=False) # net_input_set: e.g., torch.Size([23*num_cycle, 1, 8, 8])   

            out_sp = self.net(net_input_z) # e.g., spatial domain output (img) torch.Size([batch_size, 2, 128, 128])
            out_sp = out_sp.permute(0,2,3,1) 
            out_kt = []
            gt_kt = []

            for idx_b in range(batch_size):
                idx_fr = idx_frs[idx_b]
                angle = self.set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:] # (3328, 2) 3328 = 13*256
                gt_kt.append(syn_radial_ri_ts[0,:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:,:].reshape(-1,2)) # syn_radial_ri_ts torch.Size([1, 32, 299, 256, 2])
                self.mynufft.X=out_sp[idx_b,:,:,:]
                out_kt.append(self.mynufft(angle,angle.shape[0]//Nvec,Nvec,Nc,coil,denc[:,:angle.shape[0]//Nvec,:]).reshape(-1,2))

            out_kt = torch.cat(out_kt)
            gt_kt = torch.cat(gt_kt)

            mse_loss = (self.loss_fn(gt_kt[...,0],out_kt[...,0]) + self.loss_fn(gt_kt[...,1],out_kt[...,1]))*(self.img_size)**2

            total_loss = mse_loss     
            self.optimizer.zero_grad()
            total_loss.backward()

            if (opt.prune):

                if self.pruned and (step % opt.save_period == 0 or step == opt.max_steps-1): 
                    removeReparam(self.net)
                    self.pruned = False 

                if (step in opt.pruneSteps):  
                    pruneOnStep(opt, step, self.net)
                    self.pruned = True 

            self.optimizer.step()
            self.scheduler.step()
            
            self.writer.add_scalar('total loss/training_loss (mse)', total_loss, step)
            
            if opt.PLOT and (step % opt.save_period == 0 or step == opt.max_steps-1):
                self.summary_and_save(step,out_sp, idx_fr)

            step += 1
            self.step = step
        
        self.writer.close()   

        if (opt.save_residuals):
            #if save residuals, save also video of residuals
            self.save_video_residuals()

        self.save_video()

    def summary_and_save(self, step, out_sp, idx_fr):        
        max_steps = self.opt.max_steps
        
        # For visualization
        idx_fr = np.random.randint(0, self.Nfr)
        out_abs=(out_sp[0,:,:,0]**2+out_sp[0,:,:,1]**2)**.5 
        out_abs = out_abs-out_abs.min()
        out_abs = out_abs/out_abs.max()
        
        syn_radial_img_ts = torch.from_numpy(self.syn_radial_img[:,:,idx_fr]).to(self.dev).float()
        gt_cartesian_img_ts = torch.from_numpy(self.gt_cartesian_img[:,:,idx_fr]).to(self.dev).float()

        images_grid = torch.cat([syn_radial_img_ts[None],out_abs[None],gt_cartesian_img_ts[None]],dim=2)
        images_grid = F.interpolate(images_grid.unsqueeze(0), scale_factor = 4).squeeze(0)
        self.writer.add_image('recon_image', images_grid, step)
        
        # Get Average PSNR and SSIM values for entire frames
        psnr_val = 0
        ssim_val = 0
        for idx_fr in range(self.Nfr):
            ims = torch_to_np(self.net(self.net_input_set[idx_fr,:,:,:][None]))
            tmp_ims=np.sqrt(ims[0,:,:]**2+ims[1,:,:]**2)
            tmp_ims -= tmp_ims.min() #here the 
            tmp_ims /= tmp_ims.max()
            psnr_val += psnr(self.gt_cartesian_img[:,:,idx_fr], tmp_ims, data_range=1.0)
            ssim_val += ssim(self.gt_cartesian_img[:,:,idx_fr], tmp_ims, data_range=1.0)
        
        psnr_val = psnr_val/self.Nfr
        ssim_val = ssim_val/self.Nfr
        self.writer.add_scalar('metrics/psnr', psnr_val, step)
        self.writer.add_scalar('metrics/ssim', ssim_val, step)
        
        self.t2 = time.time()

        if psnr_val >= self.best_psnr:
            self.best_psnr, self.best_psnr_step = psnr_val, step
        if ssim_val >= self.best_ssim:
            self.best_ssim, self.best_ssim_step = ssim_val, step
        
        self.save(step)

        curr_lr = self.scheduler.get_lr()[0]
        eta = (self.t2-self.t1) * (max_steps-step) /self.opt.save_period / 3600

        output_string = "[{}/{}] {:.2f} {:.4f} (Best PSNR: {:.2f} SSIM {:.4f} @ {} step) LR: {}, ETA: {:.1f} hours".format(step, max_steps, psnr_val, ssim_val, self.best_psnr, self.best_ssim, self.best_psnr_step,curr_lr, eta)

        if (self.opt.measure_L):
            norms = spectralNorms(self.net.net) 
            if (self.mapnet != None):
                norms.extend(spectralNorms(self.mapnet.net))
            L = np.product(norms)
            output_string += ", L: {:.4f}".format(L)
            self.writer.add_scalar('metrics/spectralnorm', L, step)

        print(output_string)

        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):        
        raise NotImplementedError("Evaluate function is not implemented yet.")
        return 

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.net_input_set = checkpoint['net_input_set'].to(self.dev)
        self.net.load_state_dict(checkpoint['net_state_dict']) 
        self.net.to(self.dev)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.step_size, gamma=self.opt.gamma) 
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']+1
        
        
    def save(self, step):        
        print('saving ... ')
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        ckptdict = {
                'step': step,
                'net_input_set': self.net_input_set,
                'net_state_dict': self.net.state_dict(), 
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'mapnet_state_dict' : self.mapnet.state_dict() #TODO: add this only if mapnet included
                }
        best_scores = {
                'best_psnr': self.best_psnr,
                'best_ssim': self.best_ssim,
                'best_psnr_step': self.best_psnr_step,
                'best_ssim_step': self.best_ssim_step,
                }
                
        torch.save(ckptdict, save_path)
        with open(os.path.join(self.opt.ckpt_root, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)
        
    def save_video(self):
        
        #h5 save
        ims = []
        inp = []

        for idx_fr in range(self.Nfr):
            net_input_fr=self.net_input_set[idx_fr,:,:,:][np.newaxis,:,:,:]
            out_HR_np = torch_to_np(self.net(net_input_fr))
            net_input_fr=torch_to_np(net_input_fr)
            ims.append(out_HR_np)
            inp.append(net_input_fr)

        f = h5py.File(os.path.join(self.opt.ckpt_root, 'final.h5'),'w')
        f.create_dataset('input',data=inp,dtype=np.float32)
        f.create_dataset('data',data=ims,dtype=np.float32)
        f.create_dataset('angle',data=self.set_ang,dtype=np.float32)
        f.close()
        print('h5 file saved.')
        
        print('creating video.')
        fig = plt.figure(figsize=(10, 10))
        vid = []
        for idx_fr in range(self.Nfr):
            tmp_ims=np.sqrt(ims[idx_fr][0,:,:]**2+ims[idx_fr][1,:,:]**2)
            tmp_ims -= tmp_ims.min()
            tmp_ims /= tmp_ims.max()    
            ttl = plt.text(128, -5, idx_fr, horizontalalignment='center', fontsize = 20)
            vid.append([plt.imshow(tmp_ims, animated=True, cmap = 'gray', vmax=0.5),ttl])
        ani = animation.ArtistAnimation(fig, vid, interval=50, blit=True, repeat_delay=1000)

        ani.save(self.opt.ckpt_root+'/final_video.mp4')
        print('video saved')

    def save_video_residuals(self):
        
        ims = []

        for idx_fr in range(self.Nfr):

            im = torch_to_np(self.net(self.net_input_set[idx_fr,:,:,:][None]))

            out = np.sqrt(im[0,:,:]**2+im[1,:,:]**2) 
            out -= out.min()
            out /= out.max()

            ref = self.gt_cartesian_img[:,:,idx_fr]

            diff = np.absolute(ref - out)

            ims.append(diff)
        
        print('creating video of residuals...')
        fig = plt.figure(figsize=(10, 10))
        vid = []
        for idx_fr in range(self.Nfr):
            frame = ims[idx_fr]
            ttl = plt.text(128, -5, idx_fr, horizontalalignment='center', fontsize = 20)
            vid.append([plt.imshow(frame, animated=True, cmap = 'gray', vmin=0.0, vmax=0.5),ttl])
        ani = animation.ArtistAnimation(fig, vid, interval=50, blit=True, repeat_delay=1000)

        ani.save(self.opt.ckpt_root+'/final_residuals.mp4')
        print('video of residuals saved')
    
    def prepare_dataset(self):   
        fname = self.opt.fname
        num_cycle = self.opt.num_cycle
        Nfibo = self.opt.Nfibo
        seq=np.squeeze(sio.loadmat(fname)['data']) # numpy array (128, 128, 23, 32), complex128, kt-space data
        coil=sio.loadmat(fname)['b1'].astype(np.complex64) #  numpy array, coil sensitivity
        coil = np.transpose(coil,(2,0,1)) # (32, 128, 128)

        Nc=np.shape(seq)[-1] # 32 number of coils
        Nvec=np.shape(seq)[0]*2 # 256 radial sampling number (virtual k-space)
        Nfr = np.shape(seq)[2]*num_cycle # 23 number of frames * num_cycle (13)
        img_size=np.shape(seq)[0] # 128        

        gt_cartesian_kt = seq[...,np.newaxis].astype(np.complex64) # (128, 128, 23, 32, 1), complex64, kt-space data 
        gt_cartesian_kt_ri = np.concatenate((np.real(gt_cartesian_kt),np.imag(gt_cartesian_kt)),axis=-1) # (128, 128, 23, 32, 2), float32, kt-space data 
        gt_cartesian_kt_ri = np.transpose(gt_cartesian_kt_ri,(3,2,0,1,4)) # (32, 23, 128, 128, 2), kt-space data
        gt_cartesian_kt_ri = np.concatenate([gt_cartesian_kt_ri]*num_cycle,axis = 1) # (32, 23*num_cycle, 128, 128, 2), kt-space data
        gt_cartesian_kt = np.concatenate([gt_cartesian_kt]*num_cycle,axis=2) # (128, 128, 23*num_cycle, 32, 1), complex64, kt-space data 


        w1=np.linspace(1,0,Nvec//2) # [1,...,0] length 128
        w2=np.linspace(0,1,Nvec//2) # [0,...,1] length 128
        w=np.concatenate((w1,w2),axis=0)[np.newaxis,np.newaxis] # [1,...0,0,...,1] (1, 1, 256)
        wr=np.tile(w,(Nc,Nfibo,1)) # (32, 13, 256) repeated w
        denc = wr.astype(np.complex64)
        
        # For visualization: GT full sampled images
        gt_cartesian = gt_cartesian_kt.transpose(3,2,0,1,4) # (32, 23*num_cycle, 128, 128, 1)
        gt_cartesian = gt_cartesian[:,:,:,:,0] # (32, 23*num_cycle, 128, 128)
        gt_cartesian_img = np.zeros((img_size,img_size,Nfr)) # (128, 128, 23*num_cycle)
        for idx_fr in range(Nfr):
            curr_gt_cartesian_img = np.sqrt((abs(gt_cartesian[:,idx_fr,:,:]*coil)**2).mean(0))
            curr_gt_cartesian_img -= curr_gt_cartesian_img.min()
            curr_gt_cartesian_img /= curr_gt_cartesian_img.max()
            gt_cartesian_img[:,:,idx_fr] = curr_gt_cartesian_img

        # 111.246 degree - golden angle | 23.63 degree - tiny golden angle
        gA=111.246
        one_vec_y=np.linspace(-3.1293208599090576,3.1293208599090576,num=Nvec)[...,np.newaxis]
        one_vec_x=np.zeros((Nvec,1))
        one_vec=np.concatenate((one_vec_y,one_vec_x),axis=1) # (256, 2)

        
        Nang=Nfibo*Nfr
        set_ang=np.zeros((Nang*Nvec,2),np.double) # (76544, 2)
        for i in range(Nang):
            theta=gA*i
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c,-s), (s, c)))
            for j in range(Nvec):
                tmp=np.matmul(R,one_vec[j,:])
                set_ang[i*Nvec+j,0]=tmp[0]
                set_ang[i*Nvec+j,1]=tmp[1]

        data_raw_fname = 'syn_radial_data_cycle%s.mat'%num_cycle        
        # This sampling process takes a bit of time, so we save it once and use it after. 
        if os.path.isfile(data_raw_fname):
            data_raw = sio.loadmat(data_raw_fname)['data_raw']
            print('file loaded: %s' % data_raw_fname)
            
        else: 
            data_raw=np.zeros((Nc,Nfibo*Nfr,Nvec)).astype(np.complex64)
            
            # Generate down-sampled data            

            for idx_fr in range(Nfr): # Fourier transform per each frame
                print('%s/%s'%(idx_fr,Nfr), '\r', end='')
                angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:]
                mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
                tmp=mynufft_test.forward(gt_cartesian_kt_ri[:,idx_fr,:,:,:])
                tmp_c=tmp[...,0]+1j*tmp[...,1]
                tmp_disp=tmp_c.reshape(Nc,Nfibo,Nvec) # (32, 13, 256)

                data_raw[:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:]=tmp_disp

            data_raw=np.transpose(data_raw,(2,1,0))# (256, 299, 32), x-f data
            sio.savemat(data_raw_fname,{'data_raw':data_raw})
            print('file saved: %s' % data_raw_fname)

        
        # Generate down-sampled image
        syn_radial_ri = np.concatenate((np.real(data_raw[...,np.newaxis]),np.imag(data_raw[...,np.newaxis])),axis=3)
        syn_radial_ri = np.transpose(syn_radial_ri,(2,1,0,3)) # (32, 299, 256, 2)
        syn_radial_ri_ts = np_to_torch(syn_radial_ri.astype(np.float32)).cuda().detach() # torch.Size([1, 32, 299, 256, 2]), added batch dimension
        
        # Just for visualization: naive inverse Fourier of undersampled data
        # syn_radial_img
        syn_radial_img_fname = 'syn_radial_img_cycle%s.mat'%num_cycle
        if os.path.isfile(syn_radial_img_fname):            
            syn_radial_img = sio.loadmat(syn_radial_img_fname)['syn_radial_img']
            print('file loaded: %s' % syn_radial_img_fname)
        else: 
            syn_radial_img=np.zeros((img_size,img_size,Nfr)) # (128, 128, 23)
            print('Get images of the synthetic radial (down-sampled) data')
            for idx_fr in range(Nfr):
                print('%s/%s'%(idx_fr,Nfr), '\r', end='')
                angle=set_ang[idx_fr*Nfibo*Nvec:(idx_fr+1)*Nfibo*Nvec,:] # (3328, 2)
                inp= torch_to_np(syn_radial_ri_ts[:,:,idx_fr*Nfibo:(idx_fr+1)*Nfibo,:,:]) # inp: (32, 13, 256, 2), removed batch dimension

                mynufft_test = Mypnufft_cardiac_test(img_size,angle,Nfibo,Nvec,Nc,coil,denc)
                gt_re_np=mynufft_test.backward(inp.reshape((-1,2))) # (128, 128, 2)
                syn_radial_img[:,:,idx_fr]=np.sqrt(gt_re_np[:,:,0]**2+gt_re_np[:,:,1]**2) # (128, 128)
                syn_radial_img[:,:,idx_fr] = syn_radial_img[:,:,idx_fr]-syn_radial_img[:,:,idx_fr].min()
                syn_radial_img[:,:,idx_fr] = syn_radial_img[:,:,idx_fr]/syn_radial_img[:,:,idx_fr].max()

            sio.savemat(syn_radial_img_fname,{'syn_radial_img':syn_radial_img})
            print('file saved: %s' % syn_radial_img_fname)

        self.mynufft = Mypnufft_cardiac(img_size,Nc)
        self.set_ang = set_ang
        self.img_size = img_size
        self.Nc = Nc
        self.Nfr = Nfr
        self.Nvec = Nvec
        self.coil = coil
        self.denc = denc
        self.syn_radial_img = syn_radial_img
        self.syn_radial_ri_ts = syn_radial_ri_ts
        self.gt_cartesian_img = gt_cartesian_img
