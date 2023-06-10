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

from utils.common_utils import *
from Mypnufft_mc_func_cardiac import *

import h5py
import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.core.debugger import set_trace

from mask_generator import generate_mask
import mat73

class Solver():
    def __init__(self, module, opt):        
        self.opt = opt
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
             
        self.prepare_dataset()        
        
        self.step = 0
        self.t1, self.t2 = None, None
        self.best_snr, self.best_snr_step = 0, 0
        
        self.net = module.Net(opt).to(self.dev)
        self.net_input_set = get_input_manifold(opt.input_type, opt.num_cycle, self.Nfr).to(self.dev)
        
        p = get_params(opt.opt_over, self.net, self.net_input_set)
        # Compute number of parameters          
        s  = sum([np.prod(list(pnb.size())) for pnb in p]);
        print ('# params: %d' % s)
        
        if opt.input_type.endswith('mapping'):
            self.mapnet = module.MappingNet(opt).to(self.dev)
            print('... adding params of mapping network ...')
            p += self.mapnet.parameters()            
            # Compute number of parameters          
            s  = sum([np.prod(list(pnb.size())) for pnb in p]);
            print ('# params: %d' % s)

        self.optimizer = torch.optim.Adam(p, lr=opt.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                    step_size=opt.step_size, 
                                                    gamma=opt.gamma)
        self.loss_fn = nn.MSELoss()        
        if opt.isresume is not None:
            self.load(opt.isresume)

    def fit(self):
        opt = self.opt
        self.writer = SummaryWriter(opt.ckpt_root)   
        self.t1 = time.time()
        
        batch_size = opt.batch_size   
        Nfr = self.Nfr
        net_input_set = self.net_input_set
        uds_cartesian_kt = self.uds_cartesian_kt    
        uds_cartesian_kt = torch.stack([torch.real(uds_cartesian_kt), torch.imag(uds_cartesian_kt)], dim=-1) # (batch_size, 128, 128, 2)
        step = self.step       

        while step < opt.max_steps:
            # randomly pick frames to train (batch, default = 1)
            idx_fr=np.random.randint(0, Nfr)
            idx_frs = range(min(idx_fr, Nfr-batch_size), min(idx_fr+batch_size, Nfr))

            net_input_z = self.net_input_set[idx_frs,:].reshape((batch_size,-1))
            net_input_w = self.mapnet(net_input_z)
            net_input_w = net_input_w.reshape((batch_size,1,opt.style_size,opt.style_size)) 
            out_sp = self.net(net_input_w) # e.g., spatial domain output (img) torch.Size([batch_size, 2, 128, 128])
            
            out_sp = torch.complex(out_sp[:,0,...],out_sp[:,1,...]) # (batch_size, 128, 128)
            out_cartisian_kt = torch_fft2c(out_sp) # (batch_size, 128, 128)
            
            uds_out_cartisian_kt = self.mask[idx_frs,...] * out_cartisian_kt
            
            uds_out_cartisian_kt = torch.stack([torch.real(uds_out_cartisian_kt), torch.imag(uds_out_cartisian_kt)], dim=-1) # (batch_size, 128, 128, 2)

            total_loss = self.loss_fn(uds_out_cartisian_kt, uds_cartesian_kt[idx_frs,...])

#             total_loss *= (self.img_size)**2       
            self.optimizer.zero_grad()
            total_loss.backward()   
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar('total loss/training_loss (mse)', total_loss, step)
            
            if (not self.opt.noPLOT) and (step % opt.save_period == 0 or step == opt.max_steps-1):                
                self.evaluate()

            step += 1
            self.step = step
        
        self.writer.close()   
        
    @torch.no_grad()
    def evaluate(self):  
        self.issave = False
        step = self.step
        max_steps = self.opt.max_steps
        # For visualizations
        # Get Average PSNR and SSIM values for entire frames
        net_input_w_set = self.mapnet(self.net_input_set).reshape((self.Nfr,1,self.opt.style_size,self.opt.style_size)) 
        out = self.net(net_input_w_set).detach().cpu().numpy()
        out= np.sqrt(out[:,0,:,:]**2+out[:,1,:,:]**2)
        # out = torch.complex(out[:,0,...],out[:,1,...]).detach().cpu().numpy()
        snr = calc_SNR(out, self.gt_cartesian_img.detach().cpu().numpy())

        psnr_val_list = []
        ssim_val_list = []
        for idx_fr in range(self.Nfr):
            tmp_ims = self.net(net_input_w_set[idx_fr:idx_fr+1,...])
            tmp_ims = torch_to_np(tmp_ims).astype('float32') 
            tmp_ims= np.sqrt(tmp_ims[0,:,:]**2+tmp_ims[1,:,:]**2)
            tmp_ims -= tmp_ims.min()
            tmp_ims /= tmp_ims.max()
            gt_cartesian_img = np.abs(self.gt_cartesian_img[idx_fr,:,:].detach().cpu().numpy())            
            psnr_val_list += [psnr(gt_cartesian_img, tmp_ims)]
            ssim_val_list += [ssim(gt_cartesian_img, tmp_ims)]  
        
        psnr_val = np.array(psnr_val_list).sum()/self.Nfr
        ssim_val = np.array(ssim_val_list).sum()/self.Nfr 
        
        if self.opt.istest:
            print("Saving h5/video (SNR {:.2f} @ {} step)".format(snr, step))
            # self.save_video(inp, ims, psnr_val_list, ssim_val_list)
        else:
            self.writer.add_scalar('metrics/snr', snr, step)
            self.writer.add_image('recon_image', out[0:1,...], step)

            self.t2 = time.time()

            if snr >= self.best_snr:
                self.best_snr, self.best_snr_step = snr, step
                self.issave = True

            if self.issave:    
                self.save(step)
                self.issave = False

            curr_lr = self.scheduler.get_lr()[0]
            eta = (self.t2-self.t1) * (max_steps-step) /self.opt.save_period / 3600
            print("[{}/{}] {:.2f} {:.2f} {:.2f} (Best SNR: {:.2f} @ {} step) LR: {}, ETA: {:.1f} hours"
                .format(step, max_steps, snr, psnr_val, ssim_val, self.best_snr, self.best_snr_step,
                 curr_lr, eta))

            self.t1 = time.time()
        
        if step == max_steps-1:            
            self.save(step)
            # self.save_video(inp, ims, snr_val_list, ssim_val_list)

    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.net_input_set = checkpoint['net_input_set'].to(self.dev)
        self.net.load_state_dict(checkpoint['net_state_dict']) 
        self.net.to(self.dev)
        if self.opt.input_type.endswith('mapping'):
            self.mapnet.load_state_dict(checkpoint['mapnet_state_dict'])
            self.mapnet.to(self.dev)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = checkpoint['scheduler']
        self.step = checkpoint['step']
        self.best_psnr, self.best_psnr_step = checkpoint['best_psnr'], checkpoint['best_psnr_step']
        self.best_ssim, self.best_ssim_step = checkpoint['best_ssim'], checkpoint['best_ssim_step']
        if not self.opt.istest:
            self.step = checkpoint['step']+1        
        
    def save(self, step):        
        print('saving ... ')
        save_path = os.path.join(self.opt.ckpt_root, str(step)+".pt")
        ckptdict = {
                'step': step,
                'net_input_set': self.net_input_set,
                'net_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                }
        best_scores = {
                'best_snr': self.best_snr,
                'best_snr_step': self.best_snr_step,
                }
        ckptdict = {**ckptdict, **best_scores}
        
        if self.opt.input_type.endswith('mapping'):            
            ckptdict['mapnet_state_dict'] = self.mapnet.state_dict()
                
        torch.save(ckptdict, save_path)
        with open(os.path.join(self.opt.ckpt_root, 'best_scores.json'), 'w') as f:
            json.dump(best_scores, f)

    def save_video(self, inp, ims, psnr_val_list, ssim_val_list):      
        f = h5py.File(os.path.join(self.opt.ckpt_root, 'final_{}.h5'.format(self.step)),'w')
        f.create_dataset('input_w',data=inp,dtype=np.float32)
        f.create_dataset('data',data=ims,dtype=np.float32)
        f.create_dataset('psnr_val_list',data=psnr_val_list,dtype=np.float32)
        f.create_dataset('ssim_val_list',data=ssim_val_list,dtype=np.float32)
        f.close()
        print('h5 file saved.')
        
        print('creating video, (vmax=0.5).')
        fig = plt.figure(figsize=(10, 10))
        vid = []
        for idx_fr in range(self.Nfr):
            tmp_ims = ims[idx_fr].squeeze()    
            ttl = plt.text(128, -5, idx_fr, horizontalalignment='center', fontsize = 20)
            vid.append([plt.imshow(tmp_ims, animated=True, cmap = 'gray', vmax=0.5),ttl])
        ani = animation.ArtistAnimation(fig, vid, interval=50, blit=True, repeat_delay=1000)        
        ani.save(self.opt.ckpt_root+'/final_video_{}_{}.mp4'.format(os.path.basename(self.opt.ckpt_root),self.step))
        print('video saved')
    
    def prepare_dataset(self):           
        fname = self.opt.fname
        num_cycle = self.opt.num_cycle
        Nfibo = self.opt.Nfibo
        
        gt_cartesian_img = np.abs(np.squeeze(mat73.loadmat(fname)['label'])[32:-33, 8:-8, :].astype(np.complex64)) # (128, 128, 22), complex64, kt-space data
        gt_cartesian_kt = fft(gt_cartesian_img, ax=(0,1)) # (128, 128, 22), complex64, kt-space data
        
        Nfr = np.shape(gt_cartesian_img)[2]*num_cycle # 22 number of frames * num_cycle (1)
        nx, ny=np.shape(gt_cartesian_img)[0:2] # 128        

        gt_cartesian_kt = np.concatenate([gt_cartesian_kt]*num_cycle,axis=2) # (128, 128, 22*num_cycle), complex64, kt-space data 
        gt_cartesian_img = np.concatenate([gt_cartesian_img]*num_cycle,axis=2) # (128, 128, 22*num_cycle), complex64, kt-space data 
        
        mask = generate_mask(gt_cartesian_kt.shape, 13, 'radial').astype(np.complex64) # (128, 128, 22*num_cycle)
        uds_cartesian_kt = mask * gt_cartesian_kt
        uds_cartesian_img = ifft(uds_cartesian_kt, ax=(0,1)) # (128, 128, 22*num_cycle)
        
        mask = np.transpose(mask, (2,0,1)) # (22, 128, 128)
        gt_cartesian_img = np.transpose(gt_cartesian_img, (2,0,1)) # (22, 128, 128)
        uds_cartesian_img = np.transpose(uds_cartesian_img, (2,0,1)) # (22, 128, 128)
        uds_cartesian_kt = np.transpose(uds_cartesian_kt, (2,0,1)) # (22, 128, 128)

        self.img_size = gt_cartesian_img.shape
        self.Nfr = Nfr
        self.mask = torch.from_numpy(mask).cuda()
        self.gt_cartesian_img = torch.from_numpy(gt_cartesian_img).cuda()
        self.uds_cartesian_img = torch.from_numpy(uds_cartesian_img).cuda()
        self.uds_cartesian_kt = torch.from_numpy(uds_cartesian_kt).cuda()