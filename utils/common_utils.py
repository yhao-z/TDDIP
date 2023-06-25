import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
from math import pi
import torchvision.utils as vutils


fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax).astype(dtype=np.complex64) 
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax).astype(dtype=np.complex64)  

# specific fft for our defined dynamic MR data
fft2c_mri  = lambda x, ax=(-2,-1) : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax).astype(dtype=np.complex64)  
ifft2c_mri = lambda X, ax=(-2,-1) : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax).astype(dtype=np.complex64)  


def torch_fft2c(x):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=(-2,-1)), dim=(-2,-1), norm='ortho'), dim=(-2,-1))


def torch_ifft2c(X):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(X, dim=(-2,-1)), dim=(-2,-1), norm='ortho'), dim=(-2,-1))


def calc_SNR(y, y_):
    y = np.array(y).flatten()
    y_ = np.array(y_).flatten()
    err = np.linalg.norm(y_ - y) ** 2
    snr = 10 * np.log10(np.linalg.norm(y_) ** 2 / err)

    return snr


def save_image_3d(tensor, slice_idx, file_name):
    '''
    tensor: [bs, c, h, w, 1]
    '''
    image_num = len(slice_idx)
    tensor = tensor[0, slice_idx, ...].permute(0, 3, 1, 2).cpu().data  # [c, 1, h, w]
    image_grid = vutils.make_grid(tensor, nrow=image_num, padding=0, normalize=True, scale_each=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif opt == 'all':
            net_input.requires_grad = True
            params += [net_input]
            params += [x for x in net.parameters() ]
        else:
            assert False, 'what is it?'
            
    return params

def get_input_manifold(input_type, num_cycle, num_fr):
    if input_type.startswith('helix'):
        x = np.cos(np.linspace(0,num_cycle*2*pi,num_fr))
        y = np.sin(np.linspace(0,num_cycle*2*pi,num_fr))
        z = np.linspace(0,1,num_fr)
        manifold = torch.tensor([x,y,z]).float() # (3, num_fr)
        manifold = manifold.permute(1,0) # (num_fr, 3) 
    else:
        raise NotImplementedError("No such input_type.")
        
    return manifold

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

