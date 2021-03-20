import os
import sys
import json
import h5py
import shutil
import importlib
import torch
from option import get_option
from solver import Solver

import faulthandler; faulthandler.enable()
from IPython.core.debugger import set_trace
def main():
    opt = get_option()

    if torch.cuda.is_available(): 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
        torch.cuda.set_device(opt.gpu_num)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        print("Current device: idx%s | %s" %(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
        
    torch.manual_seed(opt.seed)

    module = importlib.import_module("models.{}".format(opt.model.lower()))

    if opt.isresume is None:
        print(opt.ckpt_root)
        os.makedirs(opt.ckpt_root, exist_ok=True)
        with open(os.path.join(opt.ckpt_root, 'myparam.json'), 'w') as f:
            json.dump(vars(opt), f)
        
        with open(opt.ckpt_root+"/command_line_log.txt", "w") as log_file:
            log_file.write("python %s" % " ".join(sys.argv))

        shutil.copy(os.path.join(os.getcwd(),__file__),opt.ckpt_root)
        shutil.copy(os.path.join(os.getcwd(),'solver.py'),opt.ckpt_root)
        shutil.copy(os.path.join(os.getcwd(),'option.py'),opt.ckpt_root)
    else:
        print('Resumed from ' + opt.isresume)   
        
    solver = Solver(module, opt)
    solver.fit()

if __name__ == "__main__":        
    main()
