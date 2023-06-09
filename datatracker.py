import yaml
import os
import torch
from torchvision.transforms import ToPILImage
from PIL import Image
import matplotlib.pyplot as plt

# Data Tracker

'''
    A Simple Class to track quantities during training.
    
    Advantages:
        1. Better customization than throwing data at tensorboard.
        2. Can choose how you want to organize the data you're collecting.
        3. No browser or API needed.
        4. Spyder might fail to track some quantities 
           (e.g., fails to track ToLoss() object. Maybe because it's not on the GPU?)
        
'''

class DataTracker():
    def __init__(self):
        
        self.main_log_dir = "log"
        self.file_ext = ".yaml"
        
        # ------------------------ main.py --------------------------------- #
        
        self.main_filename = os.path.join(self.main_log_dir, 
                                          "main", self.file_ext)
        self.main_dict = {}
        
        # ------------------------ train.py -------------------------------- #
        
        self.train_filename = os.path.join(self.main_log_dir, 
                                           "train", self.file_ext)
        self.train_dict = {}
        
        # ------------------------ test.py --------------------------------- #
        
        self.test_filename = os.path.join(self.main_log_dir, 
                                          "test", self.file_ext)
        self.test_dict = {}
        
        # ------------------------ loss.py --------------------------------- #
        
        self.loss_filename = os.path.join(self.main_log_dir, 
                                          "loss", self.file_ext)
        self.loss_dict = {}
        
        # ------------------------ utils.py -------------------------------- #
        
        self.utils_filename = os.path.join(self.main_log_dir, 
                                           "utils", self.file_ext)
        self.utils_dict = {}
        
        # ------------------------ params.py ------------------------------- #
        
        self.params_filename = os.path.join(self.main_log_dir, 
                                            "params", self.file_ext)
        self.params_dict = {}
        
        # ------------------------ dataset.py ------------------------------ #
        
        self.dataset_filename = os.path.join(self.main_log_dir, 
                                             "dataset", self.file_ext)
        self.dataset_dict = {}
        
        # ------------------------ dataloader.py --------------------------- #
        
        self.dataloader_filename = os.path.join(self.main_log_dir, 
                                                "dataloader", self.file_ext)
        self.dataloader_dict = {}
        
        # ------------------------ posenet.py ------------------------------ #
        
        self.posenet_filename = os.path.join(self.main_log_dir, 
                                             "posenet", self.file_ext)
        self.posenet_dict = {}
        
        # ------------------------ depthnet.py ----------------------------- #
        
        self.depthnet_filename = os.path.join(self.main_log_dir, 
                                              "depthnet", self.file_ext)
        self.depthnet_dict = {}
    
    
    def add_param(self, param_object, key, value):
        
        param_object[key] = value
        
    def write_data(self, list_of_params, param_names):
        
        for i in range(len(list_of_params)):
            stream = open(param_names[i],'w')
            yaml.dump(list_of_params[i], stream)
            stream.close()
        
    def load_data(self, param_objects, param_names):
        
        for i in range(len(param_objects)):
            param_objects[i] = yaml.safe_load(param_names[i])
        
    def write_image(self, img, epoch, img_name, ctype, colorbar):
        
        # Tensor to PIL image for saving
        pil_img = ToPILImage()(img)
        if not os.path.isdir(os.path.join(self.main_log_dir,"imgs",epoch)):
            os.mkdir(os.path.join(self.main_log_dir,"imgs",epoch))
        
        
        if colorbar:
            
            plt_img = plt.imshow(pil_img, cmap=ctype)
            cb = plt.colorbar(plt_img, location='bottom')
            plt.savefig(os.path.join(self.main_log_dir,
                                     "imgs",
                                     epoch,
                                     img_name))
            cb.remove()
            #plt.cla()
        else:
            plt_img = plt.imshow(pil_img, cmap=ctype)
            plt.savefig(os.path.join(self.main_log_dir,
                                     "imgs",
                                     epoch,
                                     img_name))
        
        
datalogger = DataTracker()


class Hook():
    def __init__(self, module):
        # Gradient hook for module backward pass
        #self.names = ""
        #self.input = 0
        #self.output = 0
        self.hook = module.register_backward_hook(self.hook_fn)
        
    def hook_fn(self, module, input_, output_):
        self.names = module.named_modules()
        self.input = input_
        self.output = output_
        
    def close_hook(self):
        self.hook.remove()
        
#hook = Hook()