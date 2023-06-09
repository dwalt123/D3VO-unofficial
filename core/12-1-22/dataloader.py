# Imports 
import os
import torch
from PIL import Image
from params import par
from utils import util
from torchvision.transforms import ToPILImage

# 2. Preprocess Images  (i.e., downsample, remove bad/glitchy images, normalize pixel values to remove mean, 
# standard deviation, and move to smaller values to learn)
from torch.utils.data import Dataset

'''
Back up is in dataloader_backup.py
'''

# GPU Usage
print("Is the GPU available?")
print(torch.cuda.is_available())
print("Device Type: " + str(par.device))

# KITTI Dataset Handler
class KITTIDataset(Dataset):
    def __init__(self, img_dir, split_file, transform, batch_size, network_type):
        #TODO: Add directory for left and right images separately.

        # Initialization
        self.img_dir = img_dir # Now the root directory to the dates
        #self.img_dirs = img_dirs # Now the sequence dates
        self.transform = transform # Transformations for the training network  
        self.batch_size = batch_size # Batch size
        self.network_type = network_type # ['depth','pose']
        
        self.left_img_dir = os.path.join("image_02","data")
        self.right_img_dir = os.path.join("image_03", "data")
        
        self.split_name = split_file

        # Network Input
        if network_type == 'pose':
          self.K = 3 # [I_t-1, I_t, I_t+1] (pose network)
        else:
          self.K = 1 # I_t (depth network)
        
    def __len__(self):
        # The number of image stacks in the entire dataset
        split_file = open(self.split_name,'r')
        file_len = len(split_file.readlines())
        split_file.close()
        return int(file_len/self.K)
    
    def __getitem__(self, img_idx, filenames):
        # img_idx: the batch index 
        # filenames: list of filenames obtained from open(...,'r')
        #            Assuming filenames have been shuffled!
        
        image_batch = []
        
        seq_idx = int(par.batch_size*img_idx)
        sequence = filenames[seq_idx:seq_idx+par.batch_size]
        # Get sequence of image indices
        nums = []
        paths = []
        cams = []
        for i in range(par.batch_size):
            [subpath,num,cam] = sequence[i].split()
            nums.append(int(num)) # frame number
            paths.append(subpath)  # path to that frame
            cams.append(cam)   # left or right stereo camera
        
        
        for j in range(par.batch_size):
            
            # path to current image sample in batch
            temp_path = os.path.join(os.path.join(self.img_dir, 
                                                  paths[j]),
                                     self.left_img_dir).replace('\\','/') 
            unordered_imgs = os.listdir(temp_path)
            # not unix images but processed in a similar way
            sorted_nums = util.sort_unix_images(unordered_imgs, False, 'kitti')
            
            img_stack = []
            if self.network_type == 'pose':
              # Camera Pose Network Input Image Stack
              # batch_idx+1 is actually the center
              frames = [sorted_nums[j], sorted_nums[j+1], sorted_nums[j+2]]
              root_path = os.path.join(os.path.join(self.img_dir, paths[j]),
                                       self.left_img_dir) 
              stack_files = [str(frames[0]).zfill(10)+".png", 
                             str(frames[1]).zfill(10)+".png", 
                             str(frames[2]).zfill(10)+".png"]
              
              # Read-in image stack
              for filename in stack_files:
                if filename.endswith(".png"):
                  rgb_img = Image.open(os.path.join(root_path,filename).replace('\\','/'))
                  #print(os.path.join(root_path,filename).replace('\\','/'))
                  img_stack.append(rgb_img)
            
            else:
              # Depth Network Input Image 'Stack'
              #print("Batch Index: " + str(batch_idx))
              frames = [sorted_nums[j+1], sorted_nums[j+1]]
              root_path = os.path.join(self.img_dir, paths[j])
              stack_files = [str(frames[0]).zfill(10)+".png", 
                             str(frames[1]).zfill(10)+".png"] # The stereo image is the same index in other folder
              # batch_idx+1 since the 'current' image is technically batch_idx+1 for camera pose
              # Read-in image stack
              img_stack.append(Image.open(os.path.join(os.path.join(root_path,self.left_img_dir),
                                                       stack_files[0]).replace('\\','/')))
              #print(os.path.join(os.path.join(root_path,self.left_img_dir), stack_files[0]).replace('\\','/'))
              img_stack.append(Image.open(os.path.join(os.path.join(root_path,self.right_img_dir),
                                                       stack_files[1]).replace('\\','/')))
              #print(os.path.join(os.path.join(root_path,self.right_img_dir), stack_files[1]).replace('\\','/'))
            
            # Transform Images
            if self.transform:
                stack = []
                for img in img_stack:
                    transformed_img = self.transform(img)
                    stack.append(transformed_img) # Modified 7/20/22 to make [-0.5,0.5]
                    
            '''if j == 0:
                i1 = ToPILImage()(stack[0])
                i1.show()
                i2 = ToPILImage()(stack[1])
                i2.show()
                i3 = ToPILImage()(stack[2])
                i3.show()'''
                
            stack_ = stack
            image = torch.cat(stack_, 0).unsqueeze(0).to(par.device) # Tensor List -> Tensor

            # Append to Image Batch List
            image_batch.append(image)
            
        image_tensors = torch.cat(image_batch, 0).to(par.device)

        return image_tensors.float(), paths, nums


# ----------------------------------------------------------------------------#

# EuRoC MAV Dataset Handler
class EuRoCMAVDataset(Dataset):
    def __init__(self, img_dir, split_file, transform, batch_size, network_type):
        #TODO: Add directory for left and right images separately.

        # Initialization
        self.img_dir = img_dir # Now the root directory to the rooms
        #self.img_dirs = img_dirs # Now the sequence dates
        self.transform = transform # Transformations for the training network  
        self.batch_size = batch_size # Batch size
        self.network_type = network_type # ['depth','pose']
        
        self.left_img_dir = os.path.join("cam0","data")
        self.right_img_dir = os.path.join("cam1", "data")
        
        self.split_name = split_file

        # Network Input
        if network_type == 'pose':
          self.K = 3 # [I_t-1, I_t, I_t+1] (pose network)
        else:
          self.K = 1 # I_t (depth network)
        
    def __len__(self):
        # The number of image stacks in the entire dataset
        split_file = open(self.split_name,'r')
        file_len = len(split_file.readlines())
        split_file.close()
        return int(file_len/self.K)
    
    def __getitem__(self, img_idx, filenames):
        # img_idx: the batch index 
        # filenames: list of filenames obtained from open(...,'r')
        #            Assuming filenames have been shuffled!
        
        image_batch = []
        
        seq_idx = int(par.batch_size*img_idx)
        sequence = filenames[seq_idx:seq_idx+par.batch_size]
        # Get sequence of image indices
        nums = []
        paths = []
        cams = []
        for i in range(par.batch_size):
            [subpath,num,cam] = sequence[i].split()
            nums.append(int(num)) # frame number
            paths.append(subpath)  # path to that frame
            cams.append(cam)   # left or right stereo camera
        
        # Image frames aren't in 0..N, they're unix time steps so you need to sort
        #sorted_nums = util.sort_unix_images(nums, False)
        
        # To ensure that you're organizing images from the same folders, you need to
        # get the list from paths[j], sort it and then proceed for every image in the batch.
        for j in range(par.batch_size):
            
            # path to current image sample in batch
            temp_path = os.path.join(os.path.join(self.img_dir, paths[j]),
                                     self.left_img_dir).replace('\\','/') 
            unordered_imgs = os.listdir(temp_path)
            sorted_nums = util.sort_unix_images(unordered_imgs, False, 'kitti')
            
            img_stack = []
            if self.network_type == 'pose':
              # Camera Pose Network Input Image Stack
              # batch_idx+1 is actually the center
              frames = [sorted_nums[j], sorted_nums[j+1], sorted_nums[j+2]]
              root_path = os.path.join(os.path.join(self.img_dir, paths[j]),
                                       self.left_img_dir)
              stack_files = [str(frames[0]) + ".png", 
                             str(frames[1]) + ".png", 
                             str(frames[2]) + ".png"]
              
              # Read-in image stack
              for filename in stack_files:
                if filename.endswith(".png"):
                  pil_img = Image.open(os.path.join(root_path,filename).replace('\\','/'))
                  rgb_img = Image.merge('RGB',[pil_img,pil_img,pil_img])
                  #rgb_img = Image.open(os.path.join(root_path,filename).replace('\\','/'))
                  #print(os.path.join(root_path,filename).replace('\\','/'))
                  img_stack.append(rgb_img)
            
            else:
              # Depth Network Input Image 'Stack'
              #print("Batch Index: " + str(batch_idx))
              frames = [sorted_nums[j+1], sorted_nums[j+1]]
              root_path = os.path.join(self.img_dir, paths[j])
              stack_files = [str(frames[0]) + ".png", 
                             str(frames[1]) + ".png"] # The stereo image is the same index in other folder
              # batch_idx+1 since the 'current' image is technically batch_idx+1 for camera pose
              # Read-in image stack
              pil_img_left = Image.open(os.path.join(os.path.join(root_path,self.left_img_dir),
                                                       stack_files[0]).replace('\\','/'))
              rgb_img_left = Image.merge('RGB',[pil_img_left,pil_img_left,pil_img_left])
              img_stack.append(rgb_img_left)
              
              pil_img_right = Image.open(os.path.join(os.path.join(root_path,self.right_img_dir),
                                                       stack_files[1]).replace('\\','/'))
              rgb_img_right = Image.merge('RGB',[pil_img_right,pil_img_right,pil_img_right])
              #print(os.path.join(os.path.join(root_path,self.left_img_dir), stack_files[0]).replace('\\','/'))
              img_stack.append(rgb_img_right)
              #print(os.path.join(os.path.join(root_path,self.right_img_dir), stack_files[1]).replace('\\','/'))
            # Transform Images
            if self.transform:
                stack = []
                for img in img_stack:
                    transformed_img = self.transform(img)
                    stack.append(transformed_img) # Modified 7/20/22 to make [-0.5,0.5]

            stack_ = stack
            image = torch.cat(stack_, 0).unsqueeze(0).to(par.device) # Tensor List -> Tensor

            # Append to Image Batch List
            image_batch.append(image)
            
        image_tensors = torch.cat(image_batch, 0).to(par.device)

        return image_tensors.float(), paths, nums