# Imports 
import os
import torch
from PIL import Image
from params import par
from utils import util


# 2. Preprocess Images  (i.e., downsample, remove bad/glitchy images, normalize pixel values to remove mean, 
# standard deviation, and move to smaller values to learn)
from torch.utils.data import Dataset

# GPU Usage
print("Is the GPU available?")
print(torch.cuda.is_available())
print("Device Type: " + str(par.device))

# Dataset Handler
class D3VODataset(Dataset):
    def __init__(self, img_dir, img_dirs, transform, batch_size, network_type):
        #TODO: Add directory for left and right images separately.

        # Initialization
        self.img_dir = img_dir # The location of sequence folders
        self.img_dirs = img_dirs # A list of the subfolders leading to the camera images
        self.transform = transform # Transformations for the training network  
        self.batch_size = batch_size # Batch size
        self.network_type = network_type # ['depth','pose']

        # Assuming KITTI Dataset
        self.stereo_img_dirs = [stereo_img.replace(stereo_img[-1],'3') for stereo_img in self.img_dirs]
        #print("Stereo Image Directories: ")
        #print(self.stereo_img_dirs)

        # Network Input
        if network_type == 'pose':
          self.K = 3 # [I_t-1, I_t, I_t+1] (pose network)
        else:
          self.K = 1 # I_t (depth network)
        
    def __len__(self):
        # The number of image stacks in the entire dataset
        num_samples = 0
        for q in range(len(self.img_dirs)):
            folder_dir = os.path.join(self.img_dir, self.img_dirs[q])
            # Ignoring Duplicates (Hard to Delete)
            for filename in os.listdir(folder_dir):
              if '(' not in filename:
                num_samples+=1

        return int(num_samples/self.K)
    
    def __getitem__(self, img_idx, folder, sequence):
        # img_idx is now the batch number
        # sequence is a list of random source images to draw from
        #c,h,w = (3,376,1241)
        folder_dir = os.path.join(self.img_dir, self.img_dirs[folder])
        stereo_dir = os.path.join(self.img_dir, self.stereo_img_dirs[folder])

        #index = sequence.index(img_idx) # img_idx is actually the value

        # Sort List of Images:
        sorted_imgs = util.sort_images(os.listdir(folder_dir))
        image_batch = []
        #pose_batch = []

        # Calculating individual folder size (not entire dataset)
        num_samples = 0
        for filename in os.listdir(folder_dir):
          if '(' not in filename:
            num_samples+=1

        #m = 1
        #MAX_SEQ_LEN = 2 # for random trajectories
        #folder_size = num_samples
        seq_idx = int(par.batch_size*img_idx)

        #print("Batch Indices: ")
        #print(sequence[seq_idx:seq_idx+batch_size])
        for batch_idx in sequence[seq_idx:seq_idx+par.batch_size]:
            # Random trajectory length
            '''if img_idx < (folder_size - MAX_SEQ_LEN):
              m = torch.randint(1, MAX_SEQ_LEN, (1,)).item()
            else:
              m = 1 # If at the end of list just pick the next image'''

            img_stack = []
            if self.network_type == 'pose':
              # Camera Pose Network Input Image Stack
              # batch_idx+1 is actually the center
              stack_files = [sorted_imgs[batch_idx], 
                             sorted_imgs[batch_idx+1], 
                             sorted_imgs[batch_idx+2]]
              
              # Read-in image stack
              for filename in stack_files:
                if filename.endswith(".png"):
                  rgb_img = Image.open(os.path.join(folder_dir,filename))
                  img_stack.append(rgb_img)
            
            else:
              # Depth Network Input Image 'Stack'
              #print("Batch Index: " + str(batch_idx))
              stack_files = [sorted_imgs[batch_idx+1],
                             sorted_imgs[batch_idx+1]] # The stereo image is the same index in other folder
              # batch_idx+1 since the 'current' image is technically batch_idx+1 for camera pose
              # Read-in image stack
              img_stack.append(Image.open(os.path.join(folder_dir,filename)))
              img_stack.append(Image.open(os.path.join(stereo_dir,filename)))
            
            # Read-in the images as PIL
            '''image_stack = []
            for filename in stack_files:
                if filename.endswith(".png"):
                  rgb_img = Image.open(os.path.join(folder_dir,filename))
                  image_stack.append(rgb_img)'''


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

        return image_tensors.float()