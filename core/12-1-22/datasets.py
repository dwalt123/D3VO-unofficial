import torchvision
import torch
from torch.utils.data import Dataset
from params import par
from dataloader import KITTIDataset,EuRoCMAVDataset

#print(os.getcwd())
#os.chdir("/content/drive/MyDrive/Colab Notebooks/")

# Strategy: Define a Dataset for KITTI and EuRoC MAV separately, and switch datasets
#           after X amount of epochs

'''class KITTIDataset(Dataset):
    def __init__(self, img_dir, split_file, transform, batch_size, network_type):
        self.img_dir = ""

class EuRoCMAVDataset(Dataset):
    def __init__(self, img_dir, split_file, transform, batch_size, network_type):
        self.img_dir = ""'''
        
# Resize
torch_resize = par.torch_resize
torch_totensor = par.torch_totensor
#torch_normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])
# Removed the normalization on both datasets because they were giving significant artefacts
torch_normalize_kitti = par.torch_normalize_kitti
torch_normalize_euroc = par.torch_normalize_euroc

# ------------------------------ KITTI DataLoaders -------------------------- #

#transform_kitti = torchvision.transforms.Compose([torch_resize, torch_totensor, torch_normalize_kitti])
transform_kitti = torchvision.transforms.Compose([torch_resize, torch_totensor])

# Dataset for Camera Pose Network
network_type = 'pose'
kitti_training_data_pose = KITTIDataset(img_dir=par.kitti_img_dir, 
                                  split_file=par.kitti_train_split, 
                                  transform=transform_kitti, 
                                  batch_size=par.batch_size,
                                  network_type=network_type)

kitti_val_data_pose = KITTIDataset(img_dir=par.kitti_img_dir, 
                                   split_file=par.kitti_val_split, 
                                   transform=transform_kitti, 
                                   batch_size=par.batch_size,
                                   network_type=network_type)

# The test datasets are currently validation splits, will generate test sequences later!
kitti_test_data_pose = KITTIDataset(img_dir=par.kitti_img_dir, 
                                    split_file=par.kitti_val_split, 
                                    transform=transform_kitti, 
                                    batch_size=1,
                                    network_type=network_type)

# Dataset for Depth Network
network_type = 'depth'
kitti_training_data_depth = KITTIDataset(img_dir=par.kitti_img_dir, 
                                         split_file=par.kitti_train_split, 
                                         transform=transform_kitti, 
                                         batch_size=par.batch_size,
                                         network_type=network_type)

kitti_val_data_depth = KITTIDataset(img_dir=par.kitti_img_dir, 
                                    split_file=par.kitti_val_split, 
                                    transform=transform_kitti, 
                                    batch_size=par.batch_size,
                                    network_type=network_type)

# Similar to pose dataset, the depth test data is currently the validation split
# It will be changed after training. (eg., Sequence 9, Sequence 10, ...)
kitti_test_data_depth = KITTIDataset(img_dir=par.kitti_img_dir, 
                                     split_file=par.kitti_val_split, 
                                     transform=transform_kitti, 
                                     batch_size=1,
                                     network_type=network_type)

# ---------------------------- EuRoC MAV DataLoaders ------------------------ #

#transform_euroc = torchvision.transforms.Compose([torch_resize, torch_totensor, torch_normalize_euroc])
transform_euroc = torchvision.transforms.Compose([torch_resize, torch_totensor])

# Dataset for Camera Pose Network
network_type = 'pose'
euroc_training_data_pose = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                           split_file=par.euroc_train_split, 
                                           transform=transform_euroc, 
                                           batch_size=par.batch_size,
                                           network_type=network_type)

euroc_val_data_pose = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                      split_file=par.euroc_val_split, 
                                      transform=transform_euroc, 
                                      batch_size=par.batch_size,
                                      network_type=network_type)

# The test datasets are currently validation splits, will generate test sequences later!
euroc_test_data_pose = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                       split_file=par.euroc_val_split, 
                                       transform=transform_euroc, 
                                       batch_size=1,
                                       network_type=network_type)

# Dataset for Depth Network
network_type = 'depth'
euroc_training_data_depth = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                            split_file=par.euroc_train_split, 
                                            transform=transform_euroc, 
                                            batch_size=par.batch_size,
                                            network_type=network_type)

euroc_val_data_depth = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                       split_file=par.euroc_val_split, 
                                       transform=transform_euroc, 
                                       batch_size=par.batch_size,
                                       network_type=network_type)

# Similar to pose dataset, the depth test data is currently the validation split
# It will be changed after training. (eg., Sequence 9, Sequence 10, ...)
euroc_test_data_depth = EuRoCMAVDataset(img_dir=par.euroc_mav_img_dir, 
                                     split_file=par.euroc_val_split, 
                                     transform=transform_euroc, 
                                     batch_size=1,
                                     network_type=network_type)

'''
index = 0
print("Number of samples in the (pose) training dataset: " + str(training_data_pose.__len__()))
print("Number of samples in the (pose) validation dataset: " + str(val_data_pose.__len__()))
print("Number of samples in the (pose) test dataset: " + str(test_data_pose.__len__()))

print("Number of samples in the (depth) training dataset: " + str(training_data_depth.__len__()))
print("Number of samples in the (depth) validation dataset: " + str(val_data_depth.__len__()))
print("Number of samples in the (depth) test dataset: " + str(test_data_depth.__len__()))'''

'''example_sequence = torch.randint(0,1000,(batch_size,)).tolist()
#p_images = training_data_pose.__getitem__(index,1,example_sequence)
d_images = training_data_depth.__getitem__(index,1,example_sequence)

#print("Image (Pose) Shape: " + str(p_images.shape))
print("Image (Depth) Shape: " + str(d_images.shape))

#del p_images
del d_images
'''