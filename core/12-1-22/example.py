import os
import torch
from PIL import Image
from PIL import ImageStat as IStat
import matplotlib.pyplot as plt
from datasets import *
from params import par
from torchvision.transforms import ToPILImage, ToTensor
from loss import TotalLoss
import kornia.geometry as kg
#import numpy as np
from posenet import posenet_model
from depthnet import depthnet_model
from datatracker import datalogger
#from train import *
#from test import *
#import test
#import train
from utils import util
import time
import loss
#from params import par
#from datasets import val_data_pose

def textline_to_filename(root_data_dir, textline):
    
    [filename,file_num,cam] = textline.split()
    
    cam_dir = "image_02\data" if cam == "l" else "image_03\data"
        
    seq_dir = os.path.join(root_data_dir, filename)
    stereo_dir = os.path.join(seq_dir, cam_dir)
        
    img_name = file_num.zfill(10) + ".png"
    full_img_path = os.path.join(stereo_dir, img_name)
    corrected_img_path = full_img_path.replace('\\','/')
    
    return corrected_img_path
        
if __name__ == "__main__":
    
    '''root_data_dir = "D:/OSUGrad/Research/Code/Datasets/KITTI/Raw"
    #root_data_dir = root_data_dir.replace("/","\r")
    train_split_dir = os.path.join('split','train_files.txt')
    test_split_dir = os.path.join('split','val_files.txt')
    
    train_file = open(train_split_dir, 'r')
    val_file = open(test_split_dir, 'r')
    
    train_filenames = train_file.readlines()
    val_filenames = val_file.readlines()
   
    for tfile in train_filenames:
        corrected_img_path = textline_to_filename(root_data_dir, tfile)
        print(corrected_img_path)
        
    print(len(train_filenames))
    last_img = Image.open(corrected_img_path)
    plt.figure()
    plt.imshow(last_img)'''
    
    '''for vfile in val_filenames:
        print("Validation")
        print(vfile)'''
    
    # Testing dataset creation on KITTI data:
    '''index = 23
    print("Number of samples in the (pose) training dataset: " + str(euroc_training_data_pose.__len__()))
    print("Number of samples in the (pose) validation dataset: " + str(euroc_val_data_pose.__len__()))
    print("Number of samples in the (pose) test dataset: " + str(euroc_test_data_pose.__len__()))
    
    print("Number of samples in the (depth) training dataset: " + str(euroc_training_data_depth.__len__()))
    print("Number of samples in the (depth) validation dataset: " + str(euroc_val_data_depth.__len__()))
    print("Number of samples in the (depth) test dataset: " + str(euroc_test_data_depth.__len__()))
    
    # Testing __getitem__()
    train_split_dir = os.path.join(os.path.join('split','EuRoC_MAV'),'train_files.txt')
    test_split_dir = os.path.join(os.path.join('split','EuRoC_MAV'),'val_files.txt')'''
    train_split_dir = os.path.join('split','KITTI','train_files.txt')
    train_file = open(train_split_dir, 'r')
    #val_file = open(test_split_dir, 'r')
    
    train_filenames = train_file.readlines()
    #val_filenames = val_file.readlines()
    
    index = 59
    #example_sequence = torch.randint(0,1000,(batch_size,)).tolist()
    p_images,ppaths,pcams = kitti_training_data_pose.__getitem__(index,train_filenames)
    #d_images,_,__ = euroc_training_data_depth.__getitem__(index,train_filenames)
    
    print("Image (PoseNet) Shape: " + str(p_images.shape))
    
    im1_batch = p_images[:,:3,:,:]
    im2_batch = p_images[:,3:6,:,:]
    im3_batch = p_images[:,6:9,:,:]
    
    print(ppaths)
    print(pcams)
    
    img1 = ToPILImage()(im1_batch[0,:,:,:])
    img2 = ToPILImage()(im2_batch[0,:,:,:])
    img3 = ToPILImage()(im3_batch[0,:,:,:])
    
    img1.show()
    img2.show()
    img3.show()
    
    # Testing Image warping using the NYU Depth Dataset V2
    # Simply testing if the image is warped reasonably given the correct depth information
    # https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2?resource=download
    
    loss_function = TotalLoss().to(par.device)
    
    img1_path = r'misc/00083_colors.png'
    depth_path = r'misc/00083_depth.png'
    source_img = Image.open(img1_path)
    depth_map = Image.open(depth_path)
    (dmin,dmax) = depth_map.getextrema()
    print(depth_map.getextrema())
    K = torch.tensor([[518.85790117450188, 0, 325.58244941119034],
                     [0, 519.46961112127485, 253.73616633400465],
                     [0, 0, 1]])
    K = K.to(par.device)
    zs = torch.unsqueeze(torch.tensor([0.2,0,0]),0)
    ZS = (zs,zs,zs,zs,zs,zs,zs,zs)
    translation_batch = torch.cat(ZS,0).to(par.device)
    rotation_batch = torch.cat(ZS,0).to(par.device)
    
    T = util.euler_to_transformation_matrix(translation_batch,
                                            rotation_batch) # no motion
    base = torch.unsqueeze(torch.tensor([0,0,0,1]).view(1,4),0)
    T_base = torch.cat((base,base,base,base,base,base,base,base),0).to(par.device)
    
    T_batch = T.to(par.device)
    T_batch_base = torch.cat((T_batch,T_base),1).to(par.device)
    img_tensor = torch.unsqueeze(ToTensor()(source_img),0).to(par.device)
    depth_tensor = torch.unsqueeze(ToTensor()(depth_map),0).to(par.device)
    
    img_batch = torch.cat((img_tensor, img_tensor,img_tensor,img_tensor,img_tensor,img_tensor,img_tensor,img_tensor),0).to(par.device)
    depth_batch = torch.cat((depth_tensor, depth_tensor,depth_tensor,depth_tensor,depth_tensor,depth_tensor, depth_tensor,depth_tensor),0).to(par.device)
    
    #time.sleep(25)
    
    #warped_img = loss_function.warp_image(img_batch,depth_batch,K,T_batch)
    K_ = torch.unsqueeze(K,0)
    K_batch = torch.cat((K_,K_,K_,K_,K_,K_,K_,K_),0).to(par.device)
    
    warp = loss.ImageWarp(K_batch,False).to(par.device)
    
    warped_img = warp(img_batch, depth_batch, T_batch_base)
    
    source_img.show()
    dmap = ToPILImage()(ToTensor()(depth_map)/dmax)
    dmap.show()
    
    warped_pil = ToPILImage()(warped_img[0,:,:,:])
    warped_pil.show()
    
    # Testing Model Performance
    '''model_number = 18
    date = '10_18_2022'
    model_root_name = os.path.join('models','d3vo_pose_model_v1_')
    pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
    depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
    
    print("Loading Pose Model Checkpoint ... " + pose_model_checkpoint_path + "\n")
    print("Loading Depth Model Checkpoint ... " + depth_model_checkpoint_path + "\n")
    pose_checkpoint = torch.load(pose_model_checkpoint_path)
    depth_checkpoint = torch.load(depth_model_checkpoint_path)
    
    print("Loading Model State Dictionaries ... ")
    posenet_model.load_state_dict(pose_checkpoint['model_state_dict'])
    depthnet_model.load_state_dict(depth_checkpoint['model_state_dict'], strict=False)
    
    posenet_model.eval()
    depthnet_model.eval()
    
    print("Finished. \n")
    val_data_depth = kitti_val_data_pose
    val_split = par.kitti_val_split
    progress_files = open(val_split,'r')
    progress_list = progress_files.readlines()
    test_img,dtemppaths,dtempcams = val_data_depth.__getitem__(45,progress_list)
    db1, db2, db3, db4 = depthnet_model(test_img[:,:3,:,:])
    sample_depth_map = db1[:,0,:,:]
    sample_stereo_depth_map = db1[:,1,:,:]
    sample_uncer_map = db1[:,2,:,:]
    
    datalogger.write_image(test_img[0,:3,:,:],"test\\1","rgb.png",None,False)
    #datalogger.write_image(test_img[0,2,:,:],"test","uncertainty_map.png")
    datalogger.write_image(db1[0,0,:,:],"test\\1","depth_map.png","magma", True)
    datalogger.write_image(db1[0,1,:,:],"test\\1","stereo_depth_map.png","magma", True)
    datalogger.write_image(db1[0,2,:,:],"test\\1","uncertainty_map.png","viridis", True)
    progress_files.close()'''
    
    # This confirms the warping image function works!
    
    #print("Image (Depth) Shape: " + str(d_images.shape))
    
    #del p_images
    #del d_images
    #train_file.close()
    
    # Calculating mean and standard deviation of training sets.
    
    # KITTI
    # Use __getitem__ for this entire particular dataset, add image recursively, then divide by N
    # same for standard deviation
    # repeat for EuRoC MAV
    '''img_mean = [0]
    img_std = [0]
    lnum=1
    for line in train_filenames:
        #print(line)
        print("Working on image " + str(lnum) + " ...")
        lnum+=1
        [subpath,img_name,cam] = line.split()
        img_path = os.path.join(par.euroc_mav_img_dir, subpath)
        full_img_path = os.path.join(img_path,
                                     "cam0/data/" + img_name + ".png").replace("\\","/")
        im = Image.open(full_img_path)
        image = IStat.Stat(im)
        imean = image.mean
        istd = image.stddev
        #print(imean)
        #print(istd)
        img_mean[0] += imean[0]
        img_std[0] += istd[0]'''
    '''for i in range(3):
            img_mean[i] += imean[i]
            img_std[i] += istd[i]'''
            
            
    '''M = len(train_filenames)
    for i in range(1):
        img_mean[i] /= M
        img_std[i] /= M
    
    print(img_mean)
    print(img_std)'''
    
    '''
    Channel-wise statistics for KITTI:
    Mean:   [90.76236815223285, 95.19403448513077, 91.30475652914224]
            ([0.35593085549895237, 0.3733099391573756, 0.3580578687417343])
    Stddev: [78.27535495750632, 79.47301481441951, 80.30937884350082]
            ([0.30696217630394634, 0.3116588816251746, 0.31493874056274834])
    
    Channel-wise statistics for EuRoC MAV:
    [98.96867976427343] -> [0.3881124696638174]
    [67.83236791117697] -> [0.2660092859261842]
    
    '''
    #print(image.mean)
    #red,green,blue = im.split()
        
    #train_file.close()
    # EuRoC MAV
    
    '''
    
    9/26/22: Test script on KITTI dataset (first 20 epochs), then add EuRoC MAV
             dataset 'mode' and test. If everything works, try training for a 
             full session! Also add spatial transformer since both Zhou and D3VO
             mention it for differentiable bilinear sampling.
             
             In summary:
            (i): Debug for KITTI dataset
            (ii) Add EuRoC MAV 'mode' for epochs 20-60 and debug
            (iii) Add differential bilinear sampling from the spatial transformer network
                  like Zhou, D3VO
    
    '''
    
    '''
    9/26/22
    
    Finished Implementation and Ready for Debug:
     (i) clamp uncertainty map to 1e-5 to 1 or so [X]
     (ii) Send depth and uncertainty maps to tensor board [X]
     
    '''
    
    