import time
import os
from params import par
#from datasets import val_data_pose, val_data_depth
import torch
from utils import util
import wandb
import cv2

# Visualization on GPU ...
#from vispy.util.transforms import ortho
#from vispy import gloo
#from vispy import app

from PIL import Image
#from params import par
#from utils import util
from torchvision.transforms import ToPILImage

import torchvision
#import torch
from torch.utils.data import Dataset
from params import par
from datasets import transform_kitti_test, transform_euroc_test
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from posenet import posenet_model
from mono_depthnet import depthnet_model

# Read-in folder sequence to evaluate, order the filenames ...
root_split_path = r'C:/Users/dhwal/Documents/OSU/Graduate/Research/Code/D3VO/split/KITTI/test'
root_seq_path = r'F:/OSUGrad/Research/Code/Datasets/KITTI/Visual Odometry/data_odometry_color/dataset/sequences'

seq_name = '01' # sequence name to test
#seq_full_path = os.path.join(root_seq_path,seq_name,'image_2').replace('\\','/')
seq_full_path = "D:/OSUGrad/Research/Code/Datasets/KITTI/Raw/2011_09_26/2011_09_26_drive_0029_sync/image_02/data/"

# ATE, RPE, Translation Error, Rotation Error metric functions ...

# Video and Plot Writer ...

# Run Model ...

# VisPy Setup

# https://vispy.org/gallery/gloo/animate_images.html

def read_test_input(root_img_dir, seq_name, img_idx, transform, network_type):
    # img_idx: the batch index 
    # filenames: list of filenames obtained from open(...,'r')
    #            Assuming filenames have been shuffled!
    
    #image_batch = []
    
    left_img_dir = "/image_2/"
    right_img_dir = "/image_3/"
    
    # path to current image sample in batch
    temp_path = root_img_dir + seq_name + left_img_dir
    unordered_imgs = os.listdir(temp_path)
    # not unix images but processed in a similar way
    sorted_nums = util.sort_unix_images(unordered_imgs, False, 'kitti')
    
    img_stack = []
    if network_type == 'pose':
      # Camera Pose Network Input Image Stack
      # batch_idx+1 is actually the center
      frames = [sorted_nums[img_idx], sorted_nums[img_idx+1]]
      root_path = root_img_dir + seq_name + left_img_dir
      stack_files = [str(frames[0]).zfill(6)+".png", 
                     str(frames[1]).zfill(6)+".png"]
      
      # Read-in image stack
      for filename in stack_files:
        if filename.endswith(".png"):
          rgb_img = Image.open(root_path + filename)
          #print(os.path.join(root_path,filename).replace('\\','/'))
          img_stack.append(rgb_img)
    
    else:
      # Depth Network Input Image 'Stack'
      #print("Batch Index: " + str(batch_idx))
      frames = [sorted_nums[img_idx], sorted_nums[img_idx]]
      root_path = root_img_dir + seq_name
      stack_files = [str(frames[0]).zfill(6)+".png", 
                     str(frames[1]).zfill(6)+".png"] 
      
      # Read-in image stack
      img_stack.append(Image.open(root_path + left_img_dir + stack_files[0]))
      img_stack.append(Image.open(root_path + right_img_dir + stack_files[1]))
    
    # Transform Images
    if transform:
        stack = []
        for img in img_stack:
            transformed_img = transform(img)
            stack.append(transformed_img)
            
    stack_ = stack
    image = torch.cat(stack_, 0).unsqueeze(0).to(par.device) # Tensor List -> Tensor

    # Append to Image Batch List
    #image_batch.append(image)
        
    #image_tensors = torch.cat(image_batch, 0).to(par.device)

    return image.float()

def gt_to_odom_gt(gt1_list, gt2_list, dataset):
    # gt1_list: list of transformation params (i-1)
    # gt2_list: list of transformation params (i)
    
    if dataset == 'kitti':
        gt_list1 = gt1_list.split()
        gt_list2 = gt2_list.split()
        
        T1 = torch.tensor([float(i) for i in gt_list1]).reshape(3,4)
        T2 = torch.tensor([float(i) for i in gt_list2]).reshape(3,4)
        #print(T1[:,:3].shape)
        #print(T1[:,3].shape)
        T1_inv = torch.cat((torch.t(T1[:,:3]),
                            torch.matmul(-1*torch.t(T1[:,:3]),T1[:,3].view(3,1))),1)
        
        T_base = torch.tensor([0, 0, 0, 1]).view(1,4)
        T2 = torch.cat((T2,T_base),0)
        T1_inv = torch.cat((T1_inv,T_base),0)
        T_delta = torch.matmul(T2, T1_inv).reshape(1,4,4)
        #print(T_delta[:,:3,4])
        t,r = util.transformation_matrix_to_euler_translation(T_delta[:,:3,:], 1)
        
        # GPS to Camera Orientation
        Rot_mat = torch.tensor([[0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0]])
        tvec = torch.tensor([t[0].item(),t[1].item(),t[2].item()])
        cam_tvec = torch.matmul(Rot_mat.view(3,3),tvec.view(3,1))
        
    return cam_tvec.view(3).tolist(),r.view(3).tolist()

def euler_to_transformation_matrix(t,r):
          roll_ang = r[:,0].view(1,1)
          pitch_ang = r[:,1].view(1,1)
          yaw_ang = r[:,2].view(1,1)
        
          rotation_batch = torch.zeros(1,3,3)
            
          ones_tensor = torch.ones(1,1,1).to(par.device)
          zeros_tensor = torch.zeros(1,1,1).to(par.device)
          sin_roll = torch.sin(roll_ang).view(1,1,1)
          cos_roll = torch.cos(roll_ang).view(1,1,1)
          sin_pitch = torch.sin(pitch_ang).view(1,1,1)
          cos_pitch = torch.cos(pitch_ang).view(1,1,1)
          sin_yaw = torch.sin(yaw_ang).view(1,1,1)
          cos_yaw = torch.cos(yaw_ang).view(1,1,1)
            
          # Roll Rotation Matrix Rows
          Rroll1 = torch.cat((ones_tensor, zeros_tensor, zeros_tensor),dim=2)
          Rroll2 = torch.cat((zeros_tensor, cos_roll, -1*sin_roll),dim=2)
          Rroll3 = torch.cat((zeros_tensor, sin_roll, cos_roll),dim=2)
          Rroll = torch.cat((Rroll1,Rroll2,Rroll3),dim=1)
            
          # Pitch Rotation Matrix Rows
          Rpitch1 = torch.cat((cos_pitch, zeros_tensor, sin_pitch),dim=2)
          Rpitch2 = torch.cat((zeros_tensor, ones_tensor, zeros_tensor),dim=2)
          Rpitch3 = torch.cat((-1*sin_pitch, zeros_tensor, cos_pitch),dim=2)
          Rpitch = torch.cat((Rpitch1,Rpitch2,Rpitch3),dim=1)
            
          # Yaw Rotation Matrix Rows
          Ryaw1 = torch.cat((cos_yaw, -1*sin_yaw, zeros_tensor), dim=2)
          Ryaw2 = torch.cat((sin_yaw, cos_yaw, zeros_tensor), dim=2)
          Ryaw3 = torch.cat((zeros_tensor, zeros_tensor, ones_tensor),dim=2)
          Ryaw = torch.cat((Ryaw1,Ryaw2,Ryaw3),dim=1)
            
          #R = torch.matmul(torch.matmul(Ryaw,Rpitch),Rroll)
          R = torch.matmul(torch.matmul(Rroll,Rpitch),Ryaw)
          rotation_batch = R.view(1,3,3)
        
          #ttest = torch.unsqueeze(torch.tensor(t),2)
          ttest = torch.unsqueeze(t,2)
          
          return torch.cat((rotation_batch,ttest),2)

def print_test_progress(img_dir, seq_dir, img_num, pose_out, a, b, t0, dataset):
    # seq_dir: the date/sequence path 
    # dataset: 'kitti' or 'euroc'
    #[seq_dir,img_num,_] = sample_id.split()
    kitti_ground_truth_dir = r'F:/OSUGrad/Research/Code/Datasets/KITTI/Visual_Odometry/data_odometry_poses/dataset/poses/'
    
    x = pose_out[:,0]
    y = pose_out[:,1]
    z = pose_out[:,2]
    roll = pose_out[:,3]
    pitch = pose_out[:,4]
    yaw = pose_out[:,5]
    
    # Printing Ground Truth to Reference while Training (Only)
    ground_truth_dirs = ["00","01","02","03","04","05","06","07","08","09","10"]
    
    if dataset == 'kitti':
        seq_bound = len(os.listdir(img_dir + seq_dir + "/image_2"))
        
        if seq_dir in ground_truth_dirs:
            if int(img_num) < (seq_bound-1):
                
                gt_filename1 = os.path.join(kitti_ground_truth_dir, seq_dir + ".txt").replace("\\","/")
                #print(gt_filename1)
                #with open(gt_filename1, 'r') as gt_data1:
                gt_data1 = open(gt_filename1,'r')
                #print(len(gt_data1.readlines()))
                gt = gt_data1.readlines()
                gt_data1.close()
                
                #print(gt)
                #print(gt[img_num+1])
                [t,r] = gt_to_odom_gt(gt[img_num], gt[img_num+1], "kitti")
                #print(t)
                #print(r)
                x1 = t[0]
                y1 = t[1]
                z1 = t[2]
                #print(r)
                roll1 = r[0]
                pitch1 = r[1]
                yaw1 = r[2]
                
                print(f"Affine Transformations: a = {a[0].item():.3f}, b = {b[0].item():.3f}")
                print(f"Est. Pose (m): x = {x[0]:.3f}, y = {y[0]:.3f}, z = {z[0]:.3f}, roll = {roll[0]:.3f}, pitch = {pitch[0]:.3f}, yaw = {yaw[0]:.3f}")
                print(f"Ground Truth (m): x = {x1:.3f}, y = {y1:.3f}, z = {z1:.3f}, roll = {pitch1:.3f}, pitch = {yaw1:.3f}, yaw = {roll1:.3f}")
                
            else:
                print("No ground truth available for this sequence ... \n")
        
        T = torch.tensor([float(i) for i in gt[img_num].split()]).reshape(3,4).to(par.device)
        T_base = torch.tensor([0, 0, 0, 1]).view(1,4).to(par.device)
        T = torch.cat((T,T_base),0).to(par.device)
        X0 = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(4,1).to(par.device)
        X_curr = torch.matmul(T, X0)
        #print(X_curr)
        
        # Estimated Pose
        T_est = euler_to_transformation_matrix(pose_out[:,:3],pose_out[:,3:]).to(par.device)
        T_est = torch.cat((T_est.view(3,4),T_base),0).to(par.device)
        t0 = torch.matmul(T_est,t0)
    return t0, X_curr[0], X_curr[2]
    
def test_loop(img_dir, seq_name, depthnet_model, posenet_model, model_name, dataset_type):

    if model_name == "d3vo":
        depthnet_model.eval() 
        posenet_model.eval()
        
        video_color = torch.zeros(1,256,512,3).to(par.device)
        video_depth = torch.zeros(1,256,512,1).to(par.device)
        
        Xest = torch.zeros(1).to(par.device)
        Zest = torch.zeros(1).to(par.device)
        Xgt = torch.zeros(1).to(par.device)
        Zgt = torch.zeros(1).to(par.device)
        t0 = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(4,1).to(par.device)
        
        et0 = time.time()
        with torch.inference_mode():
            N = len(os.listdir(img_dir + seq_name + "/image_2"))
            T_est = torch.eye(4).to(par.device)
            for j in range(N-1900):
                idx = j # Only works for one test folder
                print(idx)
                # Getting Data
                etn = time.time()
                p_images = read_test_input(par.kitti_img_dir_test, seq_name, idx, transform_kitti_test, "pose")
                d_images = p_images[:,:3,:,:]
                #p_images,_,__ = test_data_pose.__getitem__(idx,test_list) 
                #d_images,_,__ = test_data_depth.__getitem__(idx,test_list)
                p_images.float().to(par.device)
                d_images.float().to(par.device)
                
                # Prediction
                #beta=par.beta
                #sample_idx = test_list[idx]
                
                # Pose 
                translation1, rotation1, a1, b1 = posenet_model(p_images)
                T_mat = euler_to_transformation_matrix(t=translation1.reshape(1,3), 
                                                       r=rotation1.reshape(1,3)).reshape(3,4)
                T_base = torch.tensor([0,0,0,1]).reshape(1,4).to(par.device)
                T_pred = torch.cat((T_mat,T_base),0).to(par.device)
                T_est = torch.matmul(T_pred, T_est)
                
                pose_6dof_t_minus_1_t = torch.cat((translation1,rotation1),1)
                t0, x_gt, z_gt = print_test_progress(par.kitti_img_dir_test, seq_name, idx, pose_6dof_t_minus_1_t, a1, b1, t0, dataset_type)
                
                x_e = torch.tensor(T_est[0,3]).view(1).to(par.device)
                z_e = torch.tensor(T_est[2,3]).view(1).to(par.device)
                x_g = torch.tensor(x_gt).view(1).to(par.device)
                z_g = torch.tensor(z_gt).view(1).to(par.device)
                
                Xest = torch.cat((Xest, x_e),0).to(par.device)
                Zest = torch.cat((Zest, z_e),0).to(par.device)
                Xgt = torch.cat((Xgt, x_g),0).to(par.device)
                Zgt = torch.cat((Zgt, z_g),0).to(par.device)
                
                
                # Depth
                depth_block_out = depthnet_model(d_images)
                depth_block_s1 = depth_block_out[('disp', 0)]
                
                rgb_frame = d_images.permute(0,2,3,1)*255
                video_color = torch.cat((video_color, rgb_frame),0).to(par.device)
                #depth_frame = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * depth_block_s1[:,0,:,:]),1).permute(0,2,3,1)
                depth_frame = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * depth_block_s1[:,2,:,:]),1).permute(0,2,3,1)
                #depth_frame = torch.unsqueeze(depth_block_s1[:,0,:,:],1).permute(0,2,3,1)
                #print(torch.max(depth_frame))
                #print(torch.min(depth_frame))
                depth_max = torch.max(depth_frame)
                depth_frame = (depth_frame/depth_max)*255
                video_depth = torch.cat((video_depth, depth_frame),0).to(par.device)
                
                et1 = time.time()
            
    elif model_name == "monodepth2":
                
        depthnet_model[0].eval() 
        posenet_model[0].eval()
        
        depthnet_model[1].eval() 
        posenet_model[1].eval()
        
        # ------------- Below Needs to be edited ------------- #
        video_color = torch.zeros(1,190,640,3).to(par.device)
        video_depth = torch.zeros(1,190,640,1).to(par.device)
        
        Xest = torch.zeros(1).to(par.device)
        Zest = torch.zeros(1).to(par.device)
        Xgt = torch.zeros(1).to(par.device)
        Zgt = torch.zeros(1).to(par.device)
        t0 = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(4,1).to(par.device)
        
        et0 = time.time()
        with torch.inference_mode():
            N = len(os.listdir(img_dir + seq_name + "/image_2"))
            T_est = torch.eye(4).to(par.device)
            for j in range(N-1900):
                idx = j # Only works for one test folder
                print(idx)
                # Getting Data
                etn = time.time()
                p_images = read_test_input(par.kitti_img_dir_test, seq_name, idx, transform_kitti_test, "pose")
                d_images = p_images[:,:3,:,:]
                #p_images,_,__ = test_data_pose.__getitem__(idx,test_list) 
                #d_images,_,__ = test_data_depth.__getitem__(idx,test_list)
                p_images.float().to(par.device)
                d_images.float().to(par.device)
                
                # Prediction
                #beta=par.beta
                #sample_idx = test_list[idx]
                
                # Pose 
                features = posenet_model[0](p_images)
                rotation1, translation1 = posenet_model[1](features)
                #translation1, rotation1, a1, b1 = posenet_model(p_images)
                T_mat = euler_to_transformation_matrix(t=translation1.reshape(1,3), 
                                                       r=rotation1.reshape(1,3)).reshape(3,4)
                T_base = torch.tensor([0,0,0,1]).reshape(1,4).to(par.device)
                T_pred = torch.cat((T_mat,T_base),0).to(par.device)
                T_est = torch.matmul(T_pred, T_est)
                
                a1 = 1.0
                b1 = 0.0
                t0 = None
                pose_6dof_t_minus_1_t = torch.cat((translation1,rotation1),1)
                t0, x_gt, z_gt = print_test_progress(par.kitti_img_dir_test, seq_name, idx, pose_6dof_t_minus_1_t, a1, b1, t0, dataset_type)
                
                x_e = torch.tensor(T_est[0,3]).view(1).to(par.device)
                z_e = torch.tensor(T_est[2,3]).view(1).to(par.device)
                x_g = torch.tensor(x_gt).view(1).to(par.device)
                z_g = torch.tensor(z_gt).view(1).to(par.device)
                
                Xest = torch.cat((Xest, x_e),0).to(par.device)
                Zest = torch.cat((Zest, z_e),0).to(par.device)
                Xgt = torch.cat((Xgt, x_g),0).to(par.device)
                Zgt = torch.cat((Zgt, z_g),0).to(par.device)
                
                
                # Depth
                depth_block_out = depthnet_model(d_images)
                depth_block_s1 = depth_block_out[('disp', 0)]
                
                rgb_frame = d_images.permute(0,2,3,1)*255
                video_color = torch.cat((video_color, rgb_frame),0).to(par.device)
                #depth_frame = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * depth_block_s1[:,0,:,:]),1).permute(0,2,3,1)
                depth_frame = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * depth_block_s1[:,2,:,:]),1).permute(0,2,3,1)
                #depth_frame = torch.unsqueeze(depth_block_s1[:,0,:,:],1).permute(0,2,3,1)
                #print(torch.max(depth_frame))
                #print(torch.min(depth_frame))
                depth_max = torch.max(depth_frame)
                depth_frame = (depth_frame/depth_max)*255
                video_depth = torch.cat((video_depth, depth_frame),0).to(par.device)
                
                et1 = time.time()
    
    print(f"Frame Rate: {1.0/(et1 - etn)} (Hz) \n")
    
    
    print(f"Elapsed Time: {et1 - et0} (seconds) \n")
    torchvision.io.write_video("kitti_color_seq_" + seq_name + ".mp4", video_color.cpu(), 10)
    torchvision.io.write_video("kitti_depth_seq_" + seq_name + ".mp4", video_depth.repeat(1,1,1,3).cpu(), 10)
    
    Xest = Xest.cpu().numpy()
    Zest = Zest.cpu().numpy()
    Xgt = Xgt.cpu().numpy()
    Zgt = Zgt.cpu().numpy()
    
    plt.figure()
    plt.plot(Xest, Zest, 'b')
    #print(Xgt.shape)
    #print(Zgt.shape)
    plt.plot(Xgt, Zgt, 'r')
    
    plt.title("Pose for Sequence " + str(seq_name))
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("z")
    
    
    


# Load Models ...
model_name = "monodepth2" # 'd3vo', 'monodepth2'
if model_name == "d3vo":
    model_number = 19
    date = '2_25_2023'
    model_root_name = os.path.join('models','d3vo_model_')
    pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
    depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
    
    print("Loading Pose Model Checkpoint ... " + pose_model_checkpoint_path + "\n")
    print("Loading Depth Model Checkpoint ... " + depth_model_checkpoint_path + "\n")
    pose_checkpoint = torch.load(pose_model_checkpoint_path)
    depth_checkpoint = torch.load(depth_model_checkpoint_path)
    print("Loading Model State Dictionaries ... ")
    posenet_model.load_state_dict(pose_checkpoint['model_state_dict'])
    depthnet_model.load_state_dict(depth_checkpoint['model_state_dict'], strict=False)
    #posenet_model.eval()
    #depthnet_model.eval()
elif model_name == "monodepth2":
    
    model_name = "mono_640x192"
    model_root_name = os.path.join('models', model_name)
    
    # Paths ...
    pose_encoder_model_checkpoint_path = os.path.join(model_root_name, 'pose_encoder.pth')
    pose_decoder_model_checkpoint_path = os.path.join(model_root_name, 'pose.pth')
    depth_encoder_model_checkpoint_path = os.path.join(model_root_name, 'encoder.pth')
    depth_decoder_model_checkpoint_path = os.path.join(model_root_name, 'depth.pth')
    
    print("Loading Pose Model Checkpoints ... ")
    print(pose_encoder_model_checkpoint_path)
    print(pose_decoder_model_checkpoint_path)
    print("Loading Depth Model Checkpoints ... ")
    print(depth_encoder_model_checkpoint_path)
    print(depth_decoder_model_checkpoint_path)
    
    pose_encoder_checkpoint = torch.load(pose_encoder_model_checkpoint_path)
    pose_decoder_checkpoint = torch.load(pose_decoder_model_checkpoint_path)
    depth_encoder_checkpoint = torch.load(depth_encoder_model_checkpoint_path)
    depth_decoder_checkpoint = torch.load(depth_decoder_model_checkpoint_path)
    
    print("Loading Model State Dictionaries ... ")
    posenet_encoder_model.load_state_dict(pose_encoder_checkpoint['model_state_dict'])
    posenet_decoder_model.load_state_dict(pose_decoder_checkpoint['model_state_dict'])
    depthnet_encoder_model.load_state_dict(depth_encoder_checkpoint['model_state_dict'])
    depthnet_decoder_model.load_state_dict(depth_decoder_checkpoint['model_state_dict'])
    
    depthnet_model = [depthnet_encoder_model, depthnet_decoder_model]
    posenet_model = [posenet_encoder_model, posenet_decoder_model]
    
    
sequence_to_test = "05"
test_loop(par.kitti_img_dir_test, sequence_to_test, depthnet_model, posenet_model, model_name, "kitti")