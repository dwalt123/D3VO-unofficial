# Train and Test Loop Functions
#!pip install tensorboard
#from torch.utils.tensorboard import SummaryWriter
import time
import sys
import math
import torch
from params import par
#from datasets import training_data_pose, training_data_depth
from utils import util
from datatracker import datalogger, Hook
import wandb

'''

  TODO:
  1. Edit train loop to reflect D3VO (no ground truth, data shapes, number of samples, etc.)
  2. Get tensorboard setup to see smaller images, depth, uncertainty, etc. in google colab

  Edited: 9/24/22
'''
'''
def train_loop(img_dir, train_file, training_data_pose, training_data_depth, 
               posenet_model, depthnet_model, loss_fn, pose_optimizer, depth_optimizer,
               batch_size, writer, dataset_type):'
'''
def train_loop(img_dir, train_file, training_data_pose, training_data_depth, 
               posenet_model, depthnet_model, loss_fn, joint_optimizer,
               batch_size, writer, dataset_type):
    
    
    # Initializations
    avg_train_loss = 0 # tracking train loss

    # Going through all of the training data
    idx = 0 # Index of source image batch in that folder

    # Starting Training Time
    training_t0 = time.time()
    
    c = 0 # c is now the index for the shuffled data
    batch = c
    
    # Training Mode
    depthnet_model.train() 
    posenet_model.train()
    
    idx = 0
    # Read in data
    train_data_files = open(train_file,'r')
    train_list = train_data_files.readlines()
    train_data_files.close()
    # Shuffle data
    shuffled_list = util.shuffle_sequence(train_list)
    N = int(len(shuffled_list)/par.batch_size)
    #loss_grads = []
    
    #use_amp = True
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp) # For Tensor Cores ...
    for idx in range(N):
        
        print("c: " + str(c)) # Stack index
        print("Batch # (Entire Dataset): " + str(c) + ", " + "Batch Number: " + str(idx))
        
        # Get Data for Training Iteration
        '''
        if par.use_stereo:
            random_pick = torch.randint(0,2,(1,)).item()
            if random_pick == 0:
                cam_view = "left"
            elif random_pick == 1:
                cam_view = "right"
        else:
            cam_view = "left"
        '''
        cam_view = "left"
        
        p_images = training_data_pose.__getitem__(idx, shuffled_list, cam_view)
        d_images = training_data_depth.__getitem__(idx, shuffled_list, cam_view)

        #p_images.float().to(par.device)
        #d_images.float().to(par.device)
        
        p_images["t"].requires_grad = True
        p_images["t+1"].requires_grad = True
        p_images["t-1"].requires_grad = True
        d_images["t"].requires_grad = True
        d_images["ts"].requires_grad = True
        
        #p_images.requires_grad = True
        #d_images.requires_grad = True
        
        sample_idx = shuffled_list[batch_size*idx]
        #pose_subpath = ppaths[-1]
         
        # Prediction and Loss
        beta = par.beta
        
        # Intrinsic camera matrix
        intrinsic_mat = util.get_intrinsic_matrix(sample_idx, dataset_type) # 6/10/23 : Might need to normalize K (see monodepth github issues)
        
        # Training and Loss
        
        # Formerly in loss function
        pose_input = torch.cat((p_images["t-1"], p_images["t"]),1)
        #translation1, rotation1, a1, b1 = posenet_model(pose_input)
        translation1, rotation1 = posenet_model(pose_input)
        pose_6dof_t_minus_1_t = torch.cat((translation1,rotation1),1).to(par.device)
        
        reverse_tensor = torch.cat((p_images["t+1"], p_images["t"]),1)
        #translation2, rotation2, a2, b2 = posenet_model(reverse_tensor)
        translation2, rotation2 = posenet_model(reverse_tensor)
        pose_6dof_t_t_plus_1 = torch.cat((translation2,rotation2),1).to(par.device)
        
        #depth_input = d_images[:,:3,:,:]
        # Always want to feed the left image in the stereo pair for either monocular or stereo case
        '''
        if par.use_stereo:
            if d_images["cam"] == "left":
                depth_input = d_images["t"]
            elif d_images["cam"] == "right":
                depth_input = d_images["ts"]
        else:
            depth_input = d_images["t"]
        '''
        depth_input = d_images["t"]
        
        depth_block_out = depthnet_model(depth_input)
        depth_block_s1 = depth_block_out[('disp', 0)]
        depth_block_s2 = depth_block_out[('disp', 1)]
        depth_block_s3 = depth_block_out[('disp', 2)]
        depth_block_s4 = depth_block_out[('disp', 3)]
        
        # Dictionary of Scales
        scales = {'0':depth_block_s4.requires_grad_(True),
                  '1':depth_block_s3.requires_grad_(True),
                  '2':depth_block_s2.requires_grad_(True),
                  '3':depth_block_s1.requires_grad_(True)}
        
        '''
        if par.use_stereo:
            if d_images["cam"] == "left":
                # Right Cam to Left Cam
                stereo_baseline = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:]
            elif d_images["cam"] == "right":
                # Left Cam to Right Cam
                stereo_base = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:]
                # Inverse
                Rs = stereo_base[:3,:3]
                trans = stereo_base[:,3].view(3,1)
                stereo_baseline = torch.cat((torch.t(Rs),torch.matmul(-1*torch.t(Rs),trans)),1)
        else:
            stereo_baseline = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:]
            #print(stereo_baseline)
        '''
        stereo_baseline = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:].to(par.device)
        '''
        if par.use_ab:
            a = [a1, a2]
            b = [b1, b2]
        else:
            a = []
            b = []
        '''
        a = []
        b = []
        
        loss = loss_fn(p_images,
                       d_images,
                       scales,
                       stereo_baseline,
                       intrinsic_mat,
                       pose_6dof_t_minus_1_t,
                       pose_6dof_t_t_plus_1,
                       a,
                       b,
                       beta,
                       idx,
                       dataset_type).float().to(par.device)
        
        
        # Training Loss To Tensorboard
        writer.add_scalar("Training Loss",loss,batch)
        
        # Backpropagation
        joint_optimizer.zero_grad()
        loss.backward()
        
        if math.isnan(loss):
            sys.exit("Loss is NaN ...")
        
        joint_optimizer.step()
        
        a1 = 1.0
        b1 = 0.0
        inv_depth = 1.0/depth_block_s1
        mean_inv_depth = inv_depth.mean(3,False).mean(2,False).reshape(par.batch_size)
        #print(mean_inv_depth.shape)
        util.print_regression_progress(img_dir,pose_6dof_t_minus_1_t, mean_inv_depth,
                                       a1, b1, sample_idx, dataset_type)
        loss, current = loss.item(), batch 
        avg_train_loss += loss # Summing train loss to average later
        
        wandb.log({"current_train_loss": loss})
        print(f"loss: {loss:>7f} [{current+1:>5d}/{int(N):>5d}]") # formerly n/batch_size instead of epoch_size
        training_t1 = time.time()
        training_tn = util.time_stamp(training_t0, training_t1)
        print("Total Elapsed Time for Training: " + training_tn) 
        idx += 1
        c += 1
        batch += 1

    avg_train_loss = avg_train_loss / batch
    print(f"Avg. Train loss: {avg_train_loss:>8f} \n")
    
    #del p_images, pose_input
    #del d_images, depth_input, depth_block_out, depth_block_s1, depth_block_s2, 
    #depth_block_s3, depth_block_s4
    #del loss
    
    return avg_train_loss  