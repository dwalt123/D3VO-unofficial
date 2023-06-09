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
    for idx in range(N):
        
        print("c: " + str(c)) # Stack index
        print("Batch # (Entire Dataset): " + str(c) + ", " + "Batch Number: " + str(idx))
        
        # Get Data for Training Iteration
        p_images,ppaths,pframes = training_data_pose.__getitem__(idx, shuffled_list)
        d_images,dpaths,dframes = training_data_depth.__getitem__(idx, shuffled_list)

        p_images.float().to(par.device)
        d_images.float().to(par.device)

        sample_idx = shuffled_list[batch_size*idx]
        #pose_subpath = ppaths[-1]
         
        # Prediction and Loss
        beta = par.beta
        
        # Intrinsic camera matrix
        intrinsic_mat = util.get_intrinsic_matrix(sample_idx, dataset_type)
        
        # Training and Loss
        # Logging data for NaN debugging ...
        #datalogger.add_param(datalogger.train_dict,"img_dir",img_dir)
        #datalogger.add_param(datalogger.train_dict,"p_images",p_images)
        #datalogger.add_param(datalogger.train_dict,"d_images",d_images)
        #datalogger.add_param(datalogger.train_dict,"intrinsic_mat",intrinsic_mat)
        #datalogger.add_param(datalogger.train_dict,"sample_idx",sample_idx)
        
        #(self,posenet_model,depthnet_model,source_imgs,source_img,K,beta,sid)
        
        # Formerly in loss function
        pose_input = p_images[:,:6,:,:]
        pose_6dof_t_minus_1_t,a1,b1 = posenet_model(pose_input)
        #pose_6dof_t_minus_1_t,a1,b1 = posenet_model(p_images[:,:6,:,:]) # 6DOF pose, affine params for T_t-1 to T_t
      
        reverse_tensor = torch.cat((p_images[:,6:,:,:],p_images[:,3:6,:,:]),1)
      
        pose_6dof_t_t_plus_1,a2,b2 = posenet_model(reverse_tensor) # 6DOF pose, affine params for T_t+1 to T_t
      
        util.print_regression_progress(img_dir,pose_6dof_t_minus_1_t, a1, b1, sample_idx, dataset_type)
      
        #depth_block_s1, depth_block_s2, depth_block_s3, depth_block_s4 = depthnet_model(d_images[:,:3,:,:])
        depth_input = d_images[:,:3,:,:]
        
        depth_block_out = depthnet_model(depth_input)
        #depth_block_out = depthnet_model(d_images[:,:3,:,:])
        depth_block_s1 = depth_block_out[('disp', 0)]
        depth_block_s2 = depth_block_out[('disp', 1)]
        depth_block_s3 = depth_block_out[('disp', 2)]
        depth_block_s4 = depth_block_out[('disp', 3)]
        
        # Dictionary of Scales
        scales = {'0':depth_block_s4,
                  '1':depth_block_s3,
                  '2':depth_block_s2,
                  '3':depth_block_s1}
      
        stereo_baseline = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:]
        #stereo_baseline.requires_grad_(True)
        # Affine Transformations for T_t-1->T_t and T_t+1->T_t
        #a1.requires_grad_(True)
        #a2.requires_grad_(True)
        #b1.requires_grad_(True)
        #b2.requires_grad_(True)
        a = [a1, a2]
        b = [b1, b2]
        #p_images.requires_grad_(True)
        #d_images.requires_grad_(True)
        #pose_6dof_t_minus_1_t.requires_grad_(True)
        #pose_6dof_t_t_plus_1.requires_grad_(True)
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
                       idx).float().to(par.device)
        '''loss = loss_fn(img_dir,
                       posenet_model,
                       depthnet_model,
                       p_images,
                       d_images,
                       intrinsic_mat,
                       beta,
                       sample_idx,
                       dataset_type).float().to(par.device)'''
        
        
       # print(torch.autograd.grad(res_min,res_t_minus_1))
        '''if math.isnan(loss):
            datalogger.add_param(datalogger.train_dict,"loss",loss)
            print("PoseNet Parameters with NaNs: ")
            pose_grad_nans = []
            for name, param in posenet_model.named_parameters():
                if param.grad is None:
                    print(str(name) + " is None")
                elif torch.any(torch.isnan(param.grad)):
                    print(name)
                    pose_grad_nans.append(name)
                    
            datalogger.add_param(datalogger.train_dict,"pose_grad_nans",pose_grad_nans)
            print("DepthNet Parameters with NaNs: ")
            depth_grad_nans = []
            for name, param in depthnet_model.named_parameters():
                if param.grad is None:
                    print(str(name) + " is None")
                elif torch.any(torch.isnan(param.grad)):
                    print(name)
                    depth_grad_nans.append(name)
            
            datalogger.add_param(datalogger.train_dict,"depth_grad_nans",depth_grad_nans)
            
            sys.exit("Loss is NaN ...")'''
        
        # Training Loss To Tensorboard
        writer.add_scalar("Training Loss",loss,batch)
        
        # Backpropagation
        #backward_hooks_posenet = [Hook(layer[1]) for layer in list(posenet_model._modules.items())]
        #backward_hooks_depthnet = [Hook(layer[1]) for layer in list(depthnet_model._modules.items())]
        
        
        #pose_optimizer.zero_grad() # Reset gradients to 0
        #depth_optimizer.zero_grad()
        
        #loss.retain_grad()
        # Zero out gradients
        
        #pose_optimizer.zero_grad()
        #depth_optimizer.zero_grad()
        joint_optimizer.zero_grad()
        
        loss.backward() # Backpropagate 
        #print("loss grad function: ")
        #print(loss.grad_fn)
        
        if math.isnan(loss):
            
            '''print("Posenet Model Hooks: ")
            for hooks in backward_hooks_posenet: 
                print(hooks.input)
                #print(hooks.names)
                #print(hooks.input)          
                #print(hooks.output)         
                print('---'*20)
            
            print("Depthnet Model Hooks: ")
            for hooks in backward_hooks_depthnet: 
                print(hooks.input)
                #print(hooks.names)
                #print(hooks.input)          
                #print(hooks.output)         
                print('---'*20)'''
            
            sys.exit("Loss is NaN ...")
        # Model Gradient Clipping ...
        '''torch.nn.utils.clip_grad_norm_(posenet_model.parameters(),
                                       1e-5,norm_type=2.0,
                                       error_if_nonfinite=False)
        torch.nn.utils.clip_grad_norm_(depthnet_model.parameters(),
                                       1e-5,norm_type=2.0,
                                       error_if_nonfinite=False)'''
        # 10/08/22 - Also try clip_grad_value!
        
        #pose_optimizer.step() # Proceed to next optimization step
        #depth_optimizer.step()
        joint_optimizer.step()
        
        loss, current = loss.item(), batch 
        avg_train_loss += loss # Summing train loss to average later
        print(f"loss: {loss:>7f} [{current+1:>5d}/{int(N):>5d}]") # formerly n/batch_size instead of epoch_size
        training_t1 = time.time()
        training_tn = util.time_stamp(training_t0, training_t1)
        print("Total Elapsed Time for Training: " + training_tn) 
        idx += 1
        c += 1
        batch += 1

    avg_train_loss = avg_train_loss / batch
    print(f"Avg. Train loss: {avg_train_loss:>8f} \n")
    return avg_train_loss  