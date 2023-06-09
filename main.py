# Imports 
import time
import os
#import itertools
#import sys
import torch
from torch.utils.tensorboard import SummaryWriter
#import copy

# Local Imports
from loss import TotalLoss
from posenet import posenet_model
from mono_depthnet import depthnet_model
#from train import *
#from test import *
import validation
import train
from utils import util
from params import par
from datasets import *
from datatracker import datalogger

import wandb

if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True) # Detect NaN location
    # Train Model
    torch.cuda.empty_cache() # clear memory from previous runs
    t0 = time.time()
    #epochs = [20,60] # 20 on KITTI and 40 on EuRoC MAV
    epochs = [20]
    epoch_writer = SummaryWriter() # writer specifically for epochs
    
    '''
    9/24/22 Update:
    The KITTI Eigen Split has named samples in splits/eigen_zhou:
    https://github.com/nianticlabs/monodepth2/tree/master/splits
    
    - See example.py on how to properly read in an image from the raw data
    - Have to fix dataloader.py to read in image properly.
    - The backup is dataloader.py
    - Note since you're using left and right you'll have to track that for
      image warping! Read paper to clarify if they use left only or both.
      
    '''
    
    # For Checkpoint Saving
    SAVE_CHECKPOINT = 60*60*2 # time (in seconds) to save checkpoint (e.g., 10 hours = 36000 seconds)
    model_number = 0
    date = '2_25_2023'
    model_root_name = os.path.join('models','d3vo_model_')
    pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
    depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
    
    experiment_name = 'd3vo_model_' + date + '_v' + str(model_number) + '_' + str(0)
    #wandb.init(project=experiment_name) # Initialize Weights and Biases Experiment
    
    checkpoint_epoch = 0 # The epoch the model was saved at
    checkpoint_loss = 0 # The loss value the model was saved at
    
    # Optimizer
    #learning_rate = [1e-4, 1e-5] # 1e-4, 1e-5 for last 5 epochs
    learning_rate = [1e-4, 1e-5]
    loss_fn = TotalLoss()
    
    alpha1 = 0
    #pose_optimizer = torch.optim.Adam(posenet_model.parameters(), lr=learning_rate[0], weight_decay=alpha1)
    #depth_optimizer = torch.optim.Adam(depthnet_model.parameters(), lr=learning_rate[0], weight_decay=alpha1)
    
    #model_parameters = itertools.chain(posenet_model.parameters(),depthnet_model.parameters())
    model_parameters = list(posenet_model.parameters()) + list(depthnet_model.parameters())
    
    if par.optimizer_name == "sgd":
        joint_optimizer = torch.optim.SGD(model_parameters, lr=learning_rate[0], momentum = par.momentum, weight_decay=alpha1)
    else:
        joint_optimizer = torch.optim.Adam(model_parameters, lr=learning_rate[0], weight_decay=alpha1)
    
    #scheduler = None
    scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=epochs[0]-5, gamma=0.1)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, T_max=epochs[1], eta_min=1e-5, verbose=True)
    
        
    FROM_CHECKPOINT = False # If you should load a model checkpoint
    FROM_SEED = True # If starting from a particular initialization/model seed for both depthnet and posenet
    
    # Configuration
    wandb.init(project=experiment_name,
               config = {
                   "learning_rate1": learning_rate[0],
                   "learning_rate2": learning_rate[1],
                   "batch_size": par.batch_size,
                    "epochs_kitti": epochs[0],
                    #"epochs_euroc_mav": epochs[1],
                    "alpha": par.alpha,
                    "beta": par.beta,
                    "normalize_data": par.normalize_data,
                    "normalize_kitti": par.torch_normalize_kitti,
                    "normalize_euroc": par.torch_normalize_euroc,
                    "resize_shape": par.torch_resize,
                    "block_activation": par.block_activation,
                    "activation": par.activation,
                    "pose_scaling": par.pose_scaling,
                    "pose_scale": par.pose_scale,
                    "interpolate_maps": par.interpolate_maps,
                    "uncer_low": par.uncer_low,
                    "uncer_high": par.uncer_high,
                    "monodepth_scaling": par.monodepth_scaling,
                    "loss_smooth_reduction": par.loss_smooth_reduction,
                    "mean_depth_norm": par.mean_depth_norm,
                    "delta_max": par.delta_max,
                    "optimizer_name": par.optimizer_name,
                    "momentum": par.momentum,
                    "use_stereo": par.use_stereo,
                    "use_uncer": par.use_uncer,
                    "use_ab": par.use_ab
                    })
    
    
    if not FROM_CHECKPOINT and not FROM_SEED:
        # Save Initial Model States
        pose_init_path = model_root_name + date + '_v' + str(model_number) + '_pose_init.pth'
        torch.save({'checkpoint_epoch': 0,
                    'model_state_dict': posenet_model.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'optimizer_state_dict': joint_optimizer.state_dict(),
                    'checkpoint_loss': None,
                    }, pose_init_path)
        print('Initial PoseNet Model Checkpoint saved at '+ pose_init_path)
        depth_init_path = model_root_name + date + '_v' + str(model_number) + '_depth_init.pth'
        torch.save({'checkpoint_epoch': 0,
                    'model_state_dict': depthnet_model.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'optimizer_state_dict': joint_optimizer.state_dict(),
                    'checkpoint_loss': None,
                    }, depth_init_path)
        print('Initial DepthNet Model Checkpoint saved at '+ depth_init_path)
    
    # Loading model checkpoint ...
    if FROM_CHECKPOINT:
      # Assuming model, optimizer, and scheduler defined
      print("Loading Pose Model Checkpoint ... " + pose_model_checkpoint_path + "\n")
      print("Loading Depth Model Checkpoint ... " + depth_model_checkpoint_path + "\n")
      pose_checkpoint = torch.load(pose_model_checkpoint_path)
      depth_checkpoint = torch.load(depth_model_checkpoint_path)
    
      print("Loading Model State Dictionaries ... ")
      posenet_model.load_state_dict(pose_checkpoint['model_state_dict'])
      depthnet_model.load_state_dict(depth_checkpoint['model_state_dict'], strict=False)
      print("Finished. \n")
      
      print("Loading Optimizer State Dictionary ... ")
      #pose_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      #depth_optimizer.load_state_dict(depth_checkpoint['optimizer_state_dict'])
      joint_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      print("Finished. \n")
      
      
      print("Loading Scheduler State Dictionary ... ")
      scheduler.load_state_dict(pose_checkpoint['scheduler_state_dict'])
      print("The current learning rate: " + str(scheduler.get_last_lr()))
      
      print("Finished. \n")
      
      print("Starting at Epoch ... ")
      checkpoint_epoch = depth_checkpoint['checkpoint_epoch']+1
      print(str(checkpoint_epoch+1))
    
      print("Last checkpoint average loss ... ")
      some_loss = depth_checkpoint['checkpoint_loss']
      print(str(some_loss) + ".\n")
      
    elif FROM_SEED:
      pose_init_path = model_root_name + date + '_v' + str(model_number) + '_pose_init.pth'
      depth_init_path = model_root_name + date + '_v' + str(model_number) + '_depth_init.pth'
      print("Starting from model seeds ... ")
      print(pose_init_path)
      print(depth_init_path)
      print("\n")
      
      # Assuming model, optimizer, and scheduler defined
      print("Loading Pose Model Checkpoint ... " + pose_init_path + "\n")
      print("Loading Depth Model Checkpoint ... " + depth_init_path + "\n")
      pose_checkpoint = torch.load(pose_init_path)
      depth_checkpoint = torch.load(depth_init_path)
    
      print("Loading Model State Dictionaries ... ")
      posenet_model.load_state_dict(pose_checkpoint['model_state_dict'])
      depthnet_model.load_state_dict(depth_checkpoint['model_state_dict'], strict=False)
      print("Finished. \n")
      
      print("Loading Optimizer State Dictionary ... ")
      #pose_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      #depth_optimizer.load_state_dict(depth_checkpoint['optimizer_state_dict'])
      joint_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      print("Finished. \n")
      
      
      print("Loading Scheduler State Dictionary ... ")
      scheduler.load_state_dict(pose_checkpoint['scheduler_state_dict'])
      print(scheduler.get_last_lr())
      print("The current learning rate: " + str(scheduler.get_last_lr()))
      
      print("Finished. \n")
      
      print("Starting at Epoch ... ")
      checkpoint_epoch = depth_checkpoint['checkpoint_epoch']+1
      print(str(checkpoint_epoch+1))
    
      print("Last checkpoint average loss ... ")
      some_loss = depth_checkpoint['checkpoint_loss']
      print(str(some_loss) + ".\n")
      
    else:
      print("Can't load from later checkpoint and seed ...")
    
    checkpoint_timer0 = t0 # Initializing Checkpoint timer
    checkpoint_timer1 = time.time() # Initial timer for checkpoint 2
    # Training ...
    # KITTI, EuRoC MAV
    SWITCH_DATASET = False # Switch dataset
    #num_epochs = epochs[1]
    num_epochs = epochs[0]
    dataset_type = 'kitti'
    
    training_data_pose = kitti_training_data_pose
    training_data_depth = kitti_training_data_depth
    val_data_pose = kitti_val_data_pose
    val_data_depth = kitti_val_data_depth
    
    img_dir = par.kitti_img_dir
    train_split = par.kitti_train_split
    val_split = par.kitti_val_split
    
    Loss = TotalLoss().to(par.device)
    
    #wandb.watch(posenet_model)
    #wandb.watch(depthnet_model)
    
    # Optimizing for Tensor Cores
    #use_amp = True
    #scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    for t in range(checkpoint_epoch, num_epochs):
        checkpoint_timer1 = time.time()
        
        print("Epoch Number: " + str(t))
        '''
        if t >= epochs[0]-1:
            SWITCH_DATASET = True
            dataset_type = 'euroc'
            training_data_pose = euroc_training_data_pose
            training_data_depth = euroc_training_data_depth
            val_data_pose = euroc_val_data_pose
            val_data_depth = euroc_val_data_depth
            img_dir = par.euroc_mav_img_dir
            train_split = par.euroc_train_split
            val_split = par.euroc_val_split
            # Code to switch dataset
        '''
        
        # Time to save checkpoint?
        if (checkpoint_timer1 - checkpoint_timer0) > SAVE_CHECKPOINT:
          model_number += 1
          pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
          depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
          #util.save_checkpoint(posenet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
          #util.save_checkpoint(depthnet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
          util.save_checkpoint(posenet_model, joint_optimizer, scheduler, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
          util.save_checkpoint(depthnet_model, joint_optimizer, scheduler, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
          checkpoint_timer0 = time.time() # Reinitialize timer
        
        
        '''
        if checkpoint_epoch == epochs[0]-5: 
            for opt_param in joint_optimizer.param_groups:
                opt_param['lr'] = learning_rate[1]
        '''
    
        # Main Epoch Loop
        print(f"Epoch {t+1}\n-------------------------------")
        print("Train Loop \n")
    
        writer = SummaryWriter() # Writer for train loop
        train_time0 = time.time()
        
        # Casting for Tensor Core Usage ...
        train_loss = train.train_loop(img_dir, train_split, training_data_pose,
                                          training_data_depth, posenet_model, 
                                          depthnet_model, loss_fn, joint_optimizer,
                                          par.batch_size,
                                          writer,
                                          dataset_type)
            
       
        train_time1 = time.time()
        #wandb.log({"train_loop_time": train_time1-train_time0})
            
        train_t = util.time_stamp(train_time0, train_time1)
        print("Epoch " + str(t+1) + " Train Time: " + train_t)
        writer.flush()
        
        print("\nTest Loop \n")
        test_time0 = time.time()
        
        #deepvo_model.eval()
        val_loss = validation.test_loop(img_dir, val_split, val_data_pose,
                                            val_data_depth, depthnet_model, 
                                            posenet_model, loss_fn, dataset_type)
        
        test_time1 = time.time()
        
        test_t = util.time_stamp(test_time0, test_time1)
        print("Epoch " + str(t+1) + " Test Time: " + test_t)
        epoch_time = util.time_stamp(0,(train_time1-train_time0)+(test_time1-test_time0))
        print("Total Epoch Time: " + epoch_time + ".\n")
        
        # Add Average Train and Validation Loss to Tensorboard
        epoch_writer.add_scalars('Train AND Validation Losses', 
                                 {'train_loss': train_loss, 'validation_loss': val_loss},
                                 t+1)
        
        # Uncertainty Map and Depth Map Progress ...
        progress_files = open(val_split,'r')
        progress_list = progress_files.readlines()
        test_img = val_data_depth.__getitem__(0,progress_list,"left")
        #db1, db2, db3, db4 = depthnet_model(test_img[:,:3,:,:])
        
        #writer.add_scalar('Learning Rate', scheduler.get_lr()[0], t+1)
        scheduler.step()
        
        #writer.add_scalars
        # Changed to work with MonoDepth2 implementation
        depth_block_out = depthnet_model(test_img["t"])
        #db1 = depth_block_out[('disp', 0)]
        #db2 = depth_block_out[('disp', 1)]
        #db3 = depth_block_out[('disp', 2)]
        #db4 = depth_block_out[('disp', 3)]
        
        datalogger.write_image(test_img["t"][0,:,:,:],str(t),"rgb.png",None,False)
        #shuffled_list = util.shuffle_sequence(progress_list)
        #sample_idx_ =  progress_list[0]
        #stereo_baseline_ = util.get_stereo_baseline_transformation(sample_idx_, dataset_type)[:3,:]
        #intrinsic_mat_ = util.get_intrinsic_matrix(sample_idx_, dataset_type)
        #bf = stereo_baseline_[0,-1]*intrinsic_mat_[0][0,0]
        
        for m in range(1):
            db = depth_block_out[('disp', m)]
            #sample_depth_map = 1.0/torch.clamp(80*db[:,0,:,:], 1e-3, 80)
            #sample_stereo_depth_map = 1.0/torch.clamp(80*db[:,1,:,:], 1e-3, 80)
            
            # Now learning disparity images
            stereo_max = torch.max(1 / (1/100 + (1/1e-1 - 1/100) * db[0,0,:,:]))
            #sample_depth_map = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * db[0,0,:,:],0))*255.0/stereo_max
            sample_depth_map = (1 / (1/100 + (1/1e-1 - 1/100) * db[0,0,:,:]))*255.0/stereo_max
            if par.use_stereo:
                stereo_max = torch.max(1 / (1/100 + (1/1e-1 - 1/100) * db[0,1,:,:]))
                sample_stereo_depth_map = torch.unsqueeze(1 / (1/100 + (1/1e-1 - 1/100) * db[0,1,:,:]),0)*255.0/stereo_max
            if par.use_uncer:
                uncer_max = torch.max(db[0,2,:,:])
                sample_uncer_map = torch.unsqueeze(db[0,2,:,:],0)*255.0/uncer_max
            
            # Add Images to Weights and Biases ...
            wandb_mono_depth_map = wandb.Image(sample_depth_map, caption="Monocular Depth Map")
            wandb.log({"mono_depth": wandb_mono_depth_map})
            del wandb_mono_depth_map
            
            if par.use_stereo:
                wandb_stereo_depth_map = wandb.Image(sample_stereo_depth_map, caption="Stereo Depth Map")
                wandb.log({"stereo_depth": wandb_stereo_depth_map})
                del wandb_stereo_depth_map
            
            if par.use_uncer:
                wandb_uncer_map = wandb.Image(sample_uncer_map, caption="Uncertainty Map")
                wandb.log({"uncertainty": wandb_uncer_map})
                del wandb_uncer_map
            
            # Add the images to tensorboard ...
            '''
            writer.add_images('Depth Map Batch, Scale: ' + str(m),
                              torch.unsqueeze(sample_depth_map,0),0)
            if par.use_stereo:
                writer.add_images('Stereo Depth Map Batch, Scale: ' + str(m),
                                  torch.unsqueeze(sample_stereo_depth_map,0),0)
            if par.use_uncer:
                writer.add_images('Uncertainty Map Batch, Scale: ' + str(m),
                                  torch.unsqueeze(sample_uncer_map,0),0)
            '''
            # Save the images locally ...
            datalogger.write_image(sample_depth_map,str(t),"depth_map_s" + str(m) + ".png","magma", True)
            if par.use_stereo:
                datalogger.write_image(sample_stereo_depth_map,str(t),"stereo_depth_map_s" + str(m) + ".png","magma", True)
                del sample_stereo_depth_map
            if par.use_uncer:
                datalogger.write_image(sample_uncer_map,str(t),"uncertainty_map_s" + str(m) + ".png","viridis", True)
                del sample_uncer_map
            
            del sample_depth_map
            
        progress_files.close()
        
        writer.flush()
        epoch_writer.flush()
    
        checkpoint_epoch = t # The last finished epoch.
        checkpoint_loss = val_loss # Last known validation loss
        
    
    t1 = time.time()
    total_time = util.time_stamp(t0, t1)   
    print("Training Done!")
    print("Total Time: " + total_time)
    writer.close()
    epoch_writer.close()
    
    # Save Final Model
    model_number += 1
    pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
    depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
    #util.save_checkpoint(posenet_model, pose_optimizer, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
    #util.save_checkpoint(depthnet_model, depth_optimizer, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
    util.save_checkpoint(posenet_model, joint_optimizer, scheduler, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
    util.save_checkpoint(depthnet_model, joint_optimizer, scheduler, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)