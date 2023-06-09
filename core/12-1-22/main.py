# Imports 
import time
import os
import itertools
#import sys
import torch
from torch.utils.tensorboard import SummaryWriter
import copy

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


if __name__ == "__main__":
    #torch.autograd.set_detect_anomaly(True) # Detect NaN location
    # Train Model
    torch.cuda.empty_cache() # clear memory from previous runs
    t0 = time.time()
    epochs = [20,60] # 20 on KITTI and 40 on EuRoC MAV
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
    date = '11_30_2022'
    model_root_name = os.path.join('models','d3vo_model_')
    pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
    depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
    
    checkpoint_epoch = 0 # The epoch the model was saved at
    checkpoint_loss = 0 # The loss value the model was saved at
    
    # Optimizer
    #learning_rate = [1e-4, 1e-5] # 1e-4, 1e-5 for last 5 epochs
    learning_rate = [1e-4, 1e-5]
    loss_fn = TotalLoss()
    
    alpha1 = 0
    #pose_optimizer = torch.optim.Adam(posenet_model.parameters(), lr=learning_rate[0], weight_decay=alpha1)
    #depth_optimizer = torch.optim.Adam(depthnet_model.parameters(), lr=learning_rate[0], weight_decay=alpha1)
    
    model_parameters = itertools.chain(posenet_model.parameters(),depthnet_model.parameters())
    joint_optimizer = torch.optim.Adam(model_parameters, lr=learning_rate[0], weight_decay=alpha1)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, T_max=epochs[1], eta_min=1e-5, verbose=True)
    
    FROM_CHECKPOINT = False # If you should load a model checkpoint
    
    if not FROM_CHECKPOINT:
        # Save Initial Model States
        pose_init_path = model_root_name + 'pose_init.pth'
        torch.save({'checkpoint_epoch': 0,
                    'model_state_dict': posenet_model.state_dict(),
                    #'scheduler_state_dict': scheduler.state_dict(),
                    'optimizer_state_dict': joint_optimizer.state_dict(),
                    'checkpoint_loss': None,
                    }, pose_init_path)
        print('Initial PoseNet Model Checkpoint saved at '+ pose_init_path)
        depth_init_path = model_root_name + 'depth_init.pth'
        torch.save({'checkpoint_epoch': 0,
                    'model_state_dict': depthnet_model.state_dict(),
                    #'scheduler_state_dict': scheduler.state_dict(),
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
      
      print("Loading Optimizer State Dictionaries ... ")
      #pose_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      #depth_optimizer.load_state_dict(depth_checkpoint['optimizer_state_dict'])
      joint_optimizer.load_state_dict(pose_checkpoint['optimizer_state_dict'])
      print("Finished. \n")
    
      print("Starting at Epoch ... ")
      checkpoint_epoch = depth_checkpoint['checkpoint_epoch']+1
      print(str(checkpoint_epoch+1))
    
      print("Last checkpoint average loss ... ")
      some_loss = depth_checkpoint['checkpoint_loss']
      print(str(some_loss) + ".\n")
    
      #posenet_model.train()
      #depthnet_model.train()
    
    
    checkpoint_timer0 = t0 # Initializing Checkpoint timer
    checkpoint_timer1 = time.time() # Initial timer for checkpoint 2
    # Training ...
    # KITTI, EuRoC MAV
    SWITCH_DATASET = False # Switch dataset
    num_epochs = epochs[1]
    dataset_type = 'kitti'
    
    training_data_pose = kitti_training_data_pose
    training_data_depth = kitti_training_data_depth
    val_data_pose = kitti_val_data_pose
    val_data_depth = kitti_val_data_depth
    
    img_dir = par.kitti_img_dir
    train_split = par.kitti_train_split
    val_split = par.kitti_val_split
    
    Loss = TotalLoss().to(par.device)
    
    for t in range(checkpoint_epoch, num_epochs):
        checkpoint_timer1 = time.time()
        
        print(t)
        if t >= epochs[0]-1:
            SWITCH_DATASET = True
            dataset_type = 'euroc'
            training_data_pose = euroc_training_data_pose
            training_data_depth = euroc_training_data_depth
            val_data_pose = euroc_val_data_pose
            val_data_depth = euroc_val_data_depth
            img_dir = par.euroc_mav_img_dir
            train_split = par.euroc_train_split
            #train_split = r'split/EuRoC_MAV/train_subset.txt'
            #val_split = r'split/EuRoC_MAV/val_subset.txt'
            val_split = par.euroc_val_split
            # Code to switch dataset
        
        #print(train_split)
        #print(val_split)
        if t > (epochs[1]-5):
            '''for group in pose_optimizer.param_groups:
                group['lr'] = learning_rate[1]
            for group in depth_optimizer.param_groups:
                group['lr'] = learning_rate[1]'''
            for group in joint_optimizer.param_groups:
                group['lr'] = learning_rate[1]
            
        
        # Time to save checkpoint?
        if (checkpoint_timer1 - checkpoint_timer0) > SAVE_CHECKPOINT:
          model_number += 1
          pose_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_pose.pth'
          depth_model_checkpoint_path = model_root_name + date + '_v' + str(model_number) + '_depth.pth'
          #util.save_checkpoint(posenet_model, pose_optimizer, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
          #util.save_checkpoint(depthnet_model, depth_optimizer, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
          util.save_checkpoint(posenet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
          util.save_checkpoint(depthnet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
          checkpoint_timer0 = time.time() # Reiinitialize timer
    
        '''if t > (epochs[1]-5):
          alpha1 = 0
          pose_optimizer = torch.optim.Adam(posenet_model.parameters(), 
                                            lr=learning_rate[1], 
                                            weight_decay=alpha1)
          depth_optimizer = torch.optim.Adam(depthnet_model.parameters(), 
                                             lr=learning_rate[1], 
                                             weight_decay=alpha1)'''
    
        # Main Epoch Loop
        print(f"Epoch {t+1}\n-------------------------------")
        print("Train Loop \n")
    
        writer = SummaryWriter() # Writer for train loop
        train_time0 = time.time()
        '''train_loss = train.train_loop(img_dir, train_split, training_data_pose,
                                      training_data_depth, posenet_model, 
                                      depthnet_model, loss_fn, pose_optimizer,
                                      depth_optimizer,
                                      par.batch_size,
                                      writer,
                                      dataset_type)'''
        train_loss = train.train_loop(img_dir, train_split, training_data_pose,
                                      training_data_depth, posenet_model, 
                                      depthnet_model, loss_fn, joint_optimizer,
                                      par.batch_size,
                                      writer,
                                      dataset_type)
        train_time1 = time.time()
        train_t = util.time_stamp(train_time0, train_time1)
        print("Epoch " + str(t+1) + " Train Time: " + train_t)
        writer.flush()
    
        print("\nTest Loop \n")
        test_time0 = time.time()
    
        #deepvo_model.eval()
        val_loss = validation.test_loop(img_dir, val_split, val_data_pose,
                                        val_data_depth, depthnet_model, 
                                        posenet_model, loss_fn, dataset_type)
        
        posenet_weights = copy.deepcopy(posenet_model.state_dict())
        depthnet_weights = copy.deepcopy(depthnet_model.state_dict())
    
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
        test_img,dtemppaths,dtempcams = val_data_depth.__getitem__(0,progress_list)
        #db1, db2, db3, db4 = depthnet_model(test_img[:,:3,:,:])
        
        #writer.add_scalar('Learning Rate', scheduler.get_lr()[0], t+1)
        #scheduler.step()
        
        #writer.add_scalars
        # Changed to work with MonoDepth2 implementation
        depth_block_out = depthnet_model(test_img[:,:3,:,:])
        db1 = depth_block_out[('disp', 0)]
        db2 = depth_block_out[('disp', 1)]
        db3 = depth_block_out[('disp', 2)]
        db4 = depth_block_out[('disp', 3)]
        
        datalogger.write_image(test_img[0,:3,:,:],str(t),"rgb.png",None,False)
        #shuffled_list = util.shuffle_sequence(progress_list)
        sample_idx_ =  progress_list[0]
        stereo_baseline_ = util.get_stereo_baseline_transformation(sample_idx_, dataset_type)[:3,:]
        intrinsic_mat_ = util.get_intrinsic_matrix(sample_idx_, dataset_type)
        bf = stereo_baseline_[0,-1]*intrinsic_mat_[0][0,0]
        
        for m in range(4):
            db = depth_block_out[('disp', m)]
            #sample_depth_map = 1.0/torch.clamp(80*db[:,0,:,:], 1e-3, 80)
            #sample_stereo_depth_map = 1.0/torch.clamp(80*db[:,1,:,:], 1e-3, 80)
            # Now learning disparity images
            sample_depth_map = torch.unsqueeze(db[0,0,:,:],0)*bf
            sample_stereo_depth_map = torch.unsqueeze(db[0,1,:,:],0)*bf
            #sample_depth_map = torch.unsqueeze(db[0,0,:,:],0)
            #sample_stereo_depth_map = torch.unsqueeze(db[0,1,:,:],0)
            sample_uncer_map = torch.unsqueeze(db[0,2,:,:],0)
        
            # Add the images to tensorboard ...
            writer.add_images('Depth Map Batch, Scale: ' + str(m),
                              torch.unsqueeze(sample_depth_map,0),0)
            writer.add_images('Stereo Depth Map Batch, Scale: ' + str(m),
                              torch.unsqueeze(sample_stereo_depth_map,0),0)
            writer.add_images('Uncertainty Map Batch, Scale: ' + str(m),
                              torch.unsqueeze(sample_uncer_map,0),0)
            
            # Save the images locally ...
            #datalogger.write_image(test_img[0,:3,:,:],str(t),"rgb.png",None,False)
            #datalogger.write_image(test_img[0,2,:,:],str(t),"uncertainty_map.png")
            '''datalogger.write_image(db[0,0,:,:],str(t),"depth_map_s" + str(m) + ".png","magma", True)
            datalogger.write_image(db[0,1,:,:],str(t),"stereo_depth_map_s" + str(m) + ".png","magma", True)
            datalogger.write_image(db[0,2,:,:],str(t),"uncertainty_map_s" + str(m) + ".png","viridis", True)'''
            datalogger.write_image(sample_depth_map,str(t),"depth_map_s" + str(m) + ".png","magma", True)
            datalogger.write_image(sample_stereo_depth_map,str(t),"stereo_depth_map_s" + str(m) + ".png","magma", True)
            datalogger.write_image(sample_uncer_map,str(t),"uncertainty_map_s" + str(m) + ".png","viridis", True)
            
            
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
    util.save_checkpoint(posenet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, pose_model_checkpoint_path)
    util.save_checkpoint(depthnet_model, joint_optimizer, checkpoint_epoch, checkpoint_loss, depth_model_checkpoint_path)
    
    '''
    Problems:
    - Loss is negative and small. L1 Loss?
    
    TODO:
    
    1. Add more to tensorboard output (depth, uncertainty map outputs) and make a google colab to view in tensorboard
    2. It appears that the authors do KITTI first for 20 epochs and EuRoC MAV 40. 
       So after you train on KITTI, adapt code to EuRoC MAV
    3. Both Zhou and D3VO use spatial transformers. See the Zhou implementation?
    4. Debug and Evaluate
    5. Use PyCeres and the python wrappers for Sophus/manif to do backend optimization
    '''