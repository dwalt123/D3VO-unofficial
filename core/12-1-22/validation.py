import time
from params import par
#from datasets import val_data_pose, val_data_depth
import torch
from utils import util

def test_loop(img_dir, test_file, test_data_pose, test_data_depth,
              depthnet_model, posenet_model, loss_fn, dataset_type):
    
    val_loss = 0
    
    # Converting float to int for all elements
    # Read in data
    test_data_files = open(test_file,'r')
    test_list = test_data_files.readlines()
    # Shuffle data
    shuffled_list = util.shuffle_sequence(test_list)
    N = int(len(shuffled_list)/par.batch_size)
    
    et0 = time.time()
    with torch.no_grad():
        # Evaluation Mode
        depthnet_model.eval() 
        posenet_model.eval()
        for j in range(N):
            idx = j # Only works for one test folder
            
            # Getting Data
            p_images,_,__ = test_data_pose.__getitem__(idx,shuffled_list) 
            d_images,_,__ = test_data_depth.__getitem__(idx,shuffled_list)
            p_images.float().to(par.device)
            d_images.float().to(par.device)
            
            # Prediction and Loss
            beta=par.beta
            sample_idx = shuffled_list[par.batch_size*idx]
            #sample_idx = shuffled_list[batch_size*idx]
            #pose_subpath = ppaths[-1]
         
            # Prediction and Loss
            #beta = par.beta
        
            # Intrinsic camera matrix
            intrinsic_mat = util.get_intrinsic_matrix(sample_idx, dataset_type)
            
        
            #(self,posenet_model,depthnet_model,source_imgs,source_img,K,beta,sid)
        
            # Formerly in loss function
            pose_6dof_t_minus_1_t,a1,b1 = posenet_model(p_images[:,:6,:,:]) # 6DOF pose, affine params for T_t-1 to T_t
      
            reverse_tensor = torch.cat((p_images[:,6:,:,:],p_images[:,3:6,:,:]),1)
      
            pose_6dof_t_t_plus_1,a2,b2 = posenet_model(reverse_tensor) # 6DOF pose, affine params for T_t+1 to T_t
      
            util.print_regression_progress(img_dir,pose_6dof_t_minus_1_t, a1, b1, sample_idx, dataset_type)
      
            #depth_block_s1, depth_block_s2, depth_block_s3, depth_block_s4 = depthnet_model(d_images[:,:3,:,:])
            depth_block_out = depthnet_model(d_images[:,:3,:,:])
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
            # Affine Transformations for T_t-1->T_t and T_t+1->T_t
            a = [a1, a2]
            b = [b1, b2]
            curr_loss = loss_fn(p_images,
                                d_images,
                                scales,
                                stereo_baseline,
                                intrinsic_mat,
                                pose_6dof_t_minus_1_t,
                                pose_6dof_t_t_plus_1,
                                a,
                                b,
                                beta,
                                j).float().to(par.device)
            
            val_loss += curr_loss
            '''val_loss += loss_fn(img_dir,
                                posenet_model,
                                depthnet_model,
                                p_images,
                                d_images,
                                util.get_intrinsic_matrix(sample_idx,dataset_type),
                                beta,
                                sample_idx,
                                dataset_type).float().to(par.device).item() '''
            et1 = time.time()
            validation_tn = util.time_stamp(et0, et1)
            print(f"loss: {curr_loss:>7f} [{j+1:>5d}/{int(N):>5d}]")
            print("Total Elapsed Time for Validation: " + validation_tn)
    
    
    
    val_loss /= N # changed to n to make it loss per batch
    print(f"Avg. Validation loss: {val_loss:>8f} \n")
    test_data_files.close()
    return val_loss