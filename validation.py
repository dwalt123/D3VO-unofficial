import time
from params import par
#from datasets import val_data_pose, val_data_depth
import torch
from utils import util
import wandb

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
    
    # Evaluation Mode
    depthnet_model.eval() 
    posenet_model.eval()
        
    et0 = time.time()
    with torch.no_grad():
        
        for j in range(N):
            idx = j # Only works for one test folder
            
            # Getting Data
            if par.use_stereo:
                random_pick = torch.randint(0,2,(1,)).item()
                if random_pick == 0:
                    cam_view = "left"
                elif random_pick == 1:
                    cam_view = "right"
            else:
                cam_view = "left"
            
            p_images = test_data_pose.__getitem__(idx,shuffled_list,cam_view) 
            d_images = test_data_depth.__getitem__(idx,shuffled_list,cam_view)
            #p_images.float().to(par.device)
            #d_images.float().to(par.device)
            
            # Prediction and Loss
            beta=par.beta
            sample_idx = shuffled_list[par.batch_size*idx]
        
            # Intrinsic camera matrix
            intrinsic_mat = util.get_intrinsic_matrix(sample_idx, dataset_type)
            
        
            #(self,posenet_model,depthnet_model,source_imgs,source_img,K,beta,sid)
        
            # Formerly in loss function
            #pose_6dof_t_minus_1_t,a1,b1 = posenet_model(p_images[:,:6,:,:]) # 6DOF pose, affine params for T_t-1 to T_t
            pose_input = torch.cat((p_images["t-1"], p_images["t"]),1)
            translation1, rotation1, a1, b1 = posenet_model(pose_input)
            pose_6dof_t_minus_1_t = torch.cat((translation1,rotation1),1)
        
            reverse_tensor = torch.cat((p_images["t+1"], p_images["t"]),1)
            translation2, rotation2, a2, b2 = posenet_model(reverse_tensor)
            pose_6dof_t_t_plus_1 = torch.cat((translation2,rotation2),1)
        
            util.print_regression_progress(img_dir,pose_6dof_t_minus_1_t, a1, b1, sample_idx, dataset_type)
            
            if par.use_stereo:
                if d_images["cam"] == "left":
                    depth_input = d_images["t"]
                elif d_images["cam"] == "right":
                    depth_input = d_images["ts"]
            else:
                depth_input = d_images["t"]
            
            depth_block_out = depthnet_model(depth_input)
            depth_block_s1 = depth_block_out[('disp', 0)]
            depth_block_s2 = depth_block_out[('disp', 1)]
            depth_block_s3 = depth_block_out[('disp', 2)]
            depth_block_s4 = depth_block_out[('disp', 3)]
            
            # Dictionary of Scales
            scales = {'0':depth_block_s4,
                      '1':depth_block_s3,
                      '2':depth_block_s2,
                      '3':depth_block_s1}
            
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
            
            #stereo_baseline = util.get_stereo_baseline_transformation(sample_idx, dataset_type)[:3,:]
            # Affine Transformations for T_t-1->T_t and T_t+1->T_t
            if par.use_ab:
                a = [a1, a2]
                b = [b1, b2]
            else:
                a = []
                b = []
            
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
                                j,
                                dataset_type).float().to(par.device)
            
            wandb.log({"current_validation_loss": curr_loss})
            val_loss += curr_loss
            
            et1 = time.time()
            validation_tn = util.time_stamp(et0, et1)
            print(f"loss: {curr_loss:>7f} [{j+1:>5d}/{int(N):>5d}]")
            print("Total Elapsed Time for Validation: " + validation_tn)
    
    
    
    val_loss /= N # changed to n to make it loss per batch
    print(f"Avg. Validation loss: {val_loss:>8f} \n")
    test_data_files.close()
    return val_loss