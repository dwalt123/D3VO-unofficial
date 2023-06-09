from params import par
from datasets import val_data_pose, val_data_depth
import torch
from utils import util

def test_loop(img_dir, img_dirs, depthnet_model, posenet_model, loss_fn, n):
    #size = val_data_pose.__len__()
    #num_batches = val_data_pose.__len__()/par.batch_size
    val_loss = 0
    #val_dirs = img_dirs
    #ls = os.listdir(os.path.join(img_dir,val_dirs[0])) # Assumes 1 validation folder

    # Initializing indices to shuffle
    seq = []
    # Creating list [0,..,t-1,0,...,q-1,.......,0,...,r-1]
    for r in range(len(img_dirs)):
        stack_seq = util.source_files(img_dir, img_dirs, r)
        seq.extend(util.shuffle_sequence(stack_seq)) # Adds random shuffled sequence to the end

    # Converting float to int for all elements
    stack_num = [int(item.rstrip('.png')) for item in seq]

    with torch.no_grad():
        # Evaluation Mode
        depthnet_model.eval() 
        posenet_model.eval()
        for j in range(n):
            idx = j # Only works for one test folder
            fnum = 0 # Only works for one test folder
            p_images = val_data_pose.__getitem__(idx,fnum,stack_num) 
            d_images = val_data_depth.__getitem__(idx,fnum,stack_num)
            p_images.float().to(par.device)
            d_images.float().to(par.device)
            # Prediction and Loss
            beta=par.beta
            folder=0
            sample_idx = stack_num[par.batch_size*idx]
            val_loss += loss_fn(posenet_model,
                           depthnet_model,
                           p_images,
                           d_images,
                           img_dirs,
                           folder,
                           util.get_intrinsic_matrix(img_dirs[folder][:2]),
                           beta,
                           sample_idx).float().to(par.device).item() 
            
    val_loss /= n # changed to n to make it loss per batch
    print(f"Avg. Validation loss: {val_loss:>8f} \n")
    return val_loss