# Train and Test Loop Functions
#!pip install tensorboard
#from torch.utils.tensorboard import SummaryWriter
import time
from params import par
from datasets import training_data_pose, training_data_depth
from utils import util

'''

  TODO:
  1. Edit train loop to reflect D3VO (no ground truth, data shapes, number of samples, etc.)
  2. Get tensorboard setup to see smaller images, depth, uncertainty, etc. in google colab

  Edited: 9/24/22
'''

def train_loop(img_dir, img_dirs, posenet_model, depthnet_model, loss_fn, optimizer, batch_size, writer):

    # Initializations
    #n = training_data_pose.__len__()
    avg_train_loss = 0 # tracking train loss

    # Going through all of the training data
    #M = 2 # Vestigal
    fnum = 0 # Folder Number of image stack
    idx = 0 # Index of source image in that folder

    # Starting Training Time
    training_t0 = time.time()

    # Initializing indices to shuffle
    #seq = []
    # Creating list [0,..,t-1,0,...,q-1,.......,0,...,r-1]
    folder_sizes = []
    folder_seqs = []
    for r in range(len(img_dirs)):
        stack_seq = util.source_files(img_dir, img_dirs, r)
        
        # Duplicate Removal
        for filename in stack_seq:
          if ('(' in stack_seq):
            stack_seq.remove(filename)
        
        # Number of source files w/o duplicates
        f_size = len(stack_seq)

        # Removal of Out of Bounds Indices
        remove_list = []
        for filename in stack_seq:
          file_num_ = filename.rstrip('.png').lstrip('0')
          if file_num_ == '':
            file_num = 0
          else:
            file_num = int(file_num_)
          
          if int(file_num) > int(3*f_size-batch_size):
            remove_list.append(filename)
        
        for file_removal in remove_list:
          stack_seq.remove(file_removal)

        folder_sizes.append(int(len(stack_seq)/batch_size))
        stack_num = [int(item.rstrip('.png')) for item in stack_seq]
        folder_seqs.append(util.shuffle_sequence(stack_num))

    # Converting float to int for all elements
    source_img_dict = {str(i): folder_seqs[i] for i in range(len(img_dirs))}
    # For all of the data samples access image, idx and folder number, fnum
    c = 0 # c is now the index for the shuffled data
    batch = c
    #folder_sizes = [int(len(remove_duplicate_files(img_dir,folder)))/3/batch_size) for folder in img_dirs]

    # Total Iterations per Epoch
    epoch_size = 0
    for i in range(len(img_dirs)):
      samp = source_img_dict[str(i)]
      epoch_size += int(len(samp)/batch_size)

    # Training Mode
    depthnet_model.train() 
    posenet_model.train()
    for k,fsize in enumerate(folder_sizes):
        
        idx = 0
        fnum = k
        folder_samples = source_img_dict[str(fnum)]
        N = int(len(folder_samples)/batch_size)
        # For all the image stacks in the sample
        for idx in range(N):

            print("c: " + str(c)) # Stack index
            print("Batch # (Entire Dataset): " + str(c) + ", " + "Batch Number (Current Folder): " + str(idx) + ", " + "Folder #: " + str(fnum))
            print(folder_samples[idx*batch_size:(idx*batch_size)+batch_size])

            #Y,X = training_data.__getitem__(idx,fnum) #### GPU Mod
            # Use stack_num?
            p_images = training_data_pose.__getitem__(idx,fnum,folder_samples)
            d_images = training_data_depth.__getitem__(idx,fnum,folder_samples)

            p_images.float().to(par.device)
            d_images.float().to(par.device)

            sample_idx = folder_samples[batch_size*idx]
            # Prediction and Loss
            beta = par.beta
            # Training and Loss
            loss = loss_fn(posenet_model,
                           depthnet_model,
                           p_images,
                           d_images,
                           img_dirs,
                           fnum,
                           util.get_intrinsic_matrix(img_dirs[fnum][:2]),
                           beta,
                           sample_idx).float().to(par.device)
            
            # Training Loss To Tensorboard
            writer.add_scalar("Training Loss",loss,batch)
            
            # Backpropagation
            optimizer.zero_grad() # Reset gradients to 0
            loss.backward() # Backpropagate 
            optimizer.step() # Proceed to next optimization step
            
            loss, current = loss.item(), batch 
            avg_train_loss += loss # Summing train loss to average later
            print(f"loss: {loss:>7f} [{current+1:>5d}/{int(epoch_size):>5d}]") # formerly n/batch_size instead of epoch_size
            training_t1 = time.time()
            training_tn = util.time_stamp(training_t0, training_t1)
            print("Total Elapsed Time for Training: " + training_tn) 
            idx += 1
            c += 1
            batch += 1

    avg_train_loss = avg_train_loss / batch
    print(f"Avg. Train loss: {avg_train_loss:>8f} \n")
    return avg_train_loss  