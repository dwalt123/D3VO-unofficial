import os
#from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
#from datasets import *
from params import par
from utils import util
import torch

def grab_img_pair(frame1_path,frame2_path):
    # read frames in grayscale
    img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)
    #print(img1 is None)
    #print(img2 is None)
    return img1,img2

def mean_optical_flow(img1,img2):
    flow = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return np.mean(mag)

def sort_images(img_direct):
        # Sort images in a given folder directory
        new_ls = []
        # Strip file extension
        for k in range(len(img_direct)):
          str_name = img_direct[k].rstrip('.png').lstrip('0')
          #print(str_name)
          if str_name == '':
            str_name = '0'
          if '(' not in str_name:
            new_ls.append(int(str_name))
    
        sorted_imgs = sorted(new_ls,key=int)
    
        # Add back Zero Padding and Reapply File Extension
        for q in range(len(sorted_imgs)):
          padstr = "{:>10}".format(str(sorted_imgs[q]))
          sorted_imgs[q] = padstr.replace(" ","0") + ".png"
        
        return sorted_imgs
    
def sort_images2(img_direct):
        # Sort images in a given folder directory for EuRoC MAV
        new_ls = []
        # Strip file extension
        for k in range(len(img_direct)):
          str_name = img_direct[k].rstrip('.png')
          new_ls.append(int(str_name))
          
        #print(new_ls[0])
        sorted_imgs = sorted(new_ls)
        #print(sorted_imgs[0])
        
        # Add back Zero Padding and Reapply File Extension
        for q in range(len(sorted_imgs)):
          sorted_imgs[q] = str(sorted_imgs[q]) + ".png"
        
        return sorted_imgs
    

if __name__ == "__main__":
    
    # Creating data split for EuRoC MAV
    
    #valid_train_files = open('split/KITTI/train_valid_files.txt', 'w')
    #valid_val_files = open('split/KITTI/val_valid_files.txt', 'w')
    
    '''
    kitti_train_and_val_dirs = ['2011_10_03/2011_10_03_drive_0027_sync',
                                '2011_09_30/2011_09_30_drive_0016_sync',
                                '2011_09_30/2011_09_30_drive_0018_sync',
                                '2011_09_30/2011_09_30_drive_0027_sync',
                                '2011_10_03/2011_10_03_drive_0047_sync',
                                '2011_09_26/2011_09_26_drive_0002_sync',
                                '2011_09_26/2011_09_26_drive_0009_sync',
                                '2011_09_26/2011_09_26_drive_0013_sync',
                                '2011_09_26/2011_09_26_drive_0020_sync',
                                '2011_09_26/2011_09_26_drive_0023_sync',
                                '2011_09_26/2011_09_26_drive_0027_sync',
                                '2011_09_26/2011_09_26_drive_0029_sync',
                                '2011_09_26/2011_09_26_drive_0036_sync',
                                '2011_09_26/2011_09_26_drive_0046_sync',
                                '2011_09_26/2011_09_26_drive_0048_sync',
                                '2011_09_26/2011_09_26_drive_0052_sync',
                                '2011_09_26/2011_09_26_drive_0056_sync',
                                '2011_09_26/2011_09_26_drive_0059_sync',
                                '2011_09_26/2011_09_26_drive_0060_sync',
                                '2011_09_26/2011_09_26_drive_0064_sync',
                                '2011_09_26/2011_09_26_drive_0084_sync',
                                '2011_09_26/2011_09_26_drive_0086_sync',
                                '2011_09_26/2011_09_26_drive_0093_sync',
                                '2011_09_26/2011_09_26_drive_0096_sync',
                                '2011_09_26/2011_09_26_drive_0101_sync',
                                '2011_09_26/2011_09_26_drive_0106_sync',
                                '2011_09_26/2011_09_26_drive_0117_sync',
                                '2011_09_28/2011_09_28_drive_0002_sync',
                                '2011_09_29/2011_09_29_drive_0071_sync',
                                '2011_10_03/2011_10_03_drive_0047_sync']'''
    
    kitti_train_dirs = ['2011_10_03/2011_10_03_drive_0027_sync',
                        '2011_09_30/2011_09_30_drive_0016_sync',
                        '2011_09_30/2011_09_30_drive_0018_sync',
                        '2011_09_30/2011_09_30_drive_0027_sync',
                        '2011_09_26/2011_09_26_drive_0009_sync',
                        '2011_09_26/2011_09_26_drive_0013_sync',
                        '2011_09_26/2011_09_26_drive_0023_sync',
                        '2011_09_26/2011_09_26_drive_0027_sync',
                        '2011_09_26/2011_09_26_drive_0029_sync',
                        '2011_09_26/2011_09_26_drive_0036_sync',
                        '2011_09_26/2011_09_26_drive_0046_sync',
                        '2011_09_26/2011_09_26_drive_0056_sync',
                        '2011_09_26/2011_09_26_drive_0059_sync',
                        '2011_09_26/2011_09_26_drive_0064_sync',
                        '2011_09_26/2011_09_26_drive_0084_sync',
                        '2011_09_26/2011_09_26_drive_0086_sync',
                        '2011_09_26/2011_09_26_drive_0093_sync',
                        '2011_09_26/2011_09_26_drive_0096_sync',
                        '2011_09_26/2011_09_26_drive_0101_sync',
                        '2011_09_26/2011_09_26_drive_0106_sync',
                        '2011_09_26/2011_09_26_drive_0117_sync',
                        '2011_10_03/2011_10_03_drive_0047_sync',
                        '2011_09_26/2011_09_26_drive_0001_sync',
                        '2011_09_26/2011_09_26_drive_0005_sync',
                        '2011_09_26/2011_09_26_drive_0014_sync',
                        '2011_09_26/2011_09_26_drive_0019_sync',
                        '2011_09_26/2011_09_26_drive_0022_sync',
                        '2011_09_26/2011_09_26_drive_0028_sync',
                        '2011_09_26/2011_09_26_drive_0035_sync',
                        '2011_09_26/2011_09_26_drive_0061_sync',
                        '2011_09_26/2011_09_26_drive_0070_sync',
                        '2011_09_26/2011_09_26_drive_0087_sync',
                        '2011_09_26/2011_09_26_drive_0091_sync',
                        '2011_09_26/2011_09_26_drive_0104_sync',
                        '2011_09_26/2011_09_26_drive_0113_sync',
                        '2011_09_28/2011_09_28_drive_0001_sync',
                        '2011_09_29/2011_09_29_drive_0004_sync']
    
    kitti_val_dirs = ['2011_09_26/2011_09_26_drive_0011_sync',
                      '2011_09_26/2011_09_26_drive_0015_sync',
                      '2011_09_26/2011_09_26_drive_0017_sync',
                      '2011_09_26/2011_09_26_drive_0018_sync',
                      '2011_09_26/2011_09_26_drive_0032_sync',
                      '2011_09_26/2011_09_26_drive_0039_sync',
                      '2011_09_26/2011_09_26_drive_0051_sync',
                      '2011_09_26/2011_09_26_drive_0057_sync',
                      '2011_09_26/2011_09_26_drive_0079_sync',
                      '2011_09_26/2011_09_26_drive_0095_sync',
                      '2011_09_29/2011_09_29_drive_0026_sync',
                      '2011_09_26/2011_09_26_drive_0002_sync',
                      '2011_09_26/2011_09_26_drive_0020_sync',
                      '2011_09_26/2011_09_26_drive_0048_sync',
                      '2011_09_26/2011_09_26_drive_0052_sync',
                      '2011_09_26/2011_09_26_drive_0060_sync',
                      '2011_09_28/2011_09_28_drive_0002_sync',
                      '2011_09_29/2011_09_29_drive_0071_sync']
    
    " Working on Training Files ..."
    train_split = par.kitti_train_split
    train_data_files = open(train_split,'r')
    train_list = train_data_files.readlines()
    train_data_files.close()
    
    nums=[]
    paths=[]
    cams=[]
    for sequence in train_list:
        [subpath,num,cam] = sequence.split()
        nums.append(int(num)) # frame number
        paths.append(subpath)  # path to that frame
        cams.append(cam)   # left or right stereo camera
    
    X = []
    Y = []
    Z = []
    Roll = []
    Pitch = []
    Yaw = []
    N = len(train_list)
    for j in range(N):
        print("Woring on Image Pair ... " + str(j+1) + "/" + str(N))
        #print(paths[j])
        temp_path = os.path.join(par.kitti_img_dir, paths[j], "oxts", "data").replace('\\','/') 
        
        seq_bound = len(os.listdir(temp_path))
        if nums[j] < (seq_bound-2):
            #print(j)
            #print(nums[j])
            unordered_imgs = os.listdir(temp_path)
            # not unix images but processed in a similar way
            sorted_nums = util.sort_unix_images(unordered_imgs, False, 'kitti')
            #print(len(sorted_nums))
            #print(seq_bound)
            frames = [sorted_nums[nums[j]], sorted_nums[nums[j]+1]]
            img_pair_names = [str(frames[0]).zfill(10)+".txt", 
                              str(frames[1]).zfill(10)+".txt"]
            
            i1 = os.path.join(temp_path, img_pair_names[0]).replace('\\','/')
            i2 = os.path.join(temp_path, img_pair_names[1]).replace('\\','/')
            
            #kitti_root_dir = par.kitti_img_dir
            #kitti_gt_dir = r'oxts/data'
            gt_filename1 = i1
            gt_data1 = open(gt_filename1,'r')
            gt1 = gt_data1.readline().split() # list of values in ground truth
            
            gt_filename2 = i2
            gt_data2 = open(gt_filename2,'r')
            gt2 = gt_data2.readline().split()
            
            [t,r] = util.raw_gt_to_odom_gt(gt1, gt2, 'kitti')
            
            X.append(t[0])
            Y.append(t[1])
            Z.append(t[2])
            Roll.append(r[0])
            Pitch.append(r[1])
            Yaw.append(r[2])
            
            gt_data1.close()
            gt_data2.close()
            
    plt.figure()
    plt.hist(X, alpha=0.5)
    plt.title("X Total")
    plt.grid()
    plt.figure()
    plt.hist(Y, alpha=0.5)
    plt.title("Y Total")
    plt.grid()
    plt.figure()
    plt.hist(Z, alpha=0.5)
    plt.title("Z Total")
    plt.grid()
    plt.figure()
    plt.hist(Roll, alpha=0.5)
    plt.title("Roll Total")
    plt.grid()
    plt.figure()
    plt.hist(Pitch, alpha=0.5)
    plt.title("Pitch Total")
    plt.grid()
    plt.figure()
    plt.hist(Yaw, alpha=0.5)
    plt.title("Yaw Total")
    plt.grid()
    '''
    train_gt = {}
    for img_sequence in kitti_train_dirs:
        
        train_full_seq_path = os.path.join(par.kitti_img_dir, img_sequence,
                                           par.kitti_left_cam + '/data/').replace('\\','/')
        print("Working on path : " + train_full_seq_path + " ...")
        img_list = os.listdir(train_full_seq_path)
        print("There are {0} images ...".format(len(img_list)))
        #print(img_list[0])
        
        X = []
        Y = []
        Z = []
        Roll = []
        Pitch = []
        Yaw = []
        # Sort Images
        ordered_imgs = sort_images(img_list)
        for i in range(len(ordered_imgs)-1):
            print("Working on ... " + ordered_imgs[i])
            #print(ordered_imgs[300])
            i1 = os.path.join(train_full_seq_path, str(ordered_imgs[i])).replace('\\','/')
            i2 = os.path.join(train_full_seq_path, str(ordered_imgs[i+1])).replace('\\','/')
            #print(i1)
            im1,im2 = grab_img_pair(i1,i2)
            mean_flow = mean_optical_flow(im1,im2)
            #print(mean_flow)
            if mean_flow >= 1.0:
                # Get Data for Training Iteration
                #if dataset == 'kitti':
                img_dir = par.kitti_img_dir
                seq_dir = img_sequence
                img_num = i
                
                kitti_root_dir = par.kitti_img_dir
                kitti_gt_dir = r'oxts/data'
                seq_bound = len(os.listdir(os.path.join(img_dir, 
                                                        seq_dir, 
                                                        "image_02/data").replace("\\", '/')))
                if int(img_num) < (seq_bound-2):
                    gt_filename1 = os.path.join(os.path.join(img_dir, 
                                               os.path.join(seq_dir,kitti_gt_dir)),
                                               str(int(img_num)+1).zfill(10) + ".txt").replace("\\",'/') # + 1 since + 1 is considered middle index
                    gt_data1 = open(gt_filename1,'r')
                    gt1 = gt_data1.readline().split() # list of values in ground truth
                    
                    gt_filename2 = os.path.join(os.path.join(img_dir, 
                                               os.path.join(seq_dir,kitti_gt_dir)),
                                               str(int(img_num)+2).zfill(10) + ".txt").replace("\\",'/') # + 2 to get next global pose
                    gt_data2 = open(gt_filename2,'r')
                    gt2 = gt_data2.readline().split()
                    
                    [t,r] = util.raw_gt_to_odom_gt(gt1, gt2, 'kitti')
                    
                    X.append(t[0])
                    Y.append(t[1])
                    Z.append(t[2])
                    Roll.append(r[0])
                    Pitch.append(r[1])
                    Yaw.append(r[2])
                    
                    gt_data1.close()
                    gt_data2.close()
            
        
        train_gt[img_sequence] = [X,Y,Z,Roll,Pitch,Yaw]
    
    Xtot = []
    Ytot = []
    Ztot = []
    Rolltot = []
    Pitchtot = []
    Yawtot = []
    for key,value in train_gt.items():
        [X,Y,Z,Roll,Pitch,Yaw] = value
        Xtot.extend(X)
        Ytot.extend(Y)
        Ztot.extend(Z)
        Rolltot.extend(Roll)
        Pitchtot.extend(Pitch)
        Yawtot.extend(Yaw)
    '''
    '''
    random_pick = torch.randint(0,2,(1,)).item()
    if random_pick == 0:
        cam_view = "l"
    elif random_pick == 1:
        cam_view = "r"
    
    str_name = ordered_imgs[i].rstrip('.png').lstrip('0')
    if str_name == '':
        str_name = '0'
    
    print_name = img_sequence + " " + str_name + " " + cam_view
    print(print_name)
    valid_train_files.write(print_name + "\n")
    '''
    '''
    plt.figure()
    plt.hist(Xtot, alpha=0.5)
    plt.title("X Total")
    plt.grid()
    plt.figure()
    plt.hist(Ytot, alpha=0.5)
    plt.title("Y Total")
    plt.grid()
    plt.figure()
    plt.hist(Ztot, alpha=0.5)
    plt.title("Z Total")
    plt.grid()
    plt.figure()
    plt.hist(Rolltot, alpha=0.5)
    plt.title("Roll Total")
    plt.grid()
    plt.figure()
    plt.hist(Pitchtot, alpha=0.5)
    plt.title("Pitch Total")
    plt.grid()
    plt.figure()
    plt.hist(Yawtot, alpha=0.5)
    plt.title("Yaw Total")
    plt.grid()
    '''
    #valid_train_files.close()
    
    " Working on Validation Files ..."
    '''
    for img_sequence in kitti_val_dirs:
        
        val_full_seq_path = os.path.join(par.kitti_img_dir, img_sequence,
                                         par.kitti_left_cam + '/data/').replace('\\','/')
        print("Working on path : " + val_full_seq_path + " ...")
        img_list = os.listdir(val_full_seq_path)
        print("There are {0} images ...".format(len(img_list)))
        #print(img_list[0])
        # Sort Images
        ordered_imgs = sort_images(img_list)
        for i in range(len(ordered_imgs)-1):
            #print(ordered_imgs[300])
            i1 = os.path.join(val_full_seq_path, str(ordered_imgs[i])).replace('\\','/')
            i2 = os.path.join(val_full_seq_path, str(ordered_imgs[i+1])).replace('\\','/')
            #print(i1)
            im1,im2 = grab_img_pair(i1,i2)
            mean_flow = mean_optical_flow(im1,im2)
            #print(mean_flow)
            if mean_flow >= 1.0:
                # Get Data for Training Iteration
                random_pick = torch.randint(0,2,(1,)).item()
                if random_pick == 0:
                    cam_view = "l"
                elif random_pick == 1:
                    cam_view = "r"
                
                str_name = ordered_imgs[i].rstrip('.png').lstrip('0')
                if str_name == '':
                    str_name = '0'
                
                print_name = img_sequence + " " + str_name + " " + cam_view
                print(print_name)
                valid_val_files.write(print_name + "\n")
            
        
    valid_val_files.close()
    
    print("Done with training and validation valid files!")
    
    # Removing Excess files ...
    valid_train_files = open('split/KITTI/train_valid_files.txt', 'r')
    valid_val_files = open('split/KITTI/val_valid_files.txt', 'r')
    train_files = open('split/KITTI/kitti_train_files.txt', 'w')
    val_files = open('split/KITTI/kitti_val_files.txt', 'w')
    
    list_valid_train_files = util.shuffle_sequence(valid_train_files.readlines())
    list_valid_val_files = util.shuffle_sequence(valid_val_files.readlines())
    
    #print(list_valid_files)
    print("Number of Valid Training Samples: " + str(len(list_valid_train_files)))
    print("Number of Valid Validation Samples: " + str(len(list_valid_val_files)))
    
    
    Nval = int(len(list_valid_train_files)/10)
    if len(list_valid_val_files) > Nval:
        # Adjusting Validation Set ...
        for i in range(Nval):
            val_files.write(list_valid_val_files[i])
            val_files.write(list_valid_val_files[i])
    else:
        for i in range(len(list_valid_val_files)):
            val_files.write(list_valid_val_files[i])
            val_files.write(list_valid_val_files[i])
            
    for j in range(len(list_valid_train_files)):
        train_files.write(list_valid_train_files[j])
        train_files.write(list_valid_train_files[j])
    
    valid_train_files.close()
    valid_val_files.close()
    train_files.close()
    val_files.close()
    
    train_files = open('split/KITTI/kitti_train_files.txt', 'r')
    val_files = open('split/KITTI/kitti_val_files.txt', 'r')
    
    len_train = len(train_files.readlines())
    len_val = len(val_files.readlines())
    
    print("Training set has size: " + str(len_train))
    print("Validation set has size: " + str(len_val))
    
    #valid_files.close()
    train_files.close()
    val_files.close()
    
    train_files_ = open('split/KITTI/kitti_train_files.txt', 'r')
    val_files_ = open('split/KITTI/kitti_val_files.txt', 'r')
    
    kitti_train_filenames = util.shuffle_sequence(train_files_.readlines())
    kitti_val_filenames = util.shuffle_sequence(val_files_.readlines())
    
    train_files_final = open('split/KITTI/train_files.txt', 'w')
    val_files_final = open('split/KITTI/val_files.txt', 'w')
    
    for trainfile in kitti_train_filenames:
        train_files_final.write(trainfile)
        #train_files_final.write(trainfile)
    
    for valfile in kitti_val_filenames:
        val_files_final.write(valfile)
        #val_files_final.write(valfile)
        
    print("Done ...")
    train_files_.close()
    val_files_.close()
    train_files_final.close()
    val_files_final.close()
    '''
   