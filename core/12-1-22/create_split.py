import os
#from PIL import Image
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from datasets import *
from params import par
from utils import util

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
    
    valid_files = open('split/EuRoC_MAV/train_val_files.txt', 'w')
    
    euroc_train_and_val_dirs = ['machine_hall/MH_01_easy/mav0/',
                                'machine_hall/MH_02_easy/mav0/',
                                'machine_hall/MH_04_difficult/mav0/',
                                'vicon_room1/V1_01_easy/mav0/',
                                'vicon_room1/V1_02_medium/mav0/']
    
    for img_sequence in euroc_train_and_val_dirs:
        
        full_seq_path = os.path.join(os.path.join(par.euroc_mav_img_dir, img_sequence),
                                     par.euroc_left_cam + '/data').replace('\\','/')
        print("Working on path : " + full_seq_path + " ...")
        img_list = os.listdir(full_seq_path)
        #print(img_list[0])
        # Sort Images
        ordered_imgs = sort_images2(img_list)
        for i in range(len(ordered_imgs)-1):
            #print(ordered_imgs[300])
            i1 = os.path.join(full_seq_path, ordered_imgs[i]).replace('\\','/')
            i2 = os.path.join(full_seq_path, ordered_imgs[i+1]).replace('\\','/')
            #print(i1)
            im1,im2 = grab_img_pair(i1,i2)
            mean_flow = mean_optical_flow(im1,im2)
            #print(mean_flow)
            if mean_flow >= 1.0:
                print_name = img_sequence + " " + ordered_imgs[i].rstrip(".png") + " " + "cam0"
                print(print_name)
                valid_files.write(print_name + "\n")
            
        
    valid_files.close()
    
    print("Done with valid files!")
    
    valid_files = open('split/EuRoC_MAV/train_val_files.txt', 'r')
    train_files = open('split/EuRoC_MAV/train_files.txt', 'w')
    val_files = open('split/EuRoC_MAV/val_files.txt', 'w')
    
    list_valid_files = util.shuffle_sequence(valid_files.readlines())
    #print(list_valid_files)
    print("Number of Training and Validation Samples: " + str(len(list_valid_files)))
    
    counter = 1
    for ifile in list_valid_files:
        print(ifile)
        if counter < 10:
            train_files.write(ifile)
            counter+=1
        else:
            val_files.write(ifile)
            counter = 1
        
    valid_files.close()
    train_files.close()
    val_files.close()
    
    train_files = open('split/EuRoC_MAV/train_files.txt', 'r')
    val_files = open('split/EuRoC_MAV/val_files.txt', 'r')
    
    len_train = len(train_files.readlines())
    len_val = len(val_files.readlines())
    
    print("Training set has size: " + str(len_train))
    print("Validation set has size: " + str(len_val))
    
    #valid_files.close()
    train_files.close()
    val_files.close()
    # It looks like only about 11704 satisfy this constraint with generic optical flow