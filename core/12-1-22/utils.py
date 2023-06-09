# Imports 
import os
import math
import random
import numpy as np
import torch
from params import par
import yaml

class Utilities():
    def __init__(self):
        self.kitti_root_dir = par.kitti_img_dir
        self.euroc_root_dir = par.euroc_mav_img_dir
        #self.gt_main_dir = r'..\Datasets\KITTI\Visual Odometry\data_odometry_poses\dataset\poses'
        self.kitti_gt_dir = r'oxts/data'
        self.euroc_gt_dir = par.euroc_mav_cam_cal
        
    def unrasterize_transformation(self,gt_rasterized):
      y = gt_rasterized.reshape((3,4))
      R = torch.from_numpy(y[:,:3]).float()
      t = torch.from_numpy(y[:,3]).float()
      
      return R,t
    
    
    # Testing chain of rotation matrices
    def chain_transformations(self,gt, img_idx, m):
    
      # Ground Truth Calculations
      dR = torch.eye(3).float()
      dt = torch.zeros((3,1)).float()
      for i in range(m):
        # have to reshape
        R,t = self.unrasterize_transformation(gt[img_idx+i,:])
        dR = torch.matmul(dR,R)
        dt += t.reshape((3,1))
    
      M = torch.cat((dR, dt), 1).reshape((1,12))
      poses = torch.cat((torch.from_numpy(gt[img_idx,:]).reshape((1,12)), M),0).unsqueeze(0)
      return poses
    
    
    # Sort a list of images from a directory
    def sort_images(self,img_direct):
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
          padstr = "{:>6}".format(str(sorted_imgs[q]))
          sorted_imgs[q] = padstr.replace(" ","0") + ".png"
        
        return sorted_imgs
    
    def sort_unix_images(self, img_direct, FILE_EXT, dataset):
        # Sort images in a given folder directory for EuRoC MAV
        # given list of unix time stamp image names (int)
        # Returns sorted list of ints for FILE_EXT is False
        # and string of image names if True
        # dataset: 'kitti' or 'euroc'
        new_ls = []
        for k in range(len(img_direct)):
          str_name = img_direct[k].rstrip('.png')
          new_ls.append(int(str_name))
          
        sorted_imgs = sorted(new_ls)
        #print(sorted_imgs[0])
        
        # Reapply File Extension
        if dataset == 'kitti':
            if FILE_EXT:
                for q in range(len(sorted_imgs)):
                  sorted_imgs[q] = str(sorted_imgs[q]).zfill(10) + ".png"
        elif dataset == 'euroc':
            if FILE_EXT:
                for q in range(len(sorted_imgs)):
                  sorted_imgs[q] = str(sorted_imgs[q]) + ".png"
        
        return sorted_imgs
    
    def source_files(self,img_dir, img_dirs, sequence_folder):
        # Gets the source image file names for the specified folder (every 2 images) 
        source_imgs = []
        folder_dir = os.path.join(img_dir,img_dirs[sequence_folder])
        ls = os.listdir(folder_dir)
    
        sorted_imgs = self.sort_images(ls)
    
        # Modified 9/2/22 to only provide source images away from edge (-3)
        # and on 9/4/22 to provide every 3 images for posenet model
        for j in range(len(sorted_imgs)-3):
            if j%3 == 0:
                source_imgs.append(sorted_imgs[j])
    
        return source_imgs
    
    
    def transformation_matrix_to_euler_translation(self,T, batch_size):
      #T1 =T[:,0,:].view(batch_size,3,4).float()
      #T2 = T[:,1,:].view(batch_size,3,4).float()
      #T2 = T.view(batch_size,3,4).float()
      T2 = T.view(3,4).float()
    
      # Rotation Matrices
      #R1 = T1[:,:,:3]
      #R2 = T2[:,:,:3]
      R2 = T2[:,:3]
    
      roll2 = torch.atan2(R2[1,0],R2[0,0]).float()
      pitch2 = torch.atan2(-R2[2,0],torch.sqrt(R2[2,1]**2 + R2[2,2]**2)).float()
      yaw2 = torch.atan2(R2[2,1],R2[2,2]).float()
    
      #print(roll2)
      #print(pitch2)
      #print(yaw2)
      #print(type(yaw2))
      #print(yaw2.shape)
      dr = torch.cat((roll2.reshape((1,)).unsqueeze(1), 
                     pitch2.reshape((1,)).unsqueeze(1), 
                     yaw2.reshape((1,)).unsqueeze(1)), 1).float()
      dt = T2[:,3]
      
      return [dt, dr]
    
    def euler_to_transformation_matrix(self,t,r):
          roll_ang = r[:,0]
          pitch_ang = r[:,1]
          yaw_ang = r[:,2]
        
          rotation_batch = torch.zeros(par.batch_size,3,3)
            
          ones_tensor = torch.ones(par.batch_size,1,1).to(par.device)
          zeros_tensor = torch.zeros(par.batch_size,1,1).to(par.device)
          sin_roll = torch.sin(roll_ang).view(par.batch_size,1,1)
          cos_roll = torch.cos(roll_ang).view(par.batch_size,1,1)
          sin_pitch = torch.sin(pitch_ang).view(par.batch_size,1,1)
          cos_pitch = torch.cos(pitch_ang).view(par.batch_size,1,1)
          sin_yaw = torch.sin(yaw_ang).view(par.batch_size,1,1)
          cos_yaw = torch.cos(yaw_ang).view(par.batch_size,1,1)
            
          # Roll Rotation Matrix Rows
          Rroll1 = torch.cat((ones_tensor, zeros_tensor, zeros_tensor),dim=2)
          Rroll2 = torch.cat((zeros_tensor, cos_roll, -1*sin_roll),dim=2)
          Rroll3 = torch.cat((zeros_tensor, sin_roll, cos_roll),dim=2)
          Rroll = torch.cat((Rroll1,Rroll2,Rroll3),dim=1)
            
          # Pitch Rotation Matrix Rows
          Rpitch1 = torch.cat((cos_pitch, zeros_tensor, sin_pitch),dim=2)
          Rpitch2 = torch.cat((zeros_tensor, ones_tensor, zeros_tensor),dim=2)
          Rpitch3 = torch.cat((-1*sin_pitch, zeros_tensor, cos_pitch),dim=2)
          Rpitch = torch.cat((Rpitch1,Rpitch2,Rpitch3),dim=1)
            
          # Yaw Rotation Matrix Rows
          Ryaw1 = torch.cat((cos_yaw, -1*sin_yaw, zeros_tensor), dim=2)
          Ryaw2 = torch.cat((sin_yaw, cos_yaw, zeros_tensor), dim=2)
          Ryaw3 = torch.cat((zeros_tensor, zeros_tensor, ones_tensor),dim=2)
          Ryaw = torch.cat((Ryaw1,Ryaw2,Ryaw3),dim=1)
            
          R = torch.matmul(torch.matmul(Ryaw,Rpitch),Rroll)
          rotation_batch = R.view(par.batch_size,3,3)
        
          #ttest = torch.unsqueeze(torch.tensor(t),2)
          ttest = torch.unsqueeze(t,2)
          
          return torch.cat((rotation_batch,ttest),2)
    
    def extract_transformation_from_file(self, cal_path):
        # Extracts a transformation matrix from the calib_cam_to_cam.txt file
        cal_file = open(cal_path,'r')
        cal_lines = cal_file.readlines()
        
        for line in cal_lines:
            if "R_03" in line:
                R_03 = line[6:].split()
            elif "T_03" in line:
                T_03 = line[6:].split()
            elif "R_02" in line:
                R_02 = line[6:].split()
            elif "T_02" in line:
                T_02 = line[6:].split()
                
        
        R03 = torch.tensor([float(i) for i in R_03]).view(3,3)
        T03 = torch.tensor([float(i) for i in T_03]).view(3,1)
        R02 = torch.tensor([float(i) for i in R_02]).view(3,3)
        T02 = torch.tensor([float(i) for i in T_02]).view(3,1)
        
        base = torch.tensor([0,0,0,1]).view(1,4)
        
        '''T_mat_30 = torch.cat((R03,T03),1)
        T_mat_02 = torch.cat((torch.t(R02),
                              torch.matmul(-1*torch.t(R02),T02)),1) # T_02 = (T_20)^-1
        
        T_final = torch.matmul(torch.cat((T_mat_30,base),0),
                               torch.cat((T_mat_02,base),0))'''
        
        T_mat_30 = torch.cat((torch.t(R03),
                              torch.matmul(-1*torch.t(R03),T03)),1) # T_30 = (T_03)^-1
        T_mat_02 = torch.cat((R02,T02),1)
        
        T_final = torch.matmul(torch.cat((T_mat_30,base),0),
                               torch.cat((T_mat_02,base),0))
        #print(T_final)
        cal_file.close()
        return T_final
    
    def stereo_baseline_euroc(self, cam0_path, cam1_path):
        cload1 = yaml.safe_load(open(cam0_path))
        #print(cload1)
        c1T = cload1['T_BS']['data']
        cload2 = yaml.safe_load(open(cam1_path))
        c2T = cload2['T_BS']['data']
        
        T1 = torch.tensor(c1T).reshape(4,4)
        T2 = torch.tensor(c2T).reshape(4,4)
        
        [t1,r1] = self.transformation_matrix_to_euler_translation(T1[:3,:], None)
        [t2,r2] = self.transformation_matrix_to_euler_translation(T2[:3,:], None)
            
        dt = t2-t1 # left-to-right
        dr = r2-r1
        
        dr = dr.view(3)
        #print(dt.shape)
        #print(dr.shape)
        sin_roll = torch.sin(dr[0])
        cos_roll = torch.cos(dr[0])
        sin_pitch = torch.sin(dr[1])
        cos_pitch = torch.cos(dr[1])
        sin_yaw = torch.sin(dr[2])
        cos_yaw = torch.cos(dr[2])
        
        Rroll = torch.tensor([[1, 0, 0],
                              [0, cos_roll, -1*sin_roll],
                              [0, sin_roll, cos_roll]])
        Rpitch = torch.tensor([[cos_pitch, 0, sin_pitch],
                               [0, 1, 0],
                               [-1*sin_pitch, 0, cos_pitch]])
        Ryaw = torch.tensor([[cos_yaw, -1*sin_yaw, 0],
                             [sin_yaw, cos_yaw, 0],
                             [0, 0, 1]])
        
        R = torch.matmul(torch.matmul(Ryaw,Rpitch),Rroll).view(3,3)
        baseline = torch.cat((R,dt.view(3,1)),1)
        return baseline
        
    def get_stereo_baseline_transformation(self, sample_id, dataset_type):
        # Reads filename and returns transformation from Cam 3 -> Cam 0 -> Cam 2
        #img_filename = open(filename,'r')
        #img_names = img_filename.readlines()
        #img_name = img_names[img_idx]
        
        # Assumes the calibrations are relatively the same for each stereo pair
        # (Not Tracking exact stereo calibration, assumes they're the same for all sequences.)
        if dataset_type == 'kitti':
            img_dir = par.kitti_img_dir
            cam_cal = par.kitti_cam_cal
            [img_folder,_,__] = sample_id.split()
            root_cal_dir = os.path.join(img_dir,img_folder)
            path_to_cal = os.path.join(root_cal_dir,cam_cal).replace('\\','/')
            st = self.extract_transformation_from_file(path_to_cal)
        elif dataset_type == 'euroc':
            img_dir = par.euroc_mav_img_dir
            [img_folder,_,__] = sample_id.split()
            root_cal_dir = os.path.join(img_dir,img_folder)
            cam_cal0 = "cam0/" + par.euroc_mav_cam_cal
            cam_cal1 = "cam1/" + par.euroc_mav_cam_cal
            path_to_cal0 = os.path.join(root_cal_dir,cam_cal0).replace('\\','/')
            path_to_cal1 = os.path.join(root_cal_dir,cam_cal1).replace('\\','/')
            st = self.stereo_baseline_euroc(path_to_cal0, path_to_cal1)
        
        # Get Tranformation Matrix:
        baseline = st
        return baseline
    
    def lla_to_cartesian_ecef(self, lla1_list):
        # lla1_list: list of LLA coordinates for ground truth frame 1
        [la1,lo1,al1] = [torch.tensor(float(i)) for i in lla1_list]
        #print([la1,lo1,al1])
        #x1 = (par.earths_radius*la1 + al1)*torch.cos(la1*math.pi/180)*torch.cos(lo1*math.pi/180)
        #y1 = (par.earths_radius*la1 + al1)*torch.cos(la1*math.pi/180)*torch.sin(lo1*math.pi/180)
        #z1 = ((1-par.eccentricity**2)*par.earths_radius*la1 + al1)*torch.sin(la1*math.pi/180)
        
        # According to latlonToMercator.m in KITTI raw dev kit
        scale = math.cos(la1*math.pi/180.0) # latToScale.m KITTI raw dev kit
        x1 = scale*lo1*math.pi*par.earths_radius/180
        y1 = scale*par.earths_radius*math.log(math.tan((90+la1)*math.pi/360))
        z1 = al1
        #print(x1)
        #print(y1)
        #print(z1)
        return x1,y1,z1
    
    def quaternion_to_euler(self, q_list):
        # assuming qw, qx, qy, qz
        #print(q_list)
        [qw,qx,qy,qz] = q_list
        
        roll = torch.atan2(2*(qw*qx + qy*qz),1 - 2*(qx**2 + qy**2))
        pitch = torch.asin(2*(qw*qy - qz*qx))
        yaw = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        
        return [roll,pitch,yaw]
        
    
    def raw_gt_to_odom_gt(self, gt1_list, gt2_list, dataset):
        # gt1_list: list of raw gps parameters for source frame (30 parameters)
        # gt2_list: list of raw gps parameters for target frame (30 parameters)
        # [latitude, longitude, altitutde, roll, pitch, yaw, ...]
        
        if dataset == 'kitti':
            [lat1, long1, alt1, roll1, pitch1, yaw1] = [torch.tensor(float(i)) for i in gt1_list[:6]]
            [lat2, long2, alt2, roll2, pitch2, yaw2] = [torch.tensor(float(i)) for i in gt2_list[:6]]
            
            x1,y1,z1 = self.lla_to_cartesian_ecef(gt1_list[:3])
            x2,y2,z2 = self.lla_to_cartesian_ecef(gt2_list[:3])
            
            dx = x2-x1
            dy = y2-y1
            dz = z2-z1
            droll = roll2-roll1
            dpitch = pitch2-pitch1
            dyaw = yaw2-yaw1
            #print(dx)
            #print(dy)
            #print(dz)
            #print(droll)
            #print(dpitch)
            #print(dyaw)
            
            # GPS to Camera Orientation
            '''Rot_mat = torch.tensor([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]]).to(torch.long)
            tvec = torch.tensor([dx,dy,dz]).to(torch.long)'''
            Rot_mat = torch.tensor([[0.0, 1.0, 0.0],
                                    [0.0, 0.0, 1.0],
                                    [1.0, 0.0, 0.0]])
            tvec = torch.tensor([dx,dy,dz])
            #print(tvec)
            cam_tvec = torch.matmul(Rot_mat.view(3,3),tvec.view(3,1))
            #print(cam_tvec)
            #cam_tvec = torch.tensor([dx,dy,dz])
            
        elif dataset == 'euroc':
            # Assumes gtx_list is a list of 16 elements for the full transformation matrix
            #print(gt1_list[3:8])
            r1 = self.quaternion_to_euler(gt1_list[3:8])
            r2 = self.quaternion_to_euler(gt2_list[3:8])
            
            dt = torch.tensor(gt2_list[:3])-torch.tensor(gt1_list[:3])
            dr = torch.tensor(r2)-torch.tensor(r1)
            
            cam_tvec = dt
            [droll,dpitch,dyaw] = dr.tolist()
            
            
        return cam_tvec.tolist(),[droll,dpitch,dyaw]
    
    def print_regression_progress(self, img_dir, pose_out, a, b, sample_id, dataset):
          # seq_dir: the date/sequence path 
          # dataset: 'kitti' or 'euroc'
          [seq_dir,img_num,_] = sample_id.split()
          #
          x = pose_out[:,0]
          y = pose_out[:,1]
          z = pose_out[:,2]
          roll = pose_out[:,3]
          pitch = pose_out[:,4]
          yaw = pose_out[:,5]
          
          # Printing Ground Truth to Reference while Training (Only)
          if dataset == 'kitti':
              seq_bound = len(os.listdir(os.path.join(img_dir, 
                                                      seq_dir, 
                                                      "image_02/data").replace("\\", '/')))
              if int(img_num) < (seq_bound-2):
                  gt_filename1 = os.path.join(os.path.join(img_dir, 
                                             os.path.join(seq_dir,self.kitti_gt_dir)),
                                             str(int(img_num)+1).zfill(10) + ".txt").replace("\\",'/') # + 1 since + 1 is considered middle index
                  gt_data1 = open(gt_filename1,'r')
                  gt1 = gt_data1.readline().split() # list of values in ground truth
                  
                  gt_filename2 = os.path.join(os.path.join(img_dir, 
                                             os.path.join(seq_dir,self.kitti_gt_dir)),
                                             str(int(img_num)+2).zfill(10) + ".txt").replace("\\",'/') # + 2 to get next global pose
                  gt_data2 = open(gt_filename2,'r')
                  gt2 = gt_data2.readline().split()
                  
                  [t,r] = self.raw_gt_to_odom_gt(gt1, gt2, 'kitti')
                  
                  gt_data1.close()
                  gt_data2.close()
                  
              else:
                  # Approximating ground truth by looking backwards instead of forwards
                  # (Avoiding out of bounds error at the end of sequences)
                  # Alternatively the ground truth could be calibrated with the initial transformation matrix
                  # in the sequence (inverse)
                  gt_filename1 = os.path.join(os.path.join(img_dir, 
                                             os.path.join(seq_dir,self.kitti_gt_dir)),
                                             str(int(img_num)).zfill(10) + ".txt").replace("\\",'/') # + 1 since + 1 is considered middle index
                  gt_data1 = open(gt_filename1,'r')
                  gt1 = gt_data1.readline().split() # list of values in ground truth
                  
                  gt_filename2 = os.path.join(os.path.join(img_dir, 
                                             os.path.join(seq_dir,self.kitti_gt_dir)),
                                             str(int(img_num)-1).zfill(10) + ".txt").replace("\\",'/') # + 2 to get next global pose
                  gt_data2 = open(gt_filename2,'r')
                  gt2 = gt_data2.readline().split()
                  
                  [t,r] = self.raw_gt_to_odom_gt(gt2, gt1, 'kitti')
                  
                  gt_data1.close()
                  gt_data2.close()
                  
          elif dataset == 'euroc':
              # 1. Sort images
              sorted_img_list = self.sort_images(os.listdir(os.path.join(img_dir, 
                                                                         seq_dir, 
                                                                         "cam0/data").replace("\\",'/')))
              seq_bound = len(os.listdir(os.path.join(img_dir, 
                                                      seq_dir, 
                                                      "cam0/data").replace("\\", '/')))
              
              data_dir = os.path.join(img_dir, seq_dir,
                                          "state_groundtruth_estimate0",
                                          "data.csv").replace("\\",'/')
              
              gt_data = torch.from_numpy(np.loadtxt(data_dir,delimiter=","))
              
              num = np.where((img_num+".png") == np.array(sorted_img_list))[0]
              #print(num)
              #print(seq_bound)
              if num[0] < seq_bound-2:
                  # 2. The ground truth is in x,y,z, quaternion
                  gt1 = gt_data[int(num)+1, 1:8] # [x,y,z,qw,qx,qy,qz]
                  gt2 = gt_data[int(num)+2, 1:8] # [x,y,z,qw,qx,qy,qz]
                  
                  # 3. Convert quaternion to roll, pitch, yaw
                  [t,r] = self.raw_gt_to_odom_gt(gt1, gt2, 'euroc')
                  
              else:
                  # Approximate ground truth at the edge
                  # 2. The ground truth is in x,y,z, quaternion
                  gt1 = gt_data[int(num), 1:8] # [x,y,z,qw,qx,qy,qz]
                  gt2 = gt_data[int(num)-1, 1:8] # [x,y,z,qw,qx,qy,qz]
                  
                  # 3. Convert quaternion to roll, pitch, yaw
                  [t,r] = self.raw_gt_to_odom_gt(gt2, gt1, 'euroc')
                  
          else:
              print("Dataset type not detected. Please use 'kitti' or 'euroc'. ")
          
          
          x1 = t[0]
          y1 = t[1]
          z1 = t[2]
          #print(r)
          roll1 = r[0]
          pitch1 = r[1]
          yaw1 = r[2]
          
          #print([x1,y1,z1,roll1,pitch1,yaw1])
          
          print(f"Affine Transformations: a = {a[0].item():.3f}, b = {b[0].item():.3f}")
          print(f"Est. Pose: x = {x[0]:.3f}, y = {y[0]:.3f}, z = {z[0]:.3f}, roll = {roll[0]:.3f}, pitch = {pitch[0]:.3f}, yaw = {yaw[0]:.3f}")
          if dataset == 'kitti':
              print(f"Ground Truth: x = {x1[0]:.3f}, y = {y1[0]:.3f}, z = {z1[0]:.3f}, roll = {pitch1.item():.3f}, pitch = {yaw1.item():.3f}, yaw = {roll1.item():.3f}")
          elif dataset == 'euroc':
              #print(x1)
              #print(y1)
              #print(z1)
              #print(roll1)
              #print(pitch1)
              #print(yaw1)
              print(f"Ground Truth: x = {x1:.3f}, y = {y1:.3f}, z = {z1:.3f}, roll = {roll1:.3f}, pitch = {pitch1:.3f}, yaw = {yaw1:.3f}")
    # Shuffle helper function
    def shuffle_sequence(self,list_numbers):
        random.shuffle(list_numbers)
        return list_numbers
    
    
    def get_intrinsic_matrix(self, sample_id, dataset_type):
      # Path to Calibration file:
      [img_folder,_,__] = sample_id.split()
      
      if dataset_type == 'kitti':
          root_cal_dir = os.path.join(par.kitti_img_dir,img_folder)
          path_to_cal = os.path.join(root_cal_dir,par.kitti_cam_cal).replace('\\','/')
          
          cal_file = open(path_to_cal,'r')
          cal_lines = cal_file.readlines()
            
          for line in cal_lines:
              if "K_02" in line:
                  K_02 = line[6:].split()
              elif "K_03" in line:
                  K_03 = line[6:].split()
                  
          K02 = torch.tensor([float(i) for i in K_02]).view(3,3)
          K03 = torch.tensor([float(i) for i in K_03]).view(3,3)
          zvec = torch.zeros(3,1)
          K2 = torch.cat((K02,zvec),1)
          K3 = torch.cat((K03,zvec),1)
          K = [K2,K3]
          cal_file.close()
          
      elif dataset_type == 'euroc':
          root_cal_dir = os.path.join(par.euroc_mav_img_dir,img_folder)
          path_to_cal1 = os.path.join(root_cal_dir, "cam0/" + par.euroc_mav_cam_cal).replace('\\','/')
          kfile1 = yaml.safe_load(open(path_to_cal1))
          path_to_cal2 = os.path.join(root_cal_dir, "cam1/" + par.euroc_mav_cam_cal).replace('\\','/')
          kfile2 = yaml.safe_load(open(path_to_cal2))
          #print(path_to_cal)
          #print(kfile)
          #print(kfile['intrinsics'])
          [fu1,fv1,cu1,cv1] = kfile1['intrinsics']
          K1 = torch.tensor([[fu1, 0, cu1],
                             [0, fv1, cv1],
                             [0, 0, 1]])
          [fu2,fv2,cu2,cv2] = kfile2['intrinsics']
          K2 = torch.tensor([[fu2, 0, cu2],
                             [0, fv2, cv2],
                             [0, 0, 1]])
          K = [K1,K2]
      
      return K
    
    
    def remove_duplicate_files(self, root_dir, img_dir, img_list):
      # Per directory file duplicate removal
      filtered_img_list = []
      path = os.path.join(root_dir,img_dir)
      for filename in os.listdir(path):
        if "(" not in filename:
          filtered_img_list.append(filename)
      return filtered_img_list
    
    def time_stamp(self, start_time, end_time):
        elapsed_time = end_time-start_time
        d = 86400 # seconds to days
        h = 3600 # seconds to hours
        m = 60 # seconds to minutes
        days = (elapsed_time-(elapsed_time%d))/d
        days_remainder = elapsed_time%d
        hours = (days_remainder-(days_remainder%h))/h
        hours_remainder = days_remainder%h
        minutes = (hours_remainder-(hours_remainder%m))/m
        minutes_remainder = hours_remainder%m
        seconds = minutes_remainder
        the_time = str(int(days)) + " day(s), " + str(int(hours)) + " hour(s), " + str(int(minutes)) + " minute(s), " + str(int(seconds)) + " second(s)."
        return the_time
    
    
    def save_checkpoint(self, model, optimizer, checkpoint_epoch, checkpoint_loss, model_checkpoint_path):
      
      # Save Checkpoint
      torch.save({'checkpoint_epoch': checkpoint_epoch,
                'model_state_dict': model.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'checkpoint_loss': checkpoint_loss,
                }, model_checkpoint_path)
      
      print('Model Checkpoint saved at '+ model_checkpoint_path)
    
    def textline_to_filename(self, root_data_dir, textline):
        
        [filename,file_num,cam] = textline.split()
        
        cam_dir = "image_02\data" if cam == "l" else "image_03\data"
            
        seq_dir = os.path.join(root_data_dir, filename)
        stereo_dir = os.path.join(seq_dir, cam_dir)
            
        img_name = file_num.zfill(10) + ".png"
        full_img_path = os.path.join(stereo_dir, img_name)
        corrected_img_path = full_img_path.replace('\\','/')
        
        return corrected_img_path

util = Utilities()