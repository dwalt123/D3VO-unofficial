# Imports 
#import time
import os
#import sys
import numpy as np
import torch
from torch import nn
import torchvision
from torchmetrics.functional import structural_similarity_index_measure as ssim
import kornia.geometry as kg
import kornia.losses as kl

#import numpy as np

# Local Imports
from utils import util
from params import par
from datatracker import datalogger

import torch.nn.functional as F
from torch.nn.functional import l1_loss, grid_sample
from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import Resize

'''
Loss Functions:

'''

'''
    Monodepth2 Loss Classes (Explicit differentiable classes for pytorch as outlined in D3VO)
'''
class Backproject(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(Backproject, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    #@staticmethod
    def forward(self, depth, inv_K):
        #depth.requires_grad_(True)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        
    #@staticmethod
    def forward(self, points, K, T):
        '''T.requires_grad_(True)
        points.requires_grad_(True)'''
        
        P = torch.matmul(K, T[:, :3 ,:])[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

# Then use F.grid_sample to interpolate
# Also turn some loss functions into classes and apply forward where necessary
 
class ImageWarp(nn.Module):
    # Wraps kornia image warping function into a differentiable class type
    def __init__(self, K_batch, norm):
        super(ImageWarp, self).__init__()
        
        self.K_batch = K_batch.to(par.device)
        self.K_inv = torch.linalg.inv(K_batch).to(par.device)
        self.norm = norm
        
        #self.backproject = Backproject(par.batch_size,h,w).to(par.device)
        #self.cam_points = backproject(depth_map, self.K_inv).to(par.device)
     
    #@staticmethod
    def forward(self, src_img, depth_map, pose, cam_type):
        '''src_img.requires_grad_(True)
        depth_map.requires_grad_(True)
        T.requires_grad_(True)'''
        
        # BackProject
        b,c,h,w = src_img.shape
        backproject = Backproject(par.batch_size,h,w).to(par.device)
        cam_points = backproject(depth_map, self.K_inv).to(par.device)
        #print("cam_points grad function: ")
        #print(cam_points.grad_fn)
        # Project 3D
        project3d = Project3D(par.batch_size,h,w).to(par.device)
        #print("project3d grad function: ")
        #print(project3d.grad_fn)
        # Euler Angle Batch to Transformation Matrix Batch
        if cam_type == 'mono':
            # Assumes stereo baseline is constant, and T matrix already provided.
            base = torch.unsqueeze(torch.tensor([0,0,0,1]).view(1,4),0)
            T_base = base.repeat(par.batch_size,1,1).to(par.device)
            pose = util.euler_to_transformation_matrix(pose[:,:3],
                                                       pose[:,3:]).to(par.device)
            pose = torch.cat((pose,T_base),1).to(par.device)
            #print(pose)
        elif cam_type == 'stereo':
            base = torch.unsqueeze(torch.tensor([0,0,0,1]).view(1,4),0)
            T_base = base.repeat(par.batch_size,1,1).to(par.device)
            pose = torch.unsqueeze(pose,0).to(par.device)
            pose = pose.repeat(par.batch_size,1,1).to(par.device)
            pose = torch.cat((pose,T_base),1).to(par.device)
            #print(pose)
        
        pixs = project3d(cam_points, self.K_batch, pose).to(par.device)
        #print("pixs grad function: ")
        #print(pixs.grad_fn)
        # F.grid_sample
        # Take out kornia, and add monodepth2 classes 
        '''return kg.depth.warp_frame_depth(src_img, depth_map, T, 
                                         self.K_batch, normalize_points = self.norm)'''
        return grid_sample(src_img, pixs, mode='bilinear', padding_mode='border')

'''class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
    def forward():
        return'''
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class ResidualLoss(nn.Module):
    def __init__(self, win_size):
        super(ResidualLoss, self).__init__()
        
        self.win_size = win_size
        
        #self.alpha = alpha
     
    #@staticmethod
    def forward(self, target_img, warped_img, alpha):
        '''target_img.requires_grad_(True)
        warped_img.requires_grad_(True)'''
        
        # SSIM Class?
        #weighted_ssim = kl.SSIMLoss(window_size=11, reduction='none')
        
        #weighted_ssim.requires_grad_(True)
        
        weighted_l1 = (1 - alpha)*torch.nn.L1Loss(reduction='none')(target_img, warped_img)
        #print("Weighted L1 Loss: ")
        #print(torch.mean(weighted_l1))
        
        #print("weighted_l1 grad function: ")
        #print(weighted_l1.grad_fn)
        #weighted_l1.requires_grad_(True)
        weighted_ssim = SSIM()
        wssim = weighted_ssim(target_img, warped_img)
        #print("SSIM Loss: ")
        #print(torch.mean(wssim))
        
        #print("wssim grad function: ")
        #print(wssim.grad_fn)
        return alpha*wssim + weighted_l1

class LossSelf(nn.Module):
    def __init__(self):
        super(LossSelf, self).__init__()
    
    #@staticmethod
    def forward(self,source_imgs,target_img,depth_map,K,
                pose_6dof_t_minus_1_t, pose_6dof_t_t_plus_1,
                stereo_baseline,a,b,uncer_map, batch_num, scale):
        
        # Calculate Residuals
        b_,c,h,w = target_img.shape
        a1 = (a[0].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
        b1 = (b[0].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
        a2 = (a[1].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
        b2 = (b[1].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
        
        
        # Masking Out Over-exposed/Saturated pixels
        #threshold = 0.99
        threshold = torch.tensor([0.99,0.99,0.99]).view(1,3,1,1).to(par.device) # channel-wise threshold
        
        
        #a1_mask = torch.where(target_img<threshold, a1*target_img, target_img)
        #masked_img1 = torch.where(target_img<threshold, a1*target_img+b1, target_img) # if not over exposed, add b, else 0
        masked_img1 = torch.where(torch.all(target_img<threshold,1,keepdim=True), a1*target_img+b1, target_img)
        
        #a2_mask = torch.where(target_img<threshold, a2*target_img, target_img)
        #masked_img2 = torch.where(target_img<threshold, a2*target_img+b2, target_img)
        masked_img2 = torch.where(torch.all(target_img<threshold,1,keepdim=True), a2*target_img+b2, target_img)
        
        alpha = par.alpha
        
        K_1 = torch.unsqueeze(K[0][:,:3],0)
        K_2 = torch.unsqueeze(K[1][:,:3],0)
        #K_batch = torch.cat((K_,K_,K_,K_,K_,K_,K_,K_)).to(par.device)
        K_batch1 = K_1.repeat(par.batch_size,1,1).to(par.device)
        K_batch2 = K_2.repeat(par.batch_size,1,1).to(par.device)
        #print("K_batch grad function: ")
        #print(K_batch.grad_fn)
        #depth_map = 1 / (1/100 + (1/0.01 - 1/100) * depth_map)
        
        baseline = torch.abs(stereo_baseline[0,-1])
        #print(baseline)
        f1 = torch.unsqueeze(K[0][0,0],0).to(par.device)
        f2 = torch.unsqueeze(K[1][0,0],0).to(par.device)
        f = torch.cat((f1,f2),0).view(1,2,1,1).to(par.device)
        #print(baseline*f)
        #print(f.shape)
        #print(f[:,0,:,:])
        #print(f[:,1,:,:])
        #print(f)
        #depth_map_ = 1 / (1/100 + (1/1e-1 - 1/100) * depth_map)
        #depth_map_ = torch.clamp(depth_map,1e-3,80)
        depth_map_ = baseline*f*depth_map
        #print(torch.max(depth_map_))
        #print(torch.min(depth_map_))
        #depth_map_ = torch.clamp(baseline*f*depth_map, 1e-3, 80)
        # depth_map = 1 / (min_disp + (max_disp - min_disp) * value[:,:2,:,:])
        #depth_map_ = 1 / (1/100 + (1/1e-1 - 1/100) * depth_map)
        warp_mono = ImageWarp(K_batch1,False)
        warp_stereo = ImageWarp(K_batch2,False)
        residual_loss = ResidualLoss(11)
        
        # I_t-1
        warped_img1 = warp_mono(source_imgs[0], 
                                torch.unsqueeze(depth_map_[:,0,:,:],1),
                                pose_6dof_t_minus_1_t,
                                'mono')
        #print("warped_img1 grad function: ")
        #print(warped_img1.grad_fn)
        #warped_img1.requires_grad_(True)
        
        # a1*target_img+b1
        #masked_img1 = target_img
        res_t_minus_1 = residual_loss(masked_img1,warped_img1,alpha)
        res_t_minus_1 = res_t_minus_1.mean(1,True)
        #res_t_minus_1 = residual_loss(target_img,warped_img1,alpha)
        #print("res_t_minus_1 grad function: ")
        #print(res_t_minus_1.grad_fn)
        #res_t_minus_1.requires_grad_(True)
        
        # I_t+1
        warped_img2 = warp_mono(source_imgs[1], 
                                torch.unsqueeze(depth_map_[:,0,:,:],1), 
                                pose_6dof_t_t_plus_1,
                                'mono')
        #print("warped_img2 grad function: ")
        #print(warped_img2.grad_fn)
        #warped_img2.requires_grad_(True)
        
        # a2*target_img+b2
        #masked_img2 = target_img
        res_t_plus_1 = residual_loss(masked_img2,warped_img2,alpha)
        res_t_plus_1 = res_t_plus_1.mean(1,True)
        #res_t_plus_1 = residual_loss(target_img,warped_img2,alpha)
        #print("res_t_plus_1 grad function: ")
        #print(res_t_plus_1.grad_fn)
        #res_t_plus_1.requires_grad_(True)
        
        # I_ts
        warped_img3 = warp_stereo(target_img, 
                                  torch.unsqueeze(depth_map_[:,1,:,:],1),
                                  stereo_baseline,
                                  'stereo')
        #print("warped_img3 grad function: ")
        #print(warped_img3.grad_fn)
        #warped_img3.requires_grad_(True)
        
        res_t_stereo = residual_loss(source_imgs[2],warped_img3,alpha)
        res_t_stereo = res_t_stereo.mean(1,True)
        #print("res_t_stereo grad function: ")
        #print(res_t_stereo.grad_fn)
        #res_t_stereo.requires_grad_(True)
        #res_min_ = torch.minimum(res_t_minus_1, res_t_plus_1)
        #res_min = torch.minimum(torch.minimum(res_t_minus_1, res_t_plus_1), res_t_stereo) # should have the minimum residual of each element
        res_min, idxes = torch.min(torch.cat((res_t_minus_1,res_t_plus_1,res_t_stereo),1),1)
        #print("res_min grad function: ")
        #print(res_min.grad_fn)
        #res_min.requires_grad_(True)
        '''print("Res Min Max: ")
        print(torch.max(res_min))
        print("Res Min Min: ")
        print(torch.min(res_min))
        print("Log(uncertainty) Max: ")
        print(torch.max(torch.log(torch.unsqueeze(uncer_map,1))))
        print("Log(uncertainty) Min: ")
        print(torch.min(torch.log(torch.unsqueeze(uncer_map,1))))
        print("Res Min Mean: ")
        print(torch.mean(res_min))
        print("Normalized Res Min: ")
        print(torch.mean(res_min/torch.unsqueeze(uncer_map,1)))
        print("Mean Log(uncertainty):")
        print(torch.mean(torch.log(torch.unsqueeze(uncer_map,1))))'''
        #if scale == 1 and batch_num == 9951:
        batch_chkpt = [0,2500,5000,7500,9000]
        #d_ = baseline*f # back to inverse depth for visualization
        #print(source_imgs[2][0,:,:,:].shape)
        if scale == 1 and batch_num in batch_chkpt:
            folder = os.path.join("batch",str(batch_num))
            
            # Right Stereo Image
            datalogger.write_image(source_imgs[2][0,:,:,:],folder,"right_stereo_img_s" + str(scale) + ".png", None, False)
            
            # Masked Images
            datalogger.write_image(masked_img1[0,:,:,:],folder,"masked_img1_s" + str(scale) + ".png", None, False)
            datalogger.write_image(masked_img2[0,:,:,:],folder,"masked_img2_s" + str(scale) + ".png", None, False)
            
            # Residuals
            #print(res_min[0,:,:,:].shape)
            datalogger.write_image(res_min[0,:,:],folder,"res_min_s" + str(scale) + ".png", None, False)
            #datalogger.write_image(res_min_[0,:,:,:],folder,"res_min_mono_s" + str(scale) + ".png", None, False)
            datalogger.write_image(res_t_minus_1[0,:,:,:],folder,"res_t_minus_1_s" + str(scale) + ".png", None, False)
            datalogger.write_image(res_t_plus_1[0,:,:,:],folder,"res_t_plus_1_s" + str(scale) + ".png", None, False)
            datalogger.write_image(res_t_stereo[0,:,:,:],folder,"res_t_stereo_s" + str(scale) + ".png", None, False)
            
            # Target Image, Forward Warped Image, Stereo Warped Image
            datalogger.write_image(target_img[0,:,:,:],folder,"rgb.png",None,False)
            datalogger.write_image(warped_img1[0,:,:,:],folder,"warped_img1.png",None,False)
            datalogger.write_image(warped_img3[0,:,:,:],folder,"warped_img3.png",None,False)
            
            # Current Monocular, Stereo Depth Map and Uncertainty Map
            datalogger.write_image(depth_map_[0,0,:,:],folder,"depth_map_s" + str(scale) + ".png","magma", True)
            datalogger.write_image(depth_map_[0,1,:,:],folder,"stereo_depth_map_s" + str(scale) + ".png","magma", True)
            datalogger.write_image(uncer_map[0,:,:],folder,"uncertainty_map_s" + str(scale) + ".png","viridis", True)
            
        return torch.mean(res_min/torch.unsqueeze(uncer_map,1) + torch.log(torch.unsqueeze(uncer_map,1)))
        #return torch.sum(res_min/torch.unsqueeze(uncertainty_map,1) + torch.log(torch.unsqueeze(uncertainty_map,1)))

class LossAB(nn.Module):
    def __init__(self):
        super(LossAB, self).__init__()
    
    #@staticmethod
    def forward(self,a,b):
      '''a[0].requires_grad_(True)
      a[1].requires_grad_(True)
      b[0].requires_grad_(True)
      b[1].requires_grad_(True)'''
      # a,b : [batch_size,1]
      
      lab = torch.zeros(par.batch_size,1).to(par.device)
      #lab.requires_grad_(True)
      ones = torch.ones(par.batch_size,1).to(par.device)
      
      for j in range(2):
          lab += (a[j] - ones)**2 + b[j]**2
          #aj = a[j]
          #aj.requires_grad_(True)
          #bj = b[j]
          #bj.requires_grad_(True)
          #lab += (aj - ones)**2 + bj**2
      
      return torch.sum(lab)

class LossSmooth(nn.Module):
    def __init__(self):
        super(LossSmooth, self).__init__()
        
    #@staticmethod
    def forward(self,source_img, depth_map):
      # depth_map = 1 / (min_disp + (max_disp - min_disp) * value[:,:2,:,:])
      #depth_map = (1/depth_map - 1/100)/(1/1e-1 - 1/100) # original value
      #depth_map = 1.0/depth_map # pseudo disparity smoothness
      #depth_mean = torch.mean(depth_map)
      #depth_map = depth_map/depth_mean # mean-normalized inverse depth as proposed by Monodepth2
      
      #source_img.requires_grad_(True)
      #depth_map.requires_grad_(True)
      
      # Resizing
      #temp_depth_map = torch.squeeze(depth_map[:,0,:,:],1)
      
      #b,h,w = temp_depth_map.shape
      
      #source_img_ = Resize(size=(h,w))(source_img)
      
      #transformed_src_img = torch.squeeze(torchvision.transforms.Grayscale(num_output_channels=1)(source_img),1)
      
      # Depth Gradient
      #depth_grad = torch.gradient(temp_depth_map.type(torch.FloatTensor))
      #depth_grad.requires_grad_(True)
      #depth_grad_x = torch.abs(depth_grad[1])
      
      #depth_grad_x = torch.mean(torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:]),1,keepdim=True)
      depth_grad_x = torch.mean(torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:]),1,keepdim=True)
      
      #print(depth_grad_x.shape)
      #depth_grad_x.requires_grad_(True)
      
      depth_grad_y = torch.mean(torch.abs(depth_map[:, :, :-1, :] - depth_map[: , :, 1:, :]),1,keepdim=True)
      
      #print(depth_grad_y.shape)
      #depth_grad_y = torch.abs(depth_grad[0])
      #depth_grad_y.requires_grad_(True)
      
      # Source Image Gradient
      #image_grad = torch.gradient(transformed_src_img.type(torch.FloatTensor))
      #image_grad.requires_grad_(True)
      
      image_grad_x = torch.mean(torch.abs(source_img[:, :, :, :-1] - source_img[:, :, :, 1:]),1, keepdim=True)
      
      #print(image_grad_x.shape)
      #image_grad_x = torch.abs(image_grad[1])
      #image_grad_x.requires_grad_(True)
      
      image_grad_y = torch.mean(torch.abs(source_img[:, :, :-1, :] - source_img[:, :, 1:, :]),1, keepdim=True)
      
      #print(image_grad_y.shape)
      #image_grad_y = torch.abs(image_grad[0])
      #image_grad_y.requires_grad_(True)
      
      depth_grad_x *= torch.exp(-image_grad_x)
      
      #print(depth_grad_x.shape)
      #x_grad_term = depth_grad_x*torch.exp(-image_grad_x)
      #x_grad_term.requires_grad_(True)
      
      depth_grad_y *= torch.exp(-image_grad_y)
      
      #print(depth_grad_y.shape)
      #y_grad_term = depth_grad_y*torch.exp(-image_grad_y)
      #y_grad_term.requires_grad_(True)
      #return torch.mean(x_grad_term + y_grad_term)
      #return torch.sum(x_grad_term + y_grad_term)
      
      return depth_grad_x.sum() + depth_grad_y.sum()
      #return depth_grad_x.mean() + depth_grad_y.mean()
      #return torch.sum(x_grad_term + y_grad_term)

class SingleScaleLoss(nn.Module):
    def __init__(self):
        super(SingleScaleLoss, self).__init__()
        self.Loss_smooth = LossSmooth()
        self.Loss_ab = LossAB()
        self.Loss_self = LossSelf()
    #@staticmethod
    def forward(self,source_imgs,target_img,depth_map,K,pose_6dof_t_minus_1_t,
                           pose_6dof_t_t_plus_1,stereo_baseline,a,b,
                           uncer_map,beta,lamb,s,batch_num,scale_):
      
      l_smooth = self.Loss_smooth(target_img,torch.unsqueeze(depth_map[:,0,:,:],1))
      #l_smooth = 0
      
      #print(l_smooth.grad)
      
      l_ab = self.Loss_ab(a,b)
      
      #print(l_ab.grad)
      
      loss_reg = l_smooth + beta*l_ab # beta*l_ab
      
      #print(loss_reg.grad)
      #print(f"loss_reg = {loss_reg:.3f}, loss_smooth = {l_smooth:.3f}, beta*l_ab = {beta*l_ab:.3f}")
      #loss_reg.requires_grad_(True)
      
      #loss_tot = self.Loss_self(source_imgs,target_img,depth_map,K,T,a,b,uncertainty_map)**s
      #loss_tot += lamb*(loss_reg**s)
      
      #loss_tot = self.Loss_self(source_imgs,target_img,depth_map,K,T,a,b,uncertainty_map)
      
      #print(loss_tot.grad)
      #print(f"loss_residual = {loss_tot:.3f}")
      
      '''loss_tot = self.Loss_self(source_imgs,
                                target_img,
                                depth_map,
                                K,T,a,b,
                                uncertainty_map) + lamb*loss_reg'''
      
      loss_res = self.Loss_self(source_imgs,
                                target_img,
                                depth_map,
                                K,
                                pose_6dof_t_minus_1_t,
                                pose_6dof_t_t_plus_1,
                                stereo_baseline,
                                a,b,
                                uncer_map,
                                batch_num,
                                scale_)
      
      #print(f"loss_residual = {loss_res**s:.3f}")
      
      loss_tot = loss_res + lamb*(loss_reg)
      #print(loss_tot.grad)
      #print(f"loss_total = {loss_tot:.3f}, lamb*loss_reg = {lamb*(loss_reg):.3f}, loss_res = {loss_res:.5f}, lambda = {lamb:.5f}")
      #print("---------------------------------------------------------------")
      #loss_tot.requires_grad_(True)
      
      # + lamb*(self.Loss_smooth(target_img,depth_map) + beta*self.Loss_ab(a,b))
      
      return loss_tot
      '''return self.Loss_self(source_imgs,
                            target_img,
                            depth_map,
                            K,T,a,b,
                            uncertainty_map) '''
    
class TotalLoss(nn.Module):
    def __init__(self, beta=par.beta,batch_size=par.batch_size):
        super(TotalLoss, self).__init__()
    
    
    '''def Loss_total(self,source_imgs,target_img,depth_map,K,T,a,b,uncertainty_map,beta,lamb):
      # Clamp uncertainty map to avoid nan values?
      uncertainty_map = torch.clamp(uncertainty_map, 1, None) # Limiting lowerbound to avoid exploding gradients
      
      loss_reg = self.Loss_smooth(target_img,depth_map) + beta*self.Loss_ab(a,b)
      
      loss_tot = self.Loss_self(source_imgs,target_img,depth_map,K,T,a,b,uncertainty_map)
      loss_tot += lamb*loss_reg
      return loss_tot'''
    #@staticmethod
    def forward(self,source_imgs,source_img,scales,stereo_baseline,K,pose_6dof_t_minus_1_t,pose_6dof_t_t_plus_1,a,b,beta,batch_num):
      # source_imgs: the img triplet {I_t-1,I_t,I_t+1} from __getitem__() (pose)
      # source_img: I_t, I_ts from __getitem()__() (depth). Might be best to grab right image too!
      
      loss = 0
      #loss.requires_grad_(True)
      #print(scales)
      
      Loss_total = SingleScaleLoss()
      b_,c_,h_,w_ = source_img[:,:3,:,:].shape
      #min_depth = 1e-1 # Default (see Monodepth2 options.py)
      #max_depth = 100 # Default (see Monodepth2 options.py)
      #min_disp = 1/max_depth
      #max_disp = 1/min_depth
      for key,value in scales.items():
        #print("Scale: " + str(key))
        
        #lamb = 1e-3 * 1/(2**(3-int(key))) # No -1 because keys are (s-1)
        lamb = 1e-3 * 1/(2**(int(key)))
        #print("Lambda: " + str(lamb))
        # Depth and Uncertainty Map Extraction ...
        #depth_map = torch.clamp(80*value[:,:2,:,:], 1e-3, 80) # D_t and D_ts, As recommended by Monodepth2
        
        # Monodepth2
        #depth_map = 1 / (min_disp + (max_disp - min_disp) * value[:,:2,:,:]) # Since you're using Monodepth2 architecture you're actually learning disparity.
        depth_map = value[:,:2,:,:]
        '''print("Depth Min: ")
        print(torch.min(depth_map))
        print("Depth Max: ")
        print(torch.max(depth_map))'''
        #uncertainty_map = torch.clamp(100*value[:,2,:,:], 1, 100)
        #uncertainty_map = torch.clamp(value[:,2,:,:], 1, 80)
        #uncertainty_map_ = torch.clamp(value[:,2,:,:], 1e-1, 1)
        uncertainty_map_ = torch.clamp(value[:,2,:,:], 1e-2, None)
        '''print("Uncertainty Max: ")
        print(torch.max(uncertainty_map_))
        print("Uncertainty Min: ")
        print(torch.min(uncertainty_map_))'''
        '''print("Uncertainty Min: ")
        print(torch.min(uncertainty_map))
        print("Uncertainty Max: ")
        print(torch.max(uncertainty_map))'''
        
        # Resizing to Target Image Size
        #depth_map = Resize(size=(h_,w_))(depth_map)
        #uncertainty_map_ = Resize(size=(h_,w_))(uncertainty_map_)
        #print(depth_map.shape)
        #print(uncertainty_map_.shape)
        depth_map = F.interpolate(depth_map, (h_,w_), mode='bilinear')
        uncertainty_map_ = F.interpolate(torch.unsqueeze(uncertainty_map_,1), (h_,w_), mode='bilinear')
        #print(depth_map.shape)
        #print(uncertainty_map.shape)
        #depth_map = torch.nn.functional.interpolate(depth_map, (h_,w_), mode='area')
        #uncertainty_map = torch.nn.functional.interpolate(torch.unsqueeze(uncertainty_map,1), (h_,w_), mode='area')
        #b_,h_,w_ = uncertainty_map.shape

        # Downscale and Rearrange Image Triplet ...
        '''img_triplet = [Resize(size=(h_,w_))(source_imgs[:,:3,:,:]),
                       Resize(size=(h_,w_))(source_imgs[:,6:9,:,:]),
                       Resize(size=(h_,w_))(source_img[:,3:6,:,:])] # {I_t-1, I_t+1, I_ts}
        target_img = Resize(size=(h_,w_))(source_img[:,:3,:,:]) # I_t'''
        img_triplet = [source_imgs[:,:3,:,:],
                       source_imgs[:,6:9,:,:],
                       source_img[:,3:6,:,:]] # {I_t-1, I_t+1, I_ts}
        target_img = source_img[:,:3,:,:] # I_t

        # Transformation Triplet
        '''T = [util.euler_to_transformation_matrix(pose_6dof_t_minus_1_t[:,:3],
                                                 pose_6dof_t_minus_1_t[:,3:]).to(par.device),
             util.euler_to_transformation_matrix(pose_6dof_t_t_plus_1[:,:3],
                                                 pose_6dof_t_t_plus_1[:,3:]).to(par.device),
             stereo_baseline.to(par.device)]'''
        
        # Loss at Scale s
        #print(type(key))
        scale_ = 1 if key == "0" else 0
        #print(scale_)
        loss += Loss_total(img_triplet,
                           target_img,
                           depth_map,
                           K,
                           pose_6dof_t_minus_1_t,
                           pose_6dof_t_t_plus_1,
                           stereo_baseline,
                           a,
                           b,
                           uncertainty_map_,
                           beta,
                           lamb,
                           int(key)+1,
                           batch_num,
                           scale_) # used to be int(key) + 1
        #print("single scale loss grad function: ")
        #print(loss.grad_fn)
        
    
      return loss/4
