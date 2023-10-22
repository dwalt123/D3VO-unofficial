# Imports 
#import time
import os
#import sys
import numpy as np
import torch
from torch import nn
import torchlie as lie
#import torchvision
#from torchmetrics.functional import structural_similarity_index_measure as ssim
#import kornia.geometry as kg
#import kornia.losses as kl

#import numpy as np

# Local Imports
from utils import util
from params import par
from datatracker import datalogger

import torch.nn.functional as F
from torch.nn.functional import grid_sample
#from torchvision.transforms.functional import gaussian_blur
from torchvision.transforms import Resize

import wandb
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
h = 256
w = 512
backproject = Backproject(par.batch_size,h,w).to(par.device)
project3d = Project3D(par.batch_size,h,w).to(par.device) 

class ImageWarp(nn.Module):
    # Wraps kornia image warping function into a differentiable class type
    def __init__(self, K_batch, norm, height, width):
        super(ImageWarp, self).__init__()
        
        self.K_batch = K_batch.to(par.device)
        self.K_inv = torch.linalg.pinv(K_batch).to(par.device)
        self.norm = norm
        self.h = height
        self.w = width
        
        #self.backproject = Backproject(par.batch_size,self.h,self.w).to(par.device)
        #self.project3d = Project3D(par.batch_size,self.h,self.w).to(par.device)
        
    #@staticmethod
    def forward(self, src_img, depth_map, pose, cam_type):
        
        # BackProject
        #b,c,h,w = src_img.shape
        #backproject = Backproject(par.batch_size,self.h,self.w).to(par.device)
        # Project 3D
        #project3d = Project3D(par.batch_size,self.h,self.w).to(par.device)
        
        #cam_points = backproject(depth_map, self.K_inv).to(par.device)
        
        # Project 3D
        #project3d = Project3D(par.batch_size,h,w).to(par.device)
        
        # Euler Angle Batch to Transformation Matrix Batch
        '''
        if cam_type == 'mono':
            # Assumes stereo baseline is constant, and T matrix already provided.
            base = torch.unsqueeze(torch.tensor([0,0,0,1]).view(1,4),0)
            T_base = base.repeat(par.batch_size,1,1).to(par.device)
            #pose = pose.to(par.device)
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
        '''
        
        # Assumes stereo baseline is constant, and T matrix already provided.
        # project3D doesn't even use the (4x4) format so it's not needed.
        #base = torch.unsqueeze(torch.tensor([0,0,0,1]).view(1,4),0)
        #T_base = base.repeat(par.batch_size,1,1).to(par.device)
        #pose = util.euler_to_transformation_matrix(pose[:,:3],pose[:,3:]).to(par.device)
        #pose = torch.cat((pose,T_base),1).to(par.device)
        
        #inv_depth = 1.0/depth_map
        #mean_inv_depth = inv_depth.mean(3,True).mean(2,True)
        #pose[:,:3] = pose[:,:3]*mean_inv_depth # providing scale for translation
        
        #pose = lie.SE3.exp(pose) # using new torchlie SE(3) functions
        cam_points = backproject(depth_map, self.K_inv).to(par.device)
        pixs = project3d(cam_points, self.K_batch, pose).to(par.device)
        
        return grid_sample(src_img, pixs, mode='bilinear', padding_mode='border', align_corners=False)

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

ssim = SSIM()

class ResidualLoss(nn.Module):
    def __init__(self, win_size):
        super(ResidualLoss, self).__init__()
        
        self.win_size = win_size
        #self.ssim = SSIM()
        
    #@staticmethod
    def forward(self, target_img, warped_img, alpha):
        
        weighted_l1 = torch.nn.L1Loss(reduction='none')(target_img, warped_img)
        wandb.log({"L1_Loss": torch.mean(weighted_l1).detach().item()})
        
        #weighted_ssim = SSIM()
        #wssim = weighted_ssim(target_img, warped_img)
        wssim = ssim(target_img, warped_img)
        wandb.log({"SSIM": torch.mean(wssim).detach().item()})
        
        #loss_res = alpha*wssim + (1 - alpha)*weighted_l1
        wandb.log({"Residual Loss": torch.mean(alpha*wssim + (1 - alpha)*weighted_l1)})
        return alpha*wssim + (1 - alpha)*weighted_l1

residual_loss = ResidualLoss(11)
K_batch1 = par.K1.unsqueeze(0).repeat(par.batch_size,1,1) if par.K1.shape[0] != par.batch_size else par.K1
K_batch2 = par.K2.unsqueeze(0).repeat(par.batch_size,1,1) if par.K2.shape[0] != par.batch_size else par.K2
warp_mono = ImageWarp(K_batch1,False,h,w).to(par.device)
warp_stereo = ImageWarp(K_batch2,False,h,w).to(par.device)

class LossSelf(nn.Module):
    def __init__(self,height,width):
        super(LossSelf, self).__init__()
        
        #self.residual_loss = ResidualLoss(11)
        self.height = height
        self.width = width
        
    #@staticmethod
    def forward(self,pose_imgs,depth_imgs,depth_map,K,
                pose_6dof_t_minus_1_t, pose_6dof_t_t_plus_1,
                stereo_baseline,a,b,uncer_map, batch_num, scale):
        # using for debugging purposes, must realign after use
        #for i in range(1):
        #K_1 = torch.unsqueeze(K[0][:,:3],0)
        #K_2 = torch.unsqueeze(K[1][:,:3],0)
        #K_batch1 = K_1.repeat(par.batch_size,1,1).to(par.device)
        #K_batch2 = K_2.repeat(par.batch_size,1,1).to(par.device)
        #self.warp_mono = ImageWarp(K_batch1,False,self.height,self.width).to(par.device)
        #self.warp_stereo = ImageWarp(K_batch2,False,self.height,self.width).to(par.device)
        
        # Calculate Residuals
        #b_,c,h,w = depth_imgs["t"].shape # initialize somewhere else, might break computation graph
        
        # Masking Out Over-exposed/Saturated pixels
        '''
            if par.use_ab:
                a1 = (a[0].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
                b1 = (b[0].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
                a2 = (a[1].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
                b2 = (b[1].view(b_,1,1,1))*torch.ones(b_,c,h,w).to(par.device)
            
                delta_max = par.delta_max
                threshold = torch.tensor([delta_max,delta_max,delta_max]).view(1,3,1,1).to(par.device) # channel-wise threshold
        '''
        #alpha = par.alpha
        '''
            K_1 = torch.unsqueeze(K[0][:,:3],0)
            K_2 = torch.unsqueeze(K[1][:,:3],0)
            K_batch1 = K_1.repeat(par.batch_size,1,1).to(par.device)
            K_batch2 = K_2.repeat(par.batch_size,1,1).to(par.device)
        '''
        
        '''
            if par.monodepth_scaling:
                depth_map_ = 1 / (1/100 + (1/1e-1 - 1/100) * depth_map)
            else:
                baseline = torch.abs(stereo_baseline[0,-1])
                #print(baseline)
                f1 = torch.unsqueeze(K[0][0,0],0).to(par.device)
                f2 = torch.unsqueeze(K[1][0,0],0).to(par.device)
                f = torch.cat((f1,f2),0).view(1,2,1,1).to(par.device)
                depth_map_ = baseline*f*depth_map
        '''
        
        depth_map_ = 1.0/100.0 + (1.0/1e-2 - 1.0/100.0) * depth_map
        
        #inv_depth = 1.0/depth_map_
        
        #print(inv_depth.shape)
        
        #mean_inv_depth = inv_depth.mean(3,False).mean(2,False).reshape(par.batch_size,1)
        #pose[:,:3] = pose[:,:3]*mean_inv_depth # providing scale for translation
        
        #warp_mono = ImageWarp(K_batch1,False)
        #warp_stereo = ImageWarp(K_batch2,False)
        #residual_loss = ResidualLoss(11)
        
        
        '''
            if pose_imgs["cam"] == "left":
                depth_map_ = depth_map_[:,0,:,:]
            elif pose_imgs["cam"] == "right":
                depth_map_ = depth_map_[:,1,:,:]
        '''
        #depth_map_ = depth_map_[:,0,:,:]
        
        
        # Should be fine to initialize here because Monodepth2 does it in a for loop
        # so reinitializing should be fine
        
        # I_t-1
        #pose_6dof_t_minus_1_t[:,:3] = pose_6dof_t_minus_1_t[:,:3]*mean_inv_depth
        pose_6dof_t_minus_1_t = lie.SE3.exp(pose_6dof_t_minus_1_t)
        #pose_6dof_t_minus_1_t = torch.randn(par.batch_size,3,4).to(par.device)
        #warped_img1 = torch.randn(par.batch_size,3,self.height,self.width).to(par.device)
        
        warped_img1 = warp_mono(pose_imgs["t-1"], 
                                     depth_map_,
                                     pose_6dof_t_minus_1_t._t,
                                     'mono') #pose_6dof_t_minus_1_t._t
            
        '''
            if par.use_ab:
                res_ssd_1 = ((a1*pose_imgs["t"]+b1)-warped_img1)**2
                #wandb.log({"res_ssd_1": res_ssd_1})
                masked_img1 = torch.where(torch.all(res_ssd_1<threshold,1,keepdim=True), a1*pose_imgs["t"]+b1, pose_imgs["t"])
            else:
                masked_img1 = pose_imgs["t"]
        '''
        masked_img1 = pose_imgs["t"]
        
        res_t_minus_1 = residual_loss(masked_img1, warped_img1, par.alpha)
        res_t_minus_1 = res_t_minus_1.mean(1,True)
        
        # I_t+1
        #pose_6dof_t_t_plus_1[:,:3] = pose_6dof_t_t_plus_1[:,:3]*mean_inv_depth
        pose_6dof_t_t_plus_1 = lie.SE3.exp(pose_6dof_t_t_plus_1)
        #pose_6dof_t_t_plus_1 = torch.randn(par.batch_size,3,4).to(par.device)
        #warped_img2 = torch.randn(par.batch_size,3,self.height,self.width).to(par.device)
        
        warped_img2 = warp_mono(pose_imgs["t+1"], 
                                     depth_map_, 
                                     pose_6dof_t_t_plus_1._t,
                                     'mono') #pose_6dof_t_t_plus_1._t
        
        '''
            if par.use_ab:
                res_ssd_2 = ((a2*pose_imgs["t"]+b2)-warped_img2)**2
                #wandb.log({"res_ssd_2": res_ssd_2})
                masked_img2 = torch.where(torch.all(res_ssd_2<threshold,1,keepdim=True), a2*pose_imgs["t"]+b2, pose_imgs["t"])
            else:
            masked_img2 = pose_imgs["t"]
            '''
        masked_img2 = pose_imgs["t"]
        
        res_t_plus_1 = residual_loss(masked_img2,warped_img2,par.alpha)
        res_t_plus_1 = res_t_plus_1.mean(1,True)
        
        # I_ts
        '''
            baseline_inv = torch.cat((torch.t(stereo_baseline[:3,:3]),
                                  torch.matmul(-1*torch.t(stereo_baseline[:3,:3]),torch.unsqueeze(stereo_baseline[:3,3],1))),1) # (baseline)^-1
        '''
        #print(stereo_baseline)
        #warped_img3 = torch.randn(par.batch_size,3,self.height,self.width).to(par.device)
        #print("Stereo Transformation:")
        #print(stereo_baseline)
        warped_img3 = warp_stereo(depth_imgs["ts"], 
                                       depth_map_,
                                       stereo_baseline.reshape(1,3,4),
                                       'stereo')
        
        res_t_stereo = residual_loss(pose_imgs["t"],warped_img3,par.alpha)
        res_t_stereo = res_t_stereo.mean(1,True)
        res_min, idxes = torch.min(torch.cat((res_t_minus_1,res_t_plus_1,res_t_stereo),1),1)
        #res_min = res_t_stereo
        #res_min = (res_t_minus_1+res_t_plus_1+res_t_stereo)/3
        wandb.log({"mean res_min": torch.mean(res_min[0,:,:]).detach().item()})
        
        
        batch_chkpt = [0,100,250,500,1000,2500,4000]
        
        if scale == 1 and batch_num in batch_chkpt:
            #folder = os.path.join("batch",str(batch_num))
            res_min_wandb = wandb.Image(res_min[0,:,:], caption="Minimum Residual")
            wandb.log({"res_min": res_min_wandb})
            
            depth_wandb= wandb.Image(depth_map_[0,:,:], caption="Monocular Depth")
            wandb.log({"depth_map_": depth_wandb})
            
            tgt_img_wandb = wandb.Image(pose_imgs["t"][0,:,:,:], caption="Target Image")
            wandb.log({"target_img": tgt_img_wandb})

            warped_img_wandb = wandb.Image(warped_img1[0,:,:,:], caption="Warped Mono Image")
            wandb.log({"warped_img": warped_img_wandb})

            warped_stereo_img_wandb = wandb.Image(warped_img3[0,:,:,:], caption="Warped Stereo Image")
            wandb.log({"warped_stereo_img": warped_stereo_img_wandb})

            #src_img_wandb = wandb.Image(pose_imgs["t-1"][0,:,:,:], caption="Source Image")
            #wandb.log({"src_img": src_img_wandb})

            #wandb.Table(columns=["x", "y", "z", "roll", "pitch", "yaw"],
            #data=pose_6dof_t_minus_1_t.log()[0,:].reshape(6).tolist())

            #wandb.log({"pose": pose_6dof_t_minus_1_t.log()[0,:].tolist()})

        '''
            # Right Stereo Image
            #datalogger.write_image(source_imgs[2][0,:,:,:],folder,"right_stereo_img_s" + str(scale) + ".png", None, False)
            #right_stereo_img = wandb.Image(source_imgs[2][0,:,:,:], caption="Right Stereo Image")
            #wandb.log({"right_stereo_img": right_stereo_img})
            
            # Masked Images
            #datalogger.write_image(masked_img1[0,:,:,:],folder,"masked_img1_s" + str(scale) + ".png", None, False)
            #masked_img1_wandb = wandb.Image(masked_img1[0,:,:,:], caption="Masked Image 1")
            #wandb.log({"masked_img1": masked_img1_wandb})
            
            #datalogger.write_image(masked_img2[0,:,:,:],folder,"masked_img2_s" + str(scale) + ".png", None, False)
            #masked_img2_wandb = wandb.Image(masked_img2[0,:,:,:], caption="Masked Image 2")
            #wandb.log({"masked_img2": masked_img2_wandb})
            
            # Residuals
            #print(res_min[0,:,:,:].shape)
            datalogger.write_image(res_min[0,:,:],folder,"res_min_s" + str(scale) + ".png", None, False)
            res_min_wandb = wandb.Image(res_min[0,:,:], caption="Minimum Residual")
            wandb.log({"res_min": res_min_wandb})
            
            
            #datalogger.write_image(res_min_[0,:,:,:],folder,"res_min_mono_s" + str(scale) + ".png", None, False)
            datalogger.write_image(res_t_minus_1[0,:,:,:],folder,"res_t_minus_1_s" + str(scale) + ".png", None, False)
            res_t_minus_1_wandb = wandb.Image(res_t_minus_1[0,:,:,:], caption="t-1 Residual")
            wandb.log({"res_t_minus_1": res_t_minus_1_wandb})
            
            datalogger.write_image(res_t_plus_1[0,:,:,:],folder,"res_t_plus_1_s" + str(scale) + ".png", None, False)
            res_t_plus_1_wandb = wandb.Image(res_t_plus_1[0,:,:,:], caption="t+1 Residual")
            wandb.log({"res_t_plus_1": res_t_plus_1_wandb})
            
            datalogger.write_image(res_t_stereo[0,:,:,:],folder,"res_t_stereo_s" + str(scale) + ".png", None, False)
            res_t_stereo_wandb = wandb.Image(res_t_stereo[0,:,:,:], caption="Stereo Residual with Mono Depth")
            wandb.log({"res_t_stereo": res_t_stereo_wandb})
            
        ''''''datalogger.write_image(res_stereo_depth[0,:,:,:],folder,"res_stereo_depth" + str(scale) + ".png", None, False)
            res_stereo_wandb = wandb.Image(res_stereo_depth[0,:,:,:], caption="Stereo Residual with Stereo Depth")
            wandb.log({"res_stereo_depth": res_stereo_wandb})''''''
            
            # Target Image, Forward Warped Image, Stereo Warped Image
            datalogger.write_image(pose_imgs["t"][0,:,:,:],folder,"rgb.png",None,False)
            target_img_wandb = wandb.Image(pose_imgs["t"][0,:,:,:], caption="Target Image")
            wandb.log({"target_img": target_img_wandb})
            
            datalogger.write_image(warped_img1[0,:,:,:],folder,"warped_img1.png",None,False)
            warped_img1_wandb = wandb.Image(warped_img1[0,:,:,:], caption="Warped Image 1")
            wandb.log({"warped_img1": warped_img1_wandb})
            
            datalogger.write_image(warped_img3[0,:,:,:],folder,"warped_img3.png",None,False)
            warped_img3_wandb = wandb.Image(warped_img3[0,:,:,:], caption="Warped Image 3")
            wandb.log({"warped_img3": warped_img3_wandb})
            
            # Current Monocular, Stereo Depth Map and Uncertainty Map
            if pose_imgs["cam"] == "left":
                datalogger.write_image(depth_map_[0,:,:],folder,"depth_map_s" + str(scale) + ".png","magma", True)
                depth_wandb = wandb.Image(depth_map_[0,:,:], caption="Monocular Depth Map")
                wandb.log({"depth_map": depth_wandb})
            elif pose_imgs["cam"] == "right":
                datalogger.write_image(depth_map_[0,:,:],folder,"stereo_depth_map_s" + str(scale) + ".png","magma", True)
                stereo_depth_wandb = wandb.Image(depth_map_[0,:,:], caption="Stereo Depth Map")
                wandb.log({"stereo_depth_map": stereo_depth_wandb})
            
            if par.use_uncer:
                datalogger.write_image(uncer_map[0,:,:],folder,"uncertainty_map_s" + str(scale) + ".png","viridis", True)
                uncer_wandb = wandb.Image(uncer_map[0,:,:], caption="Uncertainty Map")
                wandb.log({"uncer_map": uncer_wandb})
            
            if par.use_uncer:
            if pose_imgs["cam"] == "right":
                # Warp uncertainty map to right view if stereo training ...
                uncer_map = warp_stereo(uncer_map, 
                                        torch.unsqueeze(depth_map_,1),
                                        stereo_baseline,
                                        'stereo')
                # Log uncertainty map statistics ...
                wandb.log({"uncer max": torch.max(uncer_map).item(),
                           "uncer min": torch.min(uncer_map).item(),
                           "uncer mean": torch.mean(uncer_map).item()})
                
                # Apply to uncertainty map to loss ...
                res_tot = torch.mean(res_min/torch.unsqueeze(uncer_map,1) + torch.log(torch.unsqueeze(uncer_map,1)))
                
            elif pose_imgs["cam"] == "left":
                # Log uncertainty map statistics ...
                wandb.log({"uncer max": torch.max(uncer_map).item(),
                           "uncer min": torch.min(uncer_map).item(),
                           "uncer mean": torch.mean(uncer_map).item()})
                
                # Apply to uncertainty map to loss ...
                res_tot = torch.mean(res_min/torch.unsqueeze(uncer_map,1) + torch.log(torch.unsqueeze(uncer_map,1)))
                
            else:
            res_tot = torch.mean(res_min)'''
        
        res_tot = torch.mean(res_min)

        #res_tot = torch.zeros(1).to(par.device)
        return  res_tot

class LossAB(nn.Module):
    def __init__(self):
        super(LossAB, self).__init__()
        
        self.ones = torch.ones(par.batch_size,1).to(par.device)
        
    #@staticmethod
    def forward(self,a,b):
      #lab = torch.zeros(par.batch_size,1).to(par.device)
      #ones = torch.ones(par.batch_size,1).to(par.device)
      '''
      for j in range(2):
          lab += (a[j] - ones)**2 + b[j]**2
      '''
      lab = (a[0] - self.ones)**2 + b[0]**2 + (a[1] - self.ones)**2 + b[1]**2
      return torch.sum(lab)

class LossSmooth(nn.Module):
    def __init__(self):
        super(LossSmooth, self).__init__()
        
    #@staticmethod
    def forward(self,source_img, depth_map):
      '''
      if par.mean_depth_norm:
          depth_mean = torch.mean(depth_map)
          depth_map = depth_map/depth_mean # mean-normalized inverse depth as proposed by Monodepth2
      '''
      #depth_mean = torch.mean(depth_map)
      #depth_map = depth_map/depth_mean
      
      #depth_grad_x = torch.mean(torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:]),1,keepdim=True)
      #depth_grad_y = torch.mean(torch.abs(depth_map[:, :, :-1, :] - depth_map[: , :, 1:, :]),1,keepdim=True)
      
      depth_grad_x = torch.abs(depth_map[:, :, :, :-1] - depth_map[:, :, :, 1:])
      depth_grad_y = torch.abs(depth_map[:, :, :-1, :] - depth_map[: , :, 1:, :])

      image_grad_x = torch.mean(torch.abs(source_img[:, :, :, :-1] - source_img[:, :, :, 1:]),1, keepdim=True)
      image_grad_y = torch.mean(torch.abs(source_img[:, :, :-1, :] - source_img[:, :, 1:, :]),1, keepdim=True)
      
      #image_grad_x = torch.abs(source_img[:, :, :, :-1] - source_img[:, :, :, 1:])
      #image_grad_y = torch.abs(source_img[:, :, :-1, :] - source_img[:, :, 1:, :])

      depth_grad_x *= torch.exp(-image_grad_x)
      depth_grad_y *= torch.exp(-image_grad_y)
      
      '''
      if par.loss_smooth_reduction == "mean":
          loss_sm = depth_grad_x.mean() + depth_grad_y.mean()
      else:
          # default
          loss_sm = depth_grad_x.sum() + depth_grad_y.sum()
      '''
      loss_sm = depth_grad_x.sum() + depth_grad_y.sum()
      #loss_sm = depth_grad_x.mean() + depth_grad_y.mean()
      wandb.log({"Smoothness Loss": loss_sm})
      return loss_sm

Loss_smooth = LossSmooth()
Loss_ab = LossAB()
Loss_self = LossSelf(h,w)

class SingleScaleLoss(nn.Module):
    def __init__(self,height,width):
        super(SingleScaleLoss, self).__init__()
        #self.Loss_smooth = LossSmooth()
        #self.Loss_ab = LossAB()
        #self.Loss_self = LossSelf(height,width)
    #@staticmethod
    def forward(self,pose_imgs,depth_imgs,depth_map,K,pose_6dof_t_minus_1_t,
                           pose_6dof_t_t_plus_1,stereo_baseline,a,b,
                           uncer_map,beta,lamb,s,batch_num,scale_):
      
      '''
      if depth_imgs["cam"] == "left":
          l_smooth = self.Loss_smooth(depth_imgs["t"],torch.unsqueeze(depth_map[:,0,:,:],1))
      elif depth_imgs["cam"] == "right":
          l_smooth = self.Loss_smooth(depth_imgs["t"],torch.unsqueeze(depth_map[:,1,:,:],1))
      '''  
      
      l_smooth = Loss_smooth(depth_imgs["t"],torch.unsqueeze(depth_map[:,0,:,:],1))
      
      '''
      if par.use_ab:
          l_ab = self.Loss_ab(a,b)
      else:
          l_ab = 0
      '''
      #l_ab = 0
      
      #print(l_ab.grad)
      
      #loss_reg = l_smooth + beta*l_ab # beta*l_ab
      loss_reg = l_smooth # Not predicting affine params right now
      
      
      loss_res = Loss_self(pose_imgs,
                                depth_imgs,
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
      
      loss_tot = loss_res + lamb*loss_reg
      wandb.log({"Loss Regularization": lamb*loss_reg})
      #loss_tot = loss_res
      
      '''
      if scale_ == 1:
          wandb.log({"loss_smooth": l_smooth})
          '''''''
          if par.use_ab:
              wandb.log({"loss_ab": l_ab})''''''
          wandb.log({"loss_reg": loss_reg})
          wandb.log({"loss_res": loss_res})
          wandb.log({"loss_tot": loss_tot})
      '''
      return loss_tot

Loss_total = SingleScaleLoss(h,w)

class TotalLoss(nn.Module):
    def __init__(self, beta=par.beta,batch_size=par.batch_size):
        super(TotalLoss, self).__init__()
        
    #@staticmethod
    def forward(self,pose_imgs,depth_imgs,scales,stereo_baseline,K,
                pose_6dof_t_minus_1_t,pose_6dof_t_t_plus_1,a,b,beta,
                batch_num,dataset_type):
      # source_imgs: the img triplet {I_t-1,I_t,I_t+1} from __getitem__() (pose)
      # source_img: I_t, I_ts from __getitem()__() (depth). Might be best to grab right image too!
      
      loss = 0
      #loss.requires_grad_(True)
      #print(scales)
      
      
      
      #b_,c_,h_,w_ = depth_imgs["t"].shape
      #print("Image Size: (" + str(h_) + "," + str(w_) + ")")
      # Trying original size image, b/c of intrinsics ...
      #b_ = par.batch_size
      #c_ = 3
      '''
      if dataset_type == 'kitti':
          h_ = 376
          w_ = 1241
      elif dataset_type == 'euroc':
          h_ = 480
          w_ = 752
      
      #min_depth = 1e-1 # Default (see Monodepth2 options.py)
      #max_depth = 100 # Default (see Monodepth2 options.py)
      #min_disp = 1/max_depth
      #max_disp = 1/min_depth
      
      # Original Size Images (Intrinsics)
      if par.interpolate_maps:
          pose_imgs["t"] = F.interpolate(pose_imgs["t"], (h_,w_), mode='bilinear', align_corners=False)
          pose_imgs["t+1"] = F.interpolate(pose_imgs["t+1"], (h_,w_), mode='bilinear', align_corners=False)
          pose_imgs["t-1"] = F.interpolate(pose_imgs["t-1"], (h_,w_), mode='bilinear', align_corners=False)
          depth_imgs["t"] = F.interpolate(depth_imgs["t"], (h_,w_), mode='bilinear', align_corners=False)
          depth_imgs["ts"] = F.interpolate(depth_imgs["ts"], (h_,w_), mode='bilinear', align_corners=False)
      else:
          pose_imgs["t"] = Resize(size=(h_,w_))(pose_imgs["t"])
          pose_imgs["t-1"] = Resize(size=(h_,w_))(pose_imgs["t-1"])
          pose_imgs["t+1"] = Resize(size=(h_,w_))(pose_imgs["t+1"])
          depth_imgs["t"] = Resize(size=(h_,w_))(depth_imgs["t"])
          depth_imgs["ts"] = Resize(size=(h_,w_))(depth_imgs["ts"])
            
       '''         
      #h_,w_ = (256,512) # probably not right
      #final_losses = [torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.zeros(1)]
      #Loss_total = SingleScaleLoss(h_,w_)
      
      for key,value in scales.items():
        #for i in range(1):
        
        #key = '0'
        #value = scales[key]

        #print("Scale: " + str(key))
        #print(par.scale_shapes[key])
        #b_,c_,h_,w_ = value.shape
        #print((h_,w_))
        #Loss_total = SingleScaleLoss(h_,w_)
        #lamb = 1e-3 * 1/(2**(3-int(key))) # No -1 because keys are (s-1)
        lamb = 1e-3 * 1/(2**(int(key)))
        #print((h_,w_))
        # Scale intrinsic matrix based on scale
        #Ksc1 = util.scale_intrinsics(K[0], key)
        #print(Ksc1)
        #Ksc2 = util.scale_intrinsics(K[1], key)
        #K_ = [Ksc1,Ksc2]
        K_ = K
        #print("Lambda: " + str(lamb))
        # Depth and Uncertainty Map Extraction ...
        #depth_map = torch.clamp(80*value[:,:2,:,:], 1e-3, 80) # D_t and D_ts, As recommended by Monodepth2
        
        # Monodepth2
        #depth_map = 1 / (min_disp + (max_disp - min_disp) * value[:,:2,:,:]) # Since you're using Monodepth2 architecture you're actually learning disparity.
        #depth_map = value[:,:2,:,:]
        depth_map = value
        
        #uncertainty_map = torch.clamp(100*value[:,2,:,:], 1, 100)
        #uncertainty_map = torch.clamp(value[:,2,:,:], 1, 80)
        #uncertainty_map_ = torch.clamp(value[:,2,:,:], 1e-1, 1)
        '''
        if par.use_uncer:
            uncertainty_map_ = torch.clamp(value[:,2,:,:], par.uncer_low, par.uncer_high)
        else:
            uncertainty_map_ = None
        '''
        uncertainty_map_ = None
        
        '''
        if par.interpolate_maps:
            depth_map = F.interpolate(depth_map, (h_,w_), mode='bilinear', align_corners=False)
            if par.use_uncer:
                uncertainty_map_ = F.interpolate(torch.unsqueeze(uncertainty_map_,1), (h_,w_), mode='bilinear', align_corners=False)
        else:
            depth_map = Resize(size=(h_,w_))(depth_map)
            if par.use_uncer:
                uncertainty_map_ = Resize(size=(h_,w_))(uncertainty_map_)
        '''
        depth_map = F.interpolate(depth_map, (h,w), mode='bilinear', align_corners=False)
        
        # Loss at Scale s
        #print(type(key))
        scale_ = 1 if key == "0" else 0
        #print(scale_)
        #loss = torch.sum(torch.sum(depth_map,dim=3),dim=2).reshape(par.batch_size,1)
        #print(loss.shape)
        #loss = torch.mean(depth_map)
        
        #continue
        '''
        loss = Loss_total(pose_imgs,
                          depth_imgs,
                          depth_map,
                          K_,
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
                          scale_)
        '''
        
        loss += Loss_total(pose_imgs,
                        depth_imgs,
                        depth_map,
                        K_,
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
      
      #del Loss_total
      return loss/4
      #return (final_losses[0]+final_losses[1]+final_losses[2]+final_losses[3])/4
