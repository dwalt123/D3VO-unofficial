import torch
import torchvision
from torch import nn
from params import par

# ResNet-18 with ImageNet Initialization for DepthNet
resnet18_model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
#print(resnet18_model)

# Upsampling and Concatenation Network
class UpConcat(nn.Module):
  ''' Upscaling and Concatenating Layer'''
  def __init__(self, in_channels, out_channels, layer):
    super(UpConcat, self).__init__()
    
    kernel_size=(3,3)
    stride=(1,1)
    padding=(1,1)
    self.elu = nn.ELU(inplace=True)
    self.upsample = nn.Upsample(scale_factor=2)
    self.upconv = nn.Conv2d(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
    
    self.econv_ = layer
    
   
  
  def forward(self,up,i):
    up_ = self.upsample(self.elu(self.upconv(up)))
    i_ = i
    
    return torch.cat((up_,i_),1)



# DepthNet

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()

        kernel_size = (3,3)
        stride = (1,1)
        padding = (1,1)

        # Depth Encoder (no avgpool,fc)
        self.depth_encoder = nn.Sequential(resnet18_model.conv1,
                                           resnet18_model.bn1,
                                           resnet18_model.relu,
                                           resnet18_model.maxpool,
                                           resnet18_model.layer1,
                                           resnet18_model.layer2,
                                           resnet18_model.layer3,
                                           resnet18_model.layer4
                                           )
        
        # Depth Decoder
        self.upsample = nn.Upsample(scale_factor=2).to(par.device)
        
        
        # Depth Layer 1
        self.iconv5_ = nn.Sequential(nn.Conv2d(512,
                                               256,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding),
                                     nn.ELU(inplace=True))
        
        # Depth Layer 2
        self.iconv4_ = nn.Sequential(nn.Conv2d(256,
                                               128,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding),
                                     nn.ELU(inplace=True))
        
        self.disp_uncer4 = nn.Sequential(nn.Conv2d(128,
                                                   3,
                                                   kernel_size=kernel_size,
                                                   stride=stride),
                                         nn.Sigmoid())
        
        # Depth Layer 3
        self.iconv3_ = nn.Sequential(nn.Conv2d(128,
                                               64,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding),
                                     nn.ELU(inplace=True)
                                     )
        
        self.disp_uncer3 = nn.Sequential(nn.Conv2d(64,
                                                   3,
                                                   kernel_size=kernel_size,
                                                   stride=stride),
                                         nn.Sigmoid())
        
        # Depth Layer 4
        self.iconv2_ = nn.Sequential(nn.Conv2d(64,
                                               32,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding),
                                     nn.ELU(inplace=True)
                                     )
        self.disp_uncer2 = nn.Sequential(nn.Conv2d(32,
                                                   3,
                                                   kernel_size=kernel_size,
                                                   stride=stride),
                                         nn.Sigmoid())
        

        # Depth Layer 5
        self.upconv1 = nn.Conv2d(32,
                                 16,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding)
        
        self.disp_uncer1 = nn.Sequential(nn.Conv2d(16,
                                                   3,
                                                   kernel_size=kernel_size,
                                                   stride=stride,
                                                   padding=padding),
                                         nn.Sigmoid())
        
        self.depthblock = nn.Sequential(self.upconv1,
                                        nn.ELU(inplace=True),
                                        self.upsample,
                                        nn.ELU(inplace=True),
                                        self.disp_uncer1)
        
    
    def forward(self, xd):
        # xd: input for depth (single image)
        # Reference Paper: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        #xd.requires_grad_(True)
        ########### For Depth Layer 5 ########################
        self.econv1 = nn.Sequential(resnet18_model.conv1,
                                    resnet18_model.bn1,
                                    resnet18_model.relu,
                                    resnet18_model.maxpool)
        conv_layer = nn.Conv2d(64,
                               32,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1)).to(par.device)
        i_ = self.upsample(conv_layer(self.econv1(xd)))
        #######################################################
        self.reslayer1 = resnet18_model.layer1
        #self.reslayer1.requires_grad_(True)
        self.reslayer2 = resnet18_model.layer2
        #self.reslayer2.requires_grad_(True)
        self.reslayer3 = resnet18_model.layer3
        #self.reslayer3.requires_grad_(True)
        self.reslayer4 = resnet18_model.layer4
        #self.reslayer4.requires_grad_(True)
        
        #self.econv1.requires_grad_(True)
        self.econv2 = self.reslayer1(self.econv1(xd))
        #self.econv2.requires_grad_(True)
        self.econv3 = self.reslayer2(self.econv2)
        #self.econv3.requires_grad_(True)
        self.econv4 = self.reslayer3(self.econv3)
        #######################################################
        #self.econv4.requires_grad_(True)
        
        self.elu = nn.ELU(inplace=True)
        self.sig = nn.Sigmoid()
        
        # Depth Network
        #decoder_in = self.depth_encoder(xd)
        xd = self.depth_encoder(xd)
        #decoder_in.requires_grad_(True)
        
        # ResNet Encoder Blocks (econv_)
        '''self.reslayer1 = resnet18_model.layer1
        self.reslayer1.requires_grad_(True)
        self.reslayer2 = resnet18_model.layer2
        self.reslayer2.requires_grad_(True)
        self.reslayer3 = resnet18_model.layer3
        self.reslayer3.requires_grad_(True)
        self.reslayer4 = resnet18_model.layer4
        self.reslayer4.requires_grad_(True)
        
        self.econv1 = nn.Sequential(resnet18_model.conv1,
                                    resnet18_model.bn1,
                                    resnet18_model.relu,
                                    resnet18_model.maxpool)
        self.econv1.requires_grad_(True)
        self.econv2 = self.reslayer1(self.econv1(xd))
        self.econv2.requires_grad_(True)
        self.econv3 = self.reslayer2(self.econv2)
        self.econv3.requires_grad_(True)
        self.econv4 = self.reslayer3(self.econv3)
        self.econv4.requires_grad_(True)'''
        
        # Depth Layer 1
        self.up1 = UpConcat(512,256,resnet18_model.layer3).to(par.device)
        self.up1.requires_grad_(True)
        #self.layer1 = self.iconv5_(self.up1(decoder_in,self.econv4))
        xd = self.iconv5_(self.up1(xd,self.econv4))
        #self.layer1.requires_grad_(True)
        
        # Depth Layer 2
        self.up2 = UpConcat(256,128,resnet18_model.layer2).to(par.device)
        #self.layer2 = self.iconv4_(self.up2(self.layer1,self.econv3))
        xd = self.iconv4_(self.up2(xd,self.econv3))
        #self.layer2.requires_grad_(True)
        #uncer4 = self.disp_uncer4(self.layer2)
        uncer4 = self.disp_uncer4(xd)
        #uncer4.requires_grad_(True)
        
        # Depth Layer 3
        self.up3 = UpConcat(128,64,resnet18_model.layer1).to(par.device)
        #self.layer3 = self.iconv3_(self.up3(self.layer2,self.econv2)).to(par.device)
        xd = self.iconv3_(self.up3(xd,self.econv2)).to(par.device)
        #self.layer3.requires_grad_(True)
        #uncer3 = self.disp_uncer3(self.layer3)
        uncer3 = self.disp_uncer3(xd)
        #uncer3.requires_grad_(True)
        
        # Depth Layer 4
        self.upconv2 = nn.Conv2d(64,
                                 32,
                                 kernel_size=(3,3),
                                 stride=(1,1),
                                 padding=(1,1)).to(par.device)
        
        #up_ = self.upsample(self.elu(self.upconv2(self.layer3)))
        xd = self.upsample(self.elu(self.upconv2(xd)))
        #up_.requires_grad_(True)
        #######################################################################
        '''conv_layer = nn.Conv2d(64,
                               32,
                               kernel_size=(3,3),
                               stride=(1,1),
                               padding=(1,1)).to(par.device) '''
        ################### Adjusting first encoder layer #####################
        #i_ = self.upsample(conv_layer(self.econv1(xd)))
        #i_.requires_grad_(True)
        #ic = torch.cat((up_,i_),1)
        xd = torch.cat((xd,i_),1)
        #ic.requires_grad_(True)
        xd = self.iconv2_(xd)
        #self.layer4 = self.iconv2_(ic)
        #self.layer4.requires_grad_(True)
        uncer2 = self.disp_uncer2(xd)
        #uncer2 = self.disp_uncer2(self.layer4)
        #uncer2.requires_grad_(True)
        
        # Depth Layer 5
        xd = self.depthblock(xd)
        #depth_block = self.depthblock(self.layer4)
        #depth_block.requires_grad_(True)
        return xd, uncer2, uncer3, uncer4
        #return depth_block, uncer2, uncer3, uncer4


depthnet_model = DepthNet().to(par.device)


# Loading pretrained-weights
depthnet_dict = depthnet_model.state_dict()
resnet_dict = resnet18_model.state_dict()
resnet_dict = {k: v for k, v in resnet_dict.items() if k in depthnet_dict}
depthnet_dict.update(resnet_dict)
depthnet_model.load_state_dict(depthnet_dict)
#print(depthnet_model.state_dict())

#print(depthnet_model.state_dict())
num_params = sum(p.numel() for p in depthnet_model.parameters())
print("Number of DepthNet Model Parameters: " + str(num_params))