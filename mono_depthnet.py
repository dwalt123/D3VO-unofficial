import torch
import torchvision
from torch import nn
from params import par
import depth_encoder, depth_decoder

# DepthNet

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        
        self.encoder = depth_encoder.ResnetEncoder(18, True).to(par.device)
        self.decoder = depth_decoder.DepthDecoder(self.encoder.num_ch_enc).to(par.device)
    
    def forward(self, xd):
        xd = self.encoder(xd)
        xd = self.decoder(xd)
        return xd
        #return depth_block, uncer2, uncer3, uncer4


depthnet_model = DepthNet().to(par.device)

num_params = sum(p.numel() for p in depthnet_model.parameters())
print("Number of DepthNet Model Parameters: " + str(num_params))

'''
# Monodepth loads the weights in their script as they're getting the layers
# Loading pretrained-weights
depthnet_dict = depthnet_model.state_dict()
resnet_dict = resnet18_model.state_dict()
resnet_dict = {k: v for k, v in resnet_dict.items() if k in depthnet_dict}
depthnet_dict.update(resnet_dict)
depthnet_model.load_state_dict(depthnet_dict)
#print(depthnet_model.state_dict())
'''
'''
sample_img = torch.rand(8,3,256,512).to(par.device)
depth_out = depthnet_model(sample_img)
for k in depth_out:
    print("Key: ")
    print(k)
    print("Value Shape: ")
    print(depth_out[k].shape)
#print(depthnet_model.state_dict())
num_params = sum(p.numel() for p in depthnet_model.parameters())
print("Number of DepthNet Model Parameters: " + str(num_params))
'''
'''
# Output:

Key: 
('disp', 3)
Value Shape: 
torch.Size([8, 3, 32, 64])
Key: 
('disp', 2)
Value Shape: 
torch.Size([8, 3, 64, 128])
Key: 
('disp', 1)
Value Shape: 
torch.Size([8, 3, 128, 256])
Key: 
('disp', 0)
Value Shape: 
torch.Size([8, 3, 256, 512])
Number of DepthNet Model Parameters: 14846564
'''