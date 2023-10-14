import torch
from torch import nn
from params import par
import torch.nn.functional as F
from datasets import kitti_training_data_pose
from utils import util

'''
class PoseDecoder(nn.Module):
    
    # Pose Decoder for PoseNet
    
    def __init__(self, pose_encoder_shape, pose_shape, bias):
        super(PoseDecoder, self).__init__()
        self.pose_encoder_shape = pose_encoder_shape
        self.pose_shape = pose_shape
        self.bias = bias
        
    def forward(self, x):
        return nn.Linear(self.pose_encoder_shape, self.pose_shape, bias=self.bias)
    
class ADecoder(nn.Module):
    
    #Decoder for the Affine transformation parameter, a
    
    def __init__(self, pose_encoder_shape, a_shape, bias):
        super(ADecoder, self).__init__()
        self.pose_encoder_shape = pose_encoder_shape
        self.a_shape = a_shape
        self.bias = bias
        
        self.out = nn.Sequential(nn.Linear(self.pose_encoder_shape, self.a_shape, self.bias),
                            nn.Softplus()
                            )
    def forward(self, x):
        return self.out(x)
    
class BDecoder(nn.Module):
    
    #Decoder for the Affine transformation parameter, b
    
    def __init__(self, pose_encoder_shape, b_shape, bias):
        super(BDecoder, self).__init__()
        self.pose_encoder_shape = pose_encoder_shape
        self.b_shape = b_shape
        self.bias = bias
        
        self.out = nn.Sequential(nn.Linear(self.pose_encoder_shape, self.b_shape, self.bias),
                                 nn.Tanh()
                                 )
    def forward(self, x):
        return self.out(x)
'''

class PoseNet(nn.Module):
    def __init__(self):
        super(PoseNet, self).__init__()
        
        kernel_size = 3
        stride_size = 2
        padding_size = 0
        bias = False # False

        # Layer Dimensions
        in_dim = 6
        c1_out = 16
        c2_out = 32
        c3_out = 64
        c4_out = 128
        c5_out = 256
        c6_out = 512
        c7_out = 1024

        # Pose Network Layers
        self.conv1 = self.conv_block(in_dim,c1_out,kernel_size,stride_size,padding_size,bias)
        self.conv2 = self.conv_block(c1_out,c2_out,kernel_size,stride_size,padding_size,bias)
        self.conv3 = self.conv_block(c2_out,c3_out,kernel_size,stride_size,padding_size,bias)
        self.conv4 = self.conv_block(c3_out,c4_out,kernel_size,stride_size,padding_size,bias)
        self.conv5 = self.conv_block(c4_out,c5_out,kernel_size,stride_size,padding_size,bias)
        self.conv6 = self.conv_block(c5_out,c6_out,kernel_size,stride_size,padding_size,bias)
        self.conv7 = self.conv_block(c6_out,c7_out,kernel_size,stride_size,padding_size,bias)
        #self.avg_pool = nn.AvgPool2d(kernel_size=1,stride=1)
        #self.avg_pool = nn.AvgPool3d(kernel_size=(512,3,3),stride=(0,0,0))
        
        # Out Shapes (Convenience for later output representations)
        self.pose_shape = 6 
        self.a_shape = 1
        self.b_shape = 1

        self.pose_encoder = nn.Sequential(self.conv1,
                                          self.conv2,
                                          self.conv3,
                                          self.conv4,
                                          self.conv5,
                                          self.conv6,
                                          self.conv7
                                          #self.avg_pool
                                          )

        #self.pose_encoder_shape = 3072 # supposed to be 1024 according to table
        self.pose_encoder_shape = 1024
        self.pose_out = self.out_block(self.pose_encoder_shape,
                                       self.pose_shape,
                                       bias,
                                       'pose')
        self.a_out = self.out_block(self.pose_encoder_shape,
                                    self.a_shape,
                                    bias,
                                    'a')
        self.b_out = self.out_block(self.pose_encoder_shape,
                                    self.b_shape,
                                    bias,
                                    'b')
    
    
    def conv_block(self, channel_in, channel_out, kernel_size, stride, padding, bias):
      return nn.Sequential(nn.Conv2d(channel_in, 
                           channel_out, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding, 
                           bias=bias),
                           par.activation
                           ) 
    
    
    def out_block(self, in_dim, out_dim, bias, output_type):
      # output_type : ['pose','a','b']
      
      if output_type == 'pose':
        #out = nn.Linear(in_dim, out_dim, bias=bias)
        out = nn.Conv2d(in_dim,
                        out_dim, 
                        kernel_size=1, 
                        stride=1, 
                        padding=0, 
                        bias=bias)
        
      elif output_type == 'a':
        '''out = nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                            nn.Softplus()
                            )'''
        out = nn.Sequential(nn.Conv2d(in_dim,
                                      out_dim, 
                                      kernel_size=1, 
                                      stride=1, 
                                      padding=0, 
                                      bias=bias),
                            nn.Softplus()
                            )
        
      elif output_type == 'b':
        '''out = nn.Sequential(nn.Linear(in_dim, out_dim, bias=bias),
                            nn.Tanh()
                            )'''
        out = nn.Sequential(nn.Conv2d(in_dim,
                                      out_dim, 
                                      kernel_size=1, 
                                      stride=1, 
                                      padding=0, 
                                      bias=bias),
                            nn.Tanh()
                            )
      return out
    
    def forward(self, xp):
        # xp: input for pose (image triplet)
        #xp = xp.type(torch.DoubleTensor).to(par.device)
        #xp.requires_grad_(True)
        #xp.requires_grad_(True)
        # Pose Network
        #encoded_xp = self.pose_encoder(xp).view(par.batch_size,-1)
        
        #xp = self.pose_encoder(xp).view(par.batch_size,-1)
        #xp = (xp - 0.45) / 0.225 # Mimicking ResNet encoder to achieve similar result
        xp = self.pose_encoder(xp)
        #print(xp.shape)
        #print(xp.size())
        xp = F.avg_pool2d(xp,kernel_size=xp.size()[2:]).view(par.batch_size,self.pose_encoder_shape,1,1)
        #print(xp.shape)
        #print(xp.shape)
        #encoded_xp.requires_grad_(True)
        #print("Pose Encoder Gradcheck: ")
        #print(torch.autograd.gradcheck(self.pose_encoder,xp))
        #encoded_xp.requires_grad_(True)
        
        
        #pose = self.pose_out(encoded_xp)
        
        #pose_decoder = PoseDecoder(self.pose_decoder_shape, self.pose_shape, False)
        #pose = pose_decoder(xp)
        #pose = self.pose_out(xp)
        
        #pose.requires_grad_(True)
        #print("Pose Layer Gradcheck: ")
        #print(torch.autograd.gradcheck(self.pose_out, encoded_xp))
        #pose.requires_grad_(True)
        # Affine Transformations
        #a = self.a_out(encoded_xp)
        
        #a_decoder = ADecoder(self.pose_decoder_shape, self.a_shape, False)
        #a = a_decoder(xp)
        #a = self.a_out(xp)
        
        #a = self.a_out(xp).view(par.batch_size,1) # ------------------------ Uncomment for a 
        
        #print(a.shape)
        #a = torch.sum(self.a_out(xp),3).view(par.batch_size,1)
        
        #print(a.shape)
        #a.requires_grad_(True)
        #print("A Layer Gradcheck: ")
        #print(torch.autograd.gradcheck(self.a_out, encoded_xp))
        #a.requires_grad_(True)
        #b = self.b_out(encoded_xp)
        
        #b_decoder =BDecoder(self.pose_decoder_shape, self.b_shape, False)
        #b = b_decoder(xp)
        #b = self.b_out(xp)
        
        #b = self.b_out(xp).view(par.batch_size,1) # ------------------------ Uncomment for b
        
        #b = torch.sum(self.b_out(xp),3).view(par.batch_size,1)
        
        #print(b.shape)
        #b.requires_grad_(True)
        #print("B Layer Gradcheck: ")
        #print(torch.autograd.gradcheck(self.b_out, encoded_xp))
        #b.requires_grad_(True)
        
        #xp = self.pose_out(xp)
        
        #pose = torch.mean(self.pose_out(xp),3).view(par.batch_size,6) # 1e-2
        
        '''
        if par.pose_scaling:
            scale = par.pose_scale
        else:
            scale = 1.0
        '''
        
        #pose = scale*self.pose_out(xp).view(par.batch_size,6)
        #pose = self.pose_out(xp).view(par.batch_size,6)
        xp = self.pose_out(xp).view(par.batch_size,6)
        #print(xp)
        #print(xp.shape)
        # removed pose variable to make pose regression more explicit
        rotation = xp[:,:3]
        translation = xp[:,3:]
        #return translation, rotation, a, b
        return translation, rotation

posenet_model = PoseNet().to(par.device)

'''
10/26/22: Instead of out_block, make classes for pose, a, b so that they
have forward functions. Otherwise the encoder is updating the encoder only to get the
right pose, a, b. Might explain the bad a,b values. Optimizer found that the pose was a better
indicator of a min loss, so it updated the encoder to reflect this, and a/b started to give bad results
'''
'''
# 10/08/22 - Changed ReLU -> LeakyReLU to see if the NaN problem persists!
#example_sequence = torch.randint(0,100,(par.batch_size,)).tolist()
train_data_files = open(par.kitti_train_split,'r')
train_list = train_data_files.readlines()
train_data_files.close()
# Shuffle data
shuffled_list = util.shuffle_sequence(train_list)
example_data,path,num = kitti_training_data_pose.__getitem__(0, shuffled_list)
#print(example_data.shape)
trans_test,rot_test,a,b = posenet_model(example_data[:,:6,:,:])

print("Rotation Shape: ")
print(rot_test.shape)
print("Translation Shape: ")
print(trans_test.shape)
print("Affine Scaling Shape: ")
print(a.shape)
print("Affine Shifting Shape: ")
print(b.shape)
'''
num_params = sum(p.numel() for p in posenet_model.parameters())
print("Number of PoseNet Model Parameters: " + str(num_params))

'''
for p in posenet_model.named_parameters(): 
    print(print(p[0] + ": " + str(p[1].shape)))
'''
#torch.nn.init.orthogonal_(posenet_model.parameters(), gain=1)
