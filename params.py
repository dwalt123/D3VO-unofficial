# Goal: Central Location for parameters
# Export a yaml or other file for each experiment to help with tracking
import torch
import torchvision
from torch import nn
# Compose Transforms : 
from torchvision.transforms import Resize,Normalize,ToTensor

class Params():
    def __init__(self):
        # ------------------------------ Device ---------------------------- #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ------------------------- Dataset Parameters --------------------- #
        self.scale_shapes = [(256, 512),(128, 256),(64, 128),(32, 64)]
        
        # Resize
        self.torch_resize = Resize(size=(256,512)) # Recommended by UnDeepVO authors
        #self.torch_resize = Resize(size=(128,256))
        self.torch_totensor = ToTensor()
        #torch_normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])
        # Consider updating! (DeepVO Implementation or your own)
        
        '''self.torch_normalize = Normalize([0.3475, 0.3675, 0.3612], 
                                         [0.3029, 0.3110, 0.3169])'''
        
        # Should change to mean: 0.45, std: 0.225 to match Monodepth2
        self.torch_normalize_kitti = Normalize([0.35593085549895237, 
                                                0.3733099391573756, 
                                                0.3580578687417343], 
                                               [0.30696217630394634, 
                                                0.3116588816251746, 
                                                0.31493874056274834])
        
        self.torch_normalize_euroc = Normalize([0.3881124696638174, 
                                                0.3881124696638174, 
                                                0.3881124696638174], 
                                               [0.2660092859261842, 
                                                0.2660092859261842,
                                                0.2660092859261842]) # Update this!
        '''self.torch_normalize_kitti = Normalize([0.485, 
                                                0.456, 
                                                0.406], 
                                               [0.229, 
                                                0.224,
                                                0.225])
        
        self.torch_normalize_euroc = Normalize([0.485, 
                                                0.456, 
                                                0.406], 
                                               [0.229, 
                                                0.224,
                                                0.225])'''
        
        # DataLoaders
        
        # ------------------------- KITTI Data ----------------------------- #
        #self.kitti_img_dir = r'F:/OSUGrad/Research/Code/Datasets/KITTI/Raw/'
        #self.kitti_img_dir_test = r'F:/OSUGrad/Research/Code/Datasets/KITTI/Visual_Odometry/data_odometry_color/dataset/sequences/'
        #self.linux_root = '/media/dannyw/My Passport/'
        #self.windows_root = r'F:/'
        self.os_name = 'linux'
        
        #self.os_root = self.linux_root if self.os_name == 'linux' else self.windows_root
        self.os_root = self.os_name
        #self.kitti_img_dir = self.os_root+'OSUGrad/Research/Code/Datasets/KITTI/Raw/'
        self.kitti_img_dir = '/mnt/disks/datasets/'
        #self.kitti_img_dir_test = self.os_root+'OSUGrad/Research/Code/Datasets/KITTI/Visual_Odometry/data_odometry_color/dataset/sequences/'
        '''
        self.kitti_img_dir_test_seqs = {'00': self.kitti_img_dir_test + '00/',
                                        '01': self.kitti_img_dir_test + '01/',
                                        '02': self.kitti_img_dir_test + '02/',
                                        '03': self.kitti_img_dir_test + '03/',
                                        '04': self.kitti_img_dir_test + '04/',
                                        '05': self.kitti_img_dir_test + '05/',
                                        '06': self.kitti_img_dir_test + '06/',
                                        '07': self.kitti_img_dir_test + '07/',
                                        '08': self.kitti_img_dir_test + '08/',
                                        '09': self.kitti_img_dir_test + '09/',
                                        '10': self.kitti_img_dir_test + '10/',
                                        '11': self.kitti_img_dir_test + '11/',
                                        '12': self.kitti_img_dir_test + '12/',
                                        '13': self.kitti_img_dir_test + '13/',
                                        '14': self.kitti_img_dir_test + '14/',
                                        '15': self.kitti_img_dir_test + '15/',
                                        '16': self.kitti_img_dir_test + '16/',
                                        '17': self.kitti_img_dir_test + '17/',
                                        '18': self.kitti_img_dir_test + '18/',
                                        '19': self.kitti_img_dir_test + '19/',
                                        '20': self.kitti_img_dir_test + '20/',
                                        '21': self.kitti_img_dir_test + '21/'}
        '''
        self.kitti_cam_cal = 'calib_cam_to_cam.txt'
        self.kitti_left_cam = 'image_02'
        self.kitti_right_cam = 'image_03'
        
        # TODO: 
        # Consider plotting histograms of the data, calculating mean, variance 
        # to see if there's an optimal split of data based on their distributions
        
        # D3VO datasplit:
        # The 'Eigen' split is 61 sequences in the raw dataset for KITTI
        self.kitti_train_split = r'split/KITTI/train_files.txt'
        self.kitti_val_split = r'split/KITTI/val_files.txt'
        self.kitti_test_split = r'split/KITTI/val_files.txt' # TODO: Generate your own test files .txt after training
        # Subsets to test code
        #self.kitti_train_split = 'split/KITTI/train_subset.txt'
        #self.kitti_val_split = 'split/KITTI/val_subset.txt'
        #self.kitti_test_split = 'split/KITTI/val_subset.txt' # TODO: Generate your own test files .txt after training
        
        # ----------------------- EuRoC MAV Data --------------------------- #
        self.euroc_mav_img_dir = ""
        #self.euroc_mav_sequences = []
        #self.euroc_left_cam = ''
        #self.euroc_right_cam = ''
        
        #self.euroc_mav_img_dir = self.os_root+'OSUGrad/Research/Code/Datasets/EuRoC_MAV/'
        self.euroc_mav_sequences = ['machine_hall/MH_01_easy/mav0/',
                                    'machine_hall/MH_02_easy/mav0/',
                                    'machine_hall/MH_03_medium/mav0/',
                                    'machine_hall/MH_04_difficult/mav0/',
                                    'machine_hall/MH_05_difficult/mav0/',
                                    'vicon_room1/V1_01_easy/mav0/',
                                    'vicon_room1/V1_02_medium/mav0/',
                                    'vicon_room1/V1_03_difficult/mav0/',
                                    'vicon_room2/V2_01_easy/mav0/',
                                    'vicon_room2/V2_02_medium/mav0/',
                                    'vicon_room2/V2_03_difficult/mav0/']
        
        self.euroc_left_cam = 'cam0'
        self.euroc_right_cam = 'cam1'
        
        self.euroc_mav_cam_cal = 'sensor.yaml'
        
        # TODO: 
        # Consider plotting histograms of the data, calculating mean, variance 
        # to see if there's an optimal split of data based on their distributions
        
        # D3VO datasplit:
        # The 'Eigen' split is 61 sequences in the raw dataset for KITTI
        self.euroc_train_split = 'split/EuRoC_MAV/train_files.txt'
        self.euroc_val_split = 'split/EuRoC_MAV/val_files.txt'
        self.euroc_test_split = 'split/EuRoC_MAV/val_files.txt' # TODO: Generate your own test files .txt after training
        
        # ------------------------------------------------------------------ #
        # There was no mention of normalization in the preprocessing, so it might not be needed. Investigate!
        # It looks like the resnet encoder uses the same normalization as ImageNet implicitly in the encoder,
        # so you should implicitly do it in your PoseNet forward method too.
        
        self.transform_kitti = torchvision.transforms.Compose([self.torch_resize, 
                                                               self.torch_totensor, 
                                                               self.torch_normalize_kitti])
        
        self.transform_euroc = torchvision.transforms.Compose([self.torch_resize, 
                                                               self.torch_totensor, 
                                                               self.torch_normalize_euroc])
        self.batch_size = 8 # 8 recommended by D3VO
        
        # -------------------------- Initial Intrinsics -------------------- #
        self.K1 = torch.tensor([[0.58, 0, 0.5],
                               [0, 1.92, 0.5],
                               [0, 0, 1]]).reshape(3,3)
        self.K2 = torch.tensor([[0.58, 0, 0.5],
                               [0, 1.92, 0.5],
                               [0, 0, 1]]).reshape(3,3)
        # -------------------------- Loss Parameters ----------------------- #
        self.beta=1e-2
        self.alpha=0.85
        
        # ------------------------- Model Parameters ----------------------- #
        
        # ------------------------ Training Parameters --------------------- #
        
        # ------------------- Raw GPS Conversion Parameters ---------------- #
        self.earths_radius = 6378137 # Earth's equitorial radius 
        self.eccentricity = 0.0818191908425 # Eccentricity 
        
        # ------------------------------------------------------------------ #
        # ------------------------- Decisions ------------------------------ #
        # ------------------------------------------------------------------ #
        
        # ------------------------- Datasets ------------------------------- #
        self.normalize_data = False # do it in forwards
        
        # ------------------------- DepthNet ------------------------------- #
        
        self.mono_flag = True
        self.stereo_flag = False
        self.uncer_flag = False
        
        if self.mono_flag and not self.stereo_flag and not self.uncer_flag:
            self.depth_block_channels = 1
        elif self.mono_flag and self.stereo_flag and not self.uncer_flag:
            self.depth_block_channels = 2
        elif self.mono_flag and self.stereo_flag and self.uncer_flag:
            self.depth_block_channels = 3
        else:
            print("Please fix depth block output channel flags!")
            self.depth_block_channels = 0
        
        # ------------------------- PoseNet -------------------------------- #
        
        # Conv Block Activation
        self.block_activation = "ReLU" # ReLU, LeakyReLU, RReLU, ...
        
        if self.block_activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1)
            #self.activation_params = [0.1, False]
            #print(self.activation_params)
        elif self.block_activation == "RReLU":
            self.activation = nn.RReLU()
            #self.activation_params = [0.125, 0.3333333333333, False]
        else:
            self.activation = nn.ReLU() # default
            #self.activation_params = [False]
            
        # Pose Scaling
        self.pose_scaling = False
        self.pose_scale = 0.01
        
        # -------------------------- Loss ---------------------------------- #
        # Interpolate vs. Resize
        self.interpolate_maps = True 
        
        # Uncertainty Map Bounds
        self.uncer_low = 1e-2
        self.uncer_high = None
        
        # Monodepth Scaling
        self.monodepth_scaling = True
        
        # Loss smooth reduction, normalize depth mean
        #self.loss_smooth_reduction = "sum"
        #self.mean_depth_norm = False
        self.loss_smooth_reduction = "sum"
        self.mean_depth_norm = False
        
        # delta_max
        self.delta_max = 0.2
        
        # More Training Settings
        self.use_stereo = False
        self.use_uncer = False
        self.use_ab = False
        
        self.scale_intrinsic_mat = True
        
        # -------------------------- Training ------------------------------ #
        # Adam vs. SGD
        self.optimizer_name = "adam" # adam
        self.momentum = 0.9 # for SGD
        
        
        

par = Params()
