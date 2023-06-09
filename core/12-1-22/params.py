# Goal: Central Location for parameters
# Export a yaml or other file for each experiment to help with tracking
import torch
import torchvision
# Compose Transforms : 
from torchvision.transforms import Resize,Normalize,ToTensor

class Params():
    def __init__(self):
        # ------------------------------ Device ---------------------------- #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ------------------------- Dataset Parameters --------------------- #
        # Resize
        self.torch_resize = Resize(size=(256,512)) # Recommended by UnDeepVO authors
        #self.torch_resize = Resize(size=(128,256))
        self.torch_totensor = ToTensor()
        #torch_normalize = Normalize([0.5, 0.5, 0.5], [1, 1, 1])
        # Consider updating! (DeepVO Implementation or your own)
        
        '''self.torch_normalize = Normalize([0.3475, 0.3675, 0.3612], 
                                         [0.3029, 0.3110, 0.3169])'''
        
        
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
        self.kitti_img_dir = r'D:/OSUGrad/Research/Code/Datasets/KITTI/Raw/'
        self.kitti_cam_cal = r'calib_cam_to_cam.txt'
        
        # TODO: 
        # Consider plotting histograms of the data, calculating mean, variance 
        # to see if there's an optimal split of data based on their distributions
        
        # D3VO datasplit:
        # The 'Eigen' split is 61 sequences in the raw dataset for KITTI
        self.kitti_train_split = r'split/KITTI/train_files.txt'
        self.kitti_val_split = r'split/KITTI/val_files.txt'
        self.kitti_test_split = r'split/KITTI/val_files.txt' # TODO: Generate your own test files .txt after training
        # Subsets to test code
        #self.kitti_train_split = r'split/KITTI/train_subset.txt'
        #self.kitti_val_split = r'split/KITTI/val_subset.txt'
        #self.kitti_test_split = r'split/KITTI/val_subset.txt' # TODO: Generate your own test files .txt after training
        
        # ----------------------- EuRoC MAV Data --------------------------- #
        self.euroc_mav_img_dir = r'D:/OSUGrad/Research/Code/Datasets/EuRoC_MAV/'
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
        
        self.euroc_left_cam = r'cam0'
        self.euroc_right_cam = r'cam1'
        
        self.euroc_mav_cam_cal = r'sensor.yaml'
        
        # TODO: 
        # Consider plotting histograms of the data, calculating mean, variance 
        # to see if there's an optimal split of data based on their distributions
        
        # D3VO datasplit:
        # The 'Eigen' split is 61 sequences in the raw dataset for KITTI
        self.euroc_train_split = r'split/EuRoC_MAV/train_files.txt'
        self.euroc_val_split = r'split/EuRoC_MAV/val_files.txt'
        self.euroc_test_split = r'split/EuRoC_MAV/val_files.txt' # TODO: Generate your own test files .txt after training
        
        # ------------------------------------------------------------------ #
        # There was no mention of normalization in the preprocessing, so it might not be needed. Investigate!
        
        self.transform_kitti = torchvision.transforms.Compose([self.torch_resize, 
                                                               self.torch_totensor, 
                                                               self.torch_normalize_kitti])
        
        self.transform_euroc = torchvision.transforms.Compose([self.torch_resize, 
                                                               self.torch_totensor, 
                                                               self.torch_normalize_euroc])
        self.batch_size = 5 # 8 recommended by D3VO
        
        # -------------------------- Loss Parameters ----------------------- #
        self.beta=1e-2
        self.alpha=0.85
        
        # ------------------------- Model Parameters ----------------------- #
        
        # PoseNet
        
        # DepthNet
        
        
        # ------------------------ Training Parameters --------------------- #
        
        # ------------------- Raw GPS Conversion Parameters ---------------- #
        self.earths_radius = 6378137 # Earth's equitorial radius 
        self.eccentricity = 0.0818191908425 # Eccentricity 
        
        

par = Params()