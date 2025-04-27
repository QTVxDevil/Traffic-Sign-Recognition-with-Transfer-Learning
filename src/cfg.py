import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTSRB_BASE_DIR = os.path.join(BASE_DIR, 'datasets/raw/GTSRB_Traffic_Sign')
ZALO_BASE_DIR = os.path.join(BASE_DIR, 'datasets/raw/ZAC_Traffic_Sign')

GTSRB_TRAINING_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
GTSRB_TEST_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')

ZALO_RAW_PATH = os.path.join(ZALO_BASE_DIR, 'traffic_train/traffic_train')
ZALO_CROP_OUTPUT_PATH = os.path.join(BASE_DIR, 'datasets/zalo_process')

RESNET_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'logs/resnet.pth')
RESNET_FIGURE_PATH = os.path.join(BASE_DIR, 'figures/resnet')

ZALO_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'logs/zalo_transfer_learning.pth')
ZALO_FIGURE_PATH = os.path.join(BASE_DIR, 'figures/zalo')

IMAGE_SIZE = (64, 64)
GTSRB_NUM_CLASSES = 43
ZALO_NUM_CLASSES = 7
BATCH = 64
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
NORMALIZE_PARAMETER = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PARAMS = {
    'patience': 5,
    'delta': 0.0005,
    'verbose': True
}