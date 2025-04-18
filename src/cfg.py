import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GTSRB_BASE_DIR = os.path.join(BASE_DIR, 'datasets/raw/GTSRB_Traffic_Sign')

GTSRB_TRAINING_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Training_Images/GTSRB/Final_Training/Images')
GTSRB_TEST_PATH = os.path.join(GTSRB_BASE_DIR, 'GTSRB_Final_Test_Images/GTSRB/Final_Test/Images')

RESNET_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'logs/resnet.pth')
RESNET_FIGURE_PATH = os.path.join(BASE_DIR, 'figures/resnet')

IMAGE_SIZE = (64, 64)
NUM_CLASSES = 43
BATCH = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
NORMALIZE_PARAMETER = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EARLY_STOPPING_PARAMS = {
    'patience': 5,
    'delta': 0.001,
    'verbose': True
}