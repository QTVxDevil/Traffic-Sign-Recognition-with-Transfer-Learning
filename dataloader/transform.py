from torchvision import transforms
from src.cfg import NORMALIZE_PARAMETER, IMAGE_SIZE

def get_transform(train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std']) 
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE), 
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std']) 
        ])
    return transform
