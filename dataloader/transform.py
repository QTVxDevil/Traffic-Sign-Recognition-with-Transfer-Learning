from torchvision import transforms
from src.cfg import NORMALIZE_PARAMETER, IMAGE_SIZE
from dataloader.local_contrast_normalization import LocalContrastNormalization

def get_transform(train=True):
    local_contrast = LocalContrastNormalization()

    if train:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-5)),
            transforms.Lambda(lambda x: local_contrast(x)),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std'])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.mean()) / (x.std() + 1e-5)),
            transforms.Lambda(lambda x: local_contrast(x)),
            transforms.Normalize(mean=NORMALIZE_PARAMETER['mean'],
                                 std=NORMALIZE_PARAMETER['std'])
        ])
    return transform
