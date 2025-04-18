import torch.nn as nn
from torchvision import models
from models.STN import SpatialTransformer

class ResNetWithSTN(nn.Module):
    def __init__(self, num_classes, stn_filters=(16, 32), stn_fc_units=128, input_size=(64, 64)):
        super(ResNetWithSTN, self).__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        
        try:
            self.resnet = models.resnet50(pretrained=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet pretrained model: {e}")
        
        # Add Spatial Transformer Network (STN)
        self.stn = SpatialTransformer(
            filters_1=stn_filters[0],
            filters_2=stn_filters[1],
            fc_units=stn_fc_units,
            input_size=input_size
        )

        for param in self.resnet.parameters():
            param.requires_grad = False
        
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        # Apply STN to the input
        x = self.stn(x)
        # Pass through ResNet
        return self.resnet(x)
    
    def unfreeze_layers(self, layer_names=None):
        for name, param in self.resnet.named_parameters():
            if layer_names is None or any(layer in name for layer in layer_names):
                param.requires_grad = True