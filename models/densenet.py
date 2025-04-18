import torch
import torch.nn as nn
from torchvision import models

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        
        try:
            self.densenet = models.densenet121(pretrained=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load DenseNet pretrained model: {e}")
        
        for param in self.densenet.parameters():
            param.requires_grad = False 
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_features, num_classes) 

    def forward(self, x):
        return self.densenet(x)
    
    def unfreeze_layers(self, layer_names=None):
        for name, param in self.densenet.named_parameters():
            if layer_names is None or any(layer in name for layer in layer_names):
                param.requires_grad = True
