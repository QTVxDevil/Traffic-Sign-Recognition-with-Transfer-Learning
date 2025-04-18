import torch
import torch.nn as nn
import torch.nn.functional as F

class Localization(nn.Module):
    def __init__(self, filters_1, filters_2, fc_units, kernel_size=(3, 3), pool_size=(2, 2), input_size=(64, 64)):
        super(Localization, self).__init__()
        self.pool_size = pool_size

        self.localization = nn.Sequential(
            nn.MaxPool2d(pool_size),
            nn.Conv2d(3, filters_1, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(filters_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
            nn.Conv2d(filters_1, filters_2, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(filters_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool_size),
        )

        # Dynamically calculate the flattened size
        dummy_input = torch.zeros(1, 3, *input_size)
        with torch.no_grad():
            flattened_size = self.localization(dummy_input).view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, fc_units)
        self.fc2 = nn.Linear(fc_units, 6)

        self.fc2.weight.data.zero_()
        self.fc2.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x = self.localization(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = F.relu(self.fc1(x))
        theta = self.fc2(x)
        theta = theta.view(-1, 2, 3)
        return theta

class SpatialTransformer(nn.Module):
    def __init__(self, filters_1, filters_2, fc_units, input_size):
        super(SpatialTransformer, self).__init__()
        self.localization = Localization(filters_1, filters_2, fc_units, input_size=input_size)
        self.height, self.width = input_size

    def forward(self, x):
        theta = self.localization(x)
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        x_transformed = F.grid_sample(x, grid, align_corners=False)
        return x_transformed