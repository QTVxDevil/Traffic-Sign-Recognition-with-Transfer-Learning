import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class LocalContrastNormalization(torch.nn.Module):
    def __init__(self, kernel_size=9, epsilon=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.epsilon = epsilon

        coords = torch.arange(kernel_size) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * (kernel_size / 6) ** 2))
        g = g / g.sum()
        self.kernel = g[:, None] @ g[None, :]
        self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel_buffer', self.kernel)

    def forward(self, img):
        if img.ndim != 3:
            raise ValueError("Input must be (C, H, W)")

        C, H, W = img.shape
        out = torch.zeros_like(img)
        padding = self.kernel_size // 2

        for c in range(C):
            channel = img[c:c+1, :, :].unsqueeze(0)

            mean = F.conv2d(channel, self.kernel_buffer, padding=padding)
            centered = channel - mean

            sq = centered ** 2
            std = torch.sqrt(F.conv2d(sq, self.kernel_buffer, padding=padding) + self.epsilon)

            out[c] = (centered / std).squeeze(0)

        return out
