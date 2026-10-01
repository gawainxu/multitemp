import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Layer 1: Convolutional. Input: 1 channel (Grayscale), Output: 6 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)

        # Layer 2: Convolutional. Input: 6, Output: 16 filters
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)

        # Fully Connected Layers
        # 16 * 4 * 4 comes from the spatial reduction after two 5x5 convs and two 2x2 pools
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Convolutional Block 1
        # Original LeNet used Tanh and Average Pooling
        x = F.avg_pool2d(F.tanh(self.conv1(x)), kernel_size=2, stride=2)

        # Convolutional Block 2
        x = F.avg_pool2d(F.tanh(self.conv2(x)), kernel_size=2, stride=2)

        # Flatten the feature maps for the fully connected layers
        x = x.view(-1, 16 * 4 * 4)

        # Fully Connected Block
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)

        return x