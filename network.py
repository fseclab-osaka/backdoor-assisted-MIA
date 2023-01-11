
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms, models
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import time

import os
import sys


class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.num_class = num_class
        self.fc2 = nn.Linear(512, num_class)

    def forward(self, x, raw=0):
        # x of shape [B, 1, 32, 32]
        x = F.relu(self.conv1(x))  # -> [B, 32, 28, 28]
        x = F.max_pool2d(x, 2)  # -> [B, 32, 14, 14]
        x = F.relu(self.conv2(x))  # -> [B, 64, 10, 10]
        x = F.max_pool2d(x, 2)  # -> [B, 64, 5, 5]
        x = x.view(-1, 64 * 5 * 5)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 512]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "ConvNet"


class ConvNet1d(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 64, 5)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_class)
        self.num_class = num_class

    def forward(self, x, raw=0):

        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 32, 24, 24]
        x = F.max_pool2d(x, 2)  # -> [B, 32, 12, 12]
        x = F.relu(self.conv2(x))  # -> [B, 64, 8, 8]
        x = F.max_pool2d(x, 2)  # -> [B, 64, 4, 4]
        x = x.view(-1, 64 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 512]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "ConvNet1d"
