# Implementation of DeepProbLog
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Datasets

dataset_folder = ".datasets"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

torch_datasets = {
    "train": MNIST(
        root=dataset_folder, train=True, download=True, transform=transform
    ),
    "test": MNIST(
        root=dataset_folder, train=False, download=True, transform=transform
    ),
}


class MNISTNetwork(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNISTNetwork, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x


class MNISTImage:
    def __init__(self, mnist_img):
        self.tensor = mnist_img[0]
        self.label = mnist_img[1]


datasets = {
    "train": [MNISTImage(img) for img in torch_datasets["train"]],
    "test": [MNISTImage(img) for img in torch_datasets["test"]]
}
