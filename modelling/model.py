from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
import cv2
import numpy as np
import torch

from life_expectancy.modelling.config import CONFIG


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class FaceAgeDataset(Dataset):
    def __init__(self, image_paths, ages, life_expectancies, scaling):
        self.image_paths = image_paths
        self.ages = ages
        self.mean_life_expectancy = np.mean(life_expectancies)
        self.deltas = life_expectancies - self.mean_life_expectancy
        self.targets = scaling(self.deltas)
        self.life_expectancies = life_expectancies

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = preprocess(img)
        age = self.ages[idx]
        target = self.targets[idx]
        life_expectancy = self.life_expectancies[idx]

        return img, torch.tensor([age]).float(), torch.tensor([life_expectancy]).float(), torch.tensor([target]).float()


class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained

    def _initialize_weights(self):
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 500)
        self.fc1 = nn.Linear(501, 250)
        self.fc2 = nn.Linear(250, 1)

    def forward(self, img, age):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age), dim=1)  # Concatenate age
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet50(ResNet):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__(pretrained)
        weights = ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.cnn = models.resnet50(weights=weights)
        self._initialize_weights()

        for param in self.cnn.parameters():
            param.requires_grad = False

        for param in self.cnn.layer4.parameters():
            param.requires_grad = True


