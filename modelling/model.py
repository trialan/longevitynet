from efficientnet_pytorch import EfficientNet
from torch import nn
from torch.utils.data import Dataset
from torchvision import models, transforms
import cv2
import numpy as np
import torch

from modelling.utils import min_max_scale

# Preprocessing for images
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FaceAgeDataset(Dataset):
    def __init__(self, image_paths, ages, life_expectancies):
        self.image_paths = image_paths
        self.ages = ages
        self.mean_life_expectancy = np.mean(life_expectancies)
        self.deltas = life_expectancies - self.mean_life_expectancy

        self.targets = min_max_scale(self.deltas)
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
        self.fc1 = nn.Linear(501, 250)  # 500 for image features + 1 for age
        self.fc2 = nn.Linear(250, 1)

    def forward(self, img, age):
        x1 = self.cnn(img)
        x = torch.cat((x1, age), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet50(ResNet):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__(pretrained)
        self.cnn = models.resnet50(pretrained=self.pretrained)
        self._initialize_weights()

        for param in self.cnn.parameters():
            param.requires_grad = False

        for param in self.cnn.layer4.parameters():
            param.requires_grad = True


class ResNet101(ResNet):
    def __init__(self, pretrained=True):
        super(ResNet101, self).__init__(pretrained)
        self.cnn = models.resnet101(pretrained=self.pretrained)
        self._initialize_weights()

        for param in self.cnn.parameters():
            param.requires_grad = False

        for param in self.cnn.layer4.parameters():
            param.requires_grad = True


class DenseNet121(nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        # Load pretrained DenseNet-121
        self.base_model = models.densenet121(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last dense block
        for param in self.base_model.features.denseblock4.parameters():
            param.requires_grad = True

        for param in self.base_model.features.denseblock3.parameters():
            param.requires_grad = True

        self.base_model.classifier = nn.Identity()  # Remove the classifier part of DenseNet
        
        # New layers
        self.fc1 = nn.Linear(1024+1, 512)  # 1024 for densenet features + 1 for age
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, img, age):
        x1 = self.base_model(img)
        x = torch.cat((x1, age), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__()

        # Load pretrained VGG16
        self.base_model = models.vgg16(pretrained=pretrained)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers
        for param in self.base_model.features[-4:].parameters():
            param.requires_grad = True

        self.base_model.classifier = nn.Identity()  # Remove the classifier part of VGG16
        
        # New layers
        self.fc1 = nn.Linear(512+1, 256)  # 512 for vgg16 features + 1 for age
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, img, age):
        x1 = self.base_model(img)
        x = torch.cat((x1.view(x1.size(0), -1), age), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()

        # Load pretrained EfficientNet
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last stage
        for param in self.base_model._blocks[-5:].parameters():
            param.requires_grad = True

        self.base_model._fc = nn.Identity()  # Remove the classifier part of EfficientNet

        # New layers
        self.fc1 = nn.Linear(1280+1, 640)  # 1280 for efficientnet features + 1 for age
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(640, 320)
        self.fc3 = nn.Linear(320, 1)

    def forward(self, img, age):
        x1 = self.base_model(img)
        x = torch.cat((x1, age), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

