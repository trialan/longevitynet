import timm
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.vgg import VGG16_Weights
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import torch


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

        return img, torch.tensor([age]).float(), torch.tensor([target]).float()


class ResNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained

    def _initialize_weights(self):
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, 500)
        self.fc1 = torch.nn.Linear(501, 250)
        self.fc2 = torch.nn.Linear(250, 1)

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
            param.requires_grad = True

        for param in self.cnn.layer4.parameters():
            param.requires_grad = True


class VGG(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.pretrained = pretrained

    def _initialize_weights(self):
        self.cnn.classifier[6] = torch.nn.Linear(self.cnn.classifier[6].in_features, 500)
        self.fc1 = torch.nn.Linear(501, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def forward(self, img, age):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age), dim=1)  # Concatenate age
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VGG16(VGG):
    def __init__(self, pretrained=True):
        super(VGG16, self).__init__(pretrained)
        self.cnn = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self._initialize_weights()

        for param in self.cnn.features.parameters():
            param.requires_grad = False

        for param in self.cnn.features[-4:].parameters():
            param.requires_grad = True


class EfficientNetCustom(torch.nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True):
        super(EfficientNetCustom, self).__init__()
        self.cnn = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self._initialize_weights()
        self._freeze_layers()

    def _initialize_weights(self):
        self.cnn._fc = torch.nn.Linear(self.cnn._fc.in_features, 500)
        self.fc1 = torch.nn.Linear(501, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def _freeze_layers(self):
        # Freeze all layers in _blocks except the last one
        for i, block in enumerate(self.cnn._blocks):
            if i < len(self.cnn._blocks) - 1:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, img, age):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age), dim=1)  # Concatenate age
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ViTCustom(torch.nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, dropout_prob=0.5):
        super(ViTCustom, self).__init__()
        print(f"======{dropout_prob} DROPOUT=======")
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self._initialize_weights()

        # Define dropout layers
        self.dropout_after_cnn = torch.nn.Dropout(dropout_prob)
        self.dropout_after_relu = torch.nn.Dropout(dropout_prob)

    def _initialize_weights(self):
        self.cnn.head = torch.nn.Linear(self.cnn.head.in_features, 500)
        self.fc1 = torch.nn.Linear(501, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def forward(self, img, age):
        x = self.cnn(img)
        x = self.dropout_after_cnn(x)  # Dropout after CNN
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age), dim=1)  # Concatenate age
        x = torch.relu(self.fc1(x))
        x = self.dropout_after_relu(x)  # Dropout after ReLU
        x = self.fc2(x)
        return x


