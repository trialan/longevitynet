from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.vgg import VGG16_Weights
from efficientnet_pytorch import EfficientNet
import cv2
import numpy as np
import torch

if torch.cuda.is_available():
    import timm

from life_expectancy.modelling.utils import get_gender_probs


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class FaceAgeDataset(Dataset):
    def __init__(self, image_paths, ages, life_expectancies, man_probs, woman_probs, scaling):
        self.image_paths = image_paths
        self.ages = ages
        self.mean_life_expectancy = np.mean(life_expectancies)
        self.deltas = life_expectancies - self.mean_life_expectancy
        self.targets = scaling(self.deltas)
        self.life_expectancies = life_expectancies
        self.man_probs = man_probs
        self.woman_probs = woman_probs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = preprocess(img)
        age = self.ages[idx]
        target = self.targets[idx]
        life_expectancy = self.life_expectancies[idx]
        p_man = self.man_probs[idx]
        p_woman = self.woman_probs[idx]
        item = {'img': img,
                'age': torch.tensor([age]).float(),
                'p_man': p_man,
                'p_woman': p_woman,
                'neg_p_man': 1 - p_man,
                'neg_p_woman': 1 - p_woman,
                'target': torch.tensor([target]).float()}
        return item


class ResNet(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained

    def _initialize_weights(self):
        self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, 500)
        self.fc1 = torch.nn.Linear(505, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def forward(self, img, age, p_man, p_woman, np_man, np_woman):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age, p_man, p_woman, np_man, np_woman), dim=1)
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


class VGG(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.pretrained = pretrained

    def _initialize_weights(self):
        self.cnn.classifier[6] = torch.nn.Linear(self.cnn.classifier[6].in_features, 500)
        self.fc1 = torch.nn.Linear(505, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def forward(self, img, age, p_man, p_woman, np_man, np_woman):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age, p_man, p_woman, np_man, np_woman), dim=1)
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
    def __init__(self, model_name='efficientnet-b1', pretrained=True):
        super(EfficientNetCustom, self).__init__()
        self.cnn = EfficientNet.from_pretrained(model_name) if pretrained else EfficientNet.from_name(model_name)
        self._initialize_weights()
        self._freeze_layers()

    def _initialize_weights(self):
        self.cnn._fc = torch.nn.Linear(self.cnn._fc.in_features, 500)
        self.fc1 = torch.nn.Linear(505, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def _freeze_layers(self):
        # Freeze all layers in _blocks except the last one
        for i, block in enumerate(self.cnn._blocks):
            if i < len(self.cnn._blocks) - 1:
                for param in block.parameters():
                    param.requires_grad = False

    def forward(self, img, age, p_man, p_woman, np_man, np_woman):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age, p_man, p_woman, np_man, np_woman), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ViTCustom(torch.nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTCustom, self).__init__()
        self.cnn = timm.create_model(model_name, pretrained=pretrained)
        self._initialize_weights()


    def _initialize_weights(self):
        self.cnn.head = torch.nn.Linear(self.cnn.head.in_features, 500)
        self.fc1 = torch.nn.Linear(505, 250)
        self.fc2 = torch.nn.Linear(250, 1)

    def forward(self, img, age, p_man, p_woman, np_man, np_woman):
        x = self.cnn(img)
        x = torch.flatten(x, 1)  # Flatten the CNN output
        x = torch.cat((x, age, p_man, p_woman, np_man, np_woman), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


