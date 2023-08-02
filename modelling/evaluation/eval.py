import torch
import glob
import numpy as np
from torch.utils.data import DataLoader, random_split
from life_expectancy.modelling.model import FaceAgeDataset, ResNet50
from life_expectancy.modelling.train import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1


if __name__ == "__main__":
    _, test_dataloader = get_dataloader()

    model = ResNet50()
    model.load_state_dict(torch.load('saved_model_binaries/best_model_20230802_0_0100886231469399994.pth', map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for imgs, _, _, _ in dataloader:
            imgs = imgs.to(device)

            output = model(imgs)
            predictions.append(output.item())

    print(predictions)
