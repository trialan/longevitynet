from tqdm import tqdm
import datetime
import torch
import matplotlib.pyplot as plt
import random
from life_expectancy.modelling.model import ResNet50
from life_expectancy.modelling.data import get_dataloaders
from life_expectancy.modelling.utils import set_seed, save_model
import warnings
warnings.simplefilter("ignore")

device = torch.device("mps")
N_EPOCHS = 3
SEED = 7457769
LR = 1e-3 / 3
BATCH_SIZE = 128
DS_VERSION = "v3"


if __name__ == '__main__':
    set_seed(SEED)
    fractions = [1.0, 0.65, 0.35]
    results = {}
    for fraction in fractions:
        train_dataloader, test_dataloader = get_dataloaders(DS_VERSION,
                                                            BATCH_SIZE,
                                                            SEED)
        N = int(fraction * len(train_dataloader))
        print(f" *** Fraction is {fraction}, {N} batches *** ")

        model = ResNet50().to(device)
        model.train()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_test_loss = float('inf')

        for epoch in range(N_EPOCHS):
            start_time = datetime.datetime.now()
            print(f"Starting epoch {epoch + 1} at", start_time)
            for i, (imgs, _, _, target) in enumerate(tqdm(train_dataloader)):
                if i >= N:#int(0.1*len(train_dataloader)):
                    break

                imgs = imgs.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(imgs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            end_time = datetime.datetime.now()
            print(f"Finished epoch {epoch + 1} at", end_time, "Duration:", end_time - start_time)


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, _, _, target in tqdm(test_dataloader):
                imgs = imgs.to(device)
                target = target.to(device)
                output = model(imgs)
                loss = criterion(output, target)
                test_loss += loss.item()
        test_loss = test_loss / len(test_dataloader)

        print("=======================================")

        results[fraction] = test_loss

    for fraction, loss in results.items():
        print(f"Fraction: {fraction * 100:.0f}%, Test Loss: {loss:.4f}")
