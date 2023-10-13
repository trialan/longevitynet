import tqdm
import torch
import matplotlib.pyplot as plt

from life_expectancy.modelling.model import ResNet50
from life_expectancy.modelling.data import get_dataloaders
from life_expectancy.modelling.utils import set_seed, save_model

import warnings
warnings.simplefilter("ignore")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

N_EPOCHS = 15
SEED = 7457769
LR = 1e-3 / 3
BATCH_SIZE = 128
DS_VERSION = "v3"


if __name__ == '__main__':
    set_seed(SEED)
    train_dataloader, test_dataloader = get_dataloaders(DS_VERSION,
                                                        BATCH_SIZE,
                                                        SEED)

    model = ResNet50().to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    test_losses = []

    best_test_loss = float('inf')
    # Training loop
    for epoch in range(N_EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for imgs, _, _, target in tqdm.tqdm(train_dataloader):
            imgs = imgs.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)

        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, _, _, target in tqdm.tqdm(test_dataloader):
                imgs = imgs.to(device)
                target = target.to(device)

                output = model(imgs)
                loss = criterion(output, target)
                test_loss += loss.item()
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, test_loss/len(test_dataloader), epoch)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")

    plt.plot(train_losses, 'r', label='train_loss')
    plt.plot(test_losses, 'g', label='test_loss')
    plt.savefig('loss.png')

