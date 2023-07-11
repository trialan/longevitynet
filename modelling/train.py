import tqdm
import glob
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib.pyplot as plt

from modelling.model import FaceAgeDataset, ResNet50

import warnings
warnings.simplefilter("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPOCHS = 15
SEED = 7457769
LR = 1e-3 / 3
BATCH_SIZE = 128
TEST_SET_RATIO = 0.2


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, test_loss):
    test_loss_str = str(test_loss).replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f"saved_model_binaries/best_model_{timestamp}_{test_loss_str}.pth")
    print(f"New best model saved at epoch: {epoch+1} with Test Loss: {test_loss}")
    

def get_dataloaders():
    dataset = generate_dataset()
    train_dataset, test_dataset = _get_train_test_split(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader


def generate_dataset():
    image_paths = np.array(glob.glob('modelling/datasets/dataset_v2/*.jpg'))
    image_dates = np.array([int(p.split('data:')[-1][:-4]) for p in image_paths])
    death_dates = np.array([int(p.split('death:')[-1][:4]) for p in image_paths])
    birth_dates = np.array([int(p.split('birth:')[-1][:4]) for p in image_paths])
    ages = np.array([img_date - birth_date for img_date, birth_date in zip(image_dates, birth_dates)])
    life_expectancies = np.array([death - date for death, date in zip(death_dates, image_dates)])

    good_ixs = np.where(life_expectancies > 0)[0]

    dataset = FaceAgeDataset(image_paths[good_ixs], ages[good_ixs], life_expectancies[good_ixs])
    return dataset


def _get_train_test_split(dataset):
    num_test = int(TEST_SET_RATIO * len(dataset))
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test],
                                               generator=torch.Generator().manual_seed(SEED))
    return train_dataset, test_dataset



if __name__ == '__main__': 
    set_seed(SEED)
    train_dataloader, test_dataloader = get_dataloaders()

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
        for imgs, age, life_expectancy, target in tqdm.tqdm(train_dataloader):
            imgs = imgs.to(device)
            age = age.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            output = model(imgs, age)
            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)

        # Evaluate
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for imgs, age, life_expectancy, target in tqdm.tqdm(test_dataloader):
                imgs = imgs.to(device)
                age = age.to(device)
                target = target.to(device)

                output = model(imgs, age)
                loss = criterion(output, target)
                test_loss += loss.item()
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_model(model, test_loss/len(test_dataloader))

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")

    plt.plot(train_losses, 'r', label='train_loss')
    plt.plot(test_losses, 'g', label='test_loss')
    plt.savefig('loss.png')

