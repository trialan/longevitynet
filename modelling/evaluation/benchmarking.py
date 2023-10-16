from torch.nn import MSELoss
from prettytable import PrettyTable
import glob
import numpy as np
import torch
from tqdm import tqdm

from life_expectancy.modelling.data import generate_dataset, _get_train_test_split
from life_expectancy.modelling.model import ResNet50, FaceAgeDataset
from life_expectancy.modelling.train import SEED, DS_VERSION, get_dataloaders, BATCH_SIZE
from life_expectancy.modelling.utils import set_seed

device = torch.device("mps")
MODEL = "/Users/thomasrialan/Documents/code/longevity_project/saved_model_binaries/best_model_20231013-145803_0p01107617188245058.pth"

def main():
    set_seed(SEED)

    _, test_dataloader = get_dataloaders(DS_VERSION, BATCH_SIZE, SEED)
    dataset = generate_dataset(DS_VERSION)

    test_targets = []
    for _, _, _, target in tqdm(test_dataloader, desc="Loading targets"):
        test_targets.extend(target.cpu().numpy())

    # Compute batch-wise MSELoss for the model and then average
    cumulative_mse_loss = 0.0
    model = ResNet50()
    path = max(glob.glob(MODEL))
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs, _, _, targets in tqdm(test_dataloader, desc="Computing preds"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            cumulative_mse_loss += MSELoss()(outputs, targets.to(device)).item()
    mse_loss = cumulative_mse_loss / len(test_dataloader)

    # Baseline Predictions
    train_targets = [row[3].item() for row in dataset]
    train_mean_life_expect = np.mean(train_targets)

    # Compute batch-wise MSELoss for mean_guess and then average
    cumulative_mean_guess_loss = 0.0
    mean_guesses = [train_mean_life_expect] * BATCH_SIZE
    for _, _, _, targets in tqdm(test_dataloader, desc="Mean guess"):
        cumulative_mean_guess_loss += MSELoss()(torch.tensor(mean_guesses[:len(targets)], dtype=torch.float32).to(device), targets.to(device)).item()
    mean_guess_mse = cumulative_mean_guess_loss / len(test_dataloader)

    # Compute batch-wise MSELoss for random_guess and then average
    min_life_expect = np.min(train_targets)
    max_life_expect = np.max(train_targets)
    cumulative_random_guess_loss = 0.0
    for _, _, _, targets in tqdm(test_dataloader, desc="Random guess"):
        random_guesses_batch = np.random.uniform(min_life_expect, max_life_expect, len(targets))
        cumulative_random_guess_loss += MSELoss()(torch.tensor(random_guesses_batch, dtype=torch.float32).to(device), targets.to(device)).item()
    random_guess_mse = cumulative_random_guess_loss / len(test_dataloader)

    table = PrettyTable()
    table.field_names = ["Method", "MSE"]
    table.add_row(["Model", f"{mse_loss:.6f}"])
    table.add_row(["Mean Guess", f"{mean_guess_mse:.6f}"])
    table.add_row(["Random Guess", f"{random_guess_mse:.6f}"])
    print(table)


if __name__ == '__main__':
    main()


