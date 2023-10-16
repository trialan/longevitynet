from torch.nn import MSELoss
import glob
import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable

from life_expectancy.modelling.data import generate_dataset, _get_train_test_split
from life_expectancy.modelling.model import ResNet50, FaceAgeDataset
from life_expectancy.modelling.train import SEED, DS_VERSION, get_dataloaders, BATCH_SIZE
from life_expectancy.modelling.utils import set_seed

device = torch.device("mps")
MODEL = "/Users/thomasrialan/Documents/code/longevity_project/saved_model_binaries/best_model_20231013-145803_0p01107617188245058.pth"


def get_data():
    test_dataloader = get_dataloaders(DS_VERSION, BATCH_SIZE, SEED)[1]
    dataset = generate_dataset(DS_VERSION)
    return test_dataloader, dataset


def compute_batchwise_mse(predictions, targets, test_dataloader):
    cumulative_loss = 0.0
    for i, (_, _, _, batch_targets) in enumerate(test_dataloader):
        cumulative_loss += MSELoss()(torch.tensor(predictions[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], dtype=torch.float32).to(device), batch_targets.to(device)).item()
    return cumulative_loss / len(test_dataloader)


def get_model_mse(test_dataloader):
    model = ResNet50()
    path = max(glob.glob(MODEL))
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(test_dataloader, desc="Computing preds"):
            imgs = imgs.to(device)
            output = model(imgs)
            predictions.extend(output.cpu().numpy())

    return compute_batchwise_mse(predictions, None, test_dataloader)


def get_mean_guess_mse(dataset, test_dataloader):
    train_targets = [row[3].item() for row in dataset]
    train_mean_life_expect = np.mean(train_targets)
    mean_guesses = [train_mean_life_expect] * len(test_dataloader.dataset)
    return compute_batchwise_mse(mean_guesses, None, test_dataloader)


def get_random_guess_mse(dataset, test_dataloader):
    train_targets = [row[3].item() for row in dataset]
    min_life_expect = np.min(train_targets)
    max_life_expect = np.max(train_targets)
    random_guesses = np.random.uniform(min_life_expect, max_life_expect, len(test_dataloader.dataset))
    return compute_batchwise_mse(random_guesses, None, test_dataloader)


def create_results_table(model_mse, mean_guess_mse, random_guess_mse):
    table = PrettyTable()
    table.field_names = ["Method", "MSE"]
    table.add_row(["Model", f"{model_mse:.6f}"])
    table.add_row(["Mean Guess", f"{mean_guess_mse:.6f}"])
    table.add_row(["Random Guess", f"{random_guess_mse:.6f}"])
    return table


def main():
    set_seed(SEED)
    test_dataloader, dataset = get_data()

    model_mse = get_model_mse(test_dataloader)
    mean_guess_mse = get_mean_guess_mse(dataset, test_dataloader)
    random_guess_mse = get_random_guess_mse(dataset, test_dataloader)

    table = create_results_table(model_mse, mean_guess_mse, random_guess_mse)
    print(table)


if __name__ == '__main__':
    main()


