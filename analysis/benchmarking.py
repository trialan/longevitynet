from prettytable import PrettyTable
from tqdm import tqdm
import glob
import numpy as np
import torch

from life_expectancy.modelling.config import CONFIG

BATCH_SIZE = CONFIG["BATCH_SIZE"]
DEVICE = CONFIG["MODEL_DEVICE"]


def print_validation_stats_table(model, val_dataloader, dataset, loss_criterion, benchmarks=True):
    model_stats = get_model_stats(model, dataset, val_dataloader, loss_criterion)
    if benchmarks:
        mean_guess_stats = get_mean_guess_stats(dataset, val_dataloader, loss_criterion)
        random_guess_stats = get_random_guess_stats(dataset, val_dataloader, loss_criterion)
        table = create_results_table(
            model_stats["mse"],
            model_stats["mae"],
            mean_guess_stats["mse"],
            mean_guess_stats["mae"],
            random_guess_stats["mse"],
            random_guess_stats["mae"],
        )
    else:
        table = create_simple_results_table(
            model_stats["mse"],
            model_stats["mae"],
        )
    print(table)
    return model_stats["mse"]


def compute_batchwise_mse(predictions, val_dataloader, loss_criterion):
    cumulative_loss = 0.0
    for i, (_, _, _, batch_targets) in enumerate(val_dataloader):
        cumulative_loss += loss_criterion(
            torch.tensor(
                predictions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype=torch.float32
            ).to(DEVICE),
            batch_targets.to(DEVICE),
        ).item()
    return cumulative_loss / len(val_dataloader)


def compute_batchwise_mae(predictions, val_dataloader):
    cumulative_error = 0.0
    for i, (_, _, _, batch_targets) in enumerate(val_dataloader):
        batch_predictions = torch.tensor(
            predictions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE], dtype=torch.float32
        ).to(DEVICE)
        batch_error = (
            torch.abs(batch_predictions - batch_targets.to(DEVICE)).mean().item()
        )
        cumulative_error += batch_error
    return cumulative_error / len(val_dataloader)


def get_model_stats(model, dataset, val_dataloader, loss_criterion):
    model.eval()

    predictions = []
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(val_dataloader, desc="Computing preds"):
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            predictions.extend(output.cpu().numpy())

    mse = compute_batchwise_mse(predictions, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(predictions, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "mse": mse}
    return stats


def get_mean_guess_stats(dataset, val_dataloader, loss_criterion):
    train_targets = [row[3].item() for row in dataset]
    train_mean_life_expect = np.mean(train_targets)
    mean_guesses = [train_mean_life_expect] * len(val_dataloader.dataset)
    mse = compute_batchwise_mse(mean_guesses, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(mean_guesses, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "mse": mse}
    return stats


def get_random_guess_stats(dataset, val_dataloader, loss_criterion):
    train_targets = [row[3].item() for row in dataset]
    min_life_expect = np.min(train_targets)
    max_life_expect = np.max(train_targets)
    random_guesses = np.random.uniform(
        min_life_expect, max_life_expect, len(val_dataloader.dataset)
    )
    mse = compute_batchwise_mse(random_guesses, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(random_guesses, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "mse": mse}
    return stats


def create_results_table(
    model_mse,
    model_mae,
    mean_guess_mse,
    mean_guess_mae,
    random_guess_mse,
    random_guess_mae,
):
    table = PrettyTable()
    table.field_names = ["Method", "MSE Loss", "Mean Abs. Error / years"]
    table.add_row(["Model", f"{model_mse:.6f}", f"{model_mae:.2f}"])
    table.add_row(["Mean Guess", f"{mean_guess_mse:.6f}", f"{mean_guess_mae:.2f}"])
    table.add_row( ["Random Guess", f"{random_guess_mse:.6f}", f"{random_guess_mae:.2f}"])
    return table


def create_simple_results_table(model_mse, model_mae):
    table = PrettyTable()
    table.field_names = ["Method", "MSE Loss", "Mean Abs. Error / years"]
    table.add_row(["Model", f"{model_mse:.6f}", f"{model_mae:.2f}"])
    return table


def convert_mae_to_years(scaled_mae, dataset):
    unscaled_targets = dataset.deltas
    max_data = max(unscaled_targets)
    min_data = min(unscaled_targets)
    return scaled_mae * (max_data - min_data)


