from prettytable import PrettyTable
from tqdm import tqdm
import glob
import numpy as np
import torch

from life_expectancy.modelling.config import CONFIG

DEVICE = CONFIG["MODEL_DEVICE"]


def print_validation_stats_table(model, dataset_dict, benchmarks=True):
    stats = get_model_stats(model, dataset_dict)
    #mean_stats = get_mean_guess_stats(dataset_dict)
    table = create_simple_results_table([stats])
    print(table)
    return stats["loss"]


def get_model_stats(model, dataset_dict):
    val_dataloader = dataset_dict['dataloaders']['validation']
    val_dataset = dataset_dict['datasets']['validation']

    predictions = compute_predictions(model, val_dataloader)
    loss = compute_batchwise_loss(predictions, val_dataloader)
    mae = compute_batchwise_mae(predictions, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, val_dataset)

    return {"mae": mae_in_years, "loss": loss}


def compute_batchwise_loss(predictions, val_dataloader):
    cumulative_loss = 0.0
    for i, (_, _, batch_targets) in enumerate(val_dataloader):
        _input = np.array(predictions[i * CONFIG["BATCH_SIZE"] : (i + 1) * CONFIG["BATCH_SIZE"]])

        torch_input = torch.tensor(_input, dtype=torch.float32).to(DEVICE)
        cumulative_loss += CONFIG["loss_criterion"](torch_input, batch_targets.to(DEVICE)).item()
    return cumulative_loss / len(val_dataloader)


def compute_batchwise_mae(predictions, val_dataloader):
    cumulative_error = 0.0
    for i, (_, _, batch_targets) in enumerate(val_dataloader):
        batch_predictions = torch.tensor(
            np.array(predictions[i * CONFIG["BATCH_SIZE"] : (i + 1) * CONFIG["BATCH_SIZE"]]),
            dtype=torch.float32,
        ).to(DEVICE)
        batch_error = (
            torch.abs(batch_predictions - batch_targets.to(DEVICE)).mean().item()
        )
        cumulative_error += batch_error
    return cumulative_error / len(val_dataloader)


def compute_predictions(model, dataloader):
    predictions = []
    with torch.no_grad():
        for imgs, ages, _ in tqdm(dataloader, desc="Computing preds"):
            imgs = imgs.to(DEVICE)
            ages = ages.to(DEVICE)
            output = model(imgs, ages)
            predictions.extend(output.cpu().numpy())
    return predictions


def get_mean_guess_stats(dataset_dict):
    print("Computing mean guess stats")
    train_targets = [row[2].item() for row in dataset_dict['datasets']['train']]
    train_mean_life_expect = np.mean(train_targets)
    # Mean guess is mean of the train set for zero leakage.
    mean_guesses = [np.array([train_mean_life_expect], dtype=np.float32)] * len(dataset_dict['dataloaders']['validation'].dataset)
    loss = compute_batchwise_loss(mean_guesses, dataset_dict['dataloaders']['validation'])
    mae = compute_batchwise_mae(mean_guesses, dataset_dict['dataloaders']['validation'])
    # Re-scale MAE based on Val set because it's calculated on that set.
    mae_in_years = convert_mae_to_years(mae, dataset_dict['datasets']['validation'])
    return {"mae": mae_in_years, "loss": loss}


def create_simple_results_table(rows):
    table = PrettyTable()
    table.field_names = ["Method", "Loss", "Mean Abs. Error / years"]
    for row in rows:
        table.add_row(["Model", f"{row['loss']:.6f}", f"{row['mae']:.2f}"])
    return table


def convert_mae_to_years(scaled_mae, dataset):
    unscaled_targets = dataset.deltas
    if "min_max" in str(CONFIG["TARGET_SCALER"]):
        max_data = max(unscaled_targets)
        min_data = min(unscaled_targets)
        return scaled_mae * (max_data - min_data)
    elif "whiten" in str(CONFIG["TARGET_SCALER"]):
        return scaled_mae * np.std(unscaled_targets)
    else:
        AssertionError, "Should have min-max or whitening as scaling"


