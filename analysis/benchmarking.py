from prettytable import PrettyTable
from tqdm import tqdm
import glob
import numpy as np
import torch

from life_expectancy.modelling.config import CONFIG
from life_expectancy.modelling.data import get_dataloaders, get_dataset

BATCH_SIZE = CONFIG["BATCH_SIZE"]
DEVICE = CONFIG["MODEL_DEVICE"]


def print_validation_stats_table(model, val_dataloader, dataset, loss_criterion, benchmarks=True):
    model_stats = get_model_stats(model, dataset, val_dataloader, loss_criterion)
    if benchmarks:
        mean_guess_stats = get_mean_guess_stats(dataset, val_dataloader, loss_criterion)
        random_guess_stats = get_random_guess_stats(dataset, val_dataloader, loss_criterion)
        table = create_results_table(
            model_stats["loss"],
            model_stats["mae"],
            mean_guess_stats["loss"],
            mean_guess_stats["mae"],
            random_guess_stats["loss"],
            random_guess_stats["mae"],
        )
    else:
        table = create_simple_results_table(
            model_stats["loss"],
            model_stats["mae"],
        )
    print(table)
    return model_stats["loss"]


def compute_batchwise_loss(predictions, val_dataloader, loss_criterion):
    cumulative_loss = 0.0
    for i, (_, _, _, batch_targets) in enumerate(val_dataloader):
        _input = np.array(predictions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE])

        torch_input = torch.tensor(_input, dtype=torch.float32).to(DEVICE)
        cumulative_loss += loss_criterion(torch_input,
                                          batch_targets.to(DEVICE)).item()
    return cumulative_loss / len(val_dataloader)


def compute_batchwise_mae(predictions, val_dataloader):
    cumulative_error = 0.0
    for i, (_, _, _, batch_targets) in enumerate(val_dataloader):
        batch_predictions = torch.tensor(
            np.array(predictions[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]), dtype=torch.float32
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
        for imgs, ages, _, _ in tqdm(val_dataloader, desc="Computing preds"):
            imgs = imgs.to(DEVICE)
            ages = ages.to(DEVICE)
            output = model(imgs, ages)
            predictions.extend(output.cpu().numpy())
    loss = compute_batchwise_loss(predictions, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(predictions, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "loss": loss}
    return stats


def get_mean_guess_stats(dataset, val_dataloader, loss_criterion):
    print("Computing mean guess stats")
    train_targets = [row[3].item() for row in dataset]
    train_mean_life_expect = np.mean(train_targets)
    mean_guesses = [np.array([train_mean_life_expect], dtype=np.float32)] * len(val_dataloader.dataset)
    loss = compute_batchwise_loss(mean_guesses, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(mean_guesses, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "loss": loss}
    return stats


def get_random_guess_stats(dataset, val_dataloader, loss_criterion):
    print("Computing random guess stats")
    train_targets = [row[3].item() for row in dataset]
    min_life_expect = np.min(train_targets)
    max_life_expect = np.max(train_targets)
    random_guesses = np.random.uniform(
        min_life_expect, max_life_expect, len(val_dataloader.dataset)
    )
    random_guesses = [np.array([r], dtype=np.float32) for r in random_guesses]
    loss = compute_batchwise_loss(random_guesses, val_dataloader, loss_criterion)
    mae = compute_batchwise_mae(random_guesses, val_dataloader)
    mae_in_years = convert_mae_to_years(mae, dataset)
    stats = {"mae": mae_in_years, "loss": loss}
    return stats


def create_results_table(
    model_loss,
    model_mae,
    mean_guess_loss,
    mean_guess_mae,
    random_guess_loss,
    random_guess_mae,
):
    table = PrettyTable()
    table.field_names = ["Method", "Loss", "Mean Abs. Error / years"]
    table.add_row(["Model", f"{model_loss:.6f}", f"{model_mae:.2f}"])
    table.add_row(["Mean Guess", f"{mean_guess_loss:.6f}", f"{mean_guess_mae:.2f}"])
    table.add_row( ["Random Guess", f"{random_guess_loss:.6f}", f"{random_guess_mae:.2f}"])
    return table


def create_simple_results_table(model_loss, model_mae):
    table = PrettyTable()
    table.field_names = ["Method", "Loss", "Mean Abs. Error / years"]
    table.add_row(["Model", f"{model_loss:.6f}", f"{model_mae:.2f}"])
    return table


def convert_mae_to_years(scaled_mae, dataset):
    unscaled_targets = dataset.deltas
    if True:
        print("Assuming Min-Max scaling of deltas")
        max_data = max(unscaled_targets)
        min_data = min(unscaled_targets)
        return scaled_mae * (max_data - min_data)
    if False:
        print("Assuming whiten delta scaling")
        return scaled_mae * np.std(unscaled_targets)


def load_model(model_class, model_path, *args, **kwargs):
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(model_path))
    return model


if __name__ == '__main__':
    from life_expectancy.modelling.model import ResNet50, ModifiedEfficientNetPartialFreeze, ModifiedVGG16Freeze, EnsembleModel
    model_names = [
                "saved_model_binaries/best_efficientnet_6epochs_0p01145232915330459.pth",
                "saved_model_binaries/best_resnet50_2410_VL_0p00747218941721846.pth"]

    en_model = load_model(ModifiedEfficientNetPartialFreeze, model_names[0]).to("mps")
    rn_model = load_model(ResNet50, model_names[1]).to("mps")
    ensemble = EnsembleModel([en_model, rn_model])

    dataset = get_dataset(CONFIG["DS_VERSION"])
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset,
                                                                        CONFIG)
    criterion = CONFIG["loss_criterion"]
    print_validation_stats_table(ensemble, val_dataloader, dataset, criterion, False)


