import tqdm
from torch.optim.lr_scheduler import StepLR
import torch
import matplotlib.pyplot as plt
from pprint import pprint

from life_expectancy.analysis.benchmarking import print_validation_stats_table
from life_expectancy.modelling.config import CONFIG
from life_expectancy.modelling.data import get_dataloaders, get_dataset
from life_expectancy.modelling.model import ResNet50, ModifiedEfficientNetPartialFreeze
from life_expectancy.modelling.utils import set_seed, save_model, plot_losses

device = torch.device(CONFIG["MODEL_DEVICE"])


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    dataset = get_dataset(CONFIG)
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset, CONFIG)

    model = ResNet50().to(device)
    criterion = CONFIG["loss_criterion"]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["LR"])

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(CONFIG["N_EPOCHS"]):
        model.train()
        train_loss = 0

        for idx, (imgs, ages, _, target) in enumerate(tqdm.tqdm(train_dataloader)):
            imgs = imgs.to(device)
            ages = ages.to(device)
            target = target.to(device)

            output = model(imgs, ages)
            loss = criterion(output, target)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(train_loss)

        model.eval()
        benchmarks = False  # True if epoch == 0 else False
        val_loss = print_validation_stats_table(
            model, val_dataloader, dataset, criterion, benchmarks=benchmarks
        )
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, val_loss, epoch)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Val loss: {val_loss}"
        )
