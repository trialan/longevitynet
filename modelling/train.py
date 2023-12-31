import tqdm
from torch.optim.lr_scheduler import StepLR
import torch
import matplotlib.pyplot as plt
from pprint import pprint

from longevitynet.analysis.eval import print_validation_stats_table
from longevitynet.modelling.config import CONFIG
from longevitynet.modelling.data import get_dataset_dict
from longevitynet.modelling.model import ResNet50, VGG16
from longevitynet.modelling.utils import (set_seed, save_model,
                                             unpack_model_input)

device = torch.device(CONFIG["MODEL_DEVICE"])


if __name__ == "__main__":
    set_seed(CONFIG["SEED"])

    dataset_dict = get_dataset_dict(CONFIG)
    train_dataloader = dataset_dict['dataloaders']['train']

    model = CONFIG["MODEL_CLASS"]().to(device)
    criterion = CONFIG["loss_criterion"]
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"],
                                 weight_decay=CONFIG["WEIGHT_DECAY"])

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(CONFIG["N_EPOCHS"]):
        model.train()
        train_loss = 0

        for idx, data in enumerate(tqdm.tqdm(train_dataloader)):
            model_input = unpack_model_input(data, device)
            target = data['target'].to(device)
            output = model(*model_input)
            loss = criterion(output, target)
            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_loss = print_validation_stats_table(
            model,
            dataset_dict,
            criterion,
        )

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            #save_model(model, val_loss, epoch)

        print(
            f"Epoch: {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Val loss: {val_loss}"
        )
