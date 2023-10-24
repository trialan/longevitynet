import tqdm
import torch
import matplotlib.pyplot as plt
from pprint import pprint

from life_expectancy.analysis.benchmarking import print_validation_stats_table
from life_expectancy.modelling.config import CONFIG
from life_expectancy.modelling.data import get_dataloaders, get_dataset
from life_expectancy.modelling.model import ResNet50
from life_expectancy.modelling.utils import set_seed, save_model

device = torch.device(CONFIG["MODEL_DEVICE"])

print("=========TRAINING CONFIG=========")
pprint(CONFIG)


if __name__ == '__main__':
    set_seed(CONFIG["SEED"])
    print(f"Training DS fraction: {CONFIG['TRAIN_SIZE_FRACTION']}")
    dataset = get_dataset(CONFIG["DS_VERSION"])
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(dataset,
                                                                        CONFIG)
    model = ResNet50().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LR"])
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch in range(CONFIG["N_EPOCHS"]):
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

        model.eval()
        benchmarks = True if epoch == 0 else False
        val_loss = print_validation_stats_table(model,
                                                val_dataloader,
                                                dataset,
                                                criterion,
                                                benchmarks=benchmarks)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, val_loss, epoch)

        print(f"Epoch: {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}, Val loss: {val_loss}")

    plt.plot(train_losses, 'r', label='train_loss')
    plt.plot(val_losses, 'g', label='val_loss')
    plt.savefig('loss.png')


