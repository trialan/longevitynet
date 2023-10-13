import numpy as np
import torch
import random
from datetime import datetime


CKPT_PATH = "/Users/thomasrialan/Documents/code/longevity_project/saved_model_binaries"


def min_max_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def undo_min_max_scaling(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, test_loss, epoch):
    test_loss_str = str(test_loss).replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(),
               f"{CKPT_PATH}/best_model_{timestamp}_{test_loss_str}.pth")
    print(f"New best model saved at epoch: {epoch+1} with Test Loss: {test_loss}")


