import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from datetime import datetime
from deepface import DeepFace

from life_expectancy import get_repo_dir

DIR_PATH = get_repo_dir()
CKPT_PATH = DIR_PATH + "/saved_model_binaries"

gender_map = {"Man": 0, "Woman": 1}

def get_gender_probs(image):
    out    = DeepFace.analyze(image,
                              actions=["gender"],
                              enforce_detection=False)

    gender_probs = out[0]['gender']
    p_man = gender_probs['Man']
    p_woman = gender_probs['Woman']
    return p_man, p_woman




def min_max_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def undo_min_max_scaling(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def whiten(x):
    return (x - np.mean(x)) / np.std(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_model(model, val_loss, epoch):
    val_loss_str = str(val_loss).replace(".", "p")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(),
               f"{CKPT_PATH}/best_model_{timestamp}_{val_loss_str}.pth")
    print(f"New best model saved at epoch: {epoch+1} with Val Loss: {val_loss}")


