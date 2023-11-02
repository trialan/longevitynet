import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from datetime import datetime
from deepface import DeepFace
import cv2
from torchvision import models, transforms

from longevitynet import get_repo_dir

DIR_PATH = get_repo_dir()
CKPT_PATH = DIR_PATH + "/saved_model_binaries"


def convert_prediction_to_years(pred, dataset):
    import pdb;pdb.set_trace() 
    unscaled_target = dataset.deltas
    unscaled_delta_pred = undo_min_max_scaling(pred,
                                               min(unscaled_target),
                                               max(unscaled_target))
    mean_life_expect = dataset.mean_life_expectancy
    out = unscaled_delta_pred + mean_life_expect
    return out


def unpack_model_input(data, device):
    """ data is a row of the dataloader """
    imgs = data['img'].to(device)
    ages = data['age'].to(device)
    p_man = data['p_man'].to(torch.float32).to(device).unsqueeze(-1)
    p_woman = data['p_woman'].to(torch.float32).to(device).unsqueeze(-1)
    np_woman = data['neg_p_woman'].to(torch.float32).to(device).unsqueeze(-1)
    np_man = data['neg_p_man'].to(torch.float32).to(device).unsqueeze(-1)
    return imgs, ages, p_man, p_woman, np_man, np_woman


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


def get_gender_probs(image_path):
    out    = DeepFace.analyze(image_path,
                              actions=["gender"],
                              enforce_detection=False)

    gender_probs = out[0]['gender']
    p_man = gender_probs['Man']
    p_woman = gender_probs['Woman']
    return p_man, p_woman


