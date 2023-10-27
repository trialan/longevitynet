import os
import torch
from pprint import pprint
from life_expectancy.modelling.utils import min_max_scale, whiten


if "paperspace" in os.getcwd():
    REPO_DIR = "/home/paperspace/life_expectancy"
else:
    REPO_DIR = "/Users/thomasrialan/Documents/code/longevity_project/life_expectancy"

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3/3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v4",
          "MODEL_DEVICE": "cuda" if torch.cuda.is_available() else "mps",
          "DATA_FRACTION": 1.,
          "loss_criterion": torch.nn.MSELoss(),
          "TARGET_SCALER": min_max_scale}


print("=========TRAINING CONFIG=========")
pprint(CONFIG)


