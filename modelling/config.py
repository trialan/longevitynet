import torch
from pprint import pprint
from life_expectancy.modelling.utils import min_max_scale, whiten

REPO_DIR = "/Users/thomasrialan/Documents/code/longevity_project/life_expectancy"

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3/3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v5",
          "MODEL_DEVICE": "mps",
          "DATA_FRACTION": 0.01,
          "loss_criterion": torch.nn.MSELoss(),
          "TARGET_SCALER": min_max_scale}


print("=========TRAINING CONFIG=========")
pprint(CONFIG)


