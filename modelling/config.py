import os
import torch
from pprint import pprint
from life_expectancy.modelling.utils import min_max_scale, whiten
from life_expectancy.modelling.model import (ResNet50,
                                             VGG16,
                                             EfficientNetCustom,
                                             ViTCustom)

from life_expectancy import get_repo_dir

REPO_DIR = get_repo_dir()

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3/3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v4",
          "MODEL_DEVICE": "cuda" if torch.cuda.is_available() else "mps",
          "DATA_FRACTION": 1.,
          "loss_criterion": torch.nn.MSELoss(),
          "TARGET_SCALER": min_max_scale,
          "MODEL_CLASS": ViTCustom}


print("=========TRAINING CONFIG=========")
pprint(CONFIG)


