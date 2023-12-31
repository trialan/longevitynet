import os
import torch
from pprint import pprint
from longevitynet.modelling.utils import min_max_scale, whiten
from longevitynet.modelling.model import (ResNet50,
                                             VGG16,
                                             EfficientNetCustom,
                                             ViTCustom)

from longevitynet import get_repo_dir

REPO_DIR = get_repo_dir()

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3 / 3.,
          "BATCH_SIZE" : 128, #128 uses 18GB, i have 46GB
          "DS_VERSION" : "v5",
          "MODEL_DEVICE": "cuda" if torch.cuda.is_available() else "mps",
          "DATA_FRACTION": 1.0,
          "loss_criterion": torch.nn.L1Loss(),
          "TARGET_SCALER": min_max_scale,
          "MODEL_CLASS": ResNet50,
          "WEIGHT_DECAY": 0.0}


print("=========TRAINING CONFIG=========")
pprint(CONFIG)


