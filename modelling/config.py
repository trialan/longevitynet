import torch
from pprint import pprint
from life_expectancy.modelling.utils import min_max_scale, whiten


CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3/3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v7",
          "MODEL_DEVICE": "mps",
          "TRAIN_SIZE_FRACTION": 1.0,
          "loss_criterion": torch.nn.MSELoss(),
          "TARGET_SCALER": min_max_scale}


print("=========TRAINING CONFIG=========")
pprint(CONFIG)


