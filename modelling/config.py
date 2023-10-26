import torch
from pprint import pprint
from life_expectancy.modelling.utils import min_max_scale, whiten


""" BEST CONFIG TO DATE

MAE = 6.3 years

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 1e-3 / 3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v4",
          "MODEL_DEVICE": "mps",
          "TRAIN_SIZE_FRACTION": 1.0,
          "loss_criterion": torch.nn.MSELoss(),
          "TARGET_SCALER": min_max_scale,
          "WEIGHT_DECAY": 0., #No decay
          "GRADIENT_ACC_STEPS": 1, #No gradient accumulation
          "DROPOUT": 0., #No dropout
          "LR_DECAY_STEP": 5,
          "LR_DECAY_FACTOR": 1. } #NO annealing
"""

CONFIG = {"N_EPOCHS" : 25,
          "SEED" : 7457769,
          "LR" : 3e-3,
          "BATCH_SIZE" : 128,
          "DS_VERSION" : "v4",
          "MODEL_DEVICE": "mps",
          "TRAIN_SIZE_FRACTION": 1.0,
          "loss_criterion": torch.nn.L1Loss(),
          "TARGET_SCALER": whiten,
          "WEIGHT_DECAY": 0.,
          "GRADIENT_ACC_STEPS": 8,
          "HIDDEN_UNIT_DROPOUT": 0.5,
          "INPUT_UNIT_DROPOUT": 0.2,
          "LR_DECAY_STEP": 5,
          "LR_DECAY_FACTOR": 3.,
          "MAX_NORM": 3.5,
          "BETAS": (0.99, 0.999)}

#Betas: High momentum: Srivastava et al. 2014, Appendix A.

print("=========TRAINING CONFIG=========")
pprint(CONFIG)


